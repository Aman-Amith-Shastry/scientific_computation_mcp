"""
Streamable-HTTP MCP server exposing NumPy/SymPy scientific computation tools.

Serves the MCP endpoint at $MCP_PATH (default /mcp) on $PORT (default 8081), plus a
plain GET /health for the hosting platform's liveness probe. Published to Smithery via
the URL method, so this process is the upstream that Smithery's gateway proxies to.

$MCP_TRANSPORT switches to stdio for clients that spawn the server as a child process
rather than connecting over the network. The tool surface is identical either way; only
the framing differs.

Deployment constraints:
  * Tensors live in process memory between tool calls, so this must run as a single
    always-on instance with sessions enabled (stateless_http=False). Autoscaling to
    more than one replica splits the store and breaks create -> view flows.
  * The store is keyed per MCP session, not global, so concurrent users on the shared
    hosted instance cannot see or clobber each other's tensors.

Usage:
    uv run src/main.py
    PORT=8000 ALLOWED_ORIGINS=https://smithery.ai uv run src/main.py
    MCP_TRANSPORT=stdio uv run src/main.py
"""

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.session import InitializationState, ServerSession
from mcp.types import ToolAnnotations
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from typing import Annotated
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Entry-point bootstrap: allow `python src/main.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import linear_algebra
import vector_calculus
import visualization
from schemas import Tensor

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8081"))
MCP_PATH = os.environ.get("MCP_PATH", "/mcp")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")
# Transport. Defaults to http so the hosted deployment is unaffected; stdio exists for
# clients that spawn the server as a child process and speak over its stdin/stdout,
# which is what Glama's build test and a local `claude mcp add` both do.
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "http").strip().lower()
# Smithery's gateway and the MCP inspector are browser origins; default to open CORS
# and let an operator pin it down without a code change.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


class SessionScopedStore(MutableMapping):
    """Per-session view over the tensor store, presented to tools as a plain dict.

    One hosted instance serves every connected client, so a single module-level dict
    would let one user's `view_tensor` read another's tensors and let colliding names
    overwrite each other. Each MCP session gets its own dict instead, held weakly so it
    is reclaimed when the session closes. Falls back to one shared dict when there is no
    active request context (direct imports, tests).
    """

    def __init__(self):
        self._by_session = WeakKeyDictionary()
        self._fallback = {}

    def _current(self) -> dict:
        try:
            session = mcp.get_context().session
        except (LookupError, AttributeError, ValueError):
            return self._fallback
        if session is None:
            return self._fallback
        return self._by_session.setdefault(session, {})

    def __getitem__(self, key):
        return self._current()[key]

    def __setitem__(self, key, value):
        self._current()[key] = value

    def __delitem__(self, key):
        del self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self):
        return len(self._current())


def _tolerate_missing_initialized_notification() -> None:
    """Accept requests from clients that skip `notifications/initialized`.

    The spec has the client send that notification after `initialize`, and the SDK holds
    the session in `Initializing` until it arrives, raising on anything else. Smithery's
    scanner never sends it: it initializes, then goes straight to tools/list. The raise is
    flattened by shared/session.py into `-32602 Invalid request parameters` with the
    message stripped, so every list request fails identically and the listing reports
    "No capabilities found" despite the tools being registered.

    Promoting `Initializing` -> `Initialized` on the first post-initialize request only
    widens what is accepted, so spec-compliant clients are unaffected: they send the
    notification and reach the same state one step earlier. `initialize` itself is matched
    before the state check, so re-initialization still behaves normally.
    """
    received_request = ServerSession._received_request

    async def _received_request(self, responder):
        if self._initialization_state == InitializationState.Initializing:
            self._initialization_state = InitializationState.Initialized
        return await received_request(self, responder)

    ServerSession._received_request = _received_request


_tolerate_missing_initialized_notification()

# Initialize tensor store
tensor_store = SessionScopedStore()

# Initialize MCP server with streamable HTTP transport. DNS-rebinding protection is
# disabled explicitly: behind Smithery's gateway the Host/Origin headers are the
# proxy's, not this server's, and validating them here rejects every proxied request
# with 421.
mcp = FastMCP(
    "scientific_computations",
    stateless_http=False,
    streamable_http_path=MCP_PATH,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> PlainTextResponse:
    """Liveness probe for the hosting platform."""
    return PlainTextResponse("ok")


# Matrix creation, deletion, and modification
@mcp.tool(annotations=ToolAnnotations(
    title="Create Tensor", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def create_tensor(
    shape: Annotated[list[int], Field(
        min_length=1, description="Tensor shape as a list of dimension sizes, e.g. [2, 3] for a 2x3 matrix.")],
    values: Annotated[list[float], Field(
        min_length=1, description="Flat, row-major list of values; its length must equal the product of shape.")],
    name: Annotated[str, Field(
        description="Name to store the tensor under; other tools take this name as their argument.")],
) -> Tensor:
    """
    Creates a NumPy array (matrix) with a specified shape and values.

    Args:
        shape (list[int]): The shape of the resulting array as a tuple(e.g., (2, 3)).
        values (list[float]): A flat list of values to populate the array.
        name (str): The name of the tensor to be stored.

    Returns:
        Tensor: The stored tensor as nested lists.

    Raises:
        ValueError: If the number of values does not match the product of the shape.
    """
    shape = [int(x) for x in shape]
    values = [float(x) for x in values]

    if len(values) != np.prod(shape):
        raise ValueError("Shape does not match number of values.")
    a = np.array(values).reshape(shape)

    tensor_store[name] = a
    return a.tolist()


@mcp.tool(annotations=ToolAnnotations(
    title="View Tensor", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def view_tensor(
    name: Annotated[str, Field(description="Name of the stored tensor to read back.")],
) -> Tensor:
    """
    Returns an immutable view of a previously stored NumPy tensor from the in-memory tensor store.

    Args:
        name (str): The name of the tensor as stored in the in-store dictionary

    Returns:
        Tensor: The stored tensor as nested lists.

    Raises:
        ValueError: If the tensor name is not found in the store.
    """
    if name not in tensor_store:
        raise ValueError("The tensor name is not found in the store.")

    return tensor_store[name].tolist()


@mcp.resource("data://tensor_store")
def list_tensor_names() -> str:
    """
    Lists the names of all tensors currently stored in the tensor store.

    Returns:
        str: A newline-separated list of tensor names.
    """
    return "\n".join(tensor_store.keys())


@mcp.tool(annotations=ToolAnnotations(
    title="Delete Tensor", readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
def delete_tensor(
    name: Annotated[str, Field(description="Name of the stored tensor to remove from the store.")],
) -> str:
    """
    Deletes a tensor from the in-memory tensor store.

    Args:
        name (str): The name of the tensor to delete.

    Returns:
        str: Confirmation that the tensor was removed.

    Raises:
        ValueError: If the tensor name is not found in the store or if an error occurs during deletion.
    """
    if name not in tensor_store:
        raise ValueError("The tensor name is not found in the store.")

    try:
        tensor_store.pop(name)
    except ValueError as e:
        raise ValueError(f"Error removing tensor:{e}")

    return f"Deleted tensor {name!r}."


# Register additional tools from modules
linear_algebra.register_tools(mcp, tensor_store)
vector_calculus.register_tools(mcp, tensor_store)
visualization.register_tools(mcp)


def main():
    if MCP_TRANSPORT == "stdio":
        # Nothing may be written to stdout but the JSON-RPC stream itself, so no banner
        # or port log here. The per-session tensor store still behaves: stdio serves one
        # session per process, which is the granularity it keys on.
        mcp.run(transport="stdio")
        return

    if MCP_TRANSPORT != "http":
        raise SystemExit(f"MCP_TRANSPORT must be 'http' or 'stdio', got {MCP_TRANSPORT!r}")

    # Run with streamable HTTP transport
    app = mcp.streamable_http_app()

    app = CORSMiddleware(
        app,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # MCP streamable HTTP methods
        allow_headers=["*"],
        expose_headers=["mcp-session-id", "mcp-protocol-version"],
    )

    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)


if __name__ == "__main__":
    main()
