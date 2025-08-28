from typing import Annotated, Optional
import os
import numpy as np
import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware

import linear_algebra
import vector_calculus
import visualization

tensor_store = {}

mcp = FastMCP("scientific_computations")


# Matrix creation, deletion, and modification
@mcp.tool()
def create_tensor(shape: Annotated[list[int], Field(min_items=1, description="Tensor shape as list of integers")],
                  values: Annotated[
                      list[float], Field(min_items=1, description="Flat list of floats to fill the tensor")],
                  name: str) -> np.ndarray:
    """
    Creates a NumPy array (matrix) with a specified shape and values.

    Args:
        shape (list[int]): The shape of the resulting array as a tuple(e.g., (2, 3)).
        values (list[float]): A flat list of values to populate the array.
        name (str): The name of the tensor to be stored.

    Returns:
        np.ndarray: A NumPy array with the specified shape.

    Raises:
        ValueError: If the number of values does not match the product of the shape.
    """

    shape = [int(x) for x in shape]
    values = [float(x) for x in values]

    if len(values) != np.prod(shape):
        raise ValueError("Shape does not match number of values.")
    a = np.array(values).reshape(shape)

    tensor_store[name] = a
    return a


@mcp.tool()
def view_tensor(name: str) -> dict:
    """
    Returns an immutable view of a previously stored NumPy tensor from the in-memory tensor store.

    Args:
        name (str): The name of the tensor as stored in the in-store dictionary
    Returns:
        dict: The in-store dictionary for tensors

    """
    return tensor_store[name]


@mcp.resource("data://tensor_store")
def list_tensor_names() -> str:
    """
    Lists the names of all tensors currently stored in the tensor store.

    Returns:
        str: A newline-separated list of tensor names.
    """
    return "\n".join(tensor_store.keys())


@mcp.tool()
def delete_tensor(name: str):
    """
    Deletes a tensor from the in-memory tensor store.

    Args:
        name (str): The name of the tensor to delete.

    Raises:
        ValueError: If the tensor name is not found in the store or if an error occurs during deletion.
    """

    if name not in tensor_store:
        raise ValueError("One or both tensor names not found in the store.")

    try:
        tensor_store.pop(name)
    except ValueError as e:
        raise ValueError(f"Error removing tensor:{e}")


linear_algebra.register_tools(mcp, tensor_store)
vector_calculus.register_tools(mcp, tensor_store)
visualization.register_tools(mcp)


def main():
    transport_mode = os.getenv("TRANSPORT", "stdio")

    if transport_mode == "http":
        # HTTP mode with config extraction from URL parameters
        print("Character Counter MCP Server starting in HTTP mode...")

        # Setup Starlette app with CORS for cross-origin requests
        app = mcp.streamable_http_app()

        # IMPORTANT: add CORS middleware for browser based clients
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id", "mcp-protocol-version"],
            max_age=86400,
        )

        # Use Smithery-required PORT environment variable
        port = int(os.environ.get("PORT", 8081))
        print(f"Listening on port {port}")

        uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")

    else:
        # Optional: add stdio transport for backwards compatibility
        # You can publish this to uv for users to run locally
        print("Character Counter MCP Server starting in stdio mode...")

        # Run with stdio transport (default)
        mcp.run()


if __name__ == "__main__":
    main()
