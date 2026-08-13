"""
Shared return types for the MCP tools.

FastMCP derives a tool's `outputSchema` from its return annotation, and only for
annotations Pydantic can model. `np.ndarray` is not one of them: a tool annotated that
way ships no output schema at all, and its result reaches the client as NumPy's repr
string ("[[6 8]\\n [10 12]]") rather than as JSON. Annotating with `Tensor` and
returning `array.tolist()` gives the tool a real schema and a real JSON payload.

The alias is a union rather than `list` because a bare `list` models as an untyped
array, which Pydantic emits no schema for. Ranks 0 through 3 are spelled out because
that is what the tools actually produce — a determinant is rank 0, a cross product is
rank 1, a matrix product is rank 2, and `create_tensor` accepts an arbitrary shape but
nothing in the toolset consumes higher than rank 3.
"""

Tensor = float | list[float] | list[list[float]] | list[list[list[float]]]
