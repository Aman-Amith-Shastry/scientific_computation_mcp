from typing import Annotated, Any

import numpy as np
from mcp.types import ToolAnnotations
from pydantic import Field

from schemas import Tensor

# Every tool here is a pure computation over the in-memory store: nothing reaches the
# network, so openWorldHint is False throughout. Only scale_matrix writes back to the
# store, and only when in_place is left at its default.
READ_ONLY = dict(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

# Reused parameter annotations. The Field descriptions are what a calling model reads
# when it picks arguments; the docstring is not carried into the input schema.
NameA = Annotated[str, Field(description="Name of the first stored tensor.")]
NameB = Annotated[str, Field(description="Name of the second stored tensor.")]


def register_tools(mcp, tensor_store):

    # Matrix/tensor operations
    @mcp.tool(annotations=ToolAnnotations(title="Add Matrices", **READ_ONLY))
    def add_matrices(name_a: NameA, name_b: NameB) -> Tensor:
        """
        Adds two stored tensors element-wise, computing name_a + name_b.

        Args:
            name_a (str): The name of the first tensor.
            name_b (str): The name of the second tensor.

        Returns:
            Tensor: The result of element-wise addition.

        Raises:
            ValueError: If the tensor names are not found or shapes are incompatible.
        """
        if name_a not in tensor_store or name_b not in tensor_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.add(tensor_store[name_a], tensor_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error adding tensors: {e}")

        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Subtract Matrices", **READ_ONLY))
    def subtract_matrices(
        name_a: Annotated[str, Field(description="Name of the tensor to subtract from (the minuend).")],
        name_b: Annotated[str, Field(description="Name of the tensor to subtract (the subtrahend).")],
    ) -> Tensor:
        """
        Subtracts one stored tensor from another element-wise, computing name_a - name_b.

        Args:
            name_a (str): The name of the tensor to subtract from (the minuend).
            name_b (str): The name of the tensor to subtract (the subtrahend).

        Returns:
            Tensor: The result of element-wise subtraction.

        Raises:
            ValueError: If the tensor names are not found or shapes are incompatible.
        """
        if name_a not in tensor_store or name_b not in tensor_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.subtract(tensor_store[name_a], tensor_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error subtracting tensors: {e}")

        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Multiply Matrices", **READ_ONLY))
    def multiply_matrices(
        name_a: Annotated[str, Field(description="Name of the left-hand tensor in the product.")],
        name_b: Annotated[str, Field(description="Name of the right-hand tensor in the product.")],
    ) -> Tensor:
        """
        Performs matrix multiplication between two stored tensors, computing name_a @ name_b.

        Args:
            name_a (str): The name of the first tensor.
            name_b (str): The name of the second tensor.

        Returns:
            Tensor: The result of matrix multiplication.

        Raises:
            ValueError: If either tensor is not found or their shapes are incompatible.
        """
        if name_a not in tensor_store or name_b not in tensor_store:
            raise ValueError("One or both tensor names not found in the store.")

        try:
            result = np.matmul(tensor_store[name_a], tensor_store[name_b])
        except ValueError as e:
            raise ValueError(f"Error multiplying tensors: {e}")

        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(
        title="Scale Matrix", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
    def scale_matrix(
        name: Annotated[str, Field(description="Name of the stored tensor to scale.")],
        scale_factor: Annotated[float, Field(description="Scalar value to multiply every element by.")],
        in_place: Annotated[bool, Field(
            description="Overwrite the stored tensor with the scaled result. False returns the "
                        "result and leaves the store untouched.")] = True,
    ) -> Tensor:
        """
        Scales a stored tensor by a scalar factor.

        Args:
            name (str): The name of the tensor to scale.
            scale_factor (float): The scalar value to multiply the tensor by.
            in_place (bool): If True, updates the stored tensor; otherwise, returns a new scaled tensor.

        Returns:
            Tensor: The scaled tensor.

        Raises:
            ValueError: If the tensor name is not found in the store.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        result = tensor_store[name] * scale_factor

        if in_place:
            tensor_store[name] = result

        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Matrix Inverse", **READ_ONLY))
    def matrix_inverse(
        name: Annotated[str, Field(description="Name of the stored square matrix to invert.")],
    ) -> Tensor:
        """
        Computes the inverse of a stored square matrix.

        Args:
            name (str): The name of the tensor to invert.

        Returns:
            Tensor: The inverse of the matrix.

        Raises:
            ValueError: If the matrix is not found, is not square, or is singular (non-invertible).
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            result = np.linalg.inv(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing matrix inverse: {e}")

        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Transpose", **READ_ONLY))
    def transpose(
        name: Annotated[str, Field(description="Name of the stored tensor to transpose.")],
    ) -> Tensor:
        """
        Computes the transpose of a stored tensor.

        Args:
            name (str): The name of the tensor to transpose.

        Returns:
            Tensor: The transposed tensor.

        Raises:
            ValueError: If the tensor name is not found in the store.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        return tensor_store[name].T.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Determinant", **READ_ONLY))
    def determinant(
        name: Annotated[str, Field(description="Name of the stored square matrix.")],
    ) -> float:
        """
        Computes the determinant of a stored square matrix.

        Args:
            name (str): The name of the matrix.

        Returns:
            float: The determinant of the matrix.

        Raises:
            ValueError: If the matrix is not found or is not square.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            result = np.linalg.det(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing determinant: {e}")

        return float(result)

    @mcp.tool(annotations=ToolAnnotations(title="Rank", **READ_ONLY))
    def rank(
        name: Annotated[str, Field(description="Name of the stored tensor.")],
    ) -> int | list[int]:
        """
        Computes the rank of a stored tensor.

        Args:
            name (str): The name of the tensor.

        Returns:
            int | list[int]: The rank of the matrix.

        Raises:
            ValueError: If the tensor name is not found in the store.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        result = np.linalg.matrix_rank(tensor_store[name])
        return result.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Eigenvalues and Eigenvectors", **READ_ONLY))
    def compute_eigen(
        name: Annotated[str, Field(description="Name of the stored square matrix to analyze.")],
    ) -> dict[str, Any]:
        """
        Computes the eigenvalues and right eigenvectors of a stored square matrix.

        Args:
            name (str): The name of the tensor to analyze.

        Returns:
            dict: A dictionary with keys:
                - 'eigenvalues': list of eigenvalues
                - 'eigenvectors': list of right eigenvectors, one per column of the result

        Raises:
            ValueError: If the tensor is not found or is not a square matrix.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            eigenvalues, eigenvectors = np.linalg.eig(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing eigenvalues and eigenvectors: {e}")

        # JSON has no complex number, and a rotation block gives a complex spectrum, so
        # split the parts into their own keys instead of dropping the imaginary half.
        # Purely real results keep the plain two-key shape.
        result = {"eigenvalues": eigenvalues.real.tolist(), "eigenvectors": eigenvectors.real.tolist()}
        if np.iscomplexobj(eigenvalues) and np.any(eigenvalues.imag):
            result["eigenvalues_imag"] = eigenvalues.imag.tolist()
            result["eigenvectors_imag"] = eigenvectors.imag.tolist()
        return result

    # Matrix decompositions
    @mcp.tool(annotations=ToolAnnotations(title="QR Decomposition", **READ_ONLY))
    def qr_decompose(
        name: Annotated[str, Field(description="Name of the stored matrix to decompose.")],
    ) -> dict[str, Any]:
        """
        Computes the QR decomposition of a stored matrix.

        Decomposes the matrix A into A = Q @ R, where Q is an orthogonal matrix
        and R is an upper triangular matrix.

        Args:
            name (str): The name of the matrix to decompose.

        Returns:
            dict: A dictionary with keys:
                - 'q': the orthogonal matrix Q, as nested lists
                - 'r': the upper triangular matrix R, as nested lists

        Raises:
            ValueError: If the matrix is not found or decomposition fails.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            q, r = np.linalg.qr(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing QR decomposition: {e}")

        return {'q': q.tolist(), 'r': r.tolist()}

    @mcp.tool(annotations=ToolAnnotations(title="SVD Decomposition", **READ_ONLY))
    def svd_decompose(
        name: Annotated[str, Field(description="Name of the stored matrix to decompose.")],
    ) -> dict[str, Any]:
        """
        Computes the Singular Value Decomposition (SVD) of a stored matrix.

        Decomposes the matrix A into A = U @ S @ V^T, where U and V^T are orthogonal
        matrices, and S is a diagonal matrix of singular values.

        Args:
            name (str): The name of the matrix to decompose.

        Returns:
            dict: A dictionary with keys:
                - 'u': the left singular vectors, as nested lists
                - 's': the singular values, as a flat list
                - 'v_t': the right singular vectors transposed, as nested lists

        Raises:
            ValueError: If the matrix is not found or decomposition fails.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            u, s, v_t = np.linalg.svd(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing SVD decomposition: {e}")

        return {'u': u.tolist(), 's': s.tolist(), 'v_t': v_t.tolist()}

    # Bases
    @mcp.tool(annotations=ToolAnnotations(title="Orthonormal Basis", **READ_ONLY))
    def find_orthonormal_basis(
        name: Annotated[str, Field(description="Name of the stored matrix whose column space to orthonormalize.")],
    ) -> list[list[float]]:
        """
        Finds an orthonormal basis for the column space of a stored matrix using QR decomposition.

        Args:
            name (str): The name of the matrix.

        Returns:
            list[list[float]]: A list of orthonormal basis vectors.

        Raises:
            ValueError: If the matrix is not found or decomposition fails.
        """
        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        try:
            # Call numpy directly rather than round-tripping through qr_decompose as an
            # MCP tool: that path hands back the serialized text, where 'q' is numpy's
            # string repr rather than a nested list, and fails this return annotation.
            q, _ = np.linalg.qr(tensor_store[name])
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error computing orthonormal basis: {e}")

        # Transposed so each element is a basis vector; the basis is the columns of Q.
        return q.T.tolist()

    @mcp.tool(annotations=ToolAnnotations(title="Change of Basis", **READ_ONLY))
    def change_basis(
        name: Annotated[str, Field(description="Name of the stored square matrix to re-express.")],
        new_basis: Annotated[list[list[float]], Field(
            description="The new basis as a nested list whose columns are the basis vectors.")],
    ) -> Tensor:
        """
        Changes the basis of a stored square matrix.

        Args:
            name (str): Name of the matrix in the tensor store.
            new_basis (list[list[float]]): Columns are new basis vectors.

        Returns:
            Tensor: Representation of the matrix in the new basis.

        Raises:
            ValueError: If the matrix name is not found or non-invertible.
        """

        if name not in tensor_store:
            raise ValueError("The tensor name is not found in the store.")

        basis = np.asarray(new_basis)
        try:
            result = np.linalg.inv(basis) @ tensor_store[name] @ basis
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Error changing basis: {e}")

        return result.tolist()
