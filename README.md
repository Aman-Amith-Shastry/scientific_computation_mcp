[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/aman-amith-shastry-scientific-computation-mcp-badge.png)](https://mseep.ai/app/aman-amith-shastry-scientific-computation-mcp)

# Scientific Computation MCP

[![smithery badge](https://smithery.ai/badge/@Aman-Amith-Shastry/scientific_computation_mcp)](https://smithery.ai/server/@Aman-Amith-Shastry/scientific_computation_mcp)

[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/5927ad38-70f6-4f5b-9778-e61ec902d735)

[![MCP Badge](https://lobehub.com/badge/mcp-full/aman-amith-shastry-scientific_computation_mcp)](https://lobehub.com/mcp/aman-amith-shastry-scientific_computation_mcp)

## Installation Guide

The server speaks streamable HTTP. Add it from Smithery with the Smithery CLI (Node 20+):

```bash
npm install -g smithery@latest
smithery mcp add @Aman-Amith-Shastry/scientific_computation_mcp --client claude
```

Swap `--client cursor` for Cursor, or drop `--client` to add it as a remote Smithery
connection. Restart the client afterwards so it picks up the server.

## Running Locally

```bash
uv sync
uv run src/main.py
```

The MCP endpoint is at `http://localhost:8081/mcp` and a liveness probe at
`http://localhost:8081/health`. Environment overrides: `PORT`, `HOST`, `MCP_PATH`,
`ALLOWED_ORIGINS`, `LOG_LEVEL`.

## Deployment

Smithery no longer builds or hosts containers — servers are published either as a URL
that Smithery's gateway proxies to, or as an MCPB bundle for local stdio. This server is
published by URL, so the container runs on any host that can serve HTTPS.

```bash
docker build -t scientific-computation-mcp .
docker run -p 8081:8081 -e PORT=8081 scientific-computation-mcp
```

Two constraints the host must satisfy:

- **One instance.** Tensors live in process memory between tool calls, so scaling past a
  single replica splits the store and breaks `create_tensor` → `view_tensor` flows.
- **Sessions must survive.** The transport runs stateful (`stateless_http=False`) and
  the tensor store is keyed per MCP session, which is what keeps concurrent users from
  reading each other's tensors.

### Hosting on Render

[`render.yaml`](render.yaml) deploys the Dockerfile as a single free-plan web service.
In the Render dashboard: **New → Blueprint**, then select this repo. Render injects
`PORT`, terminates TLS, and probes `/health`; no other configuration is required.

Free instances spin down after 15 minutes without *inbound* traffic and take roughly a
minute to come back. An open MCP session does not prevent this: the streamable-HTTP
stream is server-to-client, so an idle session sends nothing inbound and the service
sleeps out from under it. Two consequences worth planning around:

- **Tensors do not survive a 15-minute gap between tool calls.** The store is in process
  memory, so a spin-down takes the session and its tensors together. Active use keeps
  the service up, since each tool call is inbound traffic; a long pause mid-analysis
  does not.
- **Publish-time scans can land on a sleeping instance.** Smithery scans the URL as
  `SmitheryBot/1.0`, and a cold start can outrun its timeout. Warm `/health` first.

The free tier also grants 750 instance-hours per workspace per month against a ~730-hour
month, so one continuously running free service just fits and a second does not.

Any host that keeps one process always on avoids all of this — the container is plain
HTTP on `$PORT` with no platform-specific assumptions.

### Publishing

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://<your-host>/health
smithery mcp publish "https://<your-host>/mcp" -n @Aman-Amith-Shastry/scientific_computation_mcp
```

The server takes no user configuration, so no config schema is needed.

## Components of the Server

### Tools

#### Tensor storage
- ```create_tensor```: Creates a new tensor based on a given name, shape, and values, and adds it to the tensor store. For the purposes of this server, tensors are vectors and matrices.
- ```view_tensor```: Display the contents of a tensor from the store .
- ```delete_tensor```: Deletes a tensor based on its name in the tensor store.

#### Linear Algebra
- ```add_matrices```: Adds two matrices with the provided names, if compatible.
- ```subtract_matrices```: Subtracts two matrices with the provided names, if compatible.
- ```multiply_matrices```: Multiplies two matrices with the provided names, if compatible.
- ```scale_matrix```: Scales a matrix of the provided name by a certain factor, in-place by default.
- ```matrix_inverse```: Computes the inverse of the matrix with the provided name.
- ```transpose```: Computes the transpose of the inverse of the matrix of the provided name.
- ```determinant```: Computes the determinant of the matrix of the provided name.
- ```rank```: Computes the rank (number of pivots) of the matrix of the provided name.
- ```compute_eigen```: Calculates the eigenvectors and eigenvalues of the matrix of the provided name.
- ```qr_decompose```: Computes the QR factorization of the matrix of the provided name. The columns of Q are an orthonormal basis for the image of the matrix, and R is upper triangular.
- ```svd_decompose```: Computes the Singular Value Decomposition of the matrix of the provided name.
- ```find_orthonormal_basis```: Finds an orthonormal basis for the matrix of the provided name. The vectors returned are all pair-wise orthogonal and are of unit length.
- ```change_basis```: Computes the matrix of the provided name in the new basis.

#### Vector Calculus
- ```vector_project```: Projects a vector in the tensor store to the specified vector in the same vector space
- ```vector_dot_product```: Computes the dot product of two vectors in the tensor stores based on their provided names.
- ```vector_cross_product```: Computes the cross product of two vectors in the tensor stores based on their provided names.
- ```gradient```: Computes the gradient of a multivariable function based on the input function. Example call: ```gradient("x^2 + 2xyz + zy^3")```. Do NOT include the function name (like f(x, y, z) = ...`).
- ```curl```: Computes the curl of a vector field based on the input vector field. The input string must be formatted as a python list. Example call: ```curl("[3xy, 2z^4, 2y]"")```.
- ```divergence```Computes the divergence of a vector field based on the input vector field. The input string must be formatted as a python list. Example call: ```divergence("[3xy, 2z^4, 2y]"")```.
- ```laplacian```Computes the laplacian of a scalar function (as the divergence of the gradient) or a vector field (where a component-wise laplacian is computed). If a scalar function is the input, it must be input in the same format as in the ```gradient``` tool. If the input is a vector field, it must be input in the same manner as the ```curl/divergence``` tools.
- ```directional_deriv```: Computes the directional derivative of a function in a given direction ```u``` By default, the tool normalizes ```u``` before computing the directional derivative, as specified by the ```unit``` parameter.

#### Visualization
- ```plot_vector_field```: Plots a vector field (specified in the same format as in the curl/divergence functions). Currently, only 3d vector fields are supported. A 2d png perspective image of the vector field is returned. By default, the bounds of the graph are from -1 to 1 on each axis.
- ```plot_function```: Plots a function in 2d or 3d (based on the input variables), specified in the same format as in the ```gradient``` tool. Only the variables x and y can be used.
