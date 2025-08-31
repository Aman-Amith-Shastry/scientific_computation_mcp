import requests
import uuid
import json

# MCP server URL
MCP_URL = "http://localhost:8081/mcp"

# Generate a unique session ID
SESSION_ID = str(uuid.uuid4())

# Headers required by FastMCP
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json,text/event-stream",
    "X-MCP-Session": SESSION_ID
}


# Helper to send a JSON-RPC request
def send_rpc(method, params=None, request_id=None):
    if params is None:
        params = {}
    if request_id is None:
        request_id = str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id,
        "params": params
    }

    response = requests.post(MCP_URL, headers=HEADERS, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()


# List all tools
tools_response = send_rpc("list_tools")
print("Available tools:", tools_response["result"])

# Example: call create_tensor
tensor_params = {
    "shape": [2, 3],
    "values": [1, 2, 3, 4, 5, 6],
    "name": "my_tensor"
}
create_response = send_rpc("create_tensor", tensor_params)
print("Created tensor:", create_response["result"])