from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import os

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Check headers or query params
        api_key = request.query_params.get("api_key") or request.headers.get("Authorization")
        profile = request.query_params.get("profile") or request.headers.get("X-Profile")

        # Validate (replace "expected_key" with your actual env variable)
        expected_key = os.environ.get("API_KEY")
        expected_profile = os.environ.get("PROFILE")

        if not api_key or not profile:
            return JSONResponse({"error": "Missing credentials"}, status_code=401)

        if expected_key and api_key != expected_key:
            return JSONResponse({"error": "Invalid API key"}, status_code=403)

        if expected_profile and profile != expected_profile:
            return JSONResponse({"error": "Invalid profile"}, status_code=403)

        # If valid → forward to MCP
        return await call_next(request)
