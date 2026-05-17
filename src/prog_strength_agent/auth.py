from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request, status


@dataclass(frozen=True)
class AuthenticatedRequest:
    """Result of validating the inbound user's JWT.

    `user_id` is the validated `sub` claim — used by the agent for
    logging and (no longer for) injection into MCP tool calls.
    `token` is the raw bearer token, forwarded verbatim to MCP via
    the per-request session's Authorization header so MCP can pass
    it through to the API.
    """

    user_id: str
    token: str


def authenticate(request: Request, signing_key: str) -> AuthenticatedRequest:
    """Verify the request's `Authorization: Bearer <jwt>` and return
    both the user_id and the raw token.

    The agent does not mint tokens — it only verifies them. The token
    is the same shape the API issues on OAuth callback: HS256, signed
    with the shared JWT_SIGNING_KEY, `sub` carrying the user's ID.

    Raises 401 on any failure. The agent must NOT leak which failure
    mode (expired vs malformed vs missing) was responsible, since the
    failure response goes to a potentially-untrusted client.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header[len("bearer ") :].strip()
    try:
        payload = jwt.decode(token, signing_key, algorithms=["HS256"])
    except jwt.InvalidTokenError as e:
        # Bucket every JWT failure into a single 401. Don't echo the
        # underlying jwt exception text — it leaks signal about which
        # validation step failed.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from e

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token missing sub claim")
    return AuthenticatedRequest(user_id=sub, token=token)
