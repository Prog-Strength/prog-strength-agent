import jwt
from fastapi import HTTPException, Request, status


def authenticated_user_id(request: Request, signing_key: str) -> str:
    """Verify the request's `Authorization: Bearer <jwt>` and return `sub`.

    The agent does not mint tokens — it only verifies them. The token is
    the same shape the API issues on OAuth callback: HS256, signed with
    the shared JWT_SIGNING_KEY, `sub` carrying the user's ID.

    Raises 401 on any failure. The agent must NOT leak which failure mode
    (expired vs malformed vs missing) was responsible, since the failure
    response goes to a potentially-untrusted client.
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
    return sub
