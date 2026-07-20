"""
DeepGuard — api/auth/security.py

Authentication and authorisation for the DeepGuard REST API.

Two complementary strategies are supported:

1. **API Key** — static secret passed via the ``X-API-Key`` header.
   Suitable for server-to-server calls and the Streamlit frontend.

2. **JWT Bearer token** — short-lived access tokens issued by
   ``POST /api/v1/auth/token``.  Suitable for interactive clients.

Both strategies are implemented as FastAPI dependency functions so they
can be composed freely:

    @router.get("/protected")
    async def endpoint(user: TokenPayload = Depends(require_jwt)):
        ...

    @router.post("/admin")
    async def admin(user: TokenPayload = Depends(require_any_auth)):
        ...

Configuration is read from ``configs/api_config.yaml`` (``security``
block) and can be overridden via environment variables:

    DEEPGUARD_API_KEY   — master API key (overrides config)
    DEEPGUARD_JWT_SECRET — JWT signing secret (overrides config)
    DEEPGUARD_AUTH_DISABLED — set to "1" to bypass auth in development
"""

from __future__ import annotations

import os
import logging
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Load config once at import time ───────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "api_config.yaml"


def _load_security_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f).get("security", {})
    return {}


_SEC = _load_security_config()

# Master API key — env variable wins over config
MASTER_API_KEY: str = os.getenv("DEEPGUARD_API_KEY", _SEC.get("master_api_key", ""))

# JWT secret — env variable wins
JWT_SECRET: str = os.getenv(
    "DEEPGUARD_JWT_SECRET",
    _SEC.get("jwt", {}).get("secret_key", "change-me-in-production"),
)
JWT_ALGORITHM: str = _SEC.get("jwt", {}).get("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = _SEC.get("jwt", {}).get("access_token_expire_minutes", 30)
REFRESH_TOKEN_EXPIRE_DAYS: int = _SEC.get("jwt", {}).get("refresh_token_expire_days", 7)

API_KEY_HEADER_NAME: str = _SEC.get("api_key_header", "X-API-Key")

# Dev bypass (never enable in production)
AUTH_DISABLED: bool = os.getenv("DEEPGUARD_AUTH_DISABLED", "0") == "1"

# ── FastAPI security schemes ──────────────────────────────────────────────────

_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


# ── Token model ───────────────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    """Decoded JWT claims."""
    sub: str                    # subject — username or client ID
    scopes: list[str] = []
    exp: datetime | None = None
    iat: datetime | None = None


class TokenResponse(BaseModel):
    """Response body for /auth/token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int             # seconds


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _try_import_jose():
    try:
        from jose import JWTError, jwt as _jwt
        return _jwt, JWTError
    except ImportError:
        return None, None


def create_access_token(subject: str, scopes: list[str] | None = None) -> str:
    """Create a signed JWT access token."""
    _jwt, _ = _try_import_jose()
    if _jwt is None:
        raise RuntimeError("python-jose is not installed. Run: pip install python-jose[cryptography]")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "scopes": scopes or [],
    }
    return _jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> TokenPayload:
    """Decode and validate a JWT access token, raising HTTPException on failure."""
    _jwt, JWTError = _try_import_jose()
    if _jwt is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication requires python-jose. Run: pip install python-jose[cryptography]",
        )
    try:
        data = _jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**data)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependency functions ──────────────────────────────────────────────────────

async def require_api_key(
    api_key: Annotated[str | None, Security(_api_key_scheme)] = None,
) -> str:
    """FastAPI dependency — validates the X-API-Key header."""
    if AUTH_DISABLED:
        return "dev-bypass"
    if not MASTER_API_KEY:
        logger.warning("DEEPGUARD_API_KEY is not configured; API key auth is disabled.")
        return "unconfigured"
    if api_key != MASTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={f"WWW-Authenticate": f"ApiKey header={API_KEY_HEADER_NAME}"},
        )
    return api_key


async def require_jwt(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> TokenPayload:
    """FastAPI dependency — validates a Bearer JWT token."""
    if AUTH_DISABLED:
        return TokenPayload(sub="dev-user", scopes=["*"])
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


async def require_any_auth(
    api_key: Annotated[str | None, Security(_api_key_scheme)] = None,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)] = None,
) -> str:
    """FastAPI dependency — accepts either a valid API key OR a valid JWT."""
    if AUTH_DISABLED:
        return "dev-bypass"

    # Try API key first
    if api_key and MASTER_API_KEY and api_key == MASTER_API_KEY:
        return f"apikey:{api_key[:8]}***"

    # Fall back to JWT
    if credentials and credentials.credentials:
        payload = decode_access_token(credentials.credentials)
        return f"jwt:{payload.sub}"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Provide a valid API key or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
