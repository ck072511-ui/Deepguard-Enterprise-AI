"""
DeepGuard — api/v1/endpoints/auth.py

Authentication endpoints:

  POST /api/v1/auth/token   — issue a JWT access token (OAuth2 password flow)
  GET  /api/v1/auth/me      — return the authenticated caller's identity
  POST /api/v1/auth/refresh — refresh an access token (stub, extend as needed)

Security note: In production replace the hard-coded credential check with
a proper user database (hashed passwords via passlib/bcrypt).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from api.auth.security import (
    TokenResponse,
    TokenPayload,
    create_access_token,
    require_jwt,
    AUTH_DISABLED,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── Minimal user store (replace with DB in production) ────────────────────────
# Env var: DEEPGUARD_ADMIN_PASSWORD (defaults to "deepguard-change-me")
_ADMIN_PASSWORD = os.getenv("DEEPGUARD_ADMIN_PASSWORD", "deepguard-change-me")
_USERS: dict[str, dict] = {
    "admin": {"password": _ADMIN_PASSWORD, "scopes": ["detect", "models", "history", "admin"]},
    "viewer": {"password": "viewer-change-me", "scopes": ["detect", "history"]},
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class UserInfo(BaseModel):
    """Public caller identity returned by GET /auth/me."""
    sub: str
    scopes: list[str]
    authenticated_at: datetime
    auth_disabled: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue JWT access token",
    description=(
        "OAuth2 Password flow. Exchange username + password for a short-lived "
        "JWT Bearer token. Token is valid for **30 minutes** by default."
    ),
    responses={
        200: {"description": "Token issued successfully"},
        401: {"description": "Invalid credentials"},
    },
)
async def issue_token(
    username: str = Form(..., example="admin"),
    password: str = Form(..., example="deepguard-change-me"),
) -> TokenResponse:
    """Exchange credentials for a JWT access token."""
    user = _USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=username, scopes=user["scopes"])
    logger.info("Token issued for user '%s'", username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get(
    "/me",
    response_model=UserInfo,
    summary="Current authenticated user",
    description="Returns the identity of the currently authenticated caller.",
    responses={
        200: {"description": "Caller identity"},
        401: {"description": "Not authenticated"},
    },
)
async def current_user(
    payload: Annotated[TokenPayload, Depends(require_jwt)],
) -> UserInfo:
    """Return the authenticated caller's identity decoded from the JWT."""
    return UserInfo(
        sub=payload.sub,
        scopes=payload.scopes,
        authenticated_at=datetime.now(timezone.utc),
        auth_disabled=AUTH_DISABLED,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Validate an existing token and issue a fresh one.",
    responses={
        200: {"description": "New token issued"},
        401: {"description": "Token invalid or expired"},
    },
)
async def refresh_token(
    payload: Annotated[TokenPayload, Depends(require_jwt)],
) -> TokenResponse:
    """Re-issue an access token for a still-valid token holder."""
    new_token = create_access_token(subject=payload.sub, scopes=payload.scopes)
    logger.info("Token refreshed for user '%s'", payload.sub)
    return TokenResponse(
        access_token=new_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
