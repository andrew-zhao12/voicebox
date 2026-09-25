"""``/auth/*``: identity, media tokens and (admin-only) key management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .keystore import KEY_ID_RE, KeyRecord
from .principal import Principal, current_principal, require_admin

if TYPE_CHECKING:
    from .install import SecurityRuntime

router = APIRouter(prefix="/auth", tags=["auth"])


class MediaTokenResponse(BaseModel):
    token: str
    expires_in: int


class WhoAmIResponse(BaseModel):
    key_id: str
    role: str
    via: str
    limits: dict[str, int | None]


class ApiKeyInfo(BaseModel):
    id: str
    role: str
    created_at: str
    source: str
    limits: dict[str, int | None]


class ApiKeyCreated(ApiKeyInfo):
    key: str = Field(description="The plaintext key; it is shown only once.")


class ApiKeyList(BaseModel):
    keys: list[ApiKeyInfo]


class ApiKeyCreate(BaseModel):
    id: str = Field(min_length=1, max_length=32, pattern=KEY_ID_RE.pattern)
    role: Literal["admin", "client"] = "client"
    limits: dict[str, int | None] | None = Field(
        default=None,
        description="Per-minute overrides: requests, inference, uploads_bytes, tts_chars, max_pending_jobs (null = unlimited).",
    )


def _runtime(request: Request) -> SecurityRuntime:
    return request.app.state.security


def _info(record: KeyRecord) -> ApiKeyInfo:
    return ApiKeyInfo(
        id=record.id,
        role=record.role,
        created_at=record.created_at,
        source=record.source,
        limits=record.key_limits().as_dict(),
    )


@router.post("/media-token", response_model=MediaTokenResponse)
async def issue_media_token(request: Request, principal: Principal = Depends(current_principal)):
    """A short-lived token for browser loads that cannot send headers (``?token=`` on media routes)."""
    if not principal.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    token, expires_in = _runtime(request).tokens.issue(principal)
    return MediaTokenResponse(token=token, expires_in=expires_in)


@router.get("/whoami", response_model=WhoAmIResponse)
async def whoami(principal: Principal = Depends(current_principal)):
    if not principal.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return WhoAmIResponse(
        key_id=principal.key_id, role=principal.role, via=principal.via, limits=principal.limits.as_dict()
    )


@router.get("/keys", response_model=ApiKeyList, dependencies=[Depends(require_admin)])
async def list_keys(request: Request):
    return ApiKeyList(keys=[_info(record) for record in _runtime(request).keystore.list_keys()])


@router.post("/keys", response_model=ApiKeyCreated, status_code=201, dependencies=[Depends(require_admin)])
async def create_key(data: ApiKeyCreate, request: Request):
    try:
        record, key = _runtime(request).keystore.create(data.id, data.role, data.limits)
    except KeyError as e:
        raise HTTPException(status_code=409, detail=str(e.args[0])) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ApiKeyCreated(**_info(record).model_dump(), key=key)


@router.delete("/keys/{key_id}", status_code=204, dependencies=[Depends(require_admin)])
async def revoke_key(key_id: str, request: Request, principal: Principal = Depends(current_principal)):
    if key_id == principal.key_id:
        raise HTTPException(status_code=400, detail="A key cannot revoke itself")
    try:
        revoked = _runtime(request).keystore.revoke(key_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=204)
