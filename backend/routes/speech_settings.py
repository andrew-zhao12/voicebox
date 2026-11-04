"""Client-scoped speech settings, with administrator editing by API-key ID."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..auth import current_principal, require_admin
from ..auth.principal import Principal
from ..database import get_db
from ..services import speech_settings
from ..speech_settings_models import SpeechSettingsResponse, SpeechSettingsUpdate

router = APIRouter(prefix="/speech/settings", tags=["speech settings"])


def _existing_key(request: Request, key_id: str) -> None:
    if request.app.state.security.keystore.get(key_id) is None:
        raise HTTPException(status_code=404, detail="Key not found")


@router.get("", response_model=SpeechSettingsResponse)
def get_own_settings(
    principal: Principal = Depends(current_principal), db: Session = Depends(get_db)
) -> SpeechSettingsResponse:
    """Read or export the authenticated caller's reusable speech settings."""
    if not principal.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return speech_settings.get_settings(db, principal.key_id)


@router.put("", response_model=SpeechSettingsResponse)
def update_own_settings(
    patch: SpeechSettingsUpdate,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> SpeechSettingsResponse:
    """Update supplied fields, or import a previously exported settings document."""
    if not principal.authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return speech_settings.update_settings(db, principal.key_id, patch)


@router.get("/{key_id}", response_model=SpeechSettingsResponse, dependencies=[Depends(require_admin)])
def get_key_settings(key_id: str, request: Request, db: Session = Depends(get_db)) -> SpeechSettingsResponse:
    """Read one existing API key's settings as an administrator."""
    _existing_key(request, key_id)
    return speech_settings.get_settings(db, key_id)


@router.put("/{key_id}", response_model=SpeechSettingsResponse, dependencies=[Depends(require_admin)])
def update_key_settings(
    key_id: str, patch: SpeechSettingsUpdate, request: Request, db: Session = Depends(get_db)
) -> SpeechSettingsResponse:
    """Update one existing API key; deleted or unknown identities cannot create settings rows."""
    _existing_key(request, key_id)
    return speech_settings.update_settings(db, key_id, patch)
