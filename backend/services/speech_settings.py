"""Persist speech defaults per API key and resolve explicit request overrides."""

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ..auth.principal import Principal, principal_var
from ..database import SpeechSettings as DBSpeechSettings
from ..speech_settings_models import SpeechSettingsResponse, SpeechSettingsUpdate
from ..utils.speech_text import validate_hints, validate_pronunciations


def get_settings(db: Session, key_id: str) -> SpeechSettingsResponse:
    """Read defaults without creating a row for callers who have never saved any."""
    row = db.get(DBSpeechSettings, key_id)
    if row is None:
        return SpeechSettingsResponse()
    return SpeechSettingsResponse(
        pronunciation_replacements=row.pronunciation_replacements,
        stt_prompt=row.stt_prompt,
        stt_hotwords=row.stt_hotwords,
    )


def update_settings(db: Session, key_id: str, patch: SpeechSettingsUpdate) -> SpeechSettingsResponse:
    """Save supplied fields; an explicit null clears the corresponding default."""
    fields = patch.model_dump(exclude_unset=True)
    if not fields:
        return get_settings(db, key_id)
    empty = SpeechSettingsResponse().model_dump()
    updates = {name: empty[name] if value is None else value for name, value in fields.items()}
    # Two first-time partial PUTs must create one row and preserve each other's
    # disjoint fields. A read-then-insert races between request DB sessions.
    statement = insert(DBSpeechSettings).values(key_id=key_id, **{**empty, **updates})
    db.execute(statement.on_conflict_do_update(index_elements=["key_id"], set_=updates))
    db.commit()
    db.expire_all()
    return get_settings(db, key_id)


def caller_settings(principal: Principal | None = None, db: Session | None = None) -> SpeechSettingsResponse:
    """Read request defaults; internal tools without an authenticated context get empty settings."""
    principal = principal if principal is not None else principal_var.get()
    if principal is None or not principal.authenticated:
        return SpeechSettingsResponse()
    if db is not None:
        return get_settings(db, principal.key_id)
    from ..database import session

    # Unit-level services and pre-startup tools may run before DB initialization.
    if session.SessionLocal is None:
        return SpeechSettingsResponse()
    with session.SessionLocal() as local:
        return get_settings(local, principal.key_id)


def resolve_pronunciations(
    db: Session,
    profile: object,
    principal: Principal,
    override: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve request > voice > client, replacing whole dictionaries rather than merging.

    Null or omission inherits; an empty dictionary deliberately disables defaults.
    This only resolves settings: synthesis applies substitutions once afterward.
    """
    if override is not None:
        return dict(validate_pronunciations(override))
    saved = getattr(profile, "pronunciation_replacements", None)
    if saved is not None:
        return dict(validate_pronunciations(saved))
    return caller_settings(principal, db).pronunciation_replacements


def resolve_transcription_hints(
    prompt: str | None,
    hotwords: list[str] | tuple[str, ...] | None,
    *,
    principal: Principal | None = None,
    db: Session | None = None,
) -> tuple[str, list[str]]:
    """Resolve missing hints from caller settings; empty text/list explicitly disables them."""
    saved = caller_settings(principal, db)
    resolved_prompt = saved.stt_prompt if prompt is None else prompt
    resolved_hotwords = saved.stt_hotwords if hotwords is None else list(hotwords)
    validate_hints(resolved_prompt, resolved_hotwords)
    return resolved_prompt, resolved_hotwords
