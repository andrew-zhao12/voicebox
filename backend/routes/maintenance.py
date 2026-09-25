"""Operator endpoints that are not tied to one resource."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import retention

router = APIRouter()


@router.post("/maintenance/prune", response_model=models.PruneReportResponse)
async def prune_now(data: models.PruneRequest | None = None, db: Session = Depends(get_db)):
    """Run the retention sweep now, with ``days`` from the body or ``VOICEBOX_RETENTION_DAYS``."""
    days = data.days if data is not None and data.days is not None else retention.configured_days()
    if days is None:
        raise HTTPException(status_code=400, detail="Pass `days` or set VOICEBOX_RETENTION_DAYS")
    report = await retention.prune(db, days=days)
    return report.as_dict()
