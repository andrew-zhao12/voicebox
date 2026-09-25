"""Operator endpoints that are not tied to one resource."""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..observability import metrics
from ..services import retention

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus exposition of the server's counters (admin key; see VOICEBOX_METRICS_PORT for scrapers)."""
    try:
        body, content_type = metrics.render()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return Response(body, media_type=content_type)


@router.post("/maintenance/prune", response_model=models.PruneReportResponse)
async def prune_now(data: models.PruneRequest | None = None, db: Session = Depends(get_db)):
    """Run the retention sweep now, with ``days`` from the body or ``VOICEBOX_RETENTION_DAYS``."""
    days = data.days if data is not None and data.days is not None else retention.configured_days()
    if days is None:
        raise HTTPException(status_code=400, detail="Pass `days` or set VOICEBOX_RETENTION_DAYS")
    report = await retention.prune(db, days=days)
    return report.as_dict()
