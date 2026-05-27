"""Evaluation API endpoints.

POST /api/eval/run    - Run the full benchmark suite, return the report
GET  /api/eval/report - Retrieve the most recent benchmark report
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.models.benchmark_suite import BenchmarkSuite


router = APIRouter(prefix="/api/eval", tags=["eval"])

# Module-level singleton so GET /report can retrieve the last run.
_suite = BenchmarkSuite()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Optional overrides for a benchmark run."""
    yaml_path: str | None = Field(
        default=None,
        description="Path to eval_tasks.yaml (defaults to data/eval_tasks.yaml)",
    )
    task_ids: list[str] | None = Field(
        default=None,
        description="If provided, only run these task IDs",
    )


class RunResponse(BaseModel):
    run_id: str
    duration_ms: float
    global_metrics: dict[str, float]
    tasks: list[dict[str, Any]]


class ReportResponse(BaseModel):
    run_id: str | None
    timestamp: float | None
    duration_ms: float | None
    global_metrics: dict[str, float] | None
    tasks: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run", response_model=RunResponse)
async def run_eval(request: RunRequest | None = None) -> RunResponse:
    """Run the benchmark suite and return the full report.

    Optionally accepts a different YAML path or a filter of task IDs.
    """
    global _suite

    yaml_path = None
    task_ids = None
    if request:
        yaml_path = request.yaml_path
        task_ids = request.task_ids

    suite = BenchmarkSuite(yaml_path=yaml_path) if yaml_path else _suite

    # Load tasks (or reload if new path)
    tasks = suite.load_tasks()

    # Filter if task_ids provided
    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.get("id") in id_set]
        if not tasks:
            raise HTTPException(
                status_code=404,
                detail=f"No tasks matched IDs: {task_ids}",
            )

    report = suite.run(tasks=tasks)
    report_dict = report.to_dict()

    return RunResponse(
        run_id=report_dict["run_id"],
        duration_ms=report_dict["duration_ms"],
        global_metrics=report_dict["global_metrics"],
        tasks=report_dict["tasks"],
    )


@router.get("/report", response_model=ReportResponse)
async def get_report() -> ReportResponse:
    """Retrieve the most recent benchmark report.

    Returns 404 if no run has been executed yet.
    """
    report = _suite.last_report
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No benchmark report available. POST /api/eval/run first.",
        )
    d = report.to_dict()
    return ReportResponse(
        run_id=d["run_id"],
        timestamp=d["timestamp"],
        duration_ms=d["duration_ms"],
        global_metrics=d["global_metrics"],
        tasks=d["tasks"],
    )
