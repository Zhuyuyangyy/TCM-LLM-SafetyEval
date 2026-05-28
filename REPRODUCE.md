# REPRODUCE.md - TCM-LLM-SafetyEval

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Not required (CPU sufficient for API server)

## Install

```bash
cd TCM-LLM-SafetyEval
pip install -e .
```

Or install dependencies directly:
```bash
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

## Smoke Test

```bash
python -c "from backend.main import app; print('Import OK')"
```

```bash
pytest backend/tests/ -v
```

## Run Server

```bash
uvicorn backend.main:app --port 8023 --reload
```

## API Documentation

Access Swagger UI at: http://localhost:8023/docs

## Project Description

中医处方审核与中医药大模型安全评测基准

## Known Issues

- No external real clinical data included; uses synthetic/demo data
- torch is a heavy dependency; consider CPU-only install for API-only usage
- No hardcoded absolute paths detected in core code
