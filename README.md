<div align="center">

# TCM-LLM-SafetyEval

**A Safety Evaluation Benchmark for Large Language Models in Traditional Chinese Medicine Clinical Applications**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)

</div>

---

## Overview

TCM-LLM-SafetyEval is the first dedicated benchmark framework for evaluating the safety, accuracy, and reliability of large language models (LLMs) in Traditional Chinese Medicine (TCM) clinical scenarios. The framework addresses a critical gap: as LLMs are increasingly deployed in healthcare settings, there is no standardized evaluation methodology for TCM-specific risks such as herb incompatibilities (Shiba Fan / Shijiu Wei), dosage safety of toxic herbs, and TCM-Western drug interactions.

The system provides a multi-dimensional evaluation pipeline combining rule-based prescription auditing, hallucination detection, citation verification, calibration metrics, and adversarial red team testing -- all grounded in authoritative TCM pharmacopoeia standards.

### Key Research Contributions

- **TCM-specific safety benchmark**: 8 task categories covering prescription audit, syndrome diagnosis, safety refusal, herbal knowledge, drug interactions, dose safety, pregnancy contraindications, and adversarial robustness
- **Prescription rule engine**: Codified implementation of 28+ classical contraindication rules (Shiba Fan and Shijiu Wei), dose limits from the Chinese Pharmacopoeia, and pregnancy safety classifications
- **Multi-dimensional evaluation metrics**: Unsafe rate, hallucination rate, citation correctness, abstention utility, Expected Calibration Error (ECE), and red team pass rate
- **Adversarial red team suite**: 15 curated adversarial test cases across 5 risk categories with automated pass/fail judgment

---

## Key Features

### Prescription Rule Engine
- **Shiba Fan (18 Incompatibilities)**: 22 codified rules covering classical herb pairing violations (e.g., Wutou vs. Banxia, Gancao vs. Gansui)
- **Shijiu Wei (19 Mutual Fears)**: 11 rules for mutual avoidance pairs (e.g., Renshen vs. Wulingzhi)
- **Dose limit enforcement**: 22 toxic/potent herbs with maximum dosage limits from the Chinese Pharmacopoeia
- **Pregnancy contraindications**: 10 forbidden and 8 caution herbs for pregnancy safety
- **Herb alias normalization**: Handles processed vs. raw herb variants (e.g., Zhifuzi -> Fuzi)

### Safety Evaluator
- **Hallucination detection**: Identifies fabricated herb names, false efficacy claims, impossible dosage claims, and non-standard terminology
- **Citation verification**: Validates references to classical texts (Shanghan Lun, Bencao Gangmu), modern pharmacopoeia, and recognized TCM authorities
- **Risk categorization**: Classifies responses into risk types including dangerous self-medication, herb overdose, contraindication violations, and false efficacy claims
- **Abstention detection**: Evaluates whether the model appropriately refuses dangerous queries

### Benchmark Suite
- **8 task types**: prescription_audit, syndrome_diagnosis, safety_refusal, herbal_knowledge, drug_interaction, dose_safety, pregnancy_contraindication, red_team_adversarial
- **YAML-driven test cases**: Easily extensible evaluation task definitions
- **Aggregate reporting**: Per-task and global metrics with detailed case-level breakdowns

### Red Team Testing
- **5 adversarial categories**: Dangerous self-medication, herb overdose, ignoring contraindications, hallucinated herb effects, false drug interactions
- **15 curated test cases**: Covering critical, high, and medium severity levels
- **Automated judgment**: Pass/fail determination based on expected behavior (refuse, warn, correct_info, cite_source)

---

## Architecture

```
backend/
  main.py                        FastAPI application entry point
  config.py                      Application settings
  api/
    eval_api.py                  REST API endpoints (run benchmark, retrieve reports)
  models/
    safety_evaluator.py          Core evaluator: hallucination, citation, risk scoring
    rule_engine.py               Prescription audit engine (Shiba Fan, Shijiu Wei, dose limits)
    benchmark_suite.py           YAML-driven benchmark runner with aggregate metrics
    red_team_suite.py            Adversarial test case runner with 15 default cases
  tests/
    test_safety.py               Unit tests for safety evaluator
data/
  eval_tasks.yaml                8 evaluation tasks with 40+ test cases
scripts/
  quick_smoke_test.py            Quick validation script
  run_experiment.py              Full benchmark experiment runner
tests/
  test_smoke.py                  Integration smoke tests
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/eval/run` | POST | Run the full benchmark suite |
| `/api/eval/report` | GET | Retrieve the most recent benchmark report |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | REST API with async support |
| ML Core | scikit-learn, PyTorch | Model evaluation infrastructure |
| Data Processing | NumPy, Pandas, SciPy | Numerical computation |
| Configuration | PyYAML, Pydantic | Typed config and validation |
| Graph Analysis | NetworkX | Herb interaction network modeling |
| Testing | pytest | Unit and integration tests |
| CI/CD | GitHub Actions | Lint (ruff) + test pipeline |

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/TCM-LLM-SafetyEval.git
cd TCM-LLM-SafetyEval

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

### Run the Server

```bash
uvicorn backend.main:app --port 8029 --reload
```

The API documentation is available at `http://localhost:8029/docs` (Swagger UI).

### Run a Benchmark

```bash
# Via API
curl -X POST http://localhost:8029/api/eval/run

# Via script
python scripts/run_experiment.py
```

### Run Tests

```bash
pytest backend/tests/ -v
```

---

## Benchmark Metrics

The framework computes the following metrics per task and globally:

| Metric | Description |
|--------|-------------|
| `unsafe_rate` | Fraction of test cases with unsafe_score >= 0.5 |
| `hallucination_rate` | Fraction of test cases with hallucination_score >= 0.5 |
| `citation_correct_rate` | Fraction of test cases with valid citations |
| `abstention_utility` | Fraction of correctly abstained responses (when refusal is expected) |
| `ece` | Expected Calibration Error (confidence vs. accuracy alignment) |
| `red_team_pass_rate` | Adversarial test suite pass rate |

### Evaluation Task Coverage

| Task ID | Category | Description |
|---------|----------|-------------|
| `prescription_audit` | Prescription Review | Herb pairing violation detection (Shiba Fan, Shijiu Wei) |
| `syndrome_diagnosis` | Clinical Reasoning | TCM syndrome differentiation accuracy |
| `safety_refusal` | Safety | Model refusal of dangerous/out-of-scope queries |
| `herbal_knowledge` | Knowledge | TCM pharmacology knowledge accuracy |
| `drug_interaction` | Safety | TCM-Western medicine interaction awareness |
| `dose_safety` | Safety | Herb dosage safety knowledge |
| `pregnancy_contraindication` | Safety | Pregnancy-safe herb usage |
| `red_team_adversarial` | Robustness | Adversarial prompt resilience |

---

## Research

### Evaluation Design Philosophy

The benchmark is grounded in authoritative TCM references:
- **Chinese Pharmacopoeia** (2020 edition) for dose limits and safety classifications
- **Classical TCM texts** for contraindication rules (Shanghan Lun, Bencao Gangmu)
- **Modern pharmacovigilance** methodology for hallucination and safety scoring

### Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{tcm_llm_safetyeval,
  title   = {TCM-LLM-SafetyEval: A Safety Evaluation Benchmark for LLMs in TCM Clinical Applications},
  year    = {2025},
  url     = {https://github.com/your-org/TCM-LLM-SafetyEval}
}
```

---

## Roadmap

- [ ] Integration with live LLM APIs (GPT-4, Claude, Qwen) for automated evaluation
- [ ] Expanded test case library (100+ cases per task category)
- [ ] Multi-LLM comparative leaderboard
- [ ] Clinical expert validation study
- [ ] Support for additional TCM knowledge bases (TCMSP, SymMap, HERB)
- [ ] Web-based evaluation dashboard with visualization
- [ ] Multilingual evaluation support (Chinese, English, Japanese)

---

## Project Structure

```
TCM-LLM-SafetyEval/
|-- backend/
|   |-- api/                    # FastAPI route handlers
|   |-- models/                 # Core evaluation engines
|   |-- tests/                  # Backend unit tests
|   |-- config.py               # Application configuration
|   |-- main.py                 # FastAPI app entry point
|-- data/
|   |-- eval_tasks.yaml         # Evaluation task definitions
|-- scripts/
|   |-- quick_smoke_test.py     # Quick validation
|   |-- run_experiment.py       # Full benchmark runner
|-- tests/
|   |-- test_smoke.py           # Integration tests
|-- .github/workflows/ci.yml    # CI pipeline
|-- pyproject.toml              # Project metadata and dependencies
|-- REPRODUCE.md                # Reproduction instructions
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, collaboration, or feedback, please open an issue on GitHub or contact the maintainers.

---

<div align="center">

**Built to advance safe and responsible AI in Traditional Chinese Medicine**

</div>
