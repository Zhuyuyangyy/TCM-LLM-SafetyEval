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

## Important Limitations and Disclaimers

> **Please read before using or citing this benchmark.**

1. **No real LLM evaluation included.** The current release evaluates only
   predefined (hand-crafted) responses from YAML test cases and mock responses
   in the red team suite.  No actual LLM API (GPT-4, Claude, Qwen, etc.) has
   been queried.  All reported metrics reflect the evaluator's scoring logic on
   static data, not the safety behaviour of any production LLM.  A future
   release should integrate live LLM API calls to produce meaningful
   evaluation results.

2. **Small test corpus (38 cases).** The YAML test file contains only 38 cases
   across 8 task categories (range: 2-10 cases per task).  This is far below
   the sample size needed for statistically reliable safety claims.  Results
   should be treated as preliminary proof-of-concept, not as definitive
   benchmarks.  The roadmap targets 100+ cases per category.

3. **Circular red team testing.** The 15 red team adversarial cases include
   hand-crafted `mock_response` strings that were written to be safe and
   correct.  These responses are then evaluated by the same rule-based
   safety evaluator that checks for the very keywords the mock responses
   were designed to include (e.g., refusal language, warning terms).  This
   creates a circular validation loop: the test data is authored to pass the
   evaluator, so the pass rate reflects author intent, not model capability.
   Genuine red team testing requires evaluating *real* LLM responses to
   adversarial prompts.

4. **ECE calibration metric uses semantic proxies.** The Expected Calibration
   Error (ECE) computation uses `hallucination_score` as a stand-in for model
   confidence and `unsafe_score` as a stand-in for correctness.  These are
   semantically distinct concepts.  ECE values should be interpreted as rough
   heuristics, not as rigorous calibration measurements.  See the
   `CalibrationMetric` class docstring for details.

5. **Keyword-based evaluation only.** All scoring (safety, hallucination,
   citation) is based on keyword matching and regex patterns, not on semantic
   understanding.  This approach will miss nuanced unsafe content that avoids
   the specific keywords and may flag benign content that happens to contain
   them.

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

## References

The following works informed the design of this benchmark:

### LLM Safety Evaluation

- Weidinger, L., Mellor, J., Rauh, M., et al. (2021). "Ethical and social risks of harm from Language Models." *arXiv preprint arXiv:2112.04359*.
- Liang, P., Bommasani, R., Lee, T., et al. (2022). "Holistic Evaluation of Language Models." *arXiv preprint arXiv:2211.09110*.
- Wang, Y., Zhong, W., Li, L., et al. (2023). "Aligning Large Language Models with Human: A Survey." *arXiv preprint arXiv:2307.12966*.
- Guo, Z., Jin, R., Liu, C., et al. (2024). "Evaluating Large Language Models: A Comprehensive Survey." *arXiv preprint arXiv:2310.19736*.
- Sun, H., Zhang, Z., He, J., et al. (2024). "SafetyBench: Evaluating the Safety of Large Language Models." *Proceedings of ACL 2024*.

### TCM Toxicology and Safety

- 国家药典委员会. (2020).《中华人民共和国药典》(2020年版). 中国医药科技出版社.
- 高学敏 主编. (2007).《中药学》(第二版). 中国中医药出版社.
- 钟赣生 主编. (2012).《中药学》(全国中医药行业高等教育"十二五"规划教材). 中国中医药出版社.
- 张廷模 主编. (2016).《中药学》(第十版). 中国中医药出版社.
- 国家中医药管理局. (2010).《中药学临床药论》. 中国中医药出版社.

### LLM Applications in Healthcare / TCM

- Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., et al. (2023). "Large language models in medicine." *Nature Medicine*, 29(8), 1930-1940.
- Wang, X., Gong, Z., Wang, G., et al. (2023). "ChatGPT Performs on the Chinese National Medical Licensing Examination." *Journal of Medical Systems*, 47(1), 86.
- Liu, J., Wang, C., Liu, S., et al. (2024). "TCM-SD: A Benchmark for Probing the Syndrome Differentiation of Traditional Chinese Medicine." *Proceedings of AAAI 2024*.

---

## Paper Status

> **Status: Draft / Pre-submission.** The accompanying SCI paper
> (`docs/SCI_Paper_Skeleton.md`) is currently a skeleton with sections yet to
> be filled.  The target journals are *Qingbao Kexue* (情报科学) or
> *Shuju Fenxi yu Zhishi Faxian* (数据分析与知识发现).  A full paper with
> real LLM evaluation results, expanded corpus, and clinical expert
> validation is required before submission.

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
