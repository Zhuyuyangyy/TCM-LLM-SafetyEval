# TCM-LLM-SafetyEval

中医处方审核与中医药大模型安全评测基准

## 核心创新点

1. **TCM 专用评测基准** — 针对中医药领域的大模型安全性评测
2. **处方合理性审核** — 自动检测处方中的配伍禁忌、剂量异常
3. **多维度评测** — 安全性、准确性、一致性、可解释性
4. **红队测试** — 对抗性测试用例生成

## 快速开始

```bash
pip install -e .
uvicorn backend.main:app --port 8023 --reload
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/eval/prescription` | POST | 处方合理性审核 |
| `/api/eval/safety` | POST | 安全性评测 |
| `/api/eval/benchmark` | POST | 基准测试运行 |

## License

MIT
