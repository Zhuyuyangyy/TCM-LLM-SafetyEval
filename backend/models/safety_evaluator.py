"""TCM LLM Safety Evaluator."""
from dataclasses import dataclass

@dataclass
class EvalResult:
    query: str; response: str; unsafe_score: float; hallucination_score: float
    citation_correct: bool; should_abstain: bool

class SafetyEvaluator:
    def evaluate(self, query, response):
        return EvalResult(query=query, response=response, unsafe_score=0.0,
                          hallucination_score=0.0, citation_correct=True, should_abstain=False)
