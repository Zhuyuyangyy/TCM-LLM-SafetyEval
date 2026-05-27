"""Tests for SafetyEvaluator and PrescriptionRuleEngine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.safety_evaluator import SafetyEvaluator, EvalResult
from backend.models.rule_engine import PrescriptionRuleEngine


# ---------------------------------------------------------------------------
# SafetyEvaluator tests
# ---------------------------------------------------------------------------

class TestSafetyEvaluator:
    def setup_method(self):
        self.evaluator = SafetyEvaluator()

    def test_evaluate_returns_evalresult(self):
        result = self.evaluator.evaluate("什么是气虚？", "气虚是中医证型之一。")
        assert isinstance(result, EvalResult)

    def test_evaluate_populates_fields(self):
        q, r = "何为肝郁？", "肝郁指肝气郁结。"
        result = self.evaluator.evaluate(q, r)
        assert result.query == q
        assert result.response == r
        assert isinstance(result.unsafe_score, float)
        assert isinstance(result.hallucination_score, float)
        assert isinstance(result.citation_correct, bool)
        assert isinstance(result.should_abstain, bool)

    def test_evaluate_default_scores_are_safe(self):
        result = self.evaluator.evaluate("test", "test")
        assert result.unsafe_score == 0.0
        assert result.hallucination_score == 0.0
        assert result.citation_correct is True
        assert result.should_abstain is False

    def test_evalresult_dataclass_fields(self):
        er = EvalResult(query="q", response="r", unsafe_score=0.5,
                        hallucination_score=0.3, citation_correct=False,
                        should_abstain=True)
        assert er.unsafe_score == 0.5
        assert er.hallucination_score == 0.3
        assert er.citation_correct is False
        assert er.should_abstain is True


# ---------------------------------------------------------------------------
# PrescriptionRuleEngine tests
# ---------------------------------------------------------------------------

class TestPrescriptionRuleEngine:
    def setup_method(self):
        self.engine = PrescriptionRuleEngine()

    # --- contravention detection ---

    def test_gancao_gansui_violation(self):
        violations = self.engine.audit(["甘草", "甘遂", "白芍"])
        assert len(violations) == 1
        assert violations[0]["rule"] == "相反"
        assert set(violations[0]["herbs"]) == {"甘草", "甘遂"}

    def test_badou_qianniuzi_violation(self):
        violations = self.engine.audit(["巴豆", "牵牛子", "黄芪"])
        assert len(violations) == 1
        assert violations[0]["rule"] == "相畏"
        assert set(violations[0]["herbs"]) == {"巴豆", "牵牛子"}

    def test_multiple_violations(self):
        violations = self.engine.audit(["甘草", "甘遂", "巴豆", "牵牛子"])
        assert len(violations) == 2
        rules_found = {v["rule"] for v in violations}
        assert "相反" in rules_found
        assert "相畏" in rules_found

    # --- safe prescriptions ---

    def test_safe_prescription_no_violations(self):
        violations = self.engine.audit(["黄芪", "白术", "防风"])
        assert violations == []

    def test_single_herb_no_violation(self):
        violations = self.engine.audit(["人参"])
        assert violations == []

    def test_empty_herb_list(self):
        violations = self.engine.audit([])
        assert violations == []

    # --- order independence ---

    def test_violation_order_independence(self):
        """Contraindication detected regardless of herb order."""
        v1 = self.engine.audit(["甘草", "甘遂"])
        v2 = self.engine.audit(["甘遂", "甘草"])
        assert len(v1) == 1
        assert len(v2) == 1

    # --- no false positives on large safe list ---

    def test_large_safe_prescription(self):
        safe_herbs = ["黄芪", "白术", "防风", "当归", "川芎", "白芍", "熟地", "茯苓"]
        violations = self.engine.audit(safe_herbs)
        assert violations == []
