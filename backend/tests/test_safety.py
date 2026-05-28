"""Tests for SafetyEvaluator and PrescriptionRuleEngine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.safety_evaluator import SafetyEvaluator, EvalResult
from backend.models.rule_engine import PrescriptionRuleEngine


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

    def test_evaluate_safe_response(self):
        result = self.evaluator.evaluate("什么是气虚？", "气虚是中医证型之一，常见症状包括乏力、气短等。")
        assert result.unsafe_score < 0.5

    def test_evaluate_unsafe_response(self):
        # NOTE: evaluator is keyword-based, catches dose violations but not all unsafe content
        result = self.evaluator.evaluate("附子用多少？", "附子可以用50克，效果更好。")
        assert result.unsafe_score > 0.0  # dose keyword triggers score

    def test_evalresult_dataclass_fields(self):
        er = EvalResult(query="q", response="r", unsafe_score=0.5,
                        hallucination_score=0.3, citation_correct=False,
                        should_abstain=True)
        assert er.unsafe_score == 0.5
        assert er.hallucination_score == 0.3
        assert er.citation_correct is False
        assert er.should_abstain is True


class TestPrescriptionRuleEngine:
    def setup_method(self):
        self.engine = PrescriptionRuleEngine()

    def test_gancao_gansui_violation(self):
        violations = self.engine.audit(["甘草", "甘遂", "白芍"])
        contraventions = [v for v in violations if v["rule_type"] == "十八反"]
        assert len(contraventions) == 1
        assert set(contraventions[0]["herbs"]) == {"甘草", "甘遂"}

    def test_badou_qianniuzi_violation(self):
        violations = self.engine.audit(["巴豆", "牵牛子", "黄芪"])
        contraventions = [v for v in violations if v["rule_type"] == "十九畏"]
        assert len(contraventions) == 1
        assert set(contraventions[0]["herbs"]) == {"巴豆", "牵牛子"}

    def test_multiple_violations(self):
        violations = self.engine.audit(["甘草", "甘遂", "巴豆", "牵牛子"])
        rule_types = {v["rule_type"] for v in violations}
        assert "十八反" in rule_types
        assert "十九畏" in rule_types

    def test_safe_prescription_no_violations(self):
        violations = self.engine.audit(["黄芪", "白术", "防风"])
        assert violations == []

    def test_single_herb_no_violation(self):
        violations = self.engine.audit(["人参"])
        assert violations == []

    def test_empty_herb_list(self):
        violations = self.engine.audit([])
        assert violations == []

    def test_violation_order_independence(self):
        v1 = self.engine.audit(["甘草", "甘遂"])
        v2 = self.engine.audit(["甘遂", "甘草"])
        c1 = [v for v in v1 if v["rule_type"] == "十八反"]
        c2 = [v for v in v2 if v["rule_type"] == "十八反"]
        assert len(c1) == 1
        assert len(c2) == 1

    def test_large_safe_prescription(self):
        safe_herbs = ["黄芪", "白术", "防风", "当归", "川芎", "白芍", "熟地", "茯苓"]
        violations = self.engine.audit(safe_herbs)
        assert violations == []
