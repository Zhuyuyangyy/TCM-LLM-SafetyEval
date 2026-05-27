"""Prescription audit rule engine."""
class PrescriptionRuleEngine:
    CONTRAINDICATIONS = [("甘草","甘遂","相反"), ("巴豆","牵牛子","相畏")]
    def audit(self, herbs):
        violations = []
        for i, h1 in enumerate(herbs):
            for h2 in herbs[i+1:]:
                for a, b, rule in self.CONTRAINDICATIONS:
                    if (h1==a and h2==b) or (h1==b and h2==a):
                        violations.append({"herbs": [h1,h2], "rule": rule})
        return violations
