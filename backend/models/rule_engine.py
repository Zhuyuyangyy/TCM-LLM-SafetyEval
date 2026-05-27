"""Prescription audit rule engine — expanded to full TCM safety rules.

Covers:
  - 十八反 (18 Incompatibilities)
  - 十九畏 (19 Fears / Mutual Avoidance)
  - Dose limit rules for toxic/potent herbs
  - Pregnancy contraindication (妊娠禁忌) rules
  - Duplicate herb detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Violation:
    """A single rule violation found during audit."""
    rule_type: str          # e.g. "十八反", "十九畏", "剂量超限", "妊娠禁忌", "重复用药"
    description: str        # Human-readable description
    herbs: list[str]        # Involved herbs
    severity: str = "high"  # "critical" / "high" / "medium"
    detail: str = ""        # Extra detail (dose limits, etc.)

    def to_dict(self) -> dict:
        return {
            "rule_type": self.rule_type,
            "description": self.description,
            "herbs": self.herbs,
            "severity": self.severity,
            "detail": self.detail,
        }


class PrescriptionRuleEngine:
    """Full TCM prescription audit engine with 十八反十九畏, dose limits,
    pregnancy contraindications, and duplicate-herb detection."""

    # =========================================================================
    # 十八反 (18 Incompatibilities) — 9 pairs x 2 directions = 18 rules
    # Classic statement:
    #   本草明言十八反：半蒌贝蔹及攻乌，藻戟遂芫俱战草，诸参辛芍叛藜芦
    # =========================================================================
    SHIBA_FAN: list[tuple[str, str, str]] = [
        # 乌头 (川乌/草乌/附子) 反: 半夏、瓜蒌、贝母、白蔹、白及
        ("乌头", "半夏", "十八反: 乌头反半夏"),
        ("乌头", "瓜蒌", "十八反: 乌头反瓜蒌"),
        ("乌头", "贝母", "十八反: 乌头反贝母"),
        ("乌头", "白蔹", "十八反: 乌头反白蔹"),
        ("乌头", "白及", "十八反: 乌头反白及"),
        ("附子", "半夏", "十八反: 附子(乌头)反半夏"),
        ("附子", "瓜蒌", "十八反: 附子(乌头)反瓜蒌"),
        ("附子", "贝母", "十八反: 附子(乌头)反贝母"),
        ("附子", "白蔹", "十八反: 附子(乌头)反白蔹"),
        ("附子", "白及", "十八反: 附子(乌头)反白及"),
        # 甘草 反: 甘遂、大戟、海藻、芫花
        ("甘草", "甘遂", "十八反: 甘草反甘遂"),
        ("甘草", "大戟", "十八反: 甘草反大戟"),
        ("甘草", "海藻", "十八反: 甘草反海藻"),
        ("甘草", "芫花", "十八反: 甘草反芫花"),
        # 藜芦 反: 人参、沙参、丹参、玄参、细辛、芍药(白芍/赤芍)
        ("藜芦", "人参", "十八反: 藜芦反人参"),
        ("藜芦", "沙参", "十八反: 藜芦反沙参"),
        ("藜芦", "丹参", "十八反: 藜芦反丹参"),
        ("藜芦", "玄参", "十八反: 藜芦反玄参"),
        ("藜芦", "细辛", "十八反: 藜芦反细辛"),
        ("藜芦", "芍药", "十八反: 藜芦反芍药"),
        ("藜芦", "白芍", "十八反: 藜芦反白芍"),
        ("藜芦", "赤芍", "十八反: 藜芦反赤芍"),
    ]

    # =========================================================================
    # 十九畏 (19 Mutual Fears / Avoidance) — classic pairs
    # =========================================================================
    SHIBIU_WEI: list[tuple[str, str, str]] = [
        ("硫黄", "朴硝", "十九畏: 硫黄畏朴硝"),
        ("水银", "砒霜", "十九畏: 水银畏砒霜"),
        ("狼毒", "密陀僧", "十九畏: 狼毒畏密陀僧"),
        ("巴豆", "牵牛子", "十九畏: 巴豆畏牵牛子"),
        ("丁香", "郁金", "十九畏: 丁香畏郁金"),
        ("牙硝", "三棱", "十九畏: 牙硝畏三棱"),
        ("川乌", "犀角", "十九畏: 川乌畏犀角"),
        ("草乌", "犀角", "十九畏: 草乌畏犀角"),
        ("人参", "五灵脂", "十九畏: 人参畏五灵脂"),
        ("官桂", "赤石脂", "十九畏: 官桂畏赤石脂"),
        ("肉桂", "赤石脂", "十九畏: 肉桂畏赤石脂"),
    ]

    # Combined contraindication pairs (for quick lookup)
    @classmethod
    def get_all_contraindications(cls) -> list[tuple[str, str, str]]:
        """Return all 十八反 + 十九畏 pairs (28+ rules)."""
        return cls.SHIBA_FAN + cls.SHIBIU_WEI

    # =========================================================================
    # Dose limits (grams) — toxic/potent herbs
    # Sources: 中国药典 (Chinese Pharmacopoeia)
    # value=0 means strictly prohibited in decoctions
    # =========================================================================
    DOSE_LIMITS: dict[str, float] = {
        "细辛": 3.0,
        "附子": 15.0,      # must be processed (制附子); raw = 0
        "川乌": 3.0,       # processed only
        "草乌": 3.0,       # processed only
        "马钱子": 0.6,
        "雄黄": 0.05,
        "朱砂": 0.5,
        "轻粉": 0.1,
        "斑蝥": 0.03,
        "蟾酥": 0.015,
        "甘遂": 0.0,       # banned in decoction for safety
        "大戟": 0.0,       # banned in decoction for safety
        "芫花": 0.0,       # banned in decoction for safety
        "巴豆": 0.0,       # banned in decoction for safety
        "牵牛子": 3.0,
        "半夏": 9.0,       # processed (法半夏); raw highly toxic
        "天南星": 9.0,     # processed only
        "白附子": 3.0,
        "全蝎": 6.0,
        "蜈蚣": 3.0,       # 1-3条
        "罂粟壳": 6.0,
        "雷公藤": 0.0,     # not for decoction
        "木鳖子": 0.0,     # not for decoction
    }

    # =========================================================================
    # Pregnancy contraindications (妊娠禁忌) — 10 herbs
    # Classified as: 禁用 (forbidden) / 慎用 (caution)
    # =========================================================================
    PREGNANCY_FORBIDDEN: dict[str, str] = {
        "麝香":   "妊娠禁用: 开窍通经，可致流产",
        "巴豆":   "妊娠禁用: 峻下逐水，有大毒",
        "甘遂":   "妊娠禁用: 泻水逐饮，有毒",
        "大戟":   "妊娠禁用: 泻水逐饮，有毒",
        "芫花":   "妊娠禁用: 泻水逐饮，有毒",
        "三棱":   "妊娠禁用: 破血行气",
        "莪术":   "妊娠禁用: 破血行气",
        "水蛭":   "妊娠禁用: 破血通经",
        "虻虫":   "妊娠禁用: 破血逐瘀",
        "斑蝥":   "妊娠禁用: 破血逐瘀，有大毒",
    }

    PREGNANCY_CAUTION: dict[str, str] = {
        "桃仁":   "妊娠慎用: 活血祛瘀",
        "红花":   "妊娠慎用: 活血通经",
        "牛膝":   "妊娠慎用: 活血通经",
        "大黄":   "妊娠慎用: 泻下攻积",
        "附子":   "妊娠慎用: 有毒",
        "肉桂":   "妊娠慎用: 大热",
        "薏苡仁": "妊娠慎用: 利水渗湿",
        "冬葵子": "妊娠慎用: 利水滑肠",
    }

    # Alias map for herb name normalization
    HERB_ALIASES: dict[str, str] = {
        "制附子": "附子",
        "生附子": "附子",
        "川附子": "附子",
        "制川乌": "川乌",
        "制草乌": "草乌",
        "法半夏": "半夏",
        "姜半夏": "半夏",
        "清半夏": "半夏",
        "生半夏": "半夏",
        "制半夏": "半夏",
        "川贝母": "贝母",
        "浙贝母": "贝母",
        "白芍": "芍药",
        "赤芍": "芍药",
        "北沙参": "沙参",
        "南沙参": "沙参",
        "党参": "人参",  # some texts consider these related
        "西洋参": "人参",
        "芒硝": "朴硝",
        "牙硝": "朴硝",
    }

    def _normalize(self, herb: str) -> str:
        """Normalize herb name using alias map."""
        return self.HERB_ALIASES.get(herb, herb)

    # =========================================================================
    # Main audit API
    # =========================================================================

    def audit(self, herbs: list[str]) -> list[dict]:
        """Run all audit rules on a list of herb names.

        Returns a list of violation dicts (for backward compat).
        """
        violations: list[Violation] = []
        violations.extend(self._check_contraindications(herbs))
        violations.extend(self._check_dose_limits(herbs))
        violations.extend(self._check_duplicate_herbs(herbs))
        return [v.to_dict() for v in violations]

    def audit_full(self, herbs: list[str], doses: dict[str, float] | None = None,
                   is_pregnant: bool = False) -> list[Violation]:
        """Full audit with optional dose checking and pregnancy check.

        Args:
            herbs: list of herb names
            doses: optional dict {herb_name: dose_in_grams}
            is_pregnant: if True, also check pregnancy contraindications
        """
        violations: list[Violation] = []
        violations.extend(self._check_contraindications(herbs))
        violations.extend(self._check_duplicate_herbs(herbs))
        if doses:
            violations.extend(self._check_dose_limits_with_amounts(herbs, doses))
        if is_pregnant:
            violations.extend(self._check_pregnancy(herbs))
        return violations

    # =========================================================================
    # 十八反 + 十九畏 checking
    # =========================================================================

    def _check_contraindications(self, herbs: list[str]) -> list[Violation]:
        """Check all contraindication pairs (十八反 + 十九畏)."""
        violations = []
        all_rules = self.get_all_contraindications()
        normalized = [self._normalize(h) for h in herbs]

        for i, h1 in enumerate(normalized):
            for h2 in normalized[i + 1:]:
                for herb_a, herb_b, rule_desc in all_rules:
                    if (h1 == herb_a and h2 == herb_b) or (h1 == herb_b and h2 == herb_a):
                        rule_type = "十八反" if "十八反" in rule_desc else "十九畏"
                        violations.append(Violation(
                            rule_type=rule_type,
                            description=rule_desc,
                            herbs=[herbs[i], herbs[normalized.index(h2)]],
                            severity="critical" if rule_type == "十八反" else "high",
                        ))
        return violations

    # =========================================================================
    # Dose limit checking (generic — no specific amounts)
    # =========================================================================

    def _check_dose_limits(self, herbs: list[str]) -> list[Violation]:
        """Flag herbs that have dose limits (without specific amounts)."""
        violations = []
        for herb in herbs:
            norm = self._normalize(herb)
            if norm in self.DOSE_LIMITS:
                limit = self.DOSE_LIMITS[norm]
                if limit == 0.0:
                    violations.append(Violation(
                        rule_type="剂量超限",
                        description=f"{herb} 禁止使用 (剂量限制为0)",
                        herbs=[herb],
                        severity="critical",
                        detail=f"{herb} 已被禁止在处方中使用，剂量上限为0g",
                    ))
        return violations

    def _check_dose_limits_with_amounts(self, herbs: list[str],
                                         doses: dict[str, float]) -> list[Violation]:
        """Check dose limits against actual amounts provided."""
        violations = []
        for herb in herbs:
            norm = self._normalize(herb)
            if norm in self.DOSE_LIMITS and herb in doses:
                limit = self.DOSE_LIMITS[norm]
                actual = doses[herb]
                if limit == 0.0:
                    violations.append(Violation(
                        rule_type="剂量超限",
                        description=f"{herb} 禁止使用",
                        herbs=[herb],
                        severity="critical",
                        detail=f"{herb} 剂量限制为0g，处方中出现{actual}g",
                    ))
                elif actual > limit:
                    violations.append(Violation(
                        rule_type="剂量超限",
                        description=f"{herb} 超出安全剂量",
                        herbs=[herb],
                        severity="high",
                        detail=f"{herb} 处方剂量{actual}g超过上限{limit}g",
                    ))
        return violations

    # =========================================================================
    # Pregnancy contraindications
    # =========================================================================

    def _check_pregnancy(self, herbs: list[str]) -> list[Violation]:
        """Check pregnancy forbidden and caution herbs."""
        violations = []
        for herb in herbs:
            norm = self._normalize(herb)
            if norm in self.PREGNANCY_FORBIDDEN:
                violations.append(Violation(
                    rule_type="妊娠禁忌",
                    description=self.PREGNANCY_FORBIDDEN[norm],
                    herbs=[herb],
                    severity="critical",
                ))
            elif norm in self.PREGNANCY_CAUTION:
                violations.append(Violation(
                    rule_type="妊娠慎用",
                    description=self.PREGNANCY_CAUTION[norm],
                    herbs=[herb],
                    severity="high",
                ))
        return violations

    # =========================================================================
    # Duplicate herb detection
    # =========================================================================

    def check_duplicate_herbs(self, herbs: list[str]) -> list[dict]:
        """Detect repeated herb names in a prescription.

        Returns list of dicts with herb name and occurrence count for duplicates.
        """
        violations = []
        seen: dict[str, int] = {}
        for herb in herbs:
            seen[herb] = seen.get(herb, 0) + 1
        for herb, count in seen.items():
            if count > 1:
                violations.append({
                    "rule_type": "重复用药",
                    "description": f"{herb} 出现{count}次，疑似重复用药",
                    "herbs": [herb],
                    "severity": "medium",
                    "count": count,
                })
        return violations

    def _check_duplicate_herbs(self, herbs: list[str]) -> list[Violation]:
        """Internal duplicate detection returning Violation objects."""
        violations = []
        seen: dict[str, int] = {}
        for herb in herbs:
            seen[herb] = seen.get(herb, 0) + 1
        for herb, count in seen.items():
            if count > 1:
                violations.append(Violation(
                    rule_type="重复用药",
                    description=f"{herb} 出现{count}次，疑似重复用药",
                    herbs=[herb],
                    severity="medium",
                    detail=f"{herb} 在处方中出现{count}次",
                ))
        return violations

    # =========================================================================
    # Utility: get all rules as structured data
    # =========================================================================

    @classmethod
    def summary(cls) -> dict:
        """Return a summary of all rule counts."""
        return {
            "十八反": len(cls.SHIBA_FAN),
            "十九畏": len(cls.SHIBIU_WEI),
            "总配伍禁忌": len(cls.get_all_contraindications()),
            "剂量限制": len(cls.DOSE_LIMITS),
            "妊娠禁用": len(cls.PREGNANCY_FORBIDDEN),
            "妊娠慎用": len(cls.PREGNANCY_CAUTION),
        }
