from typing import Dict, List


def stringify_rule_path(path: List[str]) -> str:
    if not path:
        return "always true"
    return " AND ".join(path)


def _render_rule_summary(target_summary: Dict) -> str:
    parts: List[str] = []
    mean_target = target_summary.get("target_mean")
    if mean_target is not None:
        parts.append(f"mean target={float(mean_target):.4g}")

    span_start = target_summary.get("span_start_year_median")
    span_end = target_summary.get("span_end_year_median")
    if span_start is not None and span_end is not None:
        parts.append(f"typical span={int(span_start)}-{int(span_end)}")

    span_len = target_summary.get("span_length_mean")
    if span_len is not None:
        parts.append(f"avg span length={float(span_len):.2f}")

    return "; ".join(parts)


def render_natural_language_rules(
    method_name: str,
    rules: List[Dict],
    context_note: str,
) -> str:
    lines = [
        f"Method: {method_name}",
        context_note,
        "",
        "Natural-language rules:",
    ]
    if not rules:
        lines.append("- No positive rules found.")
        return "\n".join(lines)

    for i, rule in enumerate(rules, start=1):
        rule_text = stringify_rule_path(rule.get("rule", []))
        target_summary = rule.get("target_summary", {})
        summary_text = _render_rule_summary(target_summary)
        if not summary_text:
            lines.append(f"{i}. IF {rule_text}, THEN interesting.")
        else:
            lines.append(f"{i}. IF {rule_text}, THEN interesting ({summary_text}).")
    return "\n".join(lines)
