#!/usr/bin/env python3
"""
Compute Opus vs Sonnet agreement, within-model consistency, Cohen's kappa, and disagreement taxonomy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from anthropic import Anthropic

from _anthropic_util import call_anthropic, extract_json_object

DIR = Path(__file__).resolve().parent
MODEL_SONNET = "claude-sonnet-4-6"

# USD per 1M tokens — override with env vars; approximate placeholders (update to your org’s rates).
PRICE_OPUS_IN = float(os.environ.get("PRICE_OPUS_INPUT_PER_MTOK", "15"))
PRICE_OPUS_OUT = float(os.environ.get("PRICE_OPUS_OUTPUT_PER_MTOK", "75"))
PRICE_SONNET_IN = float(os.environ.get("PRICE_SONNET_INPUT_PER_MTOK", "3"))
PRICE_SONNET_OUT = float(os.environ.get("PRICE_SONNET_OUTPUT_PER_MTOK", "15"))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def norm_grade(g: str | None) -> bool | None:
    if not g:
        return None
    s = str(g).strip().upper()
    if s == "MET":
        return True
    if s == "NOT_MET":
        return False
    return None


def majority_met(votes: list[bool | None]) -> bool | None:
    clean = [v for v in votes if v is not None]
    if not clean:
        return None
    c = Counter(clean)
    met_n = c.get(True, 0)
    not_n = c.get(False, 0)
    if met_n > not_n:
        return True
    if not_n > met_n:
        return False
    return None


def cohen_kappa_binary(y1: list[bool], y2: list[bool]) -> float | None:
    if len(y1) != len(y2) or not y1:
        return None
    n = len(y1)
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    p_y1 = sum(y1) / n
    p_y2 = sum(y2) / n
    p_e = p_y1 * p_y2 + (1 - p_y1) * (1 - p_y2)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def flatten_grades(grades_obj: dict[str, Any] | None) -> list[tuple[str, str, bool | None, str]]:
    """(criterion_name, dimension_id, is_met, evidence)"""
    if not grades_obj or "grades" not in grades_obj:
        return []
    out: list[tuple[str, str, bool | None, str]] = []
    for block in grades_obj["grades"]:
        cname = (block.get("criterion_name") or "").strip()
        for dg in block.get("dimension_grades") or []:
            did = (dg.get("dimension_id") or "").strip()
            g = norm_grade(dg.get("grade"))
            ev = (dg.get("evidence") or "").strip()
            out.append((cname, did, g, ev))
    return out


def norm_key(criterion: str, dim_id: str) -> tuple[str, str]:
    return (criterion.strip().lower(), dim_id.strip().lower())


def build_canonical_dims(cases: list[dict]) -> dict[tuple[str, str, str], tuple[str, str, str, str]]:
    """(scenario_id, crit_lower, id_lower) -> (scenario_id, criterion_name, dim_id, label)"""
    m: dict[tuple[str, str, str], tuple[str, str, str, str]] = {}
    for case in cases:
        sid = case["scenario_id"]
        for crit in case["rubric"].get("rubric", []):
            cname = crit.get("name") or ""
            for dm in crit.get("dimensions") or []:
                did = dm.get("id") or ""
                lab = dm.get("label") or did
                m[(sid, cname.strip().lower(), did.strip().lower())] = (sid, cname, did, lab)
    return m


def map_to_canonical(
    sid: str,
    cname: str,
    did: str,
    canon: dict[tuple[str, str, str], tuple[str, str, str, str]],
) -> tuple[str, str] | None:
    k = (sid, cname.strip().lower(), did.strip().lower())
    if k in canon:
        _, cn, d, _ = canon[k]
        return cn, d
    # fall back: match dim id only within scenario
    for (s, cl, il), (_, cn, d, _) in canon.items():
        if s == sid and il == did.strip().lower():
            return cn, d
    return None


DISAGREE_CLASSIFY_SYSTEM = "You classify sources of disagreement between two graders. Reply with ONLY valid JSON."

DISAGREE_CLASSIFY_USER = """Two models graded the same rubric dimension on the same draft and disagreed.

**Criterion:** __CRIT__
**Dimension id:** __DIM_ID__
**Dimension label:** __DIM_LABEL__

**Draft excerpt (may be truncated):**
---
__DRAFT__
---

**Opus:** __OPUS_G__ — __OPUS_E__

**Sonnet:** __SON_G__ — __SON_E__

Classify the disagreement as exactly one of:
- "dimension_ambiguity" — the dimension wording is vague or untestable so graders guess differently
- "judgment_call" — both interpretations are defensible
- "clear_error" — one grader is clearly wrong given the dimension and draft

Respond with ONLY this JSON:
{"classification": "<dimension_ambiguity|judgment_call|clear_error>", "rationale": "<one short sentence>"}
"""


def classify_disagreement(
    client: Anthropic,
    *,
    criterion_name: str,
    dimension_id: str,
    dimension_label: str,
    draft_excerpt: str,
    opus_g: bool,
    sonnet_g: bool,
    opus_ev: str,
    sonnet_ev: str,
) -> dict[str, Any]:
    user = (
        DISAGREE_CLASSIFY_USER.replace("__CRIT__", criterion_name)
        .replace("__DIM_ID__", dimension_id)
        .replace("__DIM_LABEL__", dimension_label)
        .replace("__DRAFT__", draft_excerpt[:4000])
        .replace("__OPUS_G__", "MET" if opus_g else "NOT_MET")
        .replace("__SON_G__", "MET" if sonnet_g else "NOT_MET")
        .replace("__OPUS_E__", opus_ev or "(none)")
        .replace("__SON_E__", sonnet_ev or "(none)")
    )
    raw, _, _ = call_anthropic(
        client,
        model=MODEL_SONNET,
        system=DISAGREE_CLASSIFY_SYSTEM,
        user=user,
        max_tokens=1024,
        temperature=0.0,
    )
    data = extract_json_object(raw)
    if not data or "classification" not in data:
        return {"classification": "unknown", "rationale": raw[:200]}
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test-cases", type=Path, default=DIR / "test_cases.json")
    ap.add_argument("-g", "--grading", type=Path, default=DIR / "grading_results.json")
    ap.add_argument("-o", "--json-out", type=Path, default=DIR / "agreement_report.json")
    ap.add_argument("-m", "--md-out", type=Path, default=DIR / "agreement_report.md")
    args = ap.parse_args()

    cases = json.loads(args.test_cases.read_text(encoding="utf-8"))
    grading = json.loads(args.grading.read_text(encoding="utf-8"))

    draft_text: dict[tuple[str, str], str] = {}
    for c in cases:
        for d in c["drafts"]:
            draft_text[(c["scenario_id"], d["draft_id"])] = d["text"]

    canon = build_canonical_dims(cases)
    dim_label: dict[tuple[str, str, str], str] = {}
    for (_s, _cl, _il), (_sid, cn, did, lab) in canon.items():
        dim_label[(_sid, cn, did)] = lab

    # Index: (sid, draft_id, model, run) -> list of (crit, dim_id, grade, evidence)
    by_run: dict[tuple[str, str, str, int], list[tuple[str, str, bool | None, str]]] = defaultdict(list)
    latency: dict[str, list[int]] = defaultdict(list)
    tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    for row in grading:
        if row.get("error"):
            continue
        g = row.get("grades")
        if not g:
            continue
        sid = row["scenario_id"]
        did = row["draft_id"]
        model = row["model"]
        run = int(row["run"])
        lat = row.get("latency_ms")
        if isinstance(lat, int):
            latency[model].append(lat)
        it = row.get("input_tokens") or 0
        ot = row.get("output_tokens") or 0
        tokens[model]["in"] += int(it)
        tokens[model]["out"] += int(ot)
        for item in flatten_grades(g):
            by_run[(sid, did, model, run)].append(item)

    # Aggregate per (sid, draft, model, crit, dim_id) -> 3 grades + evidence from run 1
    dim_runs: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    keys_seen = set()
    for (sid, draft_id, model, run), items in by_run.items():
        for cname, did, g, ev in items:
            mapped = map_to_canonical(sid, cname, did, canon)
            if not mapped:
                cn, dcanonical = cname, did
            else:
                cn, dcanonical = mapped
            k = (sid, draft_id, cn, dcanonical)
            if k not in dim_runs:
                dim_runs[k] = {
                    "opus": [None, None, None],
                    "sonnet": [None, None, None],
                    "evidence_opus": ["", "", ""],
                    "evidence_sonnet": ["", "", ""],
                }
            idx = run - 1
            if 0 <= idx < 3:
                if model == "opus":
                    dim_runs[k]["opus"][idx] = g
                    dim_runs[k]["evidence_opus"][idx] = ev
                elif model == "sonnet":
                    dim_runs[k]["sonnet"][idx] = g
                    dim_runs[k]["evidence_sonnet"][idx] = ev

    # Per-dimension metrics
    dim_agreements: list[bool] = []
    opus_consistent: list[bool] = []
    sonnet_consistent: list[bool] = []
    y_opus: list[bool] = []
    y_sonnet: list[bool] = []
    per_crit_dims: dict[str, list[bool]] = defaultdict(list)
    disagreements: list[dict[str, Any]] = []

    for (sid, draft_id, cn, dcanon), rec in dim_runs.items():
        om = majority_met(rec["opus"])
        sm = majority_met(rec["sonnet"])

        def _cons(v: list[bool | None]) -> bool:
            vs = [x for x in v if x is not None]
            return len(vs) < 2 or len(set(vs)) == 1

        o_cons = _cons(rec["opus"])
        s_cons = _cons(rec["sonnet"])
        opus_consistent.append(o_cons)
        sonnet_consistent.append(s_cons)

        if om is None or sm is None:
            continue
        agree = om == sm
        dim_agreements.append(agree)
        y_opus.append(om)
        y_sonnet.append(sm)
        per_crit_dims[cn].append(agree)

        if not agree:
            lab = dim_label.get((sid, cn, dcanon), dcanon)
            draft_full = draft_text.get((sid, draft_id), "")
            # pick evidence from first run with text
            oe = next((e for e in rec["evidence_opus"] if e), "")
            se = next((e for e in rec["evidence_sonnet"] if e), "")
            disagreements.append(
                {
                    "scenario_id": sid,
                    "draft_id": draft_id,
                    "criterion_name": cn,
                    "dimension_id": dcanon,
                    "dimension_label": lab,
                    "draft_excerpt": draft_full[:2500],
                    "opus_majority_met": om,
                    "sonnet_majority_met": sm,
                    "opus_evidence": oe,
                    "sonnet_evidence": se,
                    "opus_runs": rec["opus"],
                    "sonnet_runs": rec["sonnet"],
                }
            )

    overall_agreement = mean(dim_agreements) if dim_agreements else 0.0
    kappa_all = cohen_kappa_binary(y_opus, y_sonnet)

    per_crit_rates = {c: mean(v) if v else 0.0 for c, v in per_crit_dims.items()}
    per_crit_kappa: dict[str, float | None] = {}
    for c in per_crit_rates:
        yo = []
        ys = []
        for (sid, draft_id, cn, dcanon), rec in dim_runs.items():
            if cn != c:
                continue
            om = majority_met(rec["opus"])
            sm = majority_met(rec["sonnet"])
            if om is None or sm is None:
                continue
            yo.append(om)
            ys.append(sm)
        per_crit_kappa[c] = cohen_kappa_binary(yo, ys) if yo else None

    # Per-dimension-id global rate (label for report)
    per_dimension_stats: dict[str, dict[str, Any]] = {}
    for (sid, draft_id, cn, dcanon), rec in dim_runs.items():
        om = majority_met(rec["opus"])
        sm = majority_met(rec["sonnet"])
        if om is None or sm is None:
            continue
        key = f"{cn} :: {dcanon}"
        if key not in per_dimension_stats:
            per_dimension_stats[key] = {"agree": 0, "total": 0, "label": dim_label.get((sid, cn, dcanon), "")}
        per_dimension_stats[key]["total"] += 1
        if om == sm:
            per_dimension_stats[key]["agree"] += 1

    for st in per_dimension_stats.values():
        st["rate"] = st["agree"] / st["total"] if st["total"] else 0.0

    log(f"Classifying {len(disagreements)} disagreements with Sonnet...")
    client = Anthropic()
    for d in disagreements:
        cls = classify_disagreement(
            client,
            criterion_name=d["criterion_name"],
            dimension_id=d["dimension_id"],
            dimension_label=d["dimension_label"],
            draft_excerpt=d["draft_excerpt"],
            opus_g=d["opus_majority_met"],
            sonnet_g=d["sonnet_majority_met"],
            opus_ev=d["opus_evidence"],
            sonnet_ev=d["sonnet_evidence"],
        )
        d["disagreement_classification"] = cls.get("classification")
        d["disagreement_rationale"] = cls.get("rationale")

    ambiguous_dims = sorted(
        {
            f"{d['criterion_name']} — {d['dimension_id']}"
            for d in disagreements
            if d.get("disagreement_classification") == "dimension_ambiguity"
        }
    )

    cost_opus = (
        tokens["opus"]["in"] / 1e6 * PRICE_OPUS_IN + tokens["opus"]["out"] / 1e6 * PRICE_OPUS_OUT
    )
    cost_sonnet = (
        tokens["sonnet"]["in"] / 1e6 * PRICE_SONNET_IN + tokens["sonnet"]["out"] / 1e6 * PRICE_SONNET_OUT
    )

    report = {
        "summary": {
            "overall_dimension_agreement_rate": overall_agreement,
            "cohens_kappa_overall": kappa_all,
            "within_model_consistency_opus": mean(opus_consistent) if opus_consistent else None,
            "within_model_consistency_sonnet": mean(sonnet_consistent) if sonnet_consistent else None,
            "num_dimensions_evaluated": len(dim_agreements),
            "num_disagreements": len(disagreements),
            "latency_ms_avg_opus": mean(latency["opus"]) if latency["opus"] else None,
            "latency_ms_avg_sonnet": mean(latency["sonnet"]) if latency["sonnet"] else None,
            "tokens_opus": tokens["opus"],
            "tokens_sonnet": tokens["sonnet"],
            "estimated_cost_usd_opus": cost_opus,
            "estimated_cost_usd_sonnet": cost_sonnet,
            "pricing_assumptions_usd_per_mtok": {
                "opus_input": PRICE_OPUS_IN,
                "opus_output": PRICE_OPUS_OUT,
                "sonnet_input": PRICE_SONNET_IN,
                "sonnet_output": PRICE_SONNET_OUT,
            },
        },
        "per_criterion_agreement_rate": per_crit_rates,
        "per_criterion_cohens_kappa": {k: v for k, v in per_crit_kappa.items()},
        "per_dimension_agreement": per_dimension_stats,
        "disagreements": disagreements,
        "ambiguous_dimensions_flagged": ambiguous_dims,
    }

    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {args.json_out}")

    # --- Markdown report ---
    s = report["summary"]
    lines = [
        "# Grader agreement report (Opus vs Sonnet 4.6)",
        "",
        "## Summary",
        "",
        f"- **Overall dimension agreement:** {s['overall_dimension_agreement_rate']:.1%}",
        f"- **Cohen's κ (overall):** {s['cohens_kappa_overall']:.4f}" if s["cohens_kappa_overall"] is not None else "- **Cohen's κ:** n/a",
        f"- **Within-model consistency (Opus):** {s['within_model_consistency_opus']:.1%}" if s["within_model_consistency_opus"] is not None else "",
        f"- **Within-model consistency (Sonnet):** {s['within_model_consistency_sonnet']:.1%}" if s["within_model_consistency_sonnet"] is not None else "",
        f"- **Dimensions compared:** {s['num_dimensions_evaluated']}",
        f"- **Disagreements:** {s['num_disagreements']}",
        f"- **Avg latency Opus:** {s['latency_ms_avg_opus']:.0f} ms" if s["latency_ms_avg_opus"] else "- **Avg latency Opus:** n/a",
        f"- **Avg latency Sonnet:** {s['latency_ms_avg_sonnet']:.0f} ms" if s["latency_ms_avg_sonnet"] else "- **Avg latency Sonnet:** n/a",
        f"- **Est. cost Opus:** ${s['estimated_cost_usd_opus']:.4f}",
        f"- **Est. cost Sonnet:** ${s['estimated_cost_usd_sonnet']:.4f}",
        "",
        "*Costs use placeholder $/MTok defaults; set `PRICE_*` env vars to match your billing.*",
        "",
        "## Per-criterion agreement",
        "",
        "| Criterion | Agreement | Cohen's κ |",
        "|-----------|-----------|-----------|",
    ]
    for c, rate in sorted(per_crit_rates.items(), key=lambda x: -x[1]):
        kap = per_crit_kappa.get(c)
        ks = f"{kap:.4f}" if kap is not None else "n/a"
        lines.append(f"| {c} | {rate:.1%} | {ks} |")
    lines.extend(["", "## Per-dimension agreement (rolled up by name + id)", "", "| Dimension | Agreement | n |", "|-----------|-----------|---|"])
    for key, st in sorted(per_dimension_stats.items(), key=lambda x: -x[1]["rate"]):
        lines.append(f"| {key} | {st['rate']:.1%} | {st['total']} |")

    lines.extend(["", "## Disagreements (with classification)", ""])
    for d in disagreements:
        lines.append(f"### {d['scenario_id']} / {d['draft_id']} — {d['criterion_name']} / `{d['dimension_id']}`")
        lines.append(f"- **Label:** {d['dimension_label']}")
        lines.append(f"- **Opus (majority):** {'MET' if d['opus_majority_met'] else 'NOT_MET'} — {d['opus_evidence']}")
        lines.append(f"- **Sonnet (majority):** {'MET' if d['sonnet_majority_met'] else 'NOT_MET'} — {d['sonnet_evidence']}")
        lines.append(f"- **Classification:** `{d.get('disagreement_classification')}` — {d.get('disagreement_rationale')}")
        lines.append("")
        lines.append("<details><summary>Draft excerpt</summary>\n\n```\n" + d["draft_excerpt"][:2000] + "\n```\n</details>\n")

    lines.extend(["", "## Dimensions flagged as ambiguous", ""])
    if ambiguous_dims:
        lines.extend([f"- {x}" for x in ambiguous_dims])
    else:
        lines.append("- *(none)*")

    # Recommendation heuristic
    rec_parts = []
    if s["overall_dimension_agreement_rate"] >= 0.85 and (s["cohens_kappa_overall"] or 0) >= 0.65:
        rec_parts.append(
            "Sonnet tracks Opus closely enough that using Sonnet as a **background grader** is reasonable, "
            "subject to spot-checks on high-stakes evaluations."
        )
    elif s["overall_dimension_agreement_rate"] >= 0.75:
        rec_parts.append(
            "Agreement is **moderate**. Consider Sonnet for coarse screening only, or tighten ambiguous dimensions (see list above)."
        )
    else:
        rec_parts.append(
            "Agreement is **low**; do not rely on Sonnet alone for rubric-grounded grading until dimensions are clarified or prompts are revised."
        )
    if s["within_model_consistency_sonnet"] is not None and s["within_model_consistency_sonnet"] < 0.9:
        rec_parts.append("Sonnet’s **within-model inconsistency** suggests some dimensions are underspecified.")

    lines.extend(["", "## Recommendation", "", " ".join(rec_parts), ""])

    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {args.md_out}")

    # Stdout table
    print(f"{'Metric':<40} {'Value':>12}")
    print("-" * 52)
    print(f"{'Overall dimension agreement':<40} {overall_agreement:12.1%}")
    _kap = kappa_all if kappa_all is not None else float("nan")
    print(f"{'Cohen kappa (overall)':<40} {_kap:12.4f}")
    print(f"{'Within-model consistency Opus':<40} {(mean(opus_consistent) if opus_consistent else 0):12.1%}")
    print(f"{'Within-model consistency Sonnet':<40} {(mean(sonnet_consistent) if sonnet_consistent else 0):12.1%}")
    print(f"{'Avg latency Opus (ms)':<40} {(mean(latency['opus']) if latency['opus'] else 0):12.0f}")
    print(f"{'Avg latency Sonnet (ms)':<40} {(mean(latency['sonnet']) if latency['sonnet'] else 0):12.0f}")
    print(f"{'Est. cost Opus (USD)':<40} {cost_opus:12.4f}")
    print(f"{'Est. cost Sonnet (USD)':<40} {cost_sonnet:12.4f}")


if __name__ == "__main__":
    main()
