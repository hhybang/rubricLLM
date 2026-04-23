"""Grading prompt eval: does the grader actually apply the rubric bar?

Motivating concern: drafts in the live app often score 100%, which could
indicate the grader is too lenient (false positives) rather than drafts
genuinely being that good.

This eval:
  1. Loads fixtures with hand-crafted drafts whose per-dim ground truth is
     known. Three variants per rubric: all_met, all_violated, mixed.
  2. Runs the real grade_draft_sync() pipeline on each draft.
  3. Compares every dim's predicted verdict against ground truth.
  4. Reports false-positive rate (saying MET when truth is NOT_MET -- the
     "everything's fine" failure mode), false-negative rate (over-strict),
     overall accuracy, and confidence calibration.

The grader runs on claude-sonnet-4-6 by default (MODEL_LIGHT) since that's
what the real app uses. Pass --model to override.

Usage:
    python -m grading_eval
    python -m grading_eval --fixture cold_email
    python -m grading_eval --report grading_eval/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make top-level modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rubric_writer.draft_grading import grade_draft_sync
from grading_eval.fixtures import FIXTURES, EDGE_CASE_FIXTURES


def _find_dim_verdict(grades: dict, dim_id: str) -> tuple[str | None, str | None, str]:
    """Look up a dim's verdict in the grader's output. Returns
    (grade, confidence, evidence) or (None, None, '') if not found."""
    target = (dim_id or "").strip().lower()
    for c in grades.get("grades") or []:
        for d in c.get("dimension_grades") or []:
            if (d.get("dimension_id") or "").strip().lower() == target:
                return (
                    (d.get("grade") or "").upper() or None,
                    (d.get("confidence") or "").lower() or None,
                    d.get("evidence", ""),
                )
    return None, None, ""


def _evaluate_draft(
    rubric: dict,
    draft_text: str,
    ground_truth: dict[str, str],  # dim_id -> "MET" or "NOT_MET"
    variant: str,
    fixture_name: str,
    model: str | None,
) -> dict:
    """Grade a draft and score each dim against ground truth."""
    kwargs: dict = {"rubric_dict": rubric, "draft_text": draft_text}
    if model:
        kwargs["model"] = model
    grades, latency_ms, err = grade_draft_sync(**kwargs)
    if err or not grades:
        print(f"  [{variant}] grading failed: {err}", flush=True)
        return {
            "fixture": fixture_name, "variant": variant,
            "error": err, "latency_ms": latency_ms,
            "per_dim": [],
        }

    per_dim = []
    for dim_id, truth in ground_truth.items():
        pred_grade, pred_conf, pred_evidence = _find_dim_verdict(grades, dim_id)
        correct = (pred_grade == truth)
        # Failure-mode classification
        if pred_grade is None:
            fail_type = "missing"
        elif correct:
            fail_type = None
        elif pred_grade == "MET" and truth == "NOT_MET":
            fail_type = "false_positive"
        elif pred_grade == "NOT_MET" and truth == "MET":
            fail_type = "false_negative"
        else:
            fail_type = "other"
        per_dim.append({
            "dim_id": dim_id,
            "truth": truth,
            "predicted": pred_grade,
            "confidence": pred_conf,
            "correct": correct,
            "fail_type": fail_type,
            "evidence": pred_evidence[:300],
        })
        marker = "✓" if correct else ("✗" if fail_type else "?")
        conf = pred_conf or "?"
        print(f"    {marker} [{variant}] {dim_id}: truth={truth}, "
              f"pred={pred_grade or 'MISSING'} ({conf})", flush=True)

    return {
        "fixture": fixture_name, "variant": variant,
        "latency_ms": latency_ms,
        "per_dim": per_dim,
        "raw_grades": grades,
    }


def evaluate_fixture(fixture: dict, model: str | None,
                     category: str = "baseline") -> list[dict]:
    """Run the grader on every draft variant defined in the fixture.

    Fixtures follow one of two shapes:

      1. Baseline fixtures: variant keys are `all_met` / `all_violated` /
         `mixed`, with `mixed_ground_truth` the only explicit map (all_met
         and all_violated ground truth are derived from the rubric).

      2. Edge-case fixtures: arbitrary variant keys paired with explicit
         `<variant>_ground_truth` maps. May also carry an
         `expected_confidence_not_high` list naming dims whose confidence
         should NOT be 'high' for this draft (used by ambiguous-dim tests).
    """
    name = fixture["name"]
    rubric = fixture["rubric"]
    drafts = fixture["drafts"]
    print(f"\n=== [{category}] {name} ===", flush=True)

    # Derive the list of draft variants: any key NOT ending with
    # _ground_truth and NOT equal to expected_confidence_not_high.
    meta_keys = {"expected_confidence_not_high"}
    variant_names = [
        k for k in drafts.keys()
        if not k.endswith("_ground_truth") and k not in meta_keys
    ]

    # Per-variant ground truth: look up `<variant>_ground_truth` first; if
    # missing, derive from rubric (legacy baseline fixtures).
    all_dim_ids: list[str] = []
    for crit in rubric["rubric"]:
        for dim in crit["dimensions"]:
            all_dim_ids.append(dim["id"])

    expected_conf_not_high = set(drafts.get("expected_confidence_not_high") or [])

    results: list[dict] = []
    for variant in variant_names:
        gt_key = f"{variant}_ground_truth"
        gt = drafts.get(gt_key)
        if gt is None:
            # Legacy fallback for baseline fixtures
            if variant == "all_met":
                gt = {did: "MET" for did in all_dim_ids}
            elif variant == "all_violated":
                gt = {did: "NOT_MET" for did in all_dim_ids}
            elif variant == "mixed":
                gt = drafts.get("mixed_ground_truth") or {}
            else:
                print(f"  [WARN] no ground truth for variant {variant!r}", flush=True)
                continue

        draft_text = drafts[variant]
        result = _evaluate_draft(
            rubric, draft_text, gt, variant, name, model,
        )
        result["category"] = category

        # Confidence-calibration check for ambiguous dims.
        if expected_conf_not_high:
            conf_problems = []
            for d in result.get("per_dim", []):
                if d["dim_id"] in expected_conf_not_high and d.get("confidence") == "high":
                    conf_problems.append(d["dim_id"])
            result["confidence_calibration_problems"] = conf_problems
            if conf_problems:
                print(f"    [CALIBRATION] dims returned 'high' confidence but "
                      f"should have been medium/low: {conf_problems}", flush=True)

        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Override the grader model (default: whatever RUBRIC_GRADING_MODEL env or MODEL_LIGHT picks)")
    parser.add_argument("--fixture", default=None,
                        help="Run only this fixture by name (default: all)")
    parser.add_argument("--report", default="grading_eval/report.json")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    run_list: list[tuple[dict, str]] = []
    for f in FIXTURES:
        run_list.append((f, "baseline"))
    for f in EDGE_CASE_FIXTURES:
        run_list.append((f, f.get("category", "edge_case")))

    if args.fixture:
        run_list = [(f, c) for (f, c) in run_list if f["name"] == args.fixture]
        if not run_list:
            print(f"ERROR: no fixture named {args.fixture!r}", file=sys.stderr)
            return 1

    all_results: list[dict] = []
    for fx, cat in run_list:
        all_results.extend(evaluate_fixture(fx, model=args.model, category=cat))

    # Aggregate by variant.
    print("\n" + "=" * 60)
    print("Results by variant:")

    def _agg(res_subset):
        all_dims = [d for r in res_subset for d in r.get("per_dim", [])]
        total = len(all_dims)
        correct = sum(1 for d in all_dims if d["correct"])
        fp = sum(1 for d in all_dims if d["fail_type"] == "false_positive")
        fn = sum(1 for d in all_dims if d["fail_type"] == "false_negative")
        missing = sum(1 for d in all_dims if d["fail_type"] == "missing")
        return total, correct, fp, fn, missing

    # Baseline variants first
    for variant in ("all_met", "all_violated", "mixed"):
        sub = [r for r in all_results if r["variant"] == variant]
        if not sub:
            continue
        total_v, correct_v, fp_v, fn_v, missing_v = _agg(sub)
        acc_v = (100.0 * correct_v / total_v) if total_v else 0
        print(f"  {variant:18s}: {correct_v}/{total_v} correct ({acc_v:.1f}%)  "
              f"— FP:{fp_v}, FN:{fn_v}, missing:{missing_v}")

    # Then remaining (edge-case) variants
    seen = {"all_met", "all_violated", "mixed"}
    other_variants = sorted({r["variant"] for r in all_results if r["variant"] not in seen})
    for variant in other_variants:
        sub = [r for r in all_results if r["variant"] == variant]
        total_v, correct_v, fp_v, fn_v, missing_v = _agg(sub)
        acc_v = (100.0 * correct_v / total_v) if total_v else 0
        print(f"  {variant:25s}: {correct_v}/{total_v} correct ({acc_v:.1f}%)  "
              f"— FP:{fp_v}, FN:{fn_v}, missing:{missing_v}")

    # Aggregate by category (baseline vs each edge-case type)
    print()
    print("Results by category:")
    categories = sorted({r.get("category", "baseline") for r in all_results})
    by_cat: dict = {}
    for cat in categories:
        sub = [r for r in all_results if r.get("category") == cat]
        total_c, correct_c, fp_c, fn_c, missing_c = _agg(sub)
        acc_c = (100.0 * correct_c / total_c) if total_c else 0
        print(f"  {cat:20s}: {correct_c}/{total_c} correct ({acc_c:.1f}%)  "
              f"— FP:{fp_c}, FN:{fn_c}, missing:{missing_c}")
        by_cat[cat] = {
            "total": total_c, "correct": correct_c,
            "false_positives": fp_c, "false_negatives": fn_c, "missing": missing_c,
            "accuracy": (correct_c / total_c) if total_c else None,
        }

    total, correct, fp, fn, missing = _agg(all_results)
    acc = (100.0 * correct / total) if total else 0
    fp_rate_of_notmet = 0.0
    # FP rate = false positives / total NOT_MET ground-truths
    total_notmet = sum(1 for r in all_results for d in r.get("per_dim", []) if d["truth"] == "NOT_MET")
    if total_notmet:
        fp_rate_of_notmet = 100.0 * fp / total_notmet
    fn_rate_of_met = 0.0
    total_met = sum(1 for r in all_results for d in r.get("per_dim", []) if d["truth"] == "MET")
    if total_met:
        fn_rate_of_met = 100.0 * fn / total_met

    print()
    print(f"AGGREGATE: {correct}/{total} correct ({acc:.1f}%)")
    print(f"  False-positive rate: {fp}/{total_notmet} "
          f"({fp_rate_of_notmet:.1f}% of NOT_MET dims scored as MET — the \"everything passes\" failure)")
    print(f"  False-negative rate: {fn}/{total_met} "
          f"({fn_rate_of_met:.1f}% of MET dims scored as NOT_MET — over-strict)")
    print(f"  Missing verdicts:    {missing}")

    # Confidence calibration
    by_conf: dict[str, dict] = {}
    for r in all_results:
        for d in r.get("per_dim", []):
            conf = d.get("confidence") or "unknown"
            b = by_conf.setdefault(conf, {"total": 0, "correct": 0})
            b["total"] += 1
            if d["correct"]:
                b["correct"] += 1
    print()
    print("Confidence calibration:")
    for conf in ("high", "medium", "low", "unknown"):
        if conf in by_conf:
            b = by_conf[conf]
            p = (100.0 * b["correct"] / b["total"]) if b["total"] else 0
            print(f"  {conf}: {b['correct']}/{b['total']} correct ({p:.1f}%)")
    print("=" * 60)

    # Aggregate confidence-calibration problems (ambiguous dims scored high).
    calibration_problems = []
    for r in all_results:
        for dim_id in r.get("confidence_calibration_problems") or []:
            calibration_problems.append({
                "fixture": r.get("fixture"),
                "variant": r.get("variant"),
                "dim_id": dim_id,
            })
    if calibration_problems:
        print()
        print("Confidence calibration problems (ambiguous dims returned 'high' confidence):")
        for p in calibration_problems:
            print(f"  {p['fixture']}/{p['variant']}: {p['dim_id']}")

    report = {
        "model": args.model or "default (sonnet-4-6)",
        "aggregate": {
            "total": total, "correct": correct,
            "accuracy": (correct / total) if total else None,
            "false_positives": fp,
            "false_negatives": fn,
            "missing": missing,
            "fp_rate_of_notmet": (fp / total_notmet) if total_notmet else None,
            "fn_rate_of_met": (fn / total_met) if total_met else None,
        },
        "by_category": by_cat,
        "confidence_calibration": {
            c: {"total": v["total"], "correct": v["correct"],
                "accuracy": (v["correct"] / v["total"]) if v["total"] else None}
            for c, v in by_conf.items()
        },
        "confidence_calibration_problems": calibration_problems,
        "results": all_results,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
