#!/usr/bin/env python3
"""Aggregate audit JSON dumps from multiple participants into one report."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PARTICIPANTS = [
    ("Jenny",   "/tmp/jenny_session.json"),
    ("Barish",  "/tmp/barish_session.json"),
    ("Felicia", "/tmp/felicia_session.json"),
    ("Dylan",   "/tmp/dylan_session.json"),
]

# Studies were instrumented starting 2026-04 — older Jenny project (Feb) is
# pre-instrumentation and should be excluded from the user-study cohort.
INSTRUMENTATION_CUTOFF = "2026-04-01"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_study_project(p: dict) -> bool:
    """Filter out pre-instrumentation projects (e.g. Jenny's Feb context-filtering)."""
    return (p.get("created_at") or "") >= INSTRUMENTATION_CUTOFF


def _hbar(label: str = "") -> None:
    print("\n" + "=" * 78)
    if label:
        print(label)
        print("=" * 78)


def _row(label: str, *vals) -> None:
    cells = "  ".join(f"{v:>8}" for v in vals)
    print(f"  {label:<38}{cells}")


def main() -> int:
    cohort: list[tuple[str, dict]] = []
    for name, path in PARTICIPANTS:
        if not Path(path).exists():
            print(f"[warn] missing {path}, skipping {name}", file=sys.stderr)
            continue
        cohort.append((name, _load(path)))

    # ── Per-participant headline metrics ────────────────────────────────────
    _hbar("PER-PARTICIPANT SUMMARY (study projects only)")
    headers = ["convs", "drafts", "rub_v", "verds", "props", "edits", "RQ1", "RQ2", "surveys"]
    print(f"  {'participant':<38}" + "  ".join(f"{h:>8}" for h in headers))
    print("  " + "-" * 76)

    per_participant_rows = []
    for name, data in cohort:
        study_projects = [p for p in data["projects"] if _is_study_project(p)]
        n_convs = sum(len(p["conversations"]) for p in study_projects)
        n_drafts = sum(len(c["drafts"]) for p in study_projects for c in p["conversations"])
        n_rub_v = sum(len(p["rubric_versions"]) for p in study_projects)
        n_verds = sum(p["derived"]["user_verdicts"]["total_verdicts"] for p in study_projects)
        n_props = sum(p["derived"]["refiner_proposal"]["n_unique_proposals"] for p in study_projects)
        n_edits = sum(p["derived"]["rubric_edit_event"]["total_events"] for p in study_projects)
        n_rq1 = sum(p["project_data_counts"].get("rq1_dimension_recognition", 0) for p in study_projects)
        n_rq2 = sum(p["project_data_counts"].get("rq2_threeway", 0) for p in study_projects)
        n_surv = sum(p["project_data_counts"].get("survey_responses", 0) for p in study_projects)
        per_participant_rows.append({
            "name": name, "n_convs": n_convs, "n_drafts": n_drafts,
            "n_rub_v": n_rub_v, "n_verds": n_verds, "n_props": n_props,
            "n_edits": n_edits, "n_rq1": n_rq1, "n_rq2": n_rq2, "n_surv": n_surv,
            "study_projects": study_projects,
        })
        _row(name, n_convs, n_drafts, n_rub_v, n_verds, n_props, n_edits, n_rq1, n_rq2, n_surv)

    # Totals
    print("  " + "-" * 76)
    _row("TOTAL",
         sum(r["n_convs"] for r in per_participant_rows),
         sum(r["n_drafts"] for r in per_participant_rows),
         sum(r["n_rub_v"] for r in per_participant_rows),
         sum(r["n_verds"] for r in per_participant_rows),
         sum(r["n_props"] for r in per_participant_rows),
         sum(r["n_edits"] for r in per_participant_rows),
         sum(r["n_rq1"] for r in per_participant_rows),
         sum(r["n_rq2"] for r in per_participant_rows),
         sum(r["n_surv"] for r in per_participant_rows),
    )

    # ── Rubric evolution ────────────────────────────────────────────────────
    _hbar("RUBRIC EVOLUTION")
    src_counts: Counter = Counter()
    for r in per_participant_rows:
        for p in r["study_projects"]:
            for v in p["rubric_versions"]:
                src_counts[v.get("source") or "?"] += 1
    print(f"  rubric versions by source:")
    for src, n in src_counts.most_common():
        print(f"    {src:<35} {n}")

    # ── Drift panels: condition_met vs. shown ───────────────────────────────
    _hbar("DRIFT PANELS — heuristic_diagnostic across cohort")
    heur_agg: dict[str, dict] = defaultdict(lambda: {
        "condition_met": 0, "shown": 0, "suppressed_by_priority": 0,
        "suppressed_reasons": Counter(),
    })
    n_drafts_diag = 0
    for r in per_participant_rows:
        for p in r["study_projects"]:
            h = p["derived"]["heuristic_diagnostic"]
            n_drafts_diag += h["n_drafts_with_diagnostic"]
            for hname, hstats in h["per_heuristic"].items():
                heur_agg[hname]["condition_met"] += hstats["condition_met"]
                heur_agg[hname]["shown"] += hstats["shown"]
                heur_agg[hname]["suppressed_by_priority"] += hstats["suppressed_by_priority"]
                for reason, cnt in (hstats.get("suppressed_reasons") or {}).items():
                    heur_agg[hname]["suppressed_reasons"][reason] += cnt
    print(f"  drafts with diagnostic recorded: {n_drafts_diag}")
    print(f"  {'heuristic':<25}{'cond_met':>10}{'shown':>10}{'supp_pri':>10}  fire_rate")
    print("  " + "-" * 76)
    for hname in sorted(heur_agg.keys()):
        s = heur_agg[hname]
        rate = (s["shown"] / s["condition_met"]) if s["condition_met"] else 0.0
        print(f"  {hname:<25}{s['condition_met']:>10}{s['shown']:>10}{s['suppressed_by_priority']:>10}  {rate:.0%}")

    # ── User verdicts (drift-panel clicks from messages) ────────────────────
    _hbar("USER VERDICTS (drift-panel clicks)")
    by_kind: Counter = Counter()
    by_action: Counter = Counter()
    by_reason: Counter = Counter()
    by_kind_x_reason: dict[str, Counter] = defaultdict(Counter)
    unmapped: list = []
    for r in per_participant_rows:
        for p in r["study_projects"]:
            v = p["derived"]["user_verdicts"]
            for k, n in (v.get("by_drift_kind") or {}).items():
                by_kind[k] += n
            for a, n in (v.get("by_action") or {}).items():
                by_action[a] += n
            for rc, n in (v.get("by_reason_code") or {}).items():
                by_reason[rc] += n
            for k, sub in (v.get("by_drift_kind_x_reason") or {}).items():
                for rc, n in sub.items():
                    by_kind_x_reason[k][rc] += n
            unmapped.extend(v.get("unmapped") or [])
    print(f"  total verdicts: {sum(by_kind.values())}")
    print(f"  by drift_kind: {dict(by_kind)}")
    print(f"  by action:     {dict(by_action)}")
    print(f"  by reason:     {dict(by_reason)}")
    print(f"  by drift_kind × reason:")
    for k in sorted(by_kind_x_reason.keys()):
        print(f"    {k}: {dict(by_kind_x_reason[k])}")
    if unmapped:
        print(f"  unmapped (need attention):")
        for u in unmapped:
            print(f"    {u}")

    # ── Refiner proposals ───────────────────────────────────────────────────
    _hbar("REFINER PROPOSALS")
    disp: Counter = Counter()
    by_drift: Counter = Counter()
    n_prop = 0
    for r in per_participant_rows:
        for p in r["study_projects"]:
            rp = p["derived"]["refiner_proposal"]
            n_prop += rp["n_unique_proposals"]
            for d, n in rp["disposition_counts"].items():
                disp[d] += n
            for k, n in rp["per_drift_kind"].items():
                by_drift[k] += n
    print(f"  total unique proposals: {n_prop}")
    print(f"  dispositions:           {dict(disp)}")
    print(f"  by drift_kind:          {dict(by_drift)}")

    # ── Rubric edit events ──────────────────────────────────────────────────
    _hbar("RUBRIC EDIT EVENTS")
    triggers: Counter = Counter()
    cum_added = cum_removed = cum_modified = total_events = 0
    for r in per_participant_rows:
        for p in r["study_projects"]:
            e = p["derived"]["rubric_edit_event"]
            total_events += e["total_events"]
            cum_added += e["cumulative_dims_added"]
            cum_removed += e["cumulative_dims_removed"]
            cum_modified += e["cumulative_dims_modified"]
            for t, n in e["by_trigger"].items():
                triggers[t] += n
    print(f"  total events: {total_events}")
    print(f"  by trigger:   {dict(triggers)}")
    print(f"  cumulative:   +{cum_added} -{cum_removed} ~{cum_modified} dims")

    # ── Per-draft pct_met trajectories ──────────────────────────────────────
    _hbar("DRAFT GRADE TRAJECTORIES (pct_met across drafts)")
    for r in per_participant_rows:
        for p in r["study_projects"]:
            for c in p["conversations"]:
                pcts = [d.get("pct_met") for d in c.get("drafts") or [] if d.get("pct_met") is not None]
                if pcts:
                    print(f"  {r['name']:<10} {p['project_name']:<25} conv={c['conv_id'][:8]}  pct_met={pcts}")

    # ── RQ data totals ──────────────────────────────────────────────────────
    _hbar("RESEARCH QUESTION DATA")
    rq_keys = ["rq1_dimension_recognition", "rq2_threeway", "rq2_confirmation", "rq2_fire_rate"]
    print(f"  {'data_type':<35}" + "  ".join(f"{n[:8]:>9}" for n, _ in PARTICIPANTS) + "    total")
    print("  " + "-" * 76)
    for k in rq_keys:
        vals = []
        for r in per_participant_rows:
            v = sum(p["project_data_counts"].get(k, 0) for p in r["study_projects"])
            vals.append(v)
        print(f"  {k:<35}" + "  ".join(f"{v:>9}" for v in vals) + f"    {sum(vals):>5}")

    # ── Survey totals ───────────────────────────────────────────────────────
    _hbar("SURVEY DATA")
    survey_keys = ["survey_task_a", "survey_task_b", "survey_final_review", "survey_responses"]
    print(f"  {'data_type':<35}" + "  ".join(f"{n[:8]:>9}" for n, _ in PARTICIPANTS) + "    total")
    print("  " + "-" * 76)
    for k in survey_keys:
        vals = []
        for r in per_participant_rows:
            v = sum(p["project_data_counts"].get(k, 0) for p in r["study_projects"])
            vals.append(v)
        print(f"  {k:<35}" + "  ".join(f"{v:>9}" for v in vals) + f"    {sum(vals):>5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
