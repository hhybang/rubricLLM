"""Background grade retest for alignment diagnostic."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY
from rubric_writer.api_client import _api_call_with_retry


def _run_grade_retest_bg(args):
    """Background-thread-safe: re-run rubric and generic judges for test-retest reliability.
    Does NOT access st.session_state. All data passed via args dict."""
    from scipy.stats import kendalltau as _kt

    draft_a = args.get("draft_good", "")
    draft_b = args.get("draft_degraded", "")
    rubric_json = args.get("rubric_criteria_json", "")
    conv_context = args.get("conv_context", "")
    original_rubric = args.get("original_rubric_result")
    original_generic = args.get("original_generic_result")

    if not draft_a or not draft_b:
        return

    retest_rubric = None
    retest_generic = None

    # Re-run rubric judge
    try:
        prompt = GRADING_rubric_judge_prompt(draft_a, draft_b, rubric_json, conv_context)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js = re.search(r'\{[\s\S]*\}', text)
        if js:
            retest_rubric = json.loads(js.group())
    except Exception as e:
        # print(f"[RETEST] Rubric judge retest failed: {e}")

        pass
    # Re-run generic judge
    try:
        prompt = GRADING_generic_judge_prompt(draft_a, draft_b)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js = re.search(r'\{[\s\S]*\}', text)
        if js:
            retest_generic = json.loads(js.group())
    except Exception as e:
        # print(f"[RETEST] Generic judge retest failed: {e}")

        pass
    # Compute test-retest metrics
    retest_data = {
        "timestamp": args.get("grade_eval_timestamp", ""),
        "retest_rubric_result": retest_rubric,
        "retest_generic_result": retest_generic,
        "original_rubric_result": original_rubric,
        "original_generic_result": original_generic,
        "metrics": {}
    }

    # --- Rubric judge test-retest ---
    if retest_rubric and original_rubric:
        orig_scores = {}
        for c in original_rubric.get("per_criterion", []):
            orig_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        retest_scores = {}
        for c in retest_rubric.get("per_criterion", []):
            retest_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        rubric_run1, rubric_run2 = [], []
        for crit in orig_scores:
            if crit in retest_scores:
                rubric_run1.extend(orig_scores[crit])
                rubric_run2.extend(retest_scores[crit])
        retest_data["metrics"]["rubric_run1_scores"] = rubric_run1
        retest_data["metrics"]["rubric_run2_scores"] = rubric_run2
        if len(rubric_run1) >= 4:
            tau, p = _kt(rubric_run1, rubric_run2)
            retest_data["metrics"]["rubric_tau"] = tau
            retest_data["metrics"]["rubric_tau_p"] = p
        # Keep legacy key for backward compat with Panel 4
        retest_data["metrics"]["retest_tau"] = retest_data["metrics"].get("rubric_tau")

    # --- Generic judge test-retest ---
    if retest_generic and original_generic:
        orig_g_scores = {}
        for c in original_generic.get("per_criterion", []):
            orig_g_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        retest_g_scores = {}
        for c in retest_generic.get("per_criterion", []):
            retest_g_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        generic_run1, generic_run2 = [], []
        for crit in orig_g_scores:
            if crit in retest_g_scores:
                generic_run1.extend(orig_g_scores[crit])
                generic_run2.extend(retest_g_scores[crit])
        retest_data["metrics"]["generic_run1_scores"] = generic_run1
        retest_data["metrics"]["generic_run2_scores"] = generic_run2
        if len(generic_run1) >= 4:
            tau, p = _kt(generic_run1, generic_run2)
            retest_data["metrics"]["generic_tau"] = tau
            retest_data["metrics"]["generic_tau_p"] = p

    # Save to database
    sb = args.get("supabase")
    pid = args.get("project_id")
    if sb and pid:
        try:
            _retest_data_type = args.get("data_type", "grade_retest")
            save_project_data(sb, pid, _retest_data_type, retest_data)
            # print(f"[RETEST] Saved {_retest_data_type} data. Tau={retest_data['metrics'].get('retest_tau', 'N/A')}")
        except Exception as e:
            # print(f"[RETEST] Failed to save: {e}")

            pass
    # Also append to session state list ref if provided
    results_list = args.get("results_list_ref")
    if results_list is not None:
        results_list.append(retest_data)
