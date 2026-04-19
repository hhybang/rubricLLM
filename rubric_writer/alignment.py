"""Alignment diagnostic processing."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY, MODEL_LIGHT, client
from rubric_writer.api_client import _api_call_with_retry
from rubric_writer.persistence import (
    get_active_rubric,
    load_rubric_history,
    save_rubric_history,
    invalidate_rubric_cache,
    next_version_number,
)
from rubric_writer.diff_html import _annotated_diff_html

def _process_alignment_diagnostic(rcp, user_ranking, user_reason="", status_callback=None, pipeline_messages=None, skip_suggestions=False):
    """Run per-criterion diagnostic comparing drafts (2 or 3 drafts).

    Returns a result dict with per-criterion classifications and suggested rubric changes.

    Args:
        rcp: ranking checkpoint pending state dict with "drafts", "writing_task", etc.
        user_ranking: list of source keys in ranked order, e.g. ["rubric", "preference", "generic"]
                      or ["rubric", "generic"] for 2-draft mode
        user_reason: optional free-text reason from user
        status_callback: optional callable(str) to update UI progress status
        pipeline_messages: optional list of prior conversation messages (from draft generation)
                          to continue the conversation context
        skip_suggestions: if True, skip rubric suggestion/verification steps (rubric draft was #1)
    """
    def _update_status(msg):
        if status_callback:
            status_callback(msg)
    drafts = rcp["drafts"]
    rubric_draft = drafts.get("rubric", "")
    generic_draft = drafts.get("generic", "")
    preference_draft = drafts.get("preference", "")
    is_3draft = bool(preference_draft)

    rubric_dict, _, _ = get_active_rubric()
    rubric_json = json.dumps(
        _rubric_to_json_serializable(rubric_dict), indent=2
    ) if rubric_dict else ""
    rubric_list = rubric_dict.get("rubric", []) if rubric_dict else []

    # Build conversation context for the rubric judge
    conv_text = _build_conversation_text(st.session_state.get("messages", []))
    conv_context = conv_text[-3000:] if len(conv_text) > 3000 else conv_text

    # --- Shared conversation context for the diagnostic pipeline ---
    # Continues from draft generation messages if provided, so the model has
    # full context of the writing task, drafts, scoring, and suggestions.
    _pipeline_messages = list(pipeline_messages) if pipeline_messages else []

    # --- Call rubric judge (per-criterion scores) ---
    _update_status("Scoring each draft against your rubric criteria...")
    rubric_judge_result = None
    try:
        if is_3draft:
            prompt_rubric = GRADING_rubric_judge_3draft_prompt(
                rubric_draft, generic_draft, preference_draft, rubric_json, conv_context
            )
        else:
            prompt_rubric = GRADING_rubric_judge_prompt(rubric_draft, generic_draft, rubric_json, conv_context)
        _pipeline_messages.append({"role": "user", "content": prompt_rubric})
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=_pipeline_messages
        )
        _judge_text = "".join(b.text for b in resp.content if b.type == "text")
        _pipeline_messages.append({"role": "assistant", "content": _judge_text})
        js_match = re.search(r'\{[\s\S]*\}', _judge_text)
        if js_match:
            rubric_judge_result = json.loads(js_match.group())
    except Exception as e:
        pass

    # --- Call generic judge (per-dimension scores, for retest reliability) ---
    _update_status("Running general quality comparison...")
    generic_judge_result = None
    try:
        prompt_generic = GRADING_generic_judge_prompt(rubric_draft, generic_draft)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt_generic}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js_match = re.search(r'\{[\s\S]*\}', text)
        if js_match:
            generic_judge_result = json.loads(js_match.group())
    except Exception as e:
        pass

    # --- Classify each rubric criterion ---
    # In judge prompt: Draft A = rubric, Draft B = generic, Draft C = preference (if 3-draft)
    criteria_analysis = []
    if rubric_judge_result and rubric_judge_result.get("per_criterion"):
        for crit in rubric_judge_result["per_criterion"]:
            rubric_score = crit.get("draft_a_score", 3)
            generic_score = crit.get("draft_b_score", 3)
            preference_score = crit.get("draft_c_score", None) if is_3draft else None
            gap = rubric_score - generic_score

            if is_3draft and preference_score is not None:
                # 3-draft classification
                if rubric_score >= generic_score and rubric_score >= preference_score and gap > 0:
                    classification = "DIFFERENTIATING"
                elif preference_score > rubric_score and preference_score >= generic_score:
                    classification = "PREFERENCE_GAP"
                elif generic_score > rubric_score and generic_score >= (preference_score or 0):
                    classification = "UNDERPERFORMING"
                elif rubric_score == generic_score == preference_score:
                    classification = "REDUNDANT"
                else:
                    classification = "REDUNDANT"
            else:
                # 2-draft classification (same as before)
                if gap > 0:
                    classification = "DIFFERENTIATING"
                elif gap < 0:
                    classification = "UNDERPERFORMING"
                else:
                    classification = "REDUNDANT"

            ca_entry = {
                "name": crit.get("criterion_name", "Unknown"),
                "rubric_score": rubric_score,
                "generic_score": generic_score,
                "gap": gap,
                "classification": classification,
                "reasoning": crit.get("reasoning", ""),
            }
            if preference_score is not None:
                ca_entry["preference_score"] = preference_score
            criteria_analysis.append(ca_entry)

    # Find matching priority from rubric_list
    rubric_priority_map = {}
    for c in rubric_list:
        rubric_priority_map[c.get("name", "").lower().strip()] = c.get("priority", 99)
    for ca in criteria_analysis:
        ca["priority"] = rubric_priority_map.get(ca["name"].lower().strip(), 99)

    # Sort by priority
    criteria_analysis.sort(key=lambda x: x["priority"])

    # --- Generate rubric improvements + reasons in a single call ---
    suggestion_text = ""
    suggested_rubric = None
    # Build ranking description (used by suggestion prompt and result metadata)
    _source_names = {"rubric": "rubric-guided", "generic": "generic (no rubric)", "preference": "preference-based (from original stated preferences)"}
    ranking_desc = "The user ranked the drafts: " + " > ".join(
        f"{i+1}. {_source_names.get(s, s)}" for i, s in enumerate(user_ranking)
    )
    suggestion_reasons = {}  # criterion_name -> reason string
    suggested_draft = ""
    suggested_annotated_changes = []
    verification_result = None

    if not skip_suggestions:
        _update_status("Identifying rubric improvements based on your ranking...")
    try:
        if skip_suggestions:
            raise Exception("__skip__")
        rubric_judge_json = json.dumps(rubric_judge_result, indent=2) if rubric_judge_result else "{}"

        _preferred_draft_for_prompt = drafts.get(user_ranking[0], "") if user_ranking else ""
        suggest_apply_prompt = ALIGNMENT_diagnostic_suggest_and_apply_prompt(
            rubric_json, rubric_judge_json, ranking_desc, user_reason,
            preferred_draft=_preferred_draft_for_prompt,
            rubric_guided_draft=rubric_draft,
            writing_task=rcp.get("writing_task", ""),
        )
        _pipeline_messages.append({"role": "user", "content": suggest_apply_prompt})
        resp = _api_call_with_retry(
            model=MODEL_LIGHT, max_tokens=4000,
            messages=_pipeline_messages
        )
        _sa_raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        _pipeline_messages.append({"role": "assistant", "content": _sa_raw})

        # Parse the combined JSON response containing both "reasons" and "rubric"
        _sa_json_match = re.search(r'\{[\s\S]*\}', _sa_raw)
        if _sa_json_match:
            _sa_parsed = json.loads(_sa_json_match.group())

            # Extract per-criterion reasons
            _all_reasons = _sa_parsed.get("reasons", {})
            # Fallback: try old "criteria" key
            if not _all_reasons:
                _all_reasons = _sa_parsed.get("criteria", {})
            suggestion_reasons = {k: v for k, v in _all_reasons.items() if v}

            # Extract the modified rubric array
            _rubric_arr = _sa_parsed.get("rubric", [])
            if isinstance(_rubric_arr, list) and len(_rubric_arr) > 0:
                suggested_rubric = _rubric_arr
            else:
                # Fallback: try to find a JSON array in the raw text
                _arr_match = re.search(r'\[[\s\S]*\]', _sa_raw)
                if _arr_match:
                    suggested_rubric = json.loads(_arr_match.group())

            # --- Post-process: fix mismatched reasons ---
            # If the model said "working well / kept as-is" but actually modified the criterion,
            # ask the model to explain why it made the change (it has full pipeline context).
            if suggested_rubric and rubric_list:
                _keep_phrases = ["keep unchanged", "no change", "working well", "kept as-is",
                                 "keep as-is", "no changes needed", "performing well",
                                 "leave as-is", "leave unchanged", "kept unchanged"]
                _orig_map = {c.get("name", "").lower().strip(): c for c in rubric_list}
                _new_map = {c.get("name", "").lower().strip(): c for c in suggested_rubric}
                _mismatched_criteria = []
                for _cr_name, _cr_reason in list(suggestion_reasons.items()):
                    _cr_key = _cr_name.lower().strip()
                    _is_keep = any(p in _cr_reason.lower() for p in _keep_phrases)
                    if _is_keep and _cr_key in _orig_map and _cr_key in _new_map:
                        _orig = _orig_map[_cr_key]
                        _new = _new_map[_cr_key]
                        _actually_changed = (
                            _orig.get("description", "") != _new.get("description", "")
                            or _orig.get("priority") != _new.get("priority")
                            or sorted([d.get("label", "") for d in _orig.get("dimensions", [])]) != sorted([d.get("label", "") for d in _new.get("dimensions", [])])
                        )
                        if _actually_changed:
                            _mismatched_criteria.append(_cr_name)

                # Ask the model for real reasons for mismatched criteria
                if _mismatched_criteria:
                    try:
                        _fix_names = ", ".join(f'"{c}"' for c in _mismatched_criteria)
                        _fix_prompt = (
                            f"You said the following criteria were 'working well' or 'kept as-is', "
                            f"but you actually modified them in the rubric you returned: {_fix_names}.\n\n"
                            f"For each one, explain in 1-2 sentences WHY you changed it. "
                            f"Do not describe what changed — the user can see the diff. Explain the reasoning.\n\n"
                            f"Return ONLY a JSON object mapping criterion name to reason, e.g.:\n"
                            f'{{"Criterion A": "reason", "Criterion B": "reason"}}'
                        )
                        _pipeline_messages.append({"role": "user", "content": _fix_prompt})
                        _fix_resp = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=_pipeline_messages
                        )
                        _fix_raw = "".join(b.text for b in _fix_resp.content if b.type == "text").strip()
                        _pipeline_messages.append({"role": "assistant", "content": _fix_raw})
                        _fix_match = re.search(r'\{[\s\S]*\}', _fix_raw)
                        if _fix_match:
                            _fix_reasons = json.loads(_fix_match.group())
                            for _fc_name, _fc_reason in _fix_reasons.items():
                                if _fc_reason and isinstance(_fc_reason, str):
                                    suggestion_reasons[_fc_name] = _fc_reason
                    except Exception:
                        # If follow-up fails, use a generic fallback
                        for _mc in _mismatched_criteria:
                            suggestion_reasons[_mc] = "Adjusted based on alignment diagnostic scores."

            # --- Check if suggested_rubric actually differs from the original ---
            # The model may return a rubric array with cosmetic rephrasing even when
            # all reasons say "keep unchanged". Detect this and discard the suggestion.
            if suggested_rubric and rubric_list:
                _orig_map_chk = {c.get("name", "").lower().strip(): c for c in rubric_list}
                _new_map_chk = {c.get("name", "").lower().strip(): c for c in suggested_rubric}
                _has_real_change = False
                # Check for added criteria
                for _nk in _new_map_chk:
                    if _nk not in _orig_map_chk:
                        _has_real_change = True
                        break
                # Check for removed criteria
                if not _has_real_change:
                    for _ok in _orig_map_chk:
                        if _ok not in _new_map_chk:
                            _has_real_change = True
                            break
                # Check for modified criteria
                if not _has_real_change:
                    for _ck in _orig_map_chk:
                        if _ck in _new_map_chk and _criterion_changed(_orig_map_chk[_ck], _new_map_chk[_ck]):
                            _has_real_change = True
                            break
                if not _has_real_change:
                    # No real changes — discard the suggested rubric
                    suggested_rubric = None

            # Filter out "keep unchanged" reasons — only show reasons for actual changes
            if suggestion_reasons:
                _keep_filter_phrases = ["keep unchanged", "no change", "working well", "kept as-is",
                                        "keep as-is", "no changes needed", "performing well",
                                        "leave as-is", "leave unchanged", "kept unchanged"]
                suggestion_reasons = {
                    k: v for k, v in suggestion_reasons.items()
                    if not any(p in v.lower() for p in _keep_filter_phrases)
                }

            # Build suggestion_text summary from reasons (for display / verification)
            _text_parts = []
            for _cr_name, _cr_reason in suggestion_reasons.items():
                _text_parts.append(f"- **{_cr_name}**: {_cr_reason}")
            if _text_parts:
                suggestion_text = "\n".join(_text_parts)
        else:
            suggestion_text = _sa_raw  # fallback

    except Exception as e:
        pass

    # --- Generate an annotated preview draft using the suggested rubric ---
    if not skip_suggestions:
        _update_status("Generating a new draft with the improved rubric...")
    if suggested_rubric and rcp.get("writing_task"):
        try:
            _sd_rubric_json = json.dumps(suggested_rubric, indent=2)
            _sd_prompt = ALIGNMENT_generate_annotated_draft_prompt(
                writing_task=rcp["writing_task"],
                suggested_rubric_json=_sd_rubric_json,
                original_rubric_draft=rubric_draft,
                suggestion_reasons=suggestion_reasons,
            )
            _pipeline_messages.append({"role": "user", "content": _sd_prompt})
            _sd_resp = _api_call_with_retry(
                model=MODEL_LIGHT, max_tokens=2000,
                messages=_pipeline_messages
            )
            _sd_raw = "".join(b.text for b in _sd_resp.content if b.type == "text").strip()
            _pipeline_messages.append({"role": "assistant", "content": _sd_raw})

            # Parse JSON response
            _sd_json_match = re.search(r'\{[\s\S]*\}', _sd_raw)
            if _sd_json_match:
                _sd_parsed = json.loads(_sd_json_match.group())
                suggested_draft = _sd_parsed.get("revised_draft", "")
                suggested_annotated_changes = _sd_parsed.get("annotated_changes", [])
            else:
                # Fallback: treat entire response as plain draft text
                suggested_draft = _sd_raw

            # DEBUG: print both drafts and annotated_changes for verification
            # print("\n" + "="*80)
            # print("[DEBUG] ALIGNMENT DRAFT GENERATION — PRIMARY PATH")
            # print("="*80)
            # print(f"\n--- ORIGINAL RUBRIC DRAFT ---\n{rubric_draft}")
            # print(f"\n--- SUGGESTED DRAFT (revised_draft) ---\n{suggested_draft}")
            # print(f"\n--- ANNOTATED CHANGES ({len(suggested_annotated_changes)} entries) ---")
            for _dbg_i, _dbg_ac in enumerate(suggested_annotated_changes, 1):
                # print(f"  [{_dbg_i}] original_text: {_dbg_ac.get('original_text', '')!r}")
                # print(f"       new_text:      {_dbg_ac.get('new_text', '')!r}")
                pass
                # print(f"       reason:        {_dbg_ac.get('reason', '')}")
            # print("="*80 + "\n")
        except Exception as e:
            pass

    # --- Verify suggested rubric against user preferences ---
    if not skip_suggestions:
        _update_status("Verifying the new rubric against your preferences...")
    _coldstart_prefs = st.session_state.get("infer_coldstart_text", "").strip()
    if suggested_rubric and suggested_draft and _coldstart_prefs:
        try:
            _verify_prompt = ALIGNMENT_verify_suggested_rubric_prompt(
                user_preferences=_coldstart_prefs,
                writing_task=rcp.get("writing_task", ""),
                rubric_draft=rubric_draft,
                generic_draft=generic_draft,
                preference_draft=preference_draft,
                user_ranking_description=ranking_desc,
                suggested_rubric_json=json.dumps(suggested_rubric, indent=2),
                suggested_draft=suggested_draft,
                original_suggestion_text=suggestion_text,
            )
            _pipeline_messages.append({"role": "user", "content": _verify_prompt})
            _verify_resp = _api_call_with_retry(
                model=MODEL_LIGHT, max_tokens=2000,
                messages=_pipeline_messages
            )
            _verify_text = "".join(b.text for b in _verify_resp.content if b.type == "text")
            _pipeline_messages.append({"role": "assistant", "content": _verify_text})
            _verify_match = re.search(r'\{[\s\S]*\}', _verify_text)
            if _verify_match:
                verification_result = json.loads(_verify_match.group())

                # If refinements needed, apply them and regenerate draft
                if verification_result.get("verdict") == "needs_refinement" and verification_result.get("refinements"):
                    _update_status("Refining the rubric based on verification...")
                    refinements = verification_result["refinements"]
                    # Convert refinements to bullet-point text for the apply prompt
                    _refine_bullets = []
                    for ref in refinements:
                        action = ref.get("action", "reword")
                        cname = ref.get("criterion_name", "")
                        new_text = ref.get("suggested_text", "")
                        reason = ref.get("reason", "")
                        if action == "add":
                            _refine_bullets.append(f"- Add a new criterion '{cname}': {new_text}. ({reason})")
                        elif action == "remove":
                            _refine_bullets.append(f"- Remove the criterion '{cname}'. ({reason})")
                        elif action == "adjust_weight":
                            _refine_bullets.append(f"- Adjust the priority/weight of '{cname}': {new_text}. ({reason})")
                        else:  # reword
                            _refine_bullets.append(f"- Reword '{cname}' to: {new_text}. ({reason})")
                    _refine_text = "\n".join(_refine_bullets)

                    # Apply refinements to the suggested rubric
                    try:
                        _sr_json = json.dumps(suggested_rubric, indent=2)
                        _apply_refine_prompt = RUBRIC_apply_suggestion_prompt(_sr_json, _sr_json, _refine_text)
                        _pipeline_messages.append({"role": "user", "content": _apply_refine_prompt})
                        _apply_refine_resp = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=4000,
                            messages=_pipeline_messages
                        )
                        _apply_refine_text = "".join(b.text for b in _apply_refine_resp.content if b.type == "text").strip()
                        _pipeline_messages.append({"role": "assistant", "content": _apply_refine_text})
                        _refine_js_match = re.search(r'\[[\s\S]*\]', _apply_refine_text)
                        if _refine_js_match:
                            refined_rubric = json.loads(_refine_js_match.group())
                            suggested_rubric = refined_rubric
                            suggestion_text = suggestion_text + "\n\n**Verification refinements:**\n" + _refine_text

                            # Regenerate annotated draft with refined rubric
                            _update_status("Regenerating draft with the refined rubric...")
                            _rd_rubric_json = json.dumps(refined_rubric, indent=2)
                            _rd_prompt = ALIGNMENT_generate_annotated_draft_prompt(
                                writing_task=rcp["writing_task"],
                                suggested_rubric_json=_rd_rubric_json,
                                original_rubric_draft=rubric_draft,
                                suggestion_reasons=suggestion_reasons,
                            )
                            _pipeline_messages.append({"role": "user", "content": _rd_prompt})
                            _rd_resp = _api_call_with_retry(
                                model=MODEL_LIGHT, max_tokens=2000,
                                messages=_pipeline_messages
                            )
                            _rd_raw = "".join(b.text for b in _rd_resp.content if b.type == "text").strip()
                            _pipeline_messages.append({"role": "assistant", "content": _rd_raw})

                            _rd_json_match = re.search(r'\{[\s\S]*\}', _rd_raw)
                            if _rd_json_match:
                                _rd_parsed = json.loads(_rd_json_match.group())
                                suggested_draft = _rd_parsed.get("revised_draft", "")
                                suggested_annotated_changes = _rd_parsed.get("annotated_changes", [])
                            else:
                                suggested_draft = _rd_raw

                            # DEBUG: print both drafts and annotated_changes for verification
                            # print("\n" + "="*80)
                            # print("[DEBUG] ALIGNMENT DRAFT GENERATION — REFINEMENT PATH")
                            # print("="*80)
                            # print(f"\n--- ORIGINAL RUBRIC DRAFT ---\n{rubric_draft}")
                            # print(f"\n--- SUGGESTED DRAFT (revised_draft) ---\n{suggested_draft}")
                            # print(f"\n--- ANNOTATED CHANGES ({len(suggested_annotated_changes)} entries) ---")
                            for _dbg_i, _dbg_ac in enumerate(suggested_annotated_changes, 1):
                                # print(f"  [{_dbg_i}] original_text: {_dbg_ac.get('original_text', '')!r}")
                                # print(f"       new_text:      {_dbg_ac.get('new_text', '')!r}")
                                # print(f"       reason:        {_dbg_ac.get('reason', '')}")
                                pass
                            # print("="*80 + "\n")
                    except Exception as e:
                        pass
        except Exception as e:
            pass

    # --- Build result ---
    # Convert ranking to legacy user_preference for backward compatibility
    _legacy_pref = user_ranking[0] if user_ranking else "tie"
    _update_status("Finalizing diagnostic results...")
    result = {
        "timestamp": datetime.now().isoformat(),
        "writing_task": rcp.get("writing_task", ""),
        "drafts": drafts,
        "shuffle_order": rcp.get("shuffle_order", []),
        "user_preference": _legacy_pref,
        "user_ranking": user_ranking,
        "user_reason": user_reason,
        "rubric_version": rubric_dict.get("version") if rubric_dict else None,
        "criteria_analysis": criteria_analysis,
        "rubric_judge_result": rubric_judge_result,
        "generic_judge_result": generic_judge_result,
        "suggestion_text": suggestion_text,
        "suggestion_reasons": suggestion_reasons,
        "suggested_rubric": suggested_rubric,
        "suggested_draft": suggested_draft,
        "suggested_annotated_changes": suggested_annotated_changes,
        "is_3draft": is_3draft,
        "verification_result": verification_result,
        "pipeline_messages": _pipeline_messages,
    }

    # Store the pipeline conversation so the main chat can build on it
    st.session_state.alignment_pipeline_messages = _pipeline_messages

    st.session_state.ranking_checkpoint_results.append(result)
    st.session_state.ranking_checkpoint_pending = None

    # Persist to database
    _save_sb = st.session_state.get('supabase')
    _save_pid = st.session_state.get('current_project_id')
    if _save_sb and _save_pid:
        try:
            save_project_data(_save_sb, _save_pid, "alignment_diagnostic", result)
        except Exception:
            pass

    # Launch background retest for reliability measurement
    if rubric_judge_result and generic_judge_result:
        import threading as _diag_rt_threading
        _diag_rt_args = {
            "draft_good": rubric_draft,
            "draft_degraded": generic_draft,
            "rubric_criteria_json": rubric_json,
            "conv_context": conv_context,
            "original_rubric_result": rubric_judge_result,
            "original_generic_result": generic_judge_result,
            "supabase": _save_sb,
            "project_id": _save_pid,
            "grade_eval_timestamp": result["timestamp"],
            "data_type": "diagnostic_retest",
            "results_list_ref": st.session_state.get("diagnostic_retest_history", []),
        }
        _diag_rt_threading.Thread(
            target=_run_grade_retest_bg,
            args=(_diag_rt_args,),
            daemon=True
        ).start()

    return result
