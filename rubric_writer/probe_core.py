"""Uncertainty probe commit flow."""
from rubric_writer.imports import *
from rubric_writer.persistence import get_active_rubric, _auto_save_conversation
from rubric_writer.probe_bg import _run_probe_refine_bg

def _probe_commit_choice(probe_state, chosen_label, user_reason=""):
    """Commit the user's uncertainty probe choice and launch background criterion refinement.

    Args:
        probe_state: dict from st.session_state.probe_pending
        chosen_label: "a", "b", or "skip"
        user_reason: optional free-text reason from user
    """
    import threading

    # Find the assistant message that the probe is attached to
    target_msg_id = probe_state.get("message_id")
    target_message = None
    for msg in reversed(st.session_state.messages):
        if msg.get("message_id") == target_msg_id:
            target_message = msg
            break

    # --- Build probe log message (rendered like DP confirmation) ---
    _probe_crit_name = probe_state.get("criterion_name", "")
    _probe_reason = probe_state.get("uncertainty_reason", "")
    _probe_interp_a = probe_state.get("interpretation_a", "")
    _probe_interp_b = probe_state.get("interpretation_b", "")
    _probe_variant_a = probe_state.get("variant_a", "")
    _probe_variant_b = probe_state.get("variant_b", "")
    _probe_dim_varied = probe_state.get("dimension_varied", "")

    def _build_probe_log(choice_label, choice_reason=""):
        """Build probe log content (summary only, drafts stored separately).

        Returns a dict with structured parts so the display layer can render
        them with better formatting (headers, expanders, etc.).
        The 'summary' key is a short one-liner for st.info().
        The 'full_markdown' key is the full text for backward compat / saving.
        """
        # --- short summary line for the header ---
        if choice_label == "skip":
            _choice_str = "Skipped"
        else:
            _choice_str = "Version A" if choice_label == "a" else "Version B"

        summary = f"Rubric Clarity Check — **\"{_probe_crit_name}\"** → You chose **{_choice_str}**"

        # --- full markdown (stored in message content for DB persistence) ---
        lines = [f"**Rubric Clarity Check: \"{_probe_crit_name}\"**\n"]

        # Split reason into bullet points by sentence boundaries
        if _probe_reason:
            lines.append("**Why probed:**")
            import re as _re_probe
            _reason_sentences = [s.strip() for s in _re_probe.split(r'(?<=[.!?])\s+', _probe_reason) if s.strip()]
            for _rs in _reason_sentences:
                lines.append(f"- {_rs}")
            if _probe_dim_varied:
                lines.append(f"- Dimension varied: {_probe_dim_varied}")
        elif _probe_dim_varied:
            lines.append(f"**Dimension varied:** {_probe_dim_varied}")

        lines.append(f"\n**Interpretation A:** {_probe_interp_a}")
        lines.append(f"\n**Interpretation B:** {_probe_interp_b}")
        lines.append("")
        if choice_label == "skip":
            lines.append("**Your choice:** Skipped")
        else:
            chosen_display = "Version A" if choice_label == "a" else "Version B"
            lines.append(f"**Your choice:** {chosen_display}")
            if choice_reason:
                lines.append(f"**Your reason:** {choice_reason}")
        full_markdown = "\n".join(lines)

        return {"summary": summary, "full_markdown": full_markdown}

    def _make_probe_log_msg(choice_label, choice_reason=""):
        _plm_data = _build_probe_log(choice_label, choice_reason)
        return {
            "role": "assistant",
            "content": _plm_data["full_markdown"],
            "display_content": _plm_data["full_markdown"],
            "probe_log_summary": _plm_data["summary"],
            "is_system_generated": True,
            "is_probe_log": True,
            "probe_log_data": {
                "criterion_name": _probe_crit_name,
                "variant_a": _probe_variant_a,
                "variant_b": _probe_variant_b,
                "user_choice": choice_label,
                "source_message_id": target_msg_id,
                "reason": _probe_reason,
                "dimension_varied": _probe_dim_varied,
                "interpretation_a": _probe_interp_a,
                "interpretation_b": _probe_interp_b,
            },
            "message_id": f"probe_log_{int(time.time() * 1000000)}",
        }

    if chosen_label == "skip":
        # Store minimal probe result on the message
        if target_message:
            target_message["probe_result"] = {
                "criterion_name": _probe_crit_name,
                "user_choice": "skip",
                "user_reason": "",
                "variant_a": _probe_variant_a,
                "variant_b": _probe_variant_b,
                "interpretation_a": _probe_interp_a,
                "interpretation_b": _probe_interp_b,
                "updated_criterion": None,
                "applied": False,
                "applied_version": None,
            }
        # Insert probe log message BEFORE the assistant draft message
        _skip_log_msg = _make_probe_log_msg("skip")
        _skip_target_idx = None
        for _i, _m in enumerate(st.session_state.messages):
            if _m.get("message_id") == target_msg_id:
                _skip_target_idx = _i
                break
        if _skip_target_idx is not None:
            st.session_state.messages.insert(_skip_target_idx, _skip_log_msg)
        else:
            st.session_state.messages.append(_skip_log_msg)
        # Persist skip result
        _save_sb = st.session_state.get("supabase")
        _save_pid = st.session_state.get("current_project_id")
        if _save_sb and _save_pid:
            try:
                save_project_data(_save_sb, _save_pid, "probe_results", {
                    "timestamp": datetime.now().isoformat(),
                    "rubric_version": probe_state.get("rubric_version"),
                    "conversation_id": st.session_state.get("selected_conversation", ""),
                    "criterion_name": _probe_crit_name,
                    "criterion_index": probe_state.get("criterion_index", -1),
                    "interpretation_a": _probe_interp_a,
                    "interpretation_b": _probe_interp_b,
                    "uncertainty_reason": _probe_reason,
                    "user_choice": "skip",
                    "user_reason": "",
                    "updated_criterion": None,
                    "rubric_updated": False,
                })
            except Exception:
                pass
        st.session_state.probe_pending = None
        _auto_save_conversation()
        return

    # User chose a or b
    _orig_slot = probe_state.get("original_slot", "a")
    # Determine which interpretation was chosen based on slot mapping:
    # original draft follows interpretation_a, alternative follows interpretation_b
    if chosen_label == _orig_slot:
        # User chose the original draft → they prefer interpretation_a
        chosen_interpretation = probe_state.get("interpretation_a", "")
    else:
        # User chose the alternative draft → they prefer interpretation_b
        chosen_interpretation = probe_state.get("interpretation_b", "")

    # If user chose the alternative draft, replace the assistant message's draft
    if chosen_label != _orig_slot and target_message:
        _alt_draft = probe_state.get("alternative_draft", "")
        if _alt_draft:
            _orig_content = target_message.get("content", "")
            _orig_display = target_message.get("display_content", _orig_content)
            # Replace the <draft>...</draft> content with the preferred alternative
            _new_content = re.sub(
                r'<draft>.*?</draft>',
                f'<draft>\n{_alt_draft}\n</draft>',
                _orig_content, count=1, flags=re.DOTALL
            )
            _new_display = re.sub(
                r'<draft>.*?</draft>',
                f'<draft>\n{_alt_draft}\n</draft>',
                _orig_display, count=1, flags=re.DOTALL
            )
            target_message["content"] = _new_content
            target_message["display_content"] = _new_display

    # Store probe result on the message (updated_criterion will be filled by background thread)
    probe_result = {
        "criterion_name": _probe_crit_name,
        "user_choice": chosen_label,
        "user_reason": user_reason,
        "variant_a": _probe_variant_a,
        "variant_b": _probe_variant_b,
        "interpretation_a": _probe_interp_a,
        "interpretation_b": _probe_interp_b,
        "original_slot": _orig_slot,
        "chose_original": chosen_label == _orig_slot,
        "updated_criterion": None,  # filled by background thread
        "applied": False,
        "applied_version": None,
    }
    if target_message:
        target_message["probe_result"] = probe_result

    # Insert probe log message BEFORE the assistant draft message (so it appears above)
    _probe_log_msg = _make_probe_log_msg(chosen_label, user_reason)
    _target_idx = None
    for _i, _m in enumerate(st.session_state.messages):
        if _m.get("message_id") == target_msg_id:
            _target_idx = _i
            break
    if _target_idx is not None:
        st.session_state.messages.insert(_target_idx, _probe_log_msg)
    else:
        st.session_state.messages.append(_probe_log_msg)

    # Get criterion JSON for refinement
    rubric_dict, _, _ = get_active_rubric()
    rubric_list = rubric_dict.get("rubric", []) if rubric_dict else []
    criterion_index = probe_state.get("criterion_index", -1)
    criterion_json = ""
    if 0 <= criterion_index < len(rubric_list):
        criterion_json = json.dumps(rubric_list[criterion_index], indent=2)
    else:
        # Fallback: find by name
        crit_name = _probe_crit_name.lower().strip()
        for c in rubric_list:
            if c.get("name", "").lower().strip() == crit_name:
                criterion_json = json.dumps(c, indent=2)
                break

    # Launch background thread for criterion refinement
    _refine_args = {
        "criterion_json": criterion_json,
        "chosen_interpretation": chosen_interpretation,
        "user_reason": user_reason,
        "message_data_ref": target_message,
        "probe_log_ref": _probe_log_msg,
        "results_list_ref": st.session_state.get("probe_results", []),
        "probe_state": probe_state,
        "supabase": st.session_state.get("supabase"),
        "project_id": st.session_state.get("current_project_id"),
        "conversation_id": st.session_state.get("selected_conversation", ""),
        "rubric_version": rubric_dict.get("version") if rubric_dict else None,
    }

    _refine_event = threading.Event()

    def _refine_background(args, done_event):
        try:
            _run_probe_refine_bg(args)
        except Exception as e:
            # print(f"[PROBE] Refine background failed: {e}")
            pass
        finally:
            done_event.set()

    threading.Thread(target=_refine_background, args=(_refine_args, _refine_event), daemon=True).start()
    st.session_state._probe_refine_done_event = _refine_event

    # Clear pending state
    st.session_state.probe_pending = None
    _auto_save_conversation()
