#!/usr/bin/env python3
"""
Automated synthetic user pipeline for RubricLLM evaluation.

Usage:
    python simulate.py                          # run all personas, default config
    python simulate.py --num-users 3            # first 3 personas
    python simulate.py --iterations 2           # 2 refinement iterations each
    python simulate.py --user-provider openai --user-model gpt-5.2
    python simulate.py --user-provider google --user-model gemini-3.1-pro-preview
    python simulate.py --dry-run                # print config and exit
"""

import argparse
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime

# Ensure project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim_config import SimConfig, Persona, PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GOOGLE
from simulation_engine import (
    headless_infer_rubric,
    headless_classify_criteria,
    headless_extract_dps,
    headless_infer_final_rubric,
    headless_chat_response,
    headless_alignment_diagnostic,
    generate_three_drafts,
    build_conversation_text,
    _rubric_to_json_serializable,
    extract_draft_from_response,

    classify_rubric_edits,
    format_edit_log_message,
    headless_regenerate_draft_from_rubric_changes,
    headless_suggest_rubric_from_feedback,
    headless_apply_rubric_suggestion,
    evaluate_preference_coverage,
    decompose_preferences,
)
from synthetic_user import SyntheticUser

log = logging.getLogger("sim")


def _current_iteration_messages(messages: list) -> list:
    """Return only messages from the current (most recent) iteration."""
    last_boundary = 0
    for idx, m in enumerate(messages):
        if m.get("role") == "system" and "Iteration" in m.get("content", ""):
            last_boundary = idx
    # Return messages after the boundary marker, excluding system messages
    return [m for m in messages[last_boundary:] if m.get("role") in ("user", "assistant")]


# ═════════════════════════════════════════════════════════════════════════════
# Local JSON Store
# ═════════════════════════════════════════════════════════════════════════════

class LocalStore:
    """Saves all simulation data to local JSON files.

    Directory structure:
        eval_output/sim_<timestamp>/
            <persona_name>/
                project.json           — project metadata
                coldstart.json         — cold-start preferences
                conversations.json     — all messages
                rubric_history.json    — all rubric versions
                project_data/
                    criteria_classification_feedback.json
                    decision_point_feedback.json
                    alignment_diagnostic.json
                    rubric_edit_applied.json
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def user_dir(self, persona_name: str) -> str:
        safe_name = persona_name.replace(" ", "_").lower()
        d = os.path.join(self.base_dir, safe_name)
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "project_data"), exist_ok=True)
        return d

    def save_project(self, persona_name: str, data: dict):
        path = os.path.join(self.user_dir(persona_name), "project.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_coldstart(self, persona_name: str, data: dict):
        path = os.path.join(self.user_dir(persona_name), "coldstart.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_conversation(self, persona_name: str, messages: list, rubric_data=None):
        path = os.path.join(self.user_dir(persona_name), "conversations.json")
        data = {
            "messages": messages,
            "rubric": _rubric_to_json_serializable(rubric_data) if rubric_data else None,
            "updated_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_rubric(self, persona_name: str, rubric_data: dict):
        path = os.path.join(self.user_dir(persona_name), "rubric_history.json")
        history = []
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
        history.append(_rubric_to_json_serializable(rubric_data))
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def save_project_data(self, persona_name: str, data_type: str, data: dict):
        path = os.path.join(self.user_dir(persona_name), "project_data", f"{data_type}.json")
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        existing.append(data)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# Main Simulation Loop
# ═════════════════════════════════════════════════════════════════════════════

def run_one_user(persona: Persona, config: SimConfig, store: LocalStore) -> dict:
    """Run the full simulation pipeline for one synthetic user."""
    log.info(f"\n{'='*60}\nStarting simulation for: {persona.name} ({persona.role})\n{'='*60}")

    # ── Setup ────────────────────────────────────────────────────────────
    project_id = str(uuid.uuid4())
    max_turns = config.max_chat_turns

    store.save_project(persona.name, {
        "id": project_id,
        "persona": persona.__dict__,
        "config": {
            "user_provider": config.user_provider,
            "user_model": config.user_model,
            "model_primary": config.model_primary,
            "model_light": config.model_light,
            "num_iterations": config.num_iterations,
            "max_chat_turns": max_turns,
            "satisfaction_threshold": config.satisfaction_threshold,
        },
        "created_at": datetime.now().isoformat(),
    })

    syn_user = SyntheticUser(
        persona,
        provider=config.user_provider,
        model=config.user_model,
        temperature=config.user_temperature,
    )

    rubric_data = None
    rubric_history = []
    messages = []
    version_counter = 0
    termination_log = []
    survey_responses = {"task_a": None, "task_b": []}
    coverage_trajectory = []  # per-iteration preference coverage scores

    # ── Decompose preferences once (fixed list for all coverage evals) ───
    log.info(f"[{persona.name}] Decomposing preferences into atomic items...")
    atomic_preferences = decompose_preferences(persona, model=config.model_primary)

    # ── Phase 1: Cold-start preferences ──────────────────────────────────
    log.info(f"[{persona.name}] Phase 1: Generating cold-start preferences...")
    coldstart_text = syn_user.generate_coldstart_preferences()
    store.save_coldstart(persona.name, {
        "text": coldstart_text,
        "timestamp": datetime.now().isoformat(),
        "source": "simulation",
    })
    log.info(f"[{persona.name}] Cold-start: {coldstart_text[:100]}...")

    # Track previous classifications for flip detection (P0 fix)
    prev_classifications = {}

    # ── Main iteration loop ──────────────────────────────────────────────
    for iteration in range(config.num_iterations):
        log.info(f"\n[{persona.name}] === Iteration {iteration + 1}/{config.num_iterations} ===")

        # Mark iteration boundary in messages for analysis
        messages.append({
            "role": "system",
            "content": f"--- Iteration {iteration + 1}/{config.num_iterations} ---",
            "iteration": iteration + 1,
        })

        # User provides a writing task
        phase = "initial_request" if iteration == 0 else "new_task"
        user_msg = syn_user.generate_chat_message(messages, rubric_data, phase=phase)
        messages.append({"role": "user", "content": user_msg})
        store.save_conversation(persona.name, messages, rubric_data)

        # ── Alignment Check Gate (iterations 2+ only) ─────────────────────
        # Before the chat loop, run the alignment diagnostic to check if the
        # rubric is helping. The user ranks 3 blind drafts, and the winning
        # draft becomes the starting point for the conversation.
        alignment_starting_draft = None
        if iteration > 0 and rubric_data:
            log.info(f"[{persona.name}] Alignment check gate (pre-conversation diagnostic)...")

            writing_task = user_msg
            log.info(f"[{persona.name}] Writing task: {writing_task[:80]}...")

            # Generate 3 blind drafts
            drafts = generate_three_drafts(writing_task, rubric_data, coldstart_text, model=config.model_light)

            # Shuffle and have synthetic user rank
            sources = ["rubric", "generic", "preference"]
            random.shuffle(sources)
            shuffle_order = list(zip(["A", "B", "C"], sources))
            blind_drafts = {label: drafts[src] for label, src in shuffle_order}

            ranking_labels, ranking_reason = syn_user.rank_drafts(blind_drafts, writing_task)
            label_to_source = {label: src for label, src in shuffle_order}
            user_ranking = [label_to_source[label] for label in ranking_labels if label in label_to_source]
            log.info(f"[{persona.name}] User ranked: {user_ranking} ({ranking_reason[:60]}...)")

            # Run diagnostic analysis
            diagnostic_result = headless_alignment_diagnostic(
                drafts=drafts,
                writing_task=writing_task,
                user_ranking=user_ranking,
                user_reason=ranking_reason,
                rubric_data=rubric_data,
                messages=messages,
                coldstart_text=coldstart_text,
                shuffle_order=shuffle_order,
                model_primary=config.model_primary,
                model_light=config.model_light,
            )

            # Save diagnostic
            diagnostic_result["iteration"] = iteration + 1
            diagnostic_result["phase"] = "pre_conversation"
            store.save_project_data(persona.name, "alignment_diagnostic", diagnostic_result)

            # Apply rubric suggestion if diagnostic generated one
            if diagnostic_result.get("suggested_rubric"):
                accept = syn_user.decide_on_rubric_suggestion(
                    rubric_data, diagnostic_result["suggested_rubric"],
                    diagnostic_result.get("suggestion_reasons", {}),
                )
                if accept:
                    log.info(f"[{persona.name}] Applying rubric suggestion from alignment check...")
                    version_counter += 1
                    new_rubric = {
                        "rubric": diagnostic_result["suggested_rubric"],
                        "version": version_counter,
                        "source": "alignment_suggestion",
                    }
                    for key in ("writing_type", "user_goals_summary", "coaching_notes"):
                        if key in rubric_data:
                            new_rubric[key] = rubric_data[key]

                    rubric_data = new_rubric
                    rubric_history.append(rubric_data)
                    store.save_rubric(persona.name, rubric_data)

                    store.save_project_data(persona.name, "rubric_edit_applied", {
                        "source": "alignment_diagnostic",
                        "rubric_version": version_counter,
                        "suggestion_reasons": diagnostic_result.get("suggestion_reasons", {}),
                        "timestamp": datetime.now().isoformat(),
                    })

                    # If rubric was updated, use the suggested draft (regenerated with new rubric)
                    if diagnostic_result.get("suggested_draft"):
                        alignment_starting_draft = diagnostic_result["suggested_draft"]
                        log.info(f"[{persona.name}] Using suggested draft (from updated rubric) as starting point")
                else:
                    log.info(f"[{persona.name}] User rejected rubric suggestion from alignment check")

            # If no suggested draft was used, fall back to the top-ranked draft
            if not alignment_starting_draft:
                alignment_starting_draft = drafts.get(user_ranking[0], "")
                log.info(f"[{persona.name}] Using top-ranked draft ({user_ranking[0]}) as starting point")

            # Inject the winning draft as the first assistant message
            if alignment_starting_draft:
                messages.append({
                    "role": "assistant",
                    "content": f"<draft>{alignment_starting_draft}</draft>",
                    "is_alignment_start_draft": True,
                })
                store.save_conversation(persona.name, messages, rubric_data)

        # ── Goal-based chat ───────────────────────────────────────────────
        log.info(f"[{persona.name}] Chat loop (max {max_turns} drafts, threshold {config.satisfaction_threshold})...")

        turn_metrics = []
        termination_reason = "max_drafts"
        draft_count = 0
        turn = 0
        consecutive_no_draft = 0  # guard against infinite no-draft loops
        log_changes_count = 0  # how many times rubric edit has fired this iteration

        # If alignment check injected a starting draft, evaluate it first
        if alignment_starting_draft:
            turn += 1
            draft_count += 1
            current_draft = alignment_starting_draft
            log.info(f"[{persona.name}]   Turn {turn} — evaluating alignment starting draft (draft {draft_count}/{max_turns})")

            # User decides: accept or give feedback
            user_response = syn_user.respond_to_draft(
                current_draft, messages, rubric_data,
                iteration=iteration, draft_count=draft_count,
            )

            turn_metric = {
                "turn": turn, "has_draft": True, "draft_number": draft_count,
                "satisfaction": {"score": user_response["score"], "reasoning": user_response["reasoning"]},
                "is_alignment_start": True,
                "user_accepted": user_response["accepted"],
                "user_satisfied": user_response["accepted"],
            }
            log.info(
                f"[{persona.name}]   Alignment draft — "
                f"User: {'ACCEPTED' if user_response['accepted'] else 'FEEDBACK'} "
                f"(score: {user_response['score']:.2f})"
            )
            turn_metrics.append(turn_metric)

            if user_response["accepted"]:
                # Add the acceptance message to the conversation
                accept_text = user_response["feedback"] or "This looks good — I'm happy with this draft."
                messages.append({"role": "user", "content": accept_text})
                store.save_conversation(persona.name, messages, rubric_data)
                termination_reason = "user_accepted"
                log.info(f"[{persona.name}]   User accepted alignment draft!")
            else:
                feedback_text = user_response["feedback"] or "This isn't quite there yet — can you revise it?"
                messages.append({"role": "user", "content": feedback_text})
                store.save_conversation(persona.name, messages, rubric_data)

        max_total_turns = max_turns * 3  # hard cap on total turns to prevent infinite loops
        while draft_count < max_turns and turn < max_total_turns and termination_reason != "user_accepted":
            turn += 1
            log.info(f"[{persona.name}]   Turn {turn} (drafts so far: {draft_count}/{max_turns})")

            # System responds (scoped to current iteration only)
            assistant_resp = headless_chat_response(
                _current_iteration_messages(messages), rubric_data, model=config.model_light,
            )
            messages.append({"role": "assistant", "content": assistant_resp})
            store.save_conversation(persona.name, messages, rubric_data)

            # Extract draft from response
            current_draft = extract_draft_from_response(assistant_resp)

            # ── Termination check (only if a draft was produced) ──
            turn_metric = {"turn": turn, "has_draft": current_draft is not None}

            if current_draft:
                draft_count += 1
                consecutive_no_draft = 0
                turn_metric["draft_number"] = draft_count

                # Single call: user checks preferences and decides accept/feedback
                user_response = syn_user.respond_to_draft(
                    current_draft, messages, rubric_data,
                    iteration=iteration, draft_count=draft_count,
                )
                turn_metric["satisfaction"] = {
                    "score": user_response["score"],
                    "reasoning": user_response["reasoning"],
                    "accepted": user_response["accepted"],
                }

                turn_metric["user_accepted"] = user_response["accepted"]

                log.info(
                    f"[{persona.name}]   Draft {draft_count}/{max_turns} — "
                    f"User: {'ACCEPTED' if user_response['accepted'] else 'FEEDBACK'} "
                    f"(score: {user_response['score']:.2f})"
                )

                if user_response["accepted"]:
                    # Add the acceptance message to the conversation
                    accept_text = user_response["feedback"] or "This looks good — I'm happy with this draft."
                    messages.append({"role": "user", "content": accept_text})
                    store.save_conversation(persona.name, messages, rubric_data)
                    turn_metrics.append(turn_metric)
                    termination_reason = "user_accepted"
                    log.info(f"[{persona.name}]   User accepted draft at draft {draft_count}!")
                    break
                else:
                    feedback_text = user_response["feedback"] or "This isn't quite there yet — can you revise it?"
                    messages.append({"role": "user", "content": feedback_text})
                    store.save_conversation(persona.name, messages, rubric_data)
            else:
                consecutive_no_draft += 1
                log.warning(f"[{persona.name}]   No draft in response ({consecutive_no_draft} consecutive), nudging assistant...")
                # Add a user nudge so the conversation doesn't end on assistant message
                messages.append({"role": "user", "content": "Could you write up a full draft for me based on what we've discussed?"})
                store.save_conversation(persona.name, messages, rubric_data)

            turn_metrics.append(turn_metric)

            # ── Mid-conversation Log Changes ──────────────────────────
            # The synthetic user decides when to directly edit the rubric.
            # Full flow mirrors the real app:
            #   1. User edits rubric → Log Changes
            #   2. Draft regenerated with annotated [N] markers
            #   3. User reviews each edit, can leave feedback
            #   4. If feedback exists → suggest rubric changes → user accepts/rejects
            if (config.enable_log_changes and rubric_data and current_draft
                    and draft_count < max_turns):
                should_edit = syn_user.should_edit_rubric(
                    messages, draft_count, max_turns, log_changes_count,
                )
                if should_edit:
                    log.info(f"[{persona.name}]   Log Changes: user editing rubric mid-conversation...")
                    rubric_list = rubric_data.get("rubric", [])
                    edit_result = syn_user.propose_rubric_edits(rubric_list)
                    edited_rubric = edit_result.get("edited_rubric", [])

                    if edited_rubric and edited_rubric != rubric_list:
                        edit_classification = classify_rubric_edits(rubric_list, edited_rubric)
                        has_changes = any(edit_classification[k] for k in edit_classification)

                        if has_changes:
                            log_msg = format_edit_log_message(
                                edit_classification,
                                rubric_data.get("version", 0),
                                f"{rubric_data.get('version', 0)}*",
                                "simulation_log_changes",
                            )
                            messages.append({"role": "system", "content": log_msg})

                            # Step 2: Regenerate draft based on rubric changes
                            regen_result = {}
                            annotated_changes = []
                            if current_draft:
                                regen_result = headless_regenerate_draft_from_rubric_changes(
                                    rubric_list, edited_rubric, current_draft,
                                    model=config.model_light,
                                )
                                if "revised_draft" in regen_result:
                                    annotated_changes = regen_result.get("annotated_changes", [])
                                    draft_msg = (
                                        f"<draft>{regen_result['revised_draft']}</draft>"
                                        f"\n\n*Draft updated based on rubric changes.*"
                                    )
                                    messages.append({
                                        "role": "assistant",
                                        "content": draft_msg,
                                        "rubric_revision": {
                                            "change_summary": regen_result.get("change_summary", ""),
                                            "annotated_changes": annotated_changes,
                                            "original_draft": current_draft,
                                            "old_rubric": rubric_list,
                                            "new_rubric": edited_rubric,
                                        },
                                    })
                                    draft_count += 1
                                    log.info(f"[{persona.name}]   Draft regenerated from rubric edit (draft {draft_count})")
                                else:
                                    log.warning(f"[{persona.name}]   Draft regeneration failed: {regen_result.get('error', 'unknown')}")

                            # Step 3: User reviews annotated changes
                            reviewed_changes = []
                            if annotated_changes:
                                reviewed_changes = syn_user.review_annotated_changes(
                                    annotated_changes,
                                    current_draft,
                                    regen_result.get("revised_draft", ""),
                                )

                            # Step 4: If user disagreed with any edit, suggest rubric changes
                            feedback_edits = [
                                rc for rc in reviewed_changes
                                if rc.get("user_feedback", "").strip()
                            ]
                            suggestion_applied = False
                            if feedback_edits:
                                log.info(
                                    f"[{persona.name}]   User disagreed with {len(feedback_edits)} edit(s), "
                                    f"suggesting rubric changes..."
                                )
                                suggestion_text = headless_suggest_rubric_from_feedback(
                                    rubric_list, edited_rubric, reviewed_changes,
                                    model=config.model_light,
                                )
                                if suggestion_text:
                                    suggested_rubric = headless_apply_rubric_suggestion(
                                        rubric_list, edited_rubric, suggestion_text,
                                        model=config.model_light,
                                    )
                                    if suggested_rubric:
                                        # User decides whether to accept the suggestion
                                        accept = syn_user.decide_on_rubric_suggestion(
                                            {"rubric": edited_rubric},
                                            suggested_rubric,
                                            {"feedback_suggestion": suggestion_text},
                                        )
                                        if accept:
                                            edited_rubric = suggested_rubric
                                            suggestion_applied = True
                                            log.info(f"[{persona.name}]   User accepted rubric suggestion from feedback")
                                        else:
                                            log.info(f"[{persona.name}]   User rejected rubric suggestion from feedback")

                            # If suggestion was applied, regenerate draft with the updated rubric
                            if suggestion_applied and current_draft:
                                post_regen = headless_regenerate_draft_from_rubric_changes(
                                    rubric_list, edited_rubric, current_draft,
                                    model=config.model_light,
                                )
                                if "revised_draft" in post_regen:
                                    current_draft = post_regen["revised_draft"]
                                    draft_msg = (
                                        f"<draft>{current_draft}</draft>"
                                        f"\n\n*Draft updated based on accepted rubric suggestion.*"
                                    )
                                    messages.append({
                                        "role": "assistant",
                                        "content": draft_msg,
                                        "is_post_apply_draft": True,
                                    })
                                    draft_count += 1
                                    log.info(f"[{persona.name}]   Post-suggestion draft regenerated (draft {draft_count})")

                            # Update rubric data (with either direct edits or suggestion-modified edits)
                            version_counter += 1
                            old_rubric_data = rubric_data
                            rubric_data = {
                                "rubric": edited_rubric,
                                "version": version_counter,
                                "source": "user_log_changes",
                            }
                            for key in ("writing_type", "user_goals_summary", "coaching_notes"):
                                if key in old_rubric_data:
                                    rubric_data[key] = old_rubric_data[key]

                            rubric_history.append(rubric_data)
                            store.save_rubric(persona.name, rubric_data)

                            store.save_project_data(persona.name, "log_changes", {
                                "iteration": iteration + 1,
                                "draft_number": draft_count,
                                "edit_classification": edit_classification,
                                "reasoning": edit_result.get("reasoning", ""),
                                "rubric_version": version_counter,
                                "draft_regenerated": "revised_draft" in regen_result,
                                "reviewed_changes": reviewed_changes,
                                "feedback_edits_count": len(feedback_edits),
                                "suggestion_applied": suggestion_applied,
                                "timestamp": datetime.now().isoformat(),
                            })

                            log_changes_count += 1
                            log.info(f"[{persona.name}]   Log Changes complete (rubric v{version_counter})")

                            # Save conversation after rubric edit
                            store.save_conversation(persona.name, messages, rubric_data)

        # Record termination data for this iteration
        termination_log.append({
            "iteration": iteration + 1,
            "drafts_produced": draft_count,
            "max_drafts": max_turns,
            "total_turns": turn,
            "termination_reason": termination_reason,
            "turn_metrics": turn_metrics,
        })

        if termination_reason == "max_drafts":
            log.info(f"[{persona.name}]   Reached max drafts ({max_turns}) without goal")

        # Save conversation
        store.save_conversation(persona.name, messages, rubric_data)

        # ── Survey: Task A (after first chat, before rubric) ────────────
        if iteration == 0:
            log.info(f"[{persona.name}] Survey: Completing Task A (without rubric)...")
            task_a_result = syn_user.complete_survey_task_a(messages)
            survey_responses["task_a"] = task_a_result
            store.save_project_data(persona.name, "survey_task_a", {
                **task_a_result, "iteration": 1,
            })
            log.info(f"[{persona.name}] Task A survey: understanding={task_a_result['q1']}/5, effort={task_a_result['q2']}/5")

        # ── Phase 3: Rubric inference (5-step flow) ──────────────────────
        log.info(f"[{persona.name}] Phase 3: Rubric inference...")
        version_counter += 1

        # Step 1: Infer rubric
        rubric_data = headless_infer_rubric(
            messages, previous_rubric=rubric_data,
            model=config.model_primary, version=version_counter,
        )
        if not rubric_data:
            log.error(f"[{persona.name}] Rubric inference failed, skipping iteration")
            continue

        rubric_history.append(rubric_data)
        store.save_rubric(persona.name, rubric_data)

        rubric_json = json.dumps(rubric_data.get("rubric", []), ensure_ascii=False, indent=2)

        # Step 1b: LLM classifies criteria vs cold-start
        log.info(f"[{persona.name}] Classifying criteria...")
        llm_classification = headless_classify_criteria(
            rubric_data, coldstart_text, messages, model=config.model_primary,
        )

        # Step 2: Synthetic user reviews classifications
        user_classifications = syn_user.classify_criteria(
            rubric_data.get("rubric", []), llm_classification,
        )
        importance_ranking = syn_user.rank_importance(
            rubric_data.get("rubric", []), user_classifications,
        )

        # P0: Flip detection — if a criterion was real/stated in the previous
        # iteration and is now hallucinated, re-classify it to verify.
        # This prevents stochastic LLM flips from deleting valid criteria.
        if prev_classifications:
            flipped = []
            for cname, new_cls in user_classifications.items():
                prev_cls = prev_classifications.get(cname)
                if prev_cls in ("stated", "real") and new_cls == "hallucinated":
                    flipped.append(cname)
            if flipped:
                log.warning(f"[{persona.name}] Classification flip detected: {flipped} changed from real/stated → hallucinated. Re-verifying...")
                # Re-classify just the flipped criteria with a focused prompt
                re_result = syn_user.classify_criteria(
                    rubric_data.get("rubric", []), llm_classification,
                )
                for cname in flipped:
                    second_cls = re_result.get(cname, "hallucinated")
                    if second_cls != "hallucinated":
                        log.info(f"[{persona.name}] Flip overridden: '{cname}' re-classified as '{second_cls}' (was flipped to hallucinated)")
                        user_classifications[cname] = second_cls
                    else:
                        log.info(f"[{persona.name}] Flip confirmed: '{cname}' is hallucinated on second pass too")

        # Store for next iteration's flip detection
        prev_classifications = dict(user_classifications)

        # Compute metrics
        n_criteria = len(rubric_data.get("rubric", []))
        n_stated = sum(1 for v in user_classifications.values() if v == "stated")
        n_real = sum(1 for v in user_classifications.values() if v == "real")
        n_hallucinated = sum(1 for v in user_classifications.values() if v == "hallucinated")
        precision = (n_stated + n_real) / n_criteria if n_criteria else 0
        log.info(f"[{persona.name}] Classification: {n_stated} stated, {n_real} real, {n_hallucinated} hallucinated (precision={precision:.2f})")

        classification_feedback = {
            "iteration": iteration + 1,
            "classifications": user_classifications,
            "importance_ranking": importance_ranking,
            "n_criteria": n_criteria,
            "n_stated": n_stated,
            "n_real": n_real,
            "n_hallucinated": n_hallucinated,
            "precision": precision,
            "rubric_version": version_counter,
            "ground_truth": {
                "core_preferences": persona.core_preferences,
                "hidden_preferences": persona.hidden_preferences,
                "dealbreakers": persona.dealbreakers,
            },
            "timestamp": datetime.now().isoformat(),
        }
        store.save_project_data(persona.name, "criteria_classification_feedback", classification_feedback)

        # Step 3: Extract decision points (DISABLED — saving cost)
        # log.info(f"[{persona.name}] Extracting decision points...")
        # dp_result = headless_extract_dps(
        #     messages, rubric_json, classification_feedback, model=config.model_primary,
        # )

        # Step 4: Synthetic user confirms/corrects DPs (DISABLED — saving cost)
        # corrected_dps = syn_user.provide_dp_feedback(dp_result)
        # store.save_project_data(persona.name, "decision_point_feedback", {
        #     **corrected_dps,
        #     "rubric_version": version_counter,
        #     "timestamp": datetime.now().isoformat(),
        # })

        # Step 5: Remove hallucinated criteria and reorder by importance ranking
        rubric_list = rubric_data.get("rubric", [])
        rubric_changed = False

        # Remove hallucinated criteria
        if n_hallucinated > 0:
            kept = [c for c in rubric_list if user_classifications.get(c.get("name", "?")) != "hallucinated"]
            removed_names = [c.get("name", "?") for c in rubric_list if user_classifications.get(c.get("name", "?")) == "hallucinated"]
            log.info(f"[{persona.name}] Removing {n_hallucinated} hallucinated criteria: {removed_names}")
            rubric_list = kept
            rubric_changed = True

        # Reorder by importance ranking
        if importance_ranking:
            name_to_criterion = {c.get("name", "?"): c for c in rubric_list}
            reordered = []
            for rank_name in importance_ranking:
                if rank_name in name_to_criterion:
                    reordered.append(name_to_criterion.pop(rank_name))
            # Append any criteria not in the ranking (shouldn't happen, but safe)
            reordered.extend(name_to_criterion.values())
            # Update priority numbers
            for i, c in enumerate(reordered):
                c["priority"] = i + 1
            if reordered != rubric_list:
                rubric_list = reordered
                rubric_changed = True
                log.info(f"[{persona.name}] Reordered rubric by importance: {[c.get('name', '?') for c in rubric_list]}")

        if rubric_changed:
            version_counter += 1
            old_rubric_data = rubric_data
            rubric_data = {
                "rubric": rubric_list,
                "version": version_counter,
                "source": "criteria_classification",
            }
            for key in ("writing_type", "user_goals_summary", "coaching_notes"):
                if key in old_rubric_data:
                    rubric_data[key] = old_rubric_data[key]
            rubric_history.append(rubric_data)
            store.save_rubric(persona.name, rubric_data)
            log.info(f"[{persona.name}] Rubric updated to v{version_counter} after classification")

        # Save conversation state
        store.save_conversation(persona.name, messages, rubric_data)

        # ── Survey: Task B (after rubric-enhanced iterations 2+) ────────
        if iteration > 0:
            log.info(f"[{persona.name}] Survey: Completing Task B (with rubric, iteration {iteration + 1})...")
            task_b_result = syn_user.complete_survey_task_b(
                messages, rubric_data, iteration + 1
            )
            survey_responses["task_b"].append(task_b_result)
            store.save_project_data(persona.name, "survey_task_b", {
                **task_b_result, "iteration": iteration + 1,
            })
            log.info(
                f"[{persona.name}] Task B survey (iter {iteration + 1}): "
                f"understanding={task_b_result['q1']}/5, effort={task_b_result['q2']}/5, "
                f"comparison={task_b_result['q4']}"
            )

        # ── Preference Coverage Evaluation (ground-truth recall) ────────
        if rubric_data and rubric_data.get("rubric"):
            log.info(f"[{persona.name}] Evaluating preference coverage (iteration {iteration + 1})...")
            coverage_result = evaluate_preference_coverage(
                persona, rubric_data["rubric"], model=config.model_primary,
                preference_items=atomic_preferences or None,
            )
            coverage_result["iteration"] = iteration + 1
            coverage_result["rubric_version"] = rubric_data.get("version", 0)
            coverage_result["timestamp"] = datetime.now().isoformat()
            coverage_trajectory.append(coverage_result)

            store.save_project_data(persona.name, "preference_coverage", coverage_result)

            score = coverage_result.get("coverage_score", 0)
            total = coverage_result.get("total_preferences", 0)
            covered = coverage_result.get("covered_count", 0)
            partial = coverage_result.get("partially_covered_count", 0)
            not_cov = coverage_result.get("not_covered_count", 0)
            log.info(
                f"[{persona.name}] Coverage (iter {iteration + 1}): "
                f"{score:.0%} ({covered}C + {partial}P + {not_cov}N = {total} preferences)"
            )

    # ── Save termination log ─────────────────────────────────────────────
    store.save_project_data(persona.name, "termination_log", {
        "satisfaction_threshold": config.satisfaction_threshold,
        "max_drafts": max_turns,
        "iterations": termination_log,
        "timestamp": datetime.now().isoformat(),
    })

    # ── Final summary ────────────────────────────────────────────────────
    summary = {
        "persona": persona.name,
        "project_id": project_id,
        "final_rubric_version": rubric_data.get("version") if rubric_data else 0,
        "num_rubric_versions": len(rubric_history),
        "num_messages": len(messages),
        "num_iterations": config.num_iterations,
        "termination_summary": [
            {
                "iteration": t["iteration"],
                "drafts_produced": t["drafts_produced"],
                "total_turns": t["total_turns"],
                "termination_reason": t["termination_reason"],
                "final_satisfaction": (
                    t["turn_metrics"][-1].get("satisfaction", {}).get("score")
                    if t["turn_metrics"] else None
                ),
            }
            for t in termination_log
        ],
        "survey_summary": {
            "task_a": {
                "understanding": survey_responses["task_a"]["q1"],
                "effort": survey_responses["task_a"]["q2"],
            } if survey_responses["task_a"] else None,
            "task_b_iterations": [
                {
                    "iteration": b.get("iteration"),
                    "understanding": b["q1"],
                    "effort": b["q2"],
                    "comparison": b["q4"],
                }
                for b in survey_responses["task_b"]
            ],
        },
        "coverage_trajectory": [
            {
                "iteration": c.get("iteration"),
                "coverage_score": c.get("coverage_score"),
                "total_preferences": c.get("total_preferences"),
                "covered": c.get("covered_count"),
                "partial": c.get("partially_covered_count"),
                "not_covered": c.get("not_covered_count"),
            }
            for c in coverage_trajectory
        ],
    }
    log.info(f"[{persona.name}] Complete: {json.dumps(summary, indent=2)}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RubricLLM Synthetic User Simulation")
    parser.add_argument("--num-users", type=int, default=None, help="Number of personas to simulate (default: all)")
    parser.add_argument("--iterations", type=int, default=3, help="Rubric refinement iterations per user")
    parser.add_argument("--max-chat-turns", type=int, default=10,
                        help="Max number of drafts the LLM can produce before terminating")
    parser.add_argument("--chat-turns", type=int, default=None,
                        help="DEPRECATED: use --max-chat-turns instead")
    parser.add_argument("--satisfaction-threshold", type=float, default=0.8,
                        help="Threshold (0.0-1.0) for both judges to agree draft is good enough")
    parser.add_argument("--user-provider", choices=["anthropic", "openai", "google"], default="anthropic",
                        help="LLM provider for synthetic user")
    parser.add_argument("--user-model", type=str, default=None,
                        help="Model ID for synthetic user (default: provider's default)")
    parser.add_argument("--system-primary", type=str, default="claude-opus-4-6",
                        help="Primary system model")
    parser.add_argument("--system-light", type=str, default="claude-sonnet-4-6",
                        help="Light system model")
    parser.add_argument("--personas", type=str, default="personas.json", help="Path to personas JSON")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: eval_output/sim_<timestamp>)")
    parser.add_argument("--no-log-changes", action="store_true",
                        help="Disable synthetic user rubric editing (Log Changes)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    # Handle deprecated --chat-turns flag
    max_chat_turns = args.max_chat_turns
    if args.chat_turns is not None:
        import warnings
        warnings.warn("--chat-turns is deprecated, use --max-chat-turns instead", DeprecationWarning)
        max_chat_turns = args.chat_turns

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default user models per provider
    from sim_config import DEFAULT_MODELS
    default_user_model = DEFAULT_MODELS.get(args.user_provider, {}).get("light", "claude-sonnet-4-6")

    config = SimConfig(
        num_iterations=args.iterations,
        max_chat_turns=max_chat_turns,
        satisfaction_threshold=args.satisfaction_threshold,
        model_primary=args.system_primary,
        model_light=args.system_light,
        user_provider=args.user_provider,
        user_model=args.user_model or default_user_model,
        persona_file=args.personas,
        enable_log_changes=not args.no_log_changes,
    )

    # Load personas
    personas = config.load_personas()
    if args.num_users is not None:
        personas = personas[:args.num_users]
    config.num_users = len(personas)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Config: {json.dumps(vars(config), indent=2, default=str)}")
        print(f"Personas ({len(personas)}):")
        for p in personas:
            print(f"  - {p.name} ({p.role}): {p.writing_type}")
        print(f"\nUser provider: {config.user_provider}")
        print(f"User model: {config.user_model}")
        print(f"System primary: {config.model_primary}")
        print(f"System light: {config.model_light}")
        print(f"Termination: goal-based (threshold={config.satisfaction_threshold}, max_drafts={config.max_chat_turns})")
        return

    # Output directory
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "eval_output", f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    store = LocalStore(output_dir)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Running {len(personas)} persona(s), {config.num_iterations} iterations each.")
    log.info(f"Goal-based termination: threshold={config.satisfaction_threshold}, max_drafts={config.max_chat_turns}")
    log.info(f"User model: {config.user_provider}/{config.user_model}")
    log.info(f"System: primary={config.model_primary}, light={config.model_light}")

    # Run simulation
    results = []
    for i, persona in enumerate(personas):
        log.info(f"\n{'#'*60}\n# Persona {i+1}/{len(personas)}: {persona.name}\n{'#'*60}")
        try:
            result = run_one_user(persona, config, store)
            results.append(result)
        except Exception as e:
            log.error(f"Failed for {persona.name}: {e}", exc_info=True)
            results.append({"persona": persona.name, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE — {len(results)} users")
    print(f"{'='*60}")
    for r in results:
        if "error" in r:
            print(f"  FAIL: {r['persona']} — {r['error']}")
        else:
            term_info = ""
            if r.get("termination_summary"):
                reasons = [t["termination_reason"] for t in r["termination_summary"]]
                drafts = [t["drafts_produced"] for t in r["termination_summary"]]
                term_info = f", termination: {reasons}, drafts: {drafts}"
            print(f"  OK: {r['persona']} — {r['num_rubric_versions']} rubric versions, {r['num_messages']} messages{term_info}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": vars(config),
            "results": results,
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nOutput saved to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
