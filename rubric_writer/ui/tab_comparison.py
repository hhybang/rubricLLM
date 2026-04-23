"""Evaluate: Comparison tab — blind THREE-way comparison across three arms:
  (N) no rubric          — proves having any rubric > baseline
  (I) first inferred     — proves inference alone is useful
  (C) current refined    — proves refinement adds value on top of inference

The three drafts are generated from the same task prompt and shown in a
randomized A/B/C order. The user picks a BEST and a WORST (endpoint ranking
is lower cognitive load than full ordering) or marks all the same.

Results go to the `rq2_threeway` data_type in Supabase via
`log_threeway_preference`."""
from rubric_writer.ui._deps import *
from rubric_writer.api_client import _api_call_with_retry
from rubric_writer.metrics import log_threeway_preference
from prompts import CHAT_build_system_prompt

import random
import re


_STATE_PREFIX = "cmp_tab_"
_ARMS = ("none", "early", "late")
_ARM_HUMAN = {
    "none": "no rubric (baseline)",
    "early": "first inferred rubric",
    "late": "current refined rubric",
}


def _state_key(name: str) -> str:
    return _STATE_PREFIX + name


def _reset_comparison() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(_STATE_PREFIX):
            del st.session_state[k]


def _strip_draft_tags(text: str) -> str:
    """Pull the <draft>...</draft> block out of a chat response. The chat
    system prompt instructs the model to wrap its draft in those tags; here
    we want just the draft text for side-by-side comparison. If no tag is
    found, fall back to the full response stripped of any probe signal."""
    m = re.search(r"<draft>([\s\S]*?)</draft>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Strip probe_signal blocks if the model emitted them without a draft
    text = re.sub(r"<probe_signal>[\s\S]*?</probe_signal>", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


# Forces the generator to produce a draft instead of asking clarifying
# questions. Previously users saw responses like "do you have a specific
# paper in mind?" in all three arms, which made the comparison useless --
# there's nothing to compare if no arm produced a draft. Wrapped around the
# user's task so it applies uniformly to all three arms (no_rubric / early /
# late), keeping the arms apples-to-apples.
_FORCE_DRAFT_SUFFIX = (
    "\n\nIMPORTANT: Produce a complete draft right now inside <draft>...</draft> "
    "tags. Do not ask clarifying questions and do not propose to write something "
    "different. If any details are missing, make reasonable assumptions and "
    "proceed -- the user wants to see a concrete attempt they can react to, not "
    "a clarification request."
)


def _generate_draft_from_rubric(task: str, rubric_dict: dict) -> str:
    """Generate a draft using the SAME system prompt the live chat uses.
    This keeps the Comparison arms apples-to-apples with how the rubric
    actually affects generation in real sessions -- CHAT_build_system_prompt
    includes the rubric-authority guidance, confidence-aware application,
    and <draft> tag formatting that a raw 'write based on rubric' prompt
    doesn't."""
    system_prompt = CHAT_build_system_prompt(rubric_dict)
    resp = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": task.strip() + _FORCE_DRAFT_SUFFIX}],
    )
    full = "".join(b.text for b in resp.content if b.type == "text")
    return _strip_draft_tags(full)


def _generate_draft_no_rubric(task: str) -> str:
    """Baseline: same CHAT system prompt infrastructure, but with NO rubric
    passed in -- CHAT_build_system_prompt has a no-rubric branch that still
    gives the model its <draft> tag format guidance without any criteria.
    This isolates the effect of HAVING a rubric vs. not, while keeping the
    surrounding prompt scaffolding identical."""
    # Passing an empty list triggers the no-rubric branch in the builder.
    system_prompt = CHAT_build_system_prompt([])
    resp = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": task.strip() + _FORCE_DRAFT_SUFFIX}],
    )
    full = "".join(b.text for b in resp.content if b.type == "text")
    return _strip_draft_tags(full)


def _randomize_label_to_arm() -> dict[str, str]:
    """Assign each of A/B/C to one of the three arms uniformly at random."""
    arms = list(_ARMS)
    random.shuffle(arms)
    return {"A": arms[0], "B": arms[1], "C": arms[2]}


def render_comparison_tab() -> None:
    st.header("⚖️ Evaluate: Comparison")
    st.markdown(
        "Blind three-way comparison. We generate three drafts of your task: "
        "one with **no rubric**, one with your **first inferred rubric**, one "
        "with your **current refined rubric**. The labels A/B/C are randomized "
        "so you don't know which is which. Pick the best and worst."
    )

    hist = load_rubric_history()
    if not hist:
        st.warning("No rubric history available yet. Create a rubric in Chat first.")
        return

    # Early rubric = first inferred = hist[1] if we have at least 2 versions
    # (version 1 is typically a placeholder); otherwise hist[0] as fallback.
    # Late rubric = current refined = hist[-1].
    early_rubric = hist[1] if len(hist) >= 2 else hist[0]
    late_rubric = hist[-1]
    early_version = early_rubric.get("version")
    late_version = late_rubric.get("version")

    # If early and late are the same version, we still do three-way because the
    # no-rubric arm gives us information. We just note it in the caption.
    if early_version == late_version:
        st.caption(
            f"Comparing **no rubric** vs **current rubric (v{late_version})**. "
            "The inferred and refined rubrics are currently the same version."
        )
    else:
        st.caption(
            f"Comparing **no rubric** vs **first inferred (v{early_version})** "
            f"vs **current refined (v{late_version})**. Order is randomized each time."
        )

    # --- State init ---
    if _state_key("label_to_arm") not in st.session_state:
        st.session_state[_state_key("label_to_arm")] = _randomize_label_to_arm()
    if _state_key("drafts") not in st.session_state:
        st.session_state[_state_key("drafts")] = None
    if _state_key("done") not in st.session_state:
        st.session_state[_state_key("done")] = False
    if _state_key("task") not in st.session_state:
        st.session_state[_state_key("task")] = ""
    if _state_key("best") not in st.session_state:
        st.session_state[_state_key("best")] = None
    if _state_key("worst") not in st.session_state:
        st.session_state[_state_key("worst")] = None

    drafts = st.session_state[_state_key("drafts")]

    # --- Task input (if no drafts yet) ---
    if drafts is None:
        task_input = st.text_area(
            "Writing task",
            value=st.session_state[_state_key("task")],
            placeholder="e.g. Write a project update email to my team about a delayed launch",
            key=_state_key("task_input"),
            height=120,
        )
        if st.button("Generate drafts", type="primary", key=_state_key("generate_btn")):
            task = (task_input or "").strip()
            if not task:
                st.warning("Please enter a writing task.")
                return
            st.session_state[_state_key("task")] = task
            label_to_arm = st.session_state[_state_key("label_to_arm")]
            try:
                gen_drafts: dict[str, str] = {}
                for label in ("A", "B", "C"):
                    arm = label_to_arm[label]
                    with st.spinner(f"Generating Draft {label}..."):
                        if arm == "none":
                            gen_drafts[label] = _generate_draft_no_rubric(task)
                        elif arm == "early":
                            gen_drafts[label] = _generate_draft_from_rubric(task, early_rubric)
                        else:  # "late"
                            gen_drafts[label] = _generate_draft_from_rubric(task, late_rubric)
            except Exception as e:
                st.error(f"Couldn't generate drafts: {e}")
                return
            st.session_state[_state_key("drafts")] = gen_drafts
            st.rerun()
        return

    # --- Show drafts side-by-side ---
    st.markdown(f"**Task:** _{st.session_state[_state_key('task')]}_")
    st.divider()

    col_a, col_b, col_c = st.columns(3)
    for col, label in zip((col_a, col_b, col_c), ("A", "B", "C")):
        with col:
            st.markdown(f"### Draft {label}")
            st.markdown(drafts[label])

    st.divider()

    # --- Done state: show reveal + reset ---
    if st.session_state[_state_key("done")]:
        label_to_arm = st.session_state[_state_key("label_to_arm")]
        best = st.session_state[_state_key("best")]
        worst = st.session_state[_state_key("worst")]
        if best is None and worst is None:
            st.success("Recorded: all three drafts were about the same.")
        else:
            parts = []
            if best:
                parts.append(f"best = **Draft {best}** ({_ARM_HUMAN[label_to_arm[best]]})")
            if worst:
                parts.append(f"worst = **Draft {worst}** ({_ARM_HUMAN[label_to_arm[worst]]})")
            st.success("Recorded: " + ", ".join(parts) + ".")
        with st.expander("Reveal the arm for each draft", expanded=False):
            for label in ("A", "B", "C"):
                st.markdown(f"- **Draft {label}**: {_ARM_HUMAN[label_to_arm[label]]}")
        if st.button("Try another task", key=_state_key("reset_after_done")):
            _reset_comparison()
            st.rerun()
        return

    # --- Choice UI ---
    st.markdown("**Pick a best and a worst.** You can also mark all the same if they feel equivalent.")

    current_best = st.session_state[_state_key("best")]
    current_worst = st.session_state[_state_key("worst")]

    col_best_label, col_best_a, col_best_b, col_best_c = st.columns([1.2, 1, 1, 1])
    with col_best_label:
        st.markdown("**Best draft:**")
    for col, label in zip((col_best_a, col_best_b, col_best_c), ("A", "B", "C")):
        with col:
            is_selected = current_best == label
            btn_label = f"✓ {label}" if is_selected else label
            # Disable a label as "best" if it's already selected as worst.
            disabled = (current_worst == label)
            if st.button(btn_label, key=_state_key(f"best_{label}"),
                         type=("primary" if is_selected else "secondary"),
                         disabled=disabled, use_container_width=True):
                st.session_state[_state_key("best")] = None if is_selected else label
                st.rerun()

    col_worst_label, col_worst_a, col_worst_b, col_worst_c = st.columns([1.2, 1, 1, 1])
    with col_worst_label:
        st.markdown("**Worst draft:**")
    for col, label in zip((col_worst_a, col_worst_b, col_worst_c), ("A", "B", "C")):
        with col:
            is_selected = current_worst == label
            btn_label = f"✓ {label}" if is_selected else label
            disabled = (current_best == label)
            if st.button(btn_label, key=_state_key(f"worst_{label}"),
                         type=("primary" if is_selected else "secondary"),
                         disabled=disabled, use_container_width=True):
                st.session_state[_state_key("worst")] = None if is_selected else label
                st.rerun()

    st.text_input(
        "Why? (optional)",
        key=_state_key("reason"),
        placeholder="What made the difference?",
    )

    def _submit(all_same: bool) -> None:
        label_to_arm = st.session_state[_state_key("label_to_arm")]
        best = None if all_same else st.session_state[_state_key("best")]
        worst = None if all_same else st.session_state[_state_key("worst")]
        best_arm = label_to_arm.get(best) if best else None
        worst_arm = label_to_arm.get(worst) if worst else None
        try:
            log_threeway_preference(
                task=st.session_state[_state_key("task")],
                early_rubric_version=early_version,
                late_rubric_version=late_version,
                label_to_arm=label_to_arm,
                best_label=best, worst_label=worst,
                best_arm=best_arm, worst_arm=worst_arm,
                all_same=all_same,
                user_reason=(st.session_state.get(_state_key("reason"), "") or ""),
            )
        except Exception as e:
            st.warning(f"Couldn't log preference: {e}")
        st.session_state[_state_key("done")] = True
        st.rerun()

    st.markdown("---")
    col_submit, col_same, col_reset = st.columns([1.5, 1.5, 1])
    with col_submit:
        can_submit = (current_best is not None) and (current_worst is not None)
        if st.button("Submit ranking",
                     type="primary",
                     disabled=not can_submit,
                     key=_state_key("submit_btn"),
                     use_container_width=True):
            _submit(all_same=False)
    with col_same:
        if st.button("≈ All three are about the same",
                     key=_state_key("tie_btn"),
                     use_container_width=True):
            _submit(all_same=True)
    with col_reset:
        if st.button("Cancel", key=_state_key("reset_btn"), use_container_width=True):
            _reset_comparison()
            st.rerun()
