"""Evaluate: Comparison tab — blind pairwise A/B between the first inferred
rubric and the current refined rubric. Lets the user enter any writing task,
generates two drafts (one from each rubric), randomizes A/B label assignment,
and records preferences for analysis."""
from rubric_writer.ui._deps import *
from rubric_writer.api_client import _api_call_with_retry
from rubric_writer.metrics import log_pairwise_preference

import random


_STATE_PREFIX = "cmp_tab_"


def _state_key(name: str) -> str:
    return _STATE_PREFIX + name


def _reset_comparison() -> None:
    """Clear the current comparison so the user can start a new one."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(_STATE_PREFIX):
            del st.session_state[k]


def _generate_draft_from_rubric(task: str, rubric_dict: dict) -> str:
    rubric_json = json.dumps(rubric_dict.get("rubric", []), ensure_ascii=False, indent=2)
    resp = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=4000,
        messages=[{"role": "user", "content": (
            "Write the following based on this rubric.\n\n"
            f"RUBRIC:\n{rubric_json}\n\n"
            f"TASK: {task.strip()}\n\n"
            "Write only the draft, nothing else."
        )}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def render_comparison_tab() -> None:
    st.header("⚖️ Evaluate: Comparison")
    st.markdown(
        "Blind A/B comparison between your **first inferred rubric** and your "
        "**current refined rubric**. Enter a writing task; we'll generate two "
        "drafts (one from each rubric, randomly assigned to A and B) and ask "
        "which you prefer. Used to measure whether refinement is making the "
        "rubric better."
    )

    hist = load_rubric_history()
    if not hist:
        st.warning("No rubric history available yet. Create a rubric in Chat first.")
        return
    if len(hist) < 2:
        st.info(
            "You need at least two rubric versions to compare. Refine your "
            "rubric in Chat (apply some edits) and come back."
        )
        return

    early_rubric = hist[1] if len(hist) >= 2 else hist[0]
    late_rubric = hist[-1]
    early_version = early_rubric.get("version", 1)
    late_version = late_rubric.get("version", len(hist))

    if early_version == late_version:
        st.info("First and current rubric are the same version — nothing to compare.")
        return

    st.caption(
        f"Comparing **first inferred rubric (v{early_version})** vs "
        f"**current rubric (v{late_version})**. "
        "Draft labels are blinded and the order is randomized each time."
    )

    # --- Initialize comparison state ---
    if _state_key("labels") not in st.session_state:
        # Randomize: A could be early or late.
        if random.random() < 0.5:
            st.session_state[_state_key("labels")] = {"A": "early", "B": "late"}
        else:
            st.session_state[_state_key("labels")] = {"A": "late", "B": "early"}
    if _state_key("drafts") not in st.session_state:
        st.session_state[_state_key("drafts")] = None
    if _state_key("done") not in st.session_state:
        st.session_state[_state_key("done")] = False
    if _state_key("task") not in st.session_state:
        st.session_state[_state_key("task")] = ""

    # --- Task input ---
    drafts = st.session_state[_state_key("drafts")]

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
            labels = st.session_state[_state_key("labels")]
            try:
                with st.spinner("Generating Draft A..."):
                    rubric_a = early_rubric if labels["A"] == "early" else late_rubric
                    draft_a = _generate_draft_from_rubric(task, rubric_a)
                with st.spinner("Generating Draft B..."):
                    rubric_b = late_rubric if labels["A"] == "early" else early_rubric
                    draft_b = _generate_draft_from_rubric(task, rubric_b)
            except Exception as e:
                st.error(f"Couldn't generate drafts: {e}")
                return
            st.session_state[_state_key("drafts")] = {"A": draft_a, "B": draft_b}
            st.rerun()
        return

    # --- Show drafts ---
    st.markdown(f"**Task:** _{st.session_state[_state_key('task')]}_")
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Draft A")
        st.markdown(drafts["A"])
    with col_b:
        st.markdown("### Draft B")
        st.markdown(drafts["B"])

    st.divider()

    if st.session_state[_state_key("done")]:
        labels = st.session_state[_state_key("labels")]
        chosen = st.session_state.get(_state_key("chosen"), "")
        if chosen == "tie":
            st.success("Recorded: drafts were about the same.")
        else:
            preferred_rubric = labels.get(chosen, "?")
            which = "first inferred" if preferred_rubric == "early" else "current refined"
            st.success(
                f"Recorded: you preferred **Draft {chosen}** (from the **{which}** rubric)."
            )
        if st.button("Try another task", key=_state_key("reset_after_done")):
            _reset_comparison()
            st.rerun()
        return

    st.markdown("**Which draft better matches what you wanted?**")
    reason = st.text_input(
        "Why? (optional)",
        key=_state_key("reason"),
        placeholder="What made the difference?",
    )

    col1, col2, col3 = st.columns(3)
    labels = st.session_state[_state_key("labels")]
    early_label = "A" if labels["A"] == "early" else "B"
    late_label = "B" if labels["A"] == "early" else "A"

    def _record(choice: str, preferred_rubric: str) -> None:
        try:
            log_pairwise_preference(
                early_rubric_version=early_version,
                late_rubric_version=late_version,
                early_rubric_label=early_label,
                late_rubric_label=late_label,
                user_preferred=choice,
                user_preferred_rubric=preferred_rubric,
                user_reason=(st.session_state.get(_state_key("reason"), "") or ""),
            )
        except Exception as e:
            st.warning(f"Couldn't log preference: {e}")
        st.session_state[_state_key("chosen")] = choice
        st.session_state[_state_key("done")] = True
        st.rerun()

    with col1:
        if st.button("⬅ Draft A", key=_state_key("pick_a"), use_container_width=True):
            _record("A", labels["A"])
    with col2:
        if st.button("Draft B ➡", key=_state_key("pick_b"), use_container_width=True):
            _record("B", labels["B"])
    with col3:
        if st.button("≈ About the same", key=_state_key("pick_tie"), use_container_width=True):
            _record("tie", "tie")

    st.markdown("---")
    if st.button("Cancel and start over", key=_state_key("reset_btn")):
        _reset_comparison()
        st.rerun()
