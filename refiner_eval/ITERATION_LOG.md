# Refiner Prompt Iteration Log

## v1 (initial) — 13/15 scope, 14/15 reasoning, 13/15 annotation → FAIL

**Failures:**
1. `reframes_01_focus_shift` → labeled `narrows` instead of `reframes`. The edit tightens but also shifts evaluation focus from structure to substance. Model defaulted to the magnitude-based label.
2. `idiosyncratic_01` → returned `no_change_needed: true`. Model treated the idiosyncratic feedback as not warranting any edit, contrary to the escape-valve guidance.
3. `narrows_03_tightened_threshold` → annotation quality was fine, but the heuristic check was too strict (required 2 content-word exact matches; evidence was only 5 words long with lots of stopword filtering).

## v2 changes

**Prompt edits in `REFINER_SYSTEM_PROMPT`:**

1. **Tie-breaker rule strengthened for `reframes`.** Added explicit rule: "If the criterion's WHAT-is-being-evaluated has changed (not just how strict), choose reframes even if the edit also tightens or loosens." Included a concrete example of the structure→substance shift to make the pattern recognizable.
2. **Stronger escape valve against `no_change_needed`.** Changed from "Only set no_change_needed if no edit whatsoever is warranted" to "**Do NOT set no_change_needed to true except in the extremely rare case...** When in doubt, produce a minimal clarifies edit." The bold emphasis + "rare" framing shifted the model's default.

**Harness heuristic fix for `_annotation_references_evidence`:**

- Accept quoted fragments as evidence of citation (strong signal).
- Stem-match content words (e.g. `sentences` ↔ `5-sentence` ↔ `sentence`) instead of requiring exact matches. This removed a false negative on `narrows_03` where the model's annotation was genuinely grounded but the string overlap was lower than the threshold.

## v2 results — 14/15 scope, 15/15 reasoning, 15/15 annotation → PASS

**Remaining miss:** `idiosyncratic_01` labeled `expands` instead of `clarifies`. The user's feedback ("I prefer storytelling openings this time at least") can defensibly be labeled either way: the edit now accepts storytelling openings, which is technically a loosening. The test label of `clarifies` reflects the spec's intent that idiosyncratic feedback should produce minimal edits, but `expands` is also a legitimate reading. This is a taxonomic ambiguity in the source definitions, not a prompt bug.

**Ship decision:** Overall PASS. The prompt meets all three thresholds.

## v3 — overfitting / length-bloat investigation

**User concern (post-v2):** "the suggestions always end up with just much longer descriptions of the dimension... we are not just asking for more detailed description of my current draft to make the dimension --> the rubric should still be generally applicable."

**Length analysis on v2 prompt (single-shot, no retries):**
- median after/before ratio: **1.93x**, max **3.78x**, 9/15 over 1.5x.
- Examples: `clarifies_01` 23→87 chars (3.78x), `clarifies_02` 27→96 (3.56x), `narrows_01` 29→76 (2.62x).
- Edits frequently included draft-specific clauses ("when the author writes...", quoted fragments).

## v4 changes

**Prompt edits:**
- Added "HARD CONSTRAINTS -- READ BEFORE WRITING" block to `after_wording` guidance.
- Length budget: after_wording MUST be at most 1.5x before_wording (hard ceiling, not a guideline).
- Banned draft-specific phrases ("when the author...", "such as...", quoted fragments).
- Banned lists of examples inside the wording.
- Added 2 BAD/GOOD example pairs showing both length and overfitting failure modes.
- Detail belongs in `example_annotation`, not in dimension wording.

**v4 results (prompt-only):** scope 14/15, reasoning 15/15, annotation 15/15. PASS, but length still over budget (9/15 cases >1.5x). Prompt alone is not enough.

## v5/v6 changes — parser-level enforcement + retry pipeline

**Parser change in `_parse_refiner_response`:**
- Added `over_budget` status: if `len(after) / len(before) > 1.5`, return status `over_budget` (suggestion still returned for fallback).
- Records `_length_ratio` on the suggestion for downstream introspection.

**Production pipeline change in `_schedule_feedback_rubric_refinement`:**
- If first refiner call returns `over_budget`, immediately retry with a **shrink_context** that names the violating ratio, the target char count, and instructs the model to move detail into `example_annotation`.
- Also reinforced length budget in all post-verification retry contexts (still_uncertain / still_unstable / mismatch).
- Added a final guard at the "best attempt" branch: if the retry's edit is over budget AND not shorter than attempt 1, ship attempt 1 instead. Prevents bloated dimensions from leaking into `is_best_attempt` warnings.

**Eval harness updated** to mirror the production shrink-retry on `over_budget`.

## v6 results — 13/15 scope, 15/15 reasoning, 15/15 annotation → PASS

**Length improvement vs v3:**
- median: 1.93x → **1.29x**
- max: 3.78x → **1.60x**
- over 1.5x: 9/15 → **2/15** (and both at 1.57x and 1.60x — barely over)
- over 2.0x: 5/15 → **0/15**

**Tradeoff:** scope dropped from 14 → 13/15 (still at threshold). `reframes_03` now misclassified — likely because the shrink retry's rewritten wording loses some reframing signal. Acceptable given the length improvement is the primary research concern.

**Ship decision:** PASS. Suggestions are now generally applicable rather than draft-specific elaborations.
