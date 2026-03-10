from textwrap import dedent
import json

CHAT_ASSESS_DRAFT_PROMPT = """
Please evaluate the most recent draft from our conversation against the rubric.

## Context & Purpose

You've been collaborating with the user on their writing. They've now asked for a formal assessment of how well the current draft aligns with their rubric. Review the conversation history to understand:
- The original writing goal and any constraints mentioned
- How the draft evolved through our discussion
- What the user explicitly valued or requested

Then assess the **final draft** (the most recent assistant message) against each criterion in the rubric.

## Dimension-Based Evaluation

Each criterion has **dimensions** — checkable items that can be marked as met (✓) or not met (✗).

Achievement levels are determined by how many dimensions are met:
- **⭐⭐⭐ Excellent**: 90%+ of dimensions met
- **⭐⭐ Good**: 75-89% of dimensions met
- **⭐ Fair**: 50-74% of dimensions met
- **◇ Needs Work**: 25-49% of dimensions met
- **☆ Weak**: Less than 25% of dimensions met

## Evaluation Process

For each criterion in the rubric (ordered by priority, highest first):

1. **Read the criterion carefully**: Understand what THIS specific user values (not generic writing quality)

2. **Check each dimension**: For each dimension listed under the criterion, determine if the draft meets it (yes/no)

3. **Examine the draft**: Look for evidence of whether each dimension is met. Consider:
   - Direct quotes that demonstrate the dimension is met or not met
   - Structural patterns (organization, flow, balance)
   - Tone, style, and rhetorical choices
   - What's present AND what's missing

4. **Calculate achievement level**: Based on the percentage of dimensions met

5. **Document your reasoning**: Provide specific evidence for each dimension check

## Required Output Format

Wrap your detailed evaluation in `<evaluation>` tags, following this structure for EACH criterion (ordered by priority):

### [Criterion Name from Rubric]
**Priority**: #[N]

**Dimension Checklist**:
- [✓/✗] [Dimension label]: [Quote or evidence supporting this check]
- [✓/✗] [Dimension label]: [Quote or evidence supporting this check]
- [etc. for all dimensions]

**Dimensions Met**: [X] of [Y] ([percentage]%)
**Achievement Level**: [Excellent/Good/Fair/Needs Work/Weak]

**Key Evidence**:
- [Specific quote from draft]
- [Another piece of concrete evidence]

**To improve**:
[What specific changes would check off the unchecked dimensions?]

---

[Repeat the above structure for EVERY criterion in the rubric]

## After All Criteria

After completing the evaluation for all criteria, provide a JSON summary:

```json
{
  "criteria_scores": [
    {
      "name": "<exact criterion name from rubric>",
      "priority": <unique integer 1..N, 1 = most important, no duplicates>,
      "dimensions_met": <count of dimensions checked>,
      "dimensions_total": <total dimensions>,
      "dimensions_detail": [
        {
          "id": "<dimension id>",
          "label": "<dimension label>",
          "met": true/false,
          "evidence": "<brief quote or reason>"
        }
      ],
      "achievement_level": "<Excellent|Good|Fair|Needs Work|Weak>",
      "evidence_summary": "<1-2 sentence summary of key evidence>",
      "improvement_explanation": "<What specific changes would check off the unchecked dimensions? Be concrete and actionable.>"
    }
    // Include ALL criteria, ordered by priority
  ],
  "level_counts": {
    "excellent": <count of criteria at excellent>,
    "good": <count>,
    "fair": <count>,
    "needs_work": <count>,
    "weak": <count>
  },
  "top_priorities_status": "<Summary of how the draft performs on the top 2-3 priority criteria - this is the most important indicator of draft quality>",
  "overall_assessment": "<2-3 sentence narrative: How well does this draft align with what matters most to the user? What's working well? What needs the most attention?>",
  "priority_improvements": [
    "<Most important dimension to check off, focusing on highest-priority criteria>",
    "<Second most important dimension to check off>",
    "<Third most important dimension to check off>"
  ],
  "evidence_highlights": [
    {
      "criterion": "<criterion name>",
      "quote": "<EXACT text from the draft - must match character-for-character>",
      "dimension_id": "<which dimension this evidence relates to>",
      "dimension_met": true/false,
      "relevance": "<brief explanation: if met, why this text demonstrates the dimension; if not met, why this text shows a violation or issue>"
    }
  ]
}
```

## Assessment Principles

- **Be binary on dimensions**: Each dimension is either met or not — no partial credit
- **Priority matters most**: A draft that checks off dimensions on priority #1-3 criteria is stronger
- **Be calibrated to THIS user's values**: The rubric reflects what THIS user cares about
- **Default to "not met" when uncertain**: If you can't find clear evidence, the dimension is not met
- **Quote specific evidence**: Cite actual text for each dimension check
- **Provide EXACT quotes for evidence_highlights**: The quote field must match text in the draft character-for-character so it can be highlighted. Include highlights for BOTH met dimensions (text that demonstrates success) AND unmet dimensions (text that shows violations or issues)
- **Frame improvements as dimensions to check off**: Focus on specific, actionable items

Provide your complete evaluation now, starting with the `<evaluation>` tags.
"""

RUBRIC_COMPARE_DRAFTS_PROMPT = """You are an editor comparing how two different rubrics influence writing on the same topic.

## PURPOSE
This tool helps users understand how different rubric versions affect the writing a model produces. You will:
1. Create a base draft for the given writing task
2. Revise that same draft twice - once following Rubric A, once following Rubric B
3. Highlight how each rubric leads to different writing choices

## IMPORTANT
- You have been provided with all necessary information below: a writing task and two rubrics
- Do NOT ask for clarification - proceed directly with generating the comparison
- The rubrics may be similar or very different - your job is to surface how even small differences affect the output

========================
WRITING TASK
========================
{task}

========================
RUBRIC A
========================
{rubric_a}

========================
RUBRIC B
========================
{rubric_b}

========================
RULES
========================
- Both revisions MUST start from the exact same BASE DRAFT (do not revise A from B or vice versa)
- Change ONLY what is required to satisfy each rubric. Keep content, argument order, and structure stable unless a rubric explicitly requires otherwise
- If a rubric requires additions or deletions, make them, but keep changes localized and intentional
- Mark additions with **bold** and removals with ~~strikethrough~~ relative to the base
- Keep differences attributable to rubric differences. Avoid unrelated rewrites

========================
OUTPUT FORMAT (STRICT)
========================
Return sections in this exact order with these exact headings:

### Key Rubric Differences
- List the main differences between the two rubrics that will affect the writing

### Stage 1 – Base Draft
Write a complete draft that fulfills the writing task naturally (without following any rubric).

### Stage 2 – Revisions
#### Rubric A Revision (from the base)
The full revised text. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.

#### Rubric B Revision (from the base)
The full revised text. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.

### Summary of Impact
- In 3-5 bullets, explain how Rubric A vs Rubric B affected tone, concision, evidence, structure, or polish
- Mention specific additions/deletions and why they were necessary for each rubric
"""

RUBRIC_INFER_SYSTEM_PROMPT = """
You are tasked with creating or updating a personalized writing rubric based on the conversation history between a user and an LLM collaboratively developing a piece of writing.

This rubric captures what THIS specific user values — not generic standards of "good writing."

**CRITICAL — GENERALIZABILITY**: The rubric you produce must represent the user's **transferable writing style, preferences, and values** — NOT a grading sheet for the specific piece of writing in the conversation. The user will reuse this rubric across many future writing tasks of the same type. Every criterion and dimension must make sense applied to a *different* piece of writing by the same user.

For example, if the user is writing a cold email to a specific person:
- ✅ **Transferable**: "Opening line references something specific to the recipient" (applies to any cold email)
- ❌ **Task-specific**: "Mentions the recipient's recent podcast episode" (only applies to this one email)

---

## PURPOSE

The rubric will be used to:
1. Align future writing assistance to this user's goals and preferences **across multiple writing tasks**
2. Allow the user to steer behavior by adjusting criteria priorities or dimensions
3. Support reliable LLM-based evaluation of drafts via dimension checklists

The rubric must be **concise, steerable, evaluable, and reusable across tasks**.

---

## OUTPUT REQUIREMENTS

### ✅ MUST INCLUDE
- `version`
- `writing_type`
- `user_goals_summary`
- **4–7 criteria**, each with:
  - `name`
  - `category` (from a shared set of 3–5 categories)
  - `description` (1–3 sentences max)
  - `dimensions` (3–5 per criterion) — these are **checkable items** that determine achievement level
  - `priority` (unique integer rank: 1 = most important, higher = less important)
- `coaching_notes`

### ❌ DO NOT INCLUDE
- Separate achievement level descriptions (excellent, good, fair, weak) — these are now derived from dimension counts
- Long rationales or justifications
- Detailed examples within the rubric
- Generic writing advice unsupported by conversation evidence

---

## ANALYSIS PROCESS

Before producing the rubric, write your reasoning inside `<analysis>` tags.

### Step 1: Determine Scenario
- **New rubric**: Create from scratch based on conversation evidence.
- **Update**: Preserve stable criteria; modify only where new evidence appears; adjust priority rankings only when priorities clearly shifted.

### Step 2: Extract Signals from Conversation

**Explicit signals:**
- Stated goals, preferences, constraints
- Direct feedback ("this doesn't work," "this is perfect")
- Requested revisions
- Questions revealing values (e.g., "Is this too formal?")

**Implicit signals:**
- Patterns in repeated edits
- What the user never comments on
- Direction of changes (more specific vs. abstract, shorter vs. longer)
- Trade-offs consistently accepted or rejected

**Non-negotiables:**
- Hard requirements (word limits, required structures, stylistic rules)
- Anything treated as a deal-breaker

### Step 3: Analyze Feedback on Previous Assessments (CRITICAL)

If the conversation includes user reactions to prior rubric evaluations,
treat this as the **highest-quality signal** for rubric updates. Assessment
disagreements reveal gaps between what the rubric says and what the user
actually means.

For each piece of assessment feedback:

1. **Identify the criterion** the user is responding to
2. **Classify the disagreement type** and apply the corresponding rubric fix:

   **Score disagreement** ("I'd rate this higher/lower"):
   → The dimensions are miscalibrated. Some dimensions may be too strict
     or too lenient for this user. Add, remove, or reword dimensions to
     match what the user considers meeting vs. not meeting the criterion.

   **Interpretation disagreement** ("That's not what I mean by casual"):
   → The description or dimensions use language the user interprets
     differently than the model. Replace vague terms with the user's
     own vocabulary. E.g., change "casual tone" to "conversational
     and warm but not slangy — like talking to a smart friend."

   **Priority disagreement** ("I don't care about that as much as X"):
   → Adjust priority rankings to reflect the user's actual hierarchy.

   **Definition disagreement** ("This criterion is missing the point"):
   → The criterion itself is wrong or incomplete. Rewrite the
     description and dimensions to capture what the user actually
     values, using their words from the feedback.

   **Missing dimension** ("You didn't even check whether..."):
   → The user expected a dimension that doesn't exist. Add it.

   **Irrelevant dimension** ("Why does this matter?"):
   → A dimension checks for something the user doesn't care about.
     Remove it.

3. **Preserve the user's exact language** whenever they articulate what
   they value. Their phrasing is more precise than any paraphrase.
   
---

## SEPARATING STYLE FROM TASK CONTENT

This is the most important distinction in rubric inference. The conversation is about ONE specific writing task, but the rubric must capture the user's **style and preferences** that transfer across tasks.

**Task-specific content** (DO NOT put in rubric):
- Names, companies, dates, specific topics discussed in the draft
- One-time constraints ("keep it under 300 words for this email")
- Content choices specific to this piece ("mention the Q3 results")

**Transferable preferences** (DO put in rubric):
- Tone and voice preferences (formal vs. conversational, warm vs. direct)
- Structural patterns (how they like to open/close, paragraph length preferences)
- Stylistic choices (active vs. passive voice, use of metaphors, sentence rhythm)
- Recurring values (conciseness, specificity, audience awareness, storytelling)
- How they handle evidence, examples, and claims

When the user says "make this warmer" — the rubric should capture that they prefer warm tone, not that this particular paragraph needed warming up.

When the user says "add a specific example about our product launch" — the rubric should capture that they value concrete examples, not that they want product launch references.

---

## DEFINING CRITERIA

Select **4–7 criteria** with clear conversation evidence.

For each candidate, verify:
- Did the user demonstrably care about this?
- Can you point to specific moments?
- Is it distinct from other criteria?
- **Would this criterion make sense for a DIFFERENT writing task by the same user?**

**Do not include**:
- Generic principles without user evidence
- Criteria that only apply to the specific content of this one piece

---

## CATEGORIES

Use **3–5 shared categories** across all criteria:
- Style
- Structure
- Content
- Mechanics
- Audience

Multiple criteria may share the same category.

---

## DESCRIPTIONS (KEEP SHORT)

Write **1–3 sentences** per criterion that:
- Use vocabulary from the conversation
- State what "good" means for THIS user
- Avoid generic phrasing
- **Describe the preference in terms that apply beyond this one task**

✅ Good: "Paragraphs open with mechanism-level language that names the underlying phenomenon before explaining it."
✅ Good: "Opening lines are personalized to the recipient with a specific, relevant reference — not a generic greeting."

❌ Avoid: "Writing should be clear and effective." (too generic)
❌ Avoid: "The email should mention John's podcast about AI trends." (too task-specific)

---

## DIMENSIONS AS CHECKABLE ITEMS

**CRITICAL**: Dimensions are now **checkable items** (yes/no checkpoints) that determine the achievement level.

Each criterion has **3–5 dimensions**. Each dimension is a specific, observable feature that can be checked as met or not met.

Each dimension includes:
- `id`: short, machine-friendly identifier (e.g., `mechanism_first_openings`)
- `label`: short human-readable phrase describing what to check for

**Dimension Design Principles:**
- Each dimension should be a **binary check**: either the draft meets it or it doesn't
- Dimensions should be **observable**: an evaluator can look at the text and determine yes/no
- Dimensions should be **distinct**: each captures a different aspect of the criterion
- Dimensions should be **comprehensive**: together, they cover what "meeting this criterion" means
- **Each dimension must be reusable**: it should apply to any piece of writing of this type, not just the current one

**Examples of Good Checkable Dimensions:**

For a "Conciseness" criterion:
- `no_redundant_phrases`: "No redundant phrases or repeated ideas"
- `active_voice`: "Uses active voice throughout"
- `no_filler_words`: "Avoids filler words (very, really, just, quite)"
- `direct_sentences`: "Sentences lead with main point, not qualifiers"

For an "Evidence Quality" criterion:
- `claims_supported`: "Every claim is supported by specific evidence"
- `sources_cited`: "Sources are cited for external data"
- `examples_concrete`: "Examples are concrete, not hypothetical"
- `data_current`: "Data and statistics are current (within 5 years)"

❌ Bad dimension: `mentions_q3_results`: "References Q3 financial results" (task-specific content)
✅ Good dimension: `concrete_data_points`: "Includes at least one concrete data point to support the main claim"

---

## ACHIEVEMENT LEVELS (DERIVED FROM DIMENSIONS)

Achievement levels are **automatically determined** by how many dimensions are checked:

- **⭐⭐⭐ Excellent**: 100% of dimensions met (all checked)
- **⭐⭐ Good**: 75%+ of dimensions met (most checked)
- **⭐ Fair**: 50-74% of dimensions met (some checked)
- **☆ Weak**: Less than 50% of dimensions met (few/none checked)

You do NOT need to write separate achievement level descriptions. The dimensions themselves define what each level looks like.

---

## PRIORITY RANKINGS

Assign a **priority rank** (integer) to each criterion, where:
- **1 = most important** (highest priority)
- Higher numbers = lower priority
- Each criterion gets a unique rank from 1 to N (where N = number of criteria)

Base rankings on:
- Frequency and intensity of user feedback
- Non-negotiables vs. preferences (non-negotiables rank higher)
- Revision patterns (frequently revised aspects rank higher)

Example: If you have 5 criteria, assign ranks 1, 2, 3, 4, 5 - no ties allowed.

---

## COACHING NOTES

Provide **2–3 concise insights** about this user's writing mindset:
- Non-negotiables
- Preferred trade-offs
- How they respond to feedback

Keep actionable, not explanatory.

---

## OUTPUT FORMAT

After your `<analysis>` block, output **only** this JSON:
```json
{
  "version": <number>,
  "writing_type": "<general type of writing, e.g. 'cold outreach emails', 'academic research papers', 'technical blog posts'>",
  "user_goals_summary": "<2–3 sentence summary of the user's transferable writing goals and values>",
  "rubric": [
    {
      "name": "<criterion name>",
      "category": "<shared category>",
      "description": "<1–3 sentence user-specific description — must apply beyond this one task>",
      "dimensions": [
        {
          "id": "<machine-friendly id>",
          "label": "<checkable item: what to verify as yes/no — must be reusable across tasks>"
        }
      ],
      "priority": <unique integer 1..N, 1 = most important, no duplicates>
    }
  ],
  "coaching_notes": "<2–3 concise insights>",
  "inference_decision_points": {
    "parsed_data": {
      "decision_points": [
        {
          "id": <integer, 1-based>,
          "title": "<brief descriptive title>",
          "dimension": "<tone/structure/detail/clarity/voice/etc.>",
          "assistant_message_num": <message number where AI suggestion appeared>,
          "user_message_num": <message number where user's change appeared>,
          "before_quote": "<short quote from assistant's suggestion, ~30 words max>",
          "after_quote": "<short quote showing user's change, ~30 words max>",
          "summary": "<one sentence: what choice the user made and why it matters>",
          "related_rubric_criterion": "<name of the rubric criterion this moment supports or refines>"
        }
      ],
      "overall_patterns": "<2–3 sentences describing patterns across decision points>",
      "rubric_insights": "<2–3 sentences on how these moments relate to the rubric>"
    }
  }
}
```

**INFERENCE DECISION POINTS (required):**
- Extract the **most important** decision points only — moments where the user's writing preferences are **strongly and clearly** demonstrated. Maximum **10 decision points**.
- **Only extract from conversational feedback**: Decision points come from user messages where the user gives direct feedback on a draft (e.g., "make it warmer", "too many bullet points", "get to the point faster"). Ignore system-generated rubric edit annotations (messages containing "Edits by rubric change:" or "User feedback:" labels) — these are not decision points.
- Focus on: explicit rejections, significant rewrites, strong feedback ("I don't like this", "make it more X"), and choices that reveal core preferences. Skip minor wording tweaks, trivial edits, or factual corrections.
- Use the **exact message numbers** from the conversation (e.g. the [Message #N] labels). Each decision point must have `assistant_message_num` and `user_message_num` pointing to real messages.
- Each decision point should map to a `related_rubric_criterion` (one of the criterion names in your rubric). This links the evidence to what you inferred.
- Quality over quantity: 5–10 strong decision points are better than 15+ weak ones.

**NOTE**: Do NOT include `excellent`, `good`, `fair`, or `weak` fields. Achievement levels are derived from dimension counts.
"""

def RUBRIC_infer_user_prompt(conversation_text, previous_rubric_json=""):
    return f"""Here is the conversation you need to analyze. Messages are numbered [Message #N] so you can reference them in decision points.

<conversation>
{conversation_text}
</conversation>

Here is the previous rubric (this may be empty if you're creating a new rubric from scratch):

<previous_rubric>
{previous_rubric_json}
</previous_rubric>

Analyze the conversation, infer a rubric, AND extract decision points. Return ONLY valid JSON matching the output format in your system instructions."""


# ============================================================================
# RUBRIC-ONLY INFERENCE (no DP extraction) — Step 1 of the 5-step flow
# ============================================================================

RUBRIC_INFER_ONLY_SYSTEM_PROMPT = """
You are tasked with creating or updating a personalized writing rubric based on the conversation history between a user and an LLM collaboratively developing a piece of writing.

This rubric captures what THIS specific user values — not generic standards of "good writing."

**CRITICAL — GENERALIZABILITY**: The rubric you produce must represent the user's **transferable writing style, preferences, and values** — NOT a grading sheet for the specific piece of writing in the conversation. The user will reuse this rubric across many future writing tasks of the same type. Every criterion and dimension must make sense applied to a *different* piece of writing by the same user.

For example, if the user is writing a cold email to a specific person:
- ✅ **Transferable**: "Opening line references something specific to the recipient" (applies to any cold email)
- ❌ **Task-specific**: "Mentions the recipient's recent podcast episode" (only applies to this one email)

---

## PURPOSE

The rubric will be used to:
1. Align future writing assistance to this user's goals and preferences **across multiple writing tasks**
2. Allow the user to steer behavior by adjusting criteria priorities or dimensions
3. Support reliable LLM-based evaluation of drafts via dimension checklists

The rubric must be **concise, steerable, evaluable, and reusable across tasks**.

---

## OUTPUT REQUIREMENTS

### ✅ MUST INCLUDE
- `version`
- `writing_type`
- `user_goals_summary`
- **4–7 criteria**, each with:
  - `name`
  - `category` (from a shared set of 3–5 categories)
  - `description` (1–3 sentences max)
  - `dimensions` (3–5 per criterion) — these are **checkable items** that determine achievement level
  - `priority` (unique integer rank: 1 = most important, higher = less important)
- `coaching_notes`

### ❌ DO NOT INCLUDE
- Separate achievement level descriptions (excellent, good, fair, weak) — these are now derived from dimension counts
- Long rationales or justifications
- Detailed examples within the rubric
- Generic writing advice unsupported by conversation evidence
- Decision points — these will be extracted in a separate step

---

## ANALYSIS PROCESS

Before producing the rubric, write your reasoning inside `<analysis>` tags.

### Step 1: Determine Scenario
- **New rubric**: Create from scratch based on conversation evidence.
- **Update**: Preserve stable criteria; modify only where new evidence appears; adjust priority rankings only when priorities clearly shifted.

### Step 2: Extract Signals from Conversation

**Explicit signals:**
- Stated goals, preferences, constraints
- Direct feedback ("this doesn't work," "this is perfect")
- Requested revisions
- Questions revealing values (e.g., "Is this too formal?")

**Implicit signals:**
- Patterns in repeated edits
- What the user never comments on
- Direction of changes (more specific vs. abstract, shorter vs. longer)
- Trade-offs consistently accepted or rejected

**Non-negotiables:**
- Hard requirements (word limits, required structures, stylistic rules)
- Anything treated as a deal-breaker

### Step 3: Analyze Feedback on Previous Assessments (CRITICAL)

If the conversation includes user reactions to prior rubric evaluations,
treat this as the **highest-quality signal** for rubric updates. Assessment
disagreements reveal gaps between what the rubric says and what the user
actually means.

For each piece of assessment feedback:

1. **Identify the criterion** the user is responding to
2. **Classify the disagreement type** and apply the corresponding rubric fix:

   **Score disagreement** ("I'd rate this higher/lower"):
   → The dimensions are miscalibrated. Some dimensions may be too strict
     or too lenient for this user. Add, remove, or reword dimensions to
     match what the user considers meeting vs. not meeting the criterion.

   **Interpretation disagreement** ("That's not what I mean by casual"):
   → The description or dimensions use language the user interprets
     differently than the model. Replace vague terms with the user's
     own vocabulary.

   **Priority disagreement** ("I don't care about that as much as X"):
   → Adjust priority rankings to reflect the user's actual hierarchy.

   **Definition disagreement** ("This criterion is missing the point"):
   → The criterion itself is wrong or incomplete. Rewrite the
     description and dimensions to capture what the user actually
     values, using their words from the feedback.

   **Missing dimension** ("You didn't even check whether..."):
   → The user expected a dimension that doesn't exist. Add it.

   **Irrelevant dimension** ("Why does this matter?"):
   → A dimension checks for something the user doesn't care about.
     Remove it.

3. **Preserve the user's exact language** whenever they articulate what
   they value. Their phrasing is more precise than any paraphrase.

---

## SEPARATING STYLE FROM TASK CONTENT

This is the most important distinction in rubric inference. The conversation is about ONE specific writing task, but the rubric must capture the user's **style and preferences** that transfer across tasks.

**Task-specific content** (DO NOT put in rubric):
- Names, companies, dates, specific topics discussed in the draft
- One-time constraints ("keep it under 300 words for this email")
- Content choices specific to this piece ("mention the Q3 results")

**Transferable preferences** (DO put in rubric):
- Tone and voice preferences (formal vs. conversational, warm vs. direct)
- Structural patterns (how they like to open/close, paragraph length preferences)
- Stylistic choices (active vs. passive voice, use of metaphors, sentence rhythm)
- Recurring values (conciseness, specificity, audience awareness, storytelling)
- How they handle evidence, examples, and claims

When the user says "make this warmer" — the rubric should capture that they prefer warm tone, not that this particular paragraph needed warming up.

When the user says "add a specific example about our product launch" — the rubric should capture that they value concrete examples, not that they want product launch references.

---

## DEFINING CRITERIA

Select **4–7 criteria** with clear conversation evidence.

For each candidate, verify:
- Did the user demonstrably care about this?
- Can you point to specific moments?
- Is it distinct from other criteria?
- **Would this criterion make sense for a DIFFERENT writing task by the same user?**

**Do not include**:
- Generic principles without user evidence
- Criteria that only apply to the specific content of this one piece

---

## CATEGORIES

Use **3–5 shared categories** across all criteria:
- Style
- Structure
- Content
- Mechanics
- Audience

Multiple criteria may share the same category.

---

## DESCRIPTIONS (KEEP SHORT)

Write **1–3 sentences** per criterion that:
- Use vocabulary from the conversation
- State what "good" means for THIS user
- Avoid generic phrasing
- **Describe the preference in terms that apply beyond this one task**

✅ Good: "Paragraphs open with mechanism-level language that names the underlying phenomenon before explaining it."
✅ Good: "Opening lines are personalized to the recipient with a specific, relevant reference — not a generic greeting."

❌ Avoid: "Writing should be clear and effective." (too generic)
❌ Avoid: "The email should mention John's podcast about AI trends." (too task-specific)

---

## DIMENSIONS AS CHECKABLE ITEMS

**CRITICAL**: Dimensions are now **checkable items** (yes/no checkpoints) that determine the achievement level.

Each criterion has **3–5 dimensions**. Each dimension is a specific, observable feature that can be checked as met or not met.

Each dimension includes:
- `id`: short, machine-friendly identifier (e.g., `mechanism_first_openings`)
- `label`: short human-readable phrase describing what to check for

**Dimension Design Principles:**
- Each dimension should be a **binary check**: either the draft meets it or it doesn't
- Dimensions should be **observable**: an evaluator can look at the text and determine yes/no
- Dimensions should be **distinct**: each captures a different aspect of the criterion
- Dimensions should be **comprehensive**: together, they cover what "meeting this criterion" means
- **Each dimension must be reusable**: it should apply to any piece of writing of this type, not just the current one

❌ Bad dimension: `mentions_q3_results`: "References Q3 financial results" (task-specific content)
✅ Good dimension: `concrete_data_points`: "Includes at least one concrete data point to support the main claim"

---

## ACHIEVEMENT LEVELS (DERIVED FROM DIMENSIONS)

Achievement levels are **automatically determined** by how many dimensions are checked:

- **⭐⭐⭐ Excellent**: 100% of dimensions met (all checked)
- **⭐⭐ Good**: 75%+ of dimensions met (most checked)
- **⭐ Fair**: 50-74% of dimensions met (some checked)
- **☆ Weak**: Less than 50% of dimensions met (few/none checked)

You do NOT need to write separate achievement level descriptions.

---

## PRIORITY RANKINGS

Assign a **priority rank** (integer) to each criterion, where:
- **1 = most important** (highest priority)
- Higher numbers = lower priority
- Each criterion gets a unique rank from 1 to N (where N = number of criteria)

Base rankings on:
- Frequency and intensity of user feedback
- Non-negotiables vs. preferences (non-negotiables rank higher)
- Revision patterns (frequently revised aspects rank higher)

---

## COACHING NOTES

Provide **2–3 concise insights** about this user's writing mindset:
- Non-negotiables
- Preferred trade-offs
- How they respond to feedback

Keep actionable, not explanatory.

---

## OUTPUT FORMAT

After your `<analysis>` block, output **only** this JSON:
```json
{
  "version": <number>,
  "writing_type": "<general type of writing, e.g. 'cold outreach emails', 'academic research papers', 'technical blog posts'>",
  "user_goals_summary": "<2–3 sentence summary of the user's transferable writing goals and values>",
  "rubric": [
    {
      "name": "<criterion name>",
      "category": "<shared category>",
      "description": "<1–3 sentence user-specific description — must apply beyond this one task>",
      "dimensions": [
        {
          "id": "<machine-friendly id>",
          "label": "<checkable item: what to verify as yes/no — must be reusable across tasks>"
        }
      ],
      "priority": <unique integer 1..N where N = number of criteria, 1 = most important, no duplicates>
    }
  ],
  "coaching_notes": "<2–3 concise insights>"
}
```

**NOTE**: Do NOT include `excellent`, `good`, `fair`, or `weak` fields. Achievement levels are derived from dimension counts.
Do NOT include decision points — those will be extracted separately.
"""

def RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json=""):
    return f"""Here is the conversation you need to analyze. Messages are numbered [Message #N].

<conversation>
{conversation_text}
</conversation>

Here is the previous rubric (this may be empty if you're creating a new rubric from scratch):

<previous_rubric>
{previous_rubric_json}
</previous_rubric>

Analyze the conversation and infer a rubric. Do NOT extract decision points — that will happen in a separate step. Return ONLY valid JSON matching the output format in your system instructions."""


# ============================================================================
# DECISION POINT EXTRACTION (with classification context) — Step 3 of the 5-step flow
# ============================================================================

RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT = """
You are extracting decision points from a writing collaboration conversation.

A "decision point" is a moment where the user's writing preferences are strongly and clearly demonstrated — typically when the user gives direct feedback on a draft, rejects a suggestion, or makes a significant change.

## CONTEXT YOU HAVE BEEN GIVEN

1. **Conversation history**: The full editing session between user and assistant, with messages numbered [Message #N]
2. **Rubric**: A personalized rubric inferred from this conversation
3. **Classification feedback**: The user's judgment on each rubric criterion:
   - **Stated**: The user explicitly mentioned this preference upfront
   - **Real**: The user didn't mention it, but confirmed they do care about it
   - **Hallucinated**: The model invented this criterion — the user does NOT care about it

## EXTRACTION RULES

- Extract the **most important** decision points only. Maximum **10 decision points**.
- **Only extract from conversational feedback**: Decision points come from user messages where the user gives direct feedback on a draft (e.g., "make it warmer", "too many bullet points", "get to the point faster"). Ignore system-generated rubric edit annotations (messages containing "Edits by rubric change:" or "User feedback:" labels).
- Focus on: explicit rejections, significant rewrites, strong feedback ("I don't like this", "make it more X"), and choices that reveal core preferences. Skip minor wording tweaks, trivial edits, or factual corrections.
- Use the **exact message numbers** from the conversation (the [Message #N] labels). Each decision point must have `assistant_message_num` and `user_message_num` pointing to real messages.
- Each decision point should map to a `related_rubric_criterion` (one of the criterion names in the rubric).
- Quality over quantity: 5–10 strong decision points are better than 15+ weak ones.

## CLASSIFICATION-AWARE EXTRACTION

- **Pay special attention to "real" criteria**: These are preferences the user didn't articulate upfront but validated as genuine. Decision points supporting "real" criteria are especially valuable evidence.
- **Do NOT extract decision points for "hallucinated" criteria**: These criteria will be removed from the rubric. Any decision points that only support a hallucinated criterion should be skipped.
- **"Stated" criteria** are well-established — DPs supporting them provide useful confirmation but are lower priority than "real" DPs.

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown code blocks, no preamble):
{
  "decision_points": [
    {
      "id": <integer, 1-based>,
      "title": "<brief descriptive title>",
      "dimension": "<tone/structure/detail/clarity/voice/etc.>",
      "assistant_message_num": <message number where AI suggestion appeared>,
      "user_message_num": <message number where user's change appeared>,
      "before_quote": "<short quote from assistant's suggestion, ~30 words max>",
      "after_quote": "<short quote showing user's change, ~30 words max>",
      "summary": "<one sentence: what choice the user made and why it matters>",
      "related_rubric_criterion": "<name of the rubric criterion this moment supports or refines>"
    }
  ],
  "overall_patterns": "<2–3 sentences describing patterns across decision points>",
  "rubric_insights": "<2–3 sentences on how these moments relate to the rubric>"
}
"""

def RUBRIC_extract_dps_user_prompt(conversation_text, rubric_json, classification_feedback_json):
    return f"""Here is the conversation. Messages are numbered [Message #N].

<conversation>
{conversation_text}
</conversation>

Here is the rubric inferred from this conversation:

<rubric>
{rubric_json}
</rubric>

Here are the user's classifications of each criterion (stated = user mentioned it upfront, real = user cares but didn't mention, hallucinated = model invented it):

<classification_feedback>
{classification_feedback_json}
</classification_feedback>

Extract decision points from the conversation. Focus especially on evidence for "real" criteria. Do NOT extract decision points for "hallucinated" criteria. Return ONLY valid JSON matching the output format in your system instructions."""


# ============================================================================
# FINAL RUBRIC INFERENCE (with all context) — Step 5 of the 5-step flow
# ============================================================================

RUBRIC_FINAL_INFER_SYSTEM_PROMPT = """
You are producing the FINAL version of a personalized writing rubric.

**CRITICAL — GENERALIZABILITY**: This rubric represents the user's **transferable writing style, preferences, and values** — NOT a grading sheet for the specific piece of writing in the conversation. Every criterion and dimension must make sense applied to a *different* piece of writing by the same user.

## BACKGROUND

You previously inferred an initial rubric from a writing conversation. Since then:
1. The user reviewed each criterion and classified it as "stated" (they mentioned it), "real" (they care but didn't mention), or "hallucinated" (model invented it)
2. Decision points were extracted from the conversation and the user reviewed them, potentially correcting which criterion each DP maps to or identifying preferences not captured by the rubric

Your task is to produce the definitive rubric that incorporates ALL of this feedback.

## HOW TO INCORPORATE FEEDBACK

### Classification Feedback
- **Hallucinated criteria** → REMOVE entirely. The user does not care about these. If a hallucination_reason is provided, use it to understand what the user DOES NOT value.
- **Real criteria** → KEEP and potentially strengthen. These are genuine preferences the user validated. Consider refining their descriptions using evidence from the conversation.
- **Stated criteria** → KEEP as-is. These are well-established.

### ANTI-REGRESSION RULE
When merging, consolidating, or renaming criteria, ensure that ALL dimensions from the original criteria are preserved in the result. Do NOT drop a criterion classified as "stated" or "real" unless you are certain its dimensions are fully captured by another criterion. If you merge two criteria, the merged result must cover everything both originals covered — check dimension by dimension.

### Importance Ranking
The classification feedback includes an `importance_ranking` — an ordered list of criteria from most to least important (as ranked by the user). Use this to set the `priority` field on each criterion. The user's ranking should be respected: criterion ranked #1 gets priority 1, etc. If new criteria are added (e.g. from "not_in_rubric" DPs), slot them in at a reasonable priority relative to the user's ranking.

### Decision Point Corrections
- **"correct"**: The DP's criterion mapping was right. No change needed.
- **"incorrect"**: The user says this DP maps to a DIFFERENT criterion. Update the target criterion's description/dimensions to better capture what this DP reveals.
- **"not_in_rubric"**: The user says this preference is NOT captured by any existing criterion. You MUST either:
  - Add a new criterion to capture it, OR
  - Add new dimensions to an existing criterion if it fits naturally

## SEPARATING STYLE FROM TASK CONTENT

The conversation is about ONE specific writing task, but the rubric must capture the user's **style and preferences** that transfer across tasks.

**Task-specific content** (DO NOT put in rubric):
- Names, companies, dates, specific topics discussed in the draft
- One-time constraints ("keep it under 300 words for this email")
- Content choices specific to this piece ("mention the Q3 results")

**Transferable preferences** (DO put in rubric):
- Tone and voice preferences (formal vs. conversational, warm vs. direct)
- Structural patterns (how they like to open/close, paragraph length preferences)
- Stylistic choices (active vs. passive voice, use of metaphors, sentence rhythm)
- Recurring values (conciseness, specificity, audience awareness, storytelling)

When refining, check that existing criteria and any new criteria you add are **generalizable** — not anchored to the specific content of this one piece.

## RUBRIC REQUIREMENTS

- **4–7 criteria** with clear conversation evidence
- Each criterion: `name`, `category`, `description` (1-3 sentences), `dimensions` (3-5 checkable items), `priority` (unique rank)
- Use the user's vocabulary from the conversation
- Unique priority rankings (1 = most important)
- Categories from: Style, Structure, Content, Mechanics, Audience
- **Every criterion and dimension must be reusable across tasks** — no task-specific content (names, topics, one-time constraints)

## OUTPUT FORMAT

Return ONLY valid JSON (no markdown code blocks, no preamble):
{
  "version": <number>,
  "writing_type": "<general type of writing, e.g. 'cold outreach emails', 'academic research papers'>",
  "user_goals_summary": "<2–3 sentence summary of the user's transferable writing goals and values>",
  "rubric": [
    {
      "name": "<criterion name>",
      "category": "<category>",
      "description": "<1–3 sentences>",
      "dimensions": [
        {"id": "<id>", "label": "<checkable item>"}
      ],
      "priority": <unique integer 1..N, no duplicates>
    }
  ],
  "coaching_notes": "<2–3 concise insights>",
  "refinement_summary": "<1-2 sentences: what changed and why>",
  "change_explanation": "<A user-facing explanation in markdown. For EACH change, explain: (1) what triggered it (classification or DP correction), (2) what you changed, and (3) why. Use bullet points. Be specific — reference criterion names and DP numbers.>"
}

**NOTE**: Do NOT include `excellent`, `good`, `fair`, or `weak` fields. Achievement levels are derived from dimension counts.
"""

def RUBRIC_final_infer_user_prompt(conversation_text, previous_rubric_json, classification_feedback_json, corrected_dps_json, coldstart_text=""):  # noqa: ARG001 — coldstart_text kept for backward compat but intentionally excluded from prompt
    return f"""Here is the conversation. Messages are numbered [Message #N].

<conversation>
{conversation_text}
</conversation>

Here is the current rubric (the one being refined):

<current_rubric>
{previous_rubric_json}
</current_rubric>

Here are the user's classification judgments for each criterion:

<classification_feedback>
{classification_feedback_json}
</classification_feedback>

Here are the user's decision point corrections:

<dp_corrections>
{corrected_dps_json}
</dp_corrections>

Produce the FINAL rubric incorporating all feedback. Return ONLY valid JSON matching the output format in your system instructions."""


def CHAT_build_system_prompt(rubric_dict_or_list):
    """Build system instruction with rubric (without assessment requirements).

    Args:
        rubric_dict_or_list: Either a full rubric dict (with source, template_name, rubric fields)
                            or just the rubric criteria list for backwards compatibility.
    """
    # Handle both dict and list inputs for backwards compatibility
    if isinstance(rubric_dict_or_list, dict):
        rubric = rubric_dict_or_list.get("rubric", [])
        source = rubric_dict_or_list.get("source", "inferred")
        template_name = rubric_dict_or_list.get("template_name", "")
    else:
        # Backwards compatibility: if a list is passed, treat as criteria list
        rubric = rubric_dict_or_list if rubric_dict_or_list else []
        source = "inferred"
        template_name = ""

    is_template = source == "template"

    rubric_block = ""
    if rubric:
        # Number the criteria explicitly for clarity, stripping _diff (contains non-serializable sets)
        numbered_rubric = []
        for idx, criterion in enumerate(rubric, start=1):
            numbered_crit = {k: v for k, v in criterion.items() if k != '_diff'}
            numbered_crit['index'] = idx
            numbered_rubric.append(numbered_crit)

        rubric_block = "\nRUBRIC (Always follow these criteria while co-writing):\n" + json.dumps(numbered_rubric, ensure_ascii=False, indent=2)

    # Add template-specific guidance if using a template rubric
    template_guidance = ""
    if is_template:
        template_guidance = f"""
    **TEMPLATE RUBRIC NOTE:**
    This rubric is based on the {template_name} template — a general style guide, NOT a personalized rubric inferred from this user's preferences.

    **CRITICAL: Prioritize user preferences over template rules.**
    - If the user's messages, edits, or feedback conflict with template criteria, ALWAYS follow the user's preferences
    - The template provides default guidelines, but the user's explicit requests override them
    - Pay close attention to what the user asks for, how they edit drafts, and any feedback they give
    - If the user's writing style differs from the template, adapt to THEIR style rather than forcing template conventions
    - Use the template as a fallback for aspects the user hasn't expressed preferences about

    Think of it this way: Template criteria < User's implicit preferences (from their writing/edits) < User's explicit requests
"""

    if rubric:
        system_instruction = dedent(f"""
        You are an AI co-writer. You collaborate with the user to develop their writing.
        {rubric_block}
        {template_guidance}
        **RUBRIC AUTHORITY:**
        The rubric is the user's persistent writing preferences. It is your primary guide for tone, style, structure, and approach. Follow it consistently across all turns.

        - The rubric defines HOW to write. The user's messages define WHAT to write.
        - If the user gives a task-specific instruction (e.g., "expand this paragraph"), follow it for that request — it does not change the rubric.
        - If the user's feedback seems to contradict the rubric, ask: "Should I treat this as a one-time adjustment, or a preference you'd like going forward?"
        - Never silently deviate from the rubric. If you think a criterion is producing poor results, tell the user.

        **OUTPUT FORMAT:**
        Always wrap any draft content (partial or full) in <draft></draft> tags.
        EXCEPTION: When presenting multiple options/alternatives for the user to choose between, do NOT use <draft> tags. Just present them as plain text (e.g. "Option 1: ...", "Option 2: ..."). Only use <draft> tags for a single committed draft that the user will edit and build on.

        **PROBE SIGNAL (include ONLY when you produce a <draft>):**
        After writing a draft, reflect on the rubric criteria you just applied. If you feel genuinely uncertain about how to interpret or apply any ONE criterion — the description is vague, the user's preferences seem conflicting, or you had to guess — append a probe signal tag AFTER the draft:
        <probe_signal>{{"criterion_name": "<exact criterion name>", "criterion_index": <0-based index>, "uncertainty_reason": "<1-2 sentence explanation of the ambiguity>"}}</probe_signal>

        Rules for the probe signal:
        - Only include <probe_signal> when you have REAL uncertainty, not every draft. Most drafts should NOT have one.
        - Pick the SINGLE most ambiguous criterion, not multiple.
        - If all criteria are clear and you feel confident, do NOT include any <probe_signal> tag.
        - The probe signal is processed by the system and will NOT be shown to the user.
        """).strip()
    else:
        system_instruction = dedent("""
        You are an AI co-writer. You collaborate with the user to develop their writing.

        **OUTPUT FORMAT:**
        Always wrap any draft content (partial or full) in <draft></draft> tags.
        EXCEPTION: When presenting multiple options/alternatives for the user to choose between, do NOT use <draft> tags. Just present them as plain text (e.g. "Option 1: ...", "Option 2: ..."). Only use <draft> tags for a single committed draft that the user will edit and build on.
        """).strip()

    return system_instruction

DRAFT_REVISE_AFTER_RUBRIC_CHANGE_SYSTEM_PROMPT = """You are a rubric refinement specialist. The user has manually edited a draft, and your task is to analyze their edits to infer what they value in their writing and suggest updates to their rubric.

You will be given:
1. The current rubric (with criteria, descriptions, achievement levels, and weights)
2. The original draft (before user edits)
3. The edited draft (after user edits)

Your task:
1. Carefully compare the two drafts to identify what the user changed
2. Infer WHY they made those changes - what preferences or values do the edits reveal?
3. Determine if these preferences are already captured in the rubric or if updates are needed
4. Suggest specific rubric modifications that would better capture the user's demonstrated values

Types of rubric updates to consider:
- **Weight adjustments**: If the user's edits strongly prioritize certain qualities, those criteria may deserve more weight
- **Description refinements**: If the user's edits reveal specific preferences not captured in criterion descriptions
- **Achievement level updates**: If the user's edits show what "exemplary" or other levels should look like
- **New criteria**: If the user's edits reveal values not covered by any existing criterion
- **No changes needed**: If the rubric already captures what the user demonstrated

Output format:
Return ONLY a valid JSON object with this exact structure:
{
  "edit_analysis": {
    "changes_observed": ["List of specific changes the user made"],
    "inferred_preferences": ["What these changes suggest the user values"]
  },
  "rubric_updates": {
    "has_updates": true/false,
    "rationale": "Brief explanation of why updates are or aren't needed",
    "modified_rubric": [
      // Full rubric array with any changes applied
      // If no changes, return the original rubric unchanged
    ]
  }
}

Guidelines:
- Be conservative - only suggest changes that are clearly supported by the user's edits
- Preserve the overall rubric structure unless changes are warranted
- Maintain weight totals at 100% if adjusting weights
- Focus on what the edits REVEAL about user preferences, not just what changed
"""

def DRAFT_revise_after_rubric_change_prompt(current_rubric, original_draft, edited_draft):
    """Generate user prompt for inferring rubric updates from draft edits."""
    return f"""Current rubric:
{json.dumps(current_rubric, ensure_ascii=False, indent=2)}

Original draft (before user edits):
{original_draft}

Edited draft (after user edits):
{edited_draft}

Analyze the user's edits and suggest any rubric updates that would better capture their demonstrated preferences."""

DRAFT_REGENERATE_SYSTEM_PROMPT = """You are a writing assistant tasked with revising a draft to better align with an updated rubric.

The user has made changes to their rubric criteria, and you need to revise the existing draft to fulfill those changes.

You will be given:
1. The original rubric (before changes)
2. The updated rubric (after user's changes)
3. The current draft that needs to be revised
4. (Optional) The conversation history between the user and assistant — use this to understand the user's intent, preferences, and any feedback they gave
5. (Optional) Rubric change suggestions — the reasoning behind why rubric criteria were changed
6. (Optional) User edit feedback — the user's feedback on specific draft edits, explaining what they agreed/disagreed with

Your task:
1. Identify what changed between the original and updated rubric (priorities, descriptions, dimensions)
2. Analyze how those changes should affect the draft
3. Revise the draft to better fulfill the updated rubric criteria
4. Provide an annotated version showing exactly what changed and why

Guidelines:
- Use the conversation history, rubric suggestions, and user edit feedback to understand the FULL context of what the user wants
- Focus changes on areas affected by the rubric updates AND any user feedback
- Make targeted edits rather than complete rewrites
- Ensure the revised draft still flows naturally
- If a criterion's priority increased (lower number = higher priority), give more attention to that aspect
- If dimensions were added or changed, ensure the draft meets those specific checkable items

Output format:
Return ONLY a valid JSON object with this exact structure:
{
  "rubric_changes_identified": [
    "List of specific changes between old and new rubric"
  ],
  "revision_strategy": "Brief explanation of how you'll approach the revision",
  "change_summary": "2-4 sentence summary for the user: what rubric changes led to what draft changes and why. Write for the user who just clicked 'Log Changes'.",
  "revised_draft": "The full revised draft text here (clean, no annotations)",
  "revised_draft_annotated": "Same content as revised_draft but with inline annotations showing EVERY change. CRITICAL RULES: (1) Every [N] marker MUST be immediately preceded by <ins>...</ins> and/or <del>...</del> tags — a bare [N] with no adjacent ins/del is INVALID. (2) Use <del>removed text</del> for deletions, <ins>new text</ins> for additions. (3) For replacements: <del>old</del><ins>new</ins>[N]. (4) Place [N] immediately AFTER the ins/del tags. Example: 'The <del>long-winded</del><ins>concise</ins>[1] version works better.' (5) Every entry in annotated_changes must have a corresponding [N] with visible ins/del tags. If you cannot show the change with ins/del, do NOT include the [N] marker.",
  "revised_draft_with_markers": "Same as revised_draft but with only [1], [2], ... markers inserted after each changed phrase (no ins/del tags). Use if revised_draft_annotated would be too cluttered; otherwise can duplicate revised_draft_annotated without the ins/del tags.",
  "annotated_changes": [
    {
      "original_text": "The exact text from the original draft that was changed",
      "new_text": "The new text that replaced it",
      "reason": "CriterionName (what changed about it): why this text was modified"
    }
  ]
}

IMPORTANT for annotated_changes:
- Each entry should capture one specific edit location
- "original_text" must be an exact substring from the original draft
- "new_text" is what replaced it in the revised draft
- "reason" MUST follow this format: "CriterionName (the rubric change): explanation"
  - Example: "Brevity (priority increased from #3 to #1): removed unnecessary qualifier words"
  - Example: "Tone (new dimension added: 'avoid jargon'): replaced technical term with plain language"
  - Example: "Structure (description updated to emphasize flow): reorganized for better transitions"
- IMPORTANT: Different types of rubric changes drive different draft edits. Treat each separately:
  - If a criterion has BOTH a priority change AND a description edit, these are separate rubric changes
  - Each draft change should reference the SPECIFIC rubric change that caused it
  - Example: If "Brevity" moved from #3 to #1 AND got a new dimension, a draft edit caused by the priority change should say "Brevity (priority #3→#1): ..." while an edit caused by the new dimension should say "Brevity (new dimension 'cut filler words'): ..."
  - Do NOT combine multiple rubric changes into one reason
- Include ALL changes, even small wording tweaks
- If text was deleted, new_text should be empty string
- If text was added (not replacing anything), original_text should indicate the location context

IMPORTANT for change_summary and revised_draft_annotated:
- change_summary: Write in plain language for the user. Example: "Raising Brevity to priority 1 led to trimming the opening sentence and removing two qualifiers. The new 'avoid jargon' dimension under Tone led to replacing 'utilize' with 'use'."
- revised_draft_annotated: EVERY [N] marker MUST have <ins> and/or <del> tags immediately before it. A [N] without adjacent ins/del is a formatting error. Every <ins> and <del> must be closed. Every [N] must correspond to an index in annotated_changes (1-based).
"""

def DRAFT_regenerate_prompt(original_rubric, updated_rubric, current_draft, conversation_history=None, rubric_suggestion_text=None, user_edit_feedback=None):
    """Generate user prompt for regenerating draft based on rubric changes."""
    parts = []

    if conversation_history:
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append(conversation_history)
        parts.append("")

    if rubric_suggestion_text:
        parts.append("=== RUBRIC CHANGE SUGGESTIONS ===")
        parts.append("These are the reasons why the rubric criteria were changed:")
        parts.append(rubric_suggestion_text)
        parts.append("")

    if user_edit_feedback:
        parts.append("=== USER EDIT FEEDBACK ===")
        parts.append("The user provided this feedback on specific draft edits:")
        parts.append(user_edit_feedback)
        parts.append("")

    parts.append(f"""Original rubric (before changes):
{json.dumps(original_rubric, ensure_ascii=False, indent=2)}

Updated rubric (after your changes):
{json.dumps(updated_rubric, ensure_ascii=False, indent=2)}

Current draft to revise:
{current_draft}

Analyze the rubric changes and revise the draft to better fulfill the updated criteria. Use the conversation history, rubric suggestions, and user edit feedback (if provided) to understand the full context of the user's intent.""")

    return "\n".join(parts)

INLINE_REGEN_SYSTEM_PROMPT = """You are a writing assistant performing targeted edits on specific sentences in a draft.

You will be given:
1. The full draft (for context — do NOT rewrite the full draft)
2. One or more sentences the user has selected for editing, each labeled S1, S2, etc.
3. The user's instruction for how to change the selected sentences
4. Optionally, rubric criteria to follow

Your task:
- Regenerate EACH selected sentence individually according to the user's instruction
- Each replacement must fit seamlessly into its original position in the draft (matching tone, tense, style)
- Maintain approximately the same length per sentence unless the instruction explicitly asks for expansion or reduction
- Follow the rubric criteria if provided

Output ONLY a valid JSON object (no markdown fences, no extra text):
{
  "replacements": [
    {"original": "The exact original sentence S1", "replacement": "The rephrased version of S1"},
    {"original": "The exact original sentence S2", "replacement": "The rephrased version of S2"}
  ],
  "explanation": "Brief explanation of what was changed and why (1-2 sentences)"
}

IMPORTANT: Return one entry per selected sentence, in the same order they were given. Each sentence is replaced independently — they may be non-adjacent in the draft."""


def INLINE_regen_user_prompt(full_draft, selected_sentences, instruction, rubric_list=None):
    """Generate prompt for inline text regeneration.

    selected_sentences: list of sentence strings to rephrase (in order).
    """
    rubric_section = ""
    if rubric_list:
        rubric_section = f"\n\nRubric criteria to follow:\n{json.dumps(rubric_list, ensure_ascii=False, indent=2)}"

    sentences_block = "\n".join(f"S{i+1}: {s}" for i, s in enumerate(selected_sentences))

    return f"""Full draft (for context only — do NOT rewrite the entire draft):
{full_draft}

Sentences to edit (each must be rephrased independently):
{sentences_block}

User's instruction:
{instruction}{rubric_section}

Regenerate EACH sentence individually. Return JSON with a "replacements" array (one entry per sentence, same order). Return JSON only."""


def RUBRIC_refine_from_corrected_dps_prompt(current_rubric_json, corrected_dps_json):
    """Build prompt for refining a rubric based on user's DP corrections.

    Called when user marks DPs as 'incorrect' or 'not in rubric' and clicks Confirm.
    Takes the current rubric (v0) and the corrected DPs to produce a refined rubric (v1).
    """
    return f"""You are refining a personalized writing rubric based on user feedback on decision points.

The user reviewed the decision points extracted from their conversation and made corrections. Use these corrections to improve the rubric.

## Current Rubric

{current_rubric_json}

## User's Decision Point Corrections

{corrected_dps_json}

## How to interpret corrections:

- **"correct"**: The DP's criterion mapping was right. No rubric change needed for this DP.
- **"incorrect"**: The user says this DP maps to a DIFFERENT criterion than suggested. Check if the target criterion's description/dimensions adequately capture what this DP reveals. If not, update them.
- **"not_in_rubric"**: The user says this preference is NOT captured by any existing criterion. You MUST either:
  - Add a new criterion to capture it, OR
  - Add new dimensions to an existing criterion if it fits naturally

## Rules

- Keep ALL existing criteria that are working well (the "correct" DPs confirm these)
- For "incorrect" DPs: update the target criterion's description or dimensions to better reflect the preference
- For "not_in_rubric" DPs: add new criteria or dimensions. Use the user's `not_in_rubric_reason` if provided
- Maintain unique priority rankings (1 = most important)
- Keep 4-7 criteria total. If adding new ones would exceed 7, consider merging related criteria

## Output

Return ONLY valid JSON matching this structure (no markdown fences, no preamble):
{{
  "version": 2,
  "writing_type": "<from current rubric>",
  "user_goals_summary": "<updated if needed>",
  "rubric": [
    {{
      "name": "<criterion name>",
      "category": "<category>",
      "description": "<1-3 sentences>",
      "dimensions": [
        {{"id": "<id>", "label": "<checkable item>"}}
      ],
      "priority": <unique integer 1..N, no duplicates>
    }}
  ],
  "coaching_notes": "<updated if needed>",
  "refinement_summary": "<1-2 sentences: what changed and why>",
  "change_explanation": "<A user-facing explanation in markdown. For EACH change you made to the rubric, explain: (1) which DP correction triggered it, (2) what you changed (added/modified/merged criterion or dimension), and (3) why. Use bullet points. Be specific — reference criterion names and DP numbers. Example: '- **DP#3** was marked *not in rubric* because you prefer paragraph transitions over bullet lists. Added new criterion **Flow & Continuity** to capture this.\n- **DP#5** was remapped to **Tone**, so I expanded its dimensions to include emotional warmth, which your correction highlighted.' Keep it concise but informative.>"
}}"""

def RUBRIC_suggest_changes_from_feedback_prompt(active_rubric_json: str, edited_rubric_json: str, edits_with_feedback: list):
    """
    Build prompt for suggesting how to change the rubric based on user feedback on edits.
    The model sees both the active (saved) rubric and the edited rubric so it knows what changed.
    active_rubric_json = saved version from history; edited_rubric_json = user's edits (used for the draft).
    edits_with_feedback is a list of {"reason", "original_text", "new_text", "user_feedback"}.
    """
    feedback_block = "\n\n".join(
        f"- [Edit] {e.get('reason', '')} — original: \"{(e.get('original_text') or '')[:80]}{'…' if len(e.get('original_text', '') or '') > 80 else ''}\" → new: \"{(e.get('new_text', '') or '')[:80]}{'…' if len(e.get('new_text', '') or '') > 80 else ''}\"\n  **User feedback:** {e.get('user_feedback', '').strip() or '(none)'}"
        for e in (edits_with_feedback or [])
    )
    return f"""You are helping refine a writing rubric based on the user's feedback on a regenerated draft.

**Active rubric** (saved version, before the user's manual edits):
{active_rubric_json}

**Edited rubric** (the user's current changes in Rubric Configuration — this version was used to regenerate the draft):
{edited_rubric_json}

**Edits the user gave feedback on** (they may not agree with the edit; their feedback explains why or what they would have done differently):
{feedback_block or "(no feedback provided)"}

**Your task:** Using the user's feedback and the context of both rubrics above, suggest specific changes to the **edited rubric** so that in future draft regenerations, the model would not be led to make edits the user disagrees with. Be concrete: name criteria, dimensions, or descriptions to add, remove, soften, or reword. Explain in 2–5 short bullet points. Write for the user who will then edit the rubric (e.g. in the Rubric tab)."""

def RUBRIC_apply_suggestion_prompt(active_rubric_json: str, edited_rubric_json: str, suggestion_text: str):
    """
    Build prompt to turn the suggestion (bullet points) into an actual modified rubric JSON.
    The model sees both active (saved) and edited rubric; it applies the suggestion to the edited rubric.
    Output: ONLY a valid JSON array of criteria, same structure.
    """
    return f"""You are a rubric editor. The user received the following suggestion for how to change their rubric (based on their feedback on draft edits):

{suggestion_text}

**Active rubric** (saved version, before the user's manual edits):
{active_rubric_json}

**Edited rubric** (current user changes — apply the suggestion TO this version):
{edited_rubric_json}

**Your task:** Produce the MODIFIED rubric by applying the suggestion to the **edited rubric**. Output ONLY a valid JSON array of criteria, no markdown or explanation. Each criterion must have:
- "name": string
- "category": string
- "description": string
- "dimensions": array of {{"id": "unique_id", "label": "string"}}
- "priority": integer (1 = most important, unique per criterion)

Preserve the same overall structure and any criteria/dimensions not mentioned in the suggestion. Only change what the suggestion asks to change. If the suggestion adds a criterion or dimension, generate a new "id" (e.g. "new_1", "dim_abc"). Output nothing but the JSON array."""

def GRADING_generate_writing_task_prompt(conversation_text: str, project_task_examples: str = "") -> str:
    """Generate a NEW writing task similar in domain/type to the project's writing tasks but a different specific scenario,
    so the same rubric would still apply. Task and resulting drafts should be short for quick comparison."""
    project_section = ""
    if project_task_examples:
        project_section = f"""## Writing tasks from this project's conversations

These are the kinds of writing tasks the user has been working on in this project. The new task MUST be the same type/domain of writing.

{project_task_examples}

"""
    conversation_section = ""
    if conversation_text:
        conversation_section = f"""## Reference conversation

{conversation_text}

"""
    return f"""{project_section}{conversation_section}Your job: Write ONE **short** writing instruction (1-2 sentences) for a **new** task that:
1. Is in the SAME domain and type of writing as the tasks above (e.g. if the tasks are about emails, the new task must be to write an email; if they are recommendation letters, write a recommendation letter).
2. Is a **different** specific scenario or prompt — not the same as any task shown above. For example, if a task was "declining a meeting," the new task might be "asking for an extension on a deadline" or "thanking a colleague for feedback."
3. Is similar enough that a rubric inferred from the conversations would still apply to drafts written for this new task.
4. Asks for a **brief** output so drafts stay short: e.g. "Write a 2-4 sentence email...", "Write one short paragraph...", "In 2-3 sentences...". The user will read 3 drafts side by side — keep the task scope small (no long essays or multi-paragraph pieces).

Output ONLY the writing instruction itself. No preamble, no "Here is the task:", no JSON. Just the instruction. Keep it concise.
"""

def GRADING_generate_draft_from_coldstart_prompt(writing_task: str, coldstart_text: str) -> str:
    """Generate a draft for the task using ONLY the user's cold-start preference description (no rubric)."""
    return f"""You are a skilled writer. Write a complete draft for the following task, following ONLY the user's stated preferences below. Do not use any rubric or other criteria.

WRITING TASK:
{writing_task}

USER'S PREFERENCES (follow these — they describe how the user wants the writing to sound and what they care about):
{coldstart_text}

Keep the draft to around 100 words (or to the length the task specifies). Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Just the draft itself.
FORMATTING: Do NOT use markdown headings (#, ##, ###), bold (**), or any other large/emphasized formatting. Write in plain prose paragraphs only."""

def GRADING_generate_draft_generic_prompt(writing_task: str) -> str:
    """Generate a draft for the task with no rubric or user preferences — generic writing."""
    return f"""You are a skilled writer. Write a complete draft for the following task.

WRITING TASK:
{writing_task}

Keep the draft to around 100 words (or to the length the task specifies). Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Just the draft itself. Fulfill the task in a clear, competent way.
FORMATTING: Do NOT use markdown headings (#, ##, ###), bold (**), or any other large/emphasized formatting. Write in plain prose paragraphs only."""

def RUBRIC_compare_to_coldstart_prompt(rubric_json: str, coldstart_text: str, conversation_text: str = "") -> str:
    """
    Compare rubric criteria against a user's cold-start preference description
    to determine which criteria the user stated vs. which are absent.
    Optionally includes conversation context for better understanding.
    """
    conversation_block = ""
    if conversation_text:
        conversation_block = f"""
CONVERSATION CONTEXT (the editing session from which the rubric was inferred — use this
to better understand what each criterion captures in practice, which helps you judge
whether the cold-start text covers the same concern):

<conversation>
{conversation_text}
</conversation>

"""
    return f"""You are analyzing a user's writing preferences.

TASK: Compare a structured rubric (inferred from the user's editing behavior) against
a free-text "cold-start" description the user wrote BEFORE seeing the rubric.

Determine which rubric criteria the user's cold-start description covers (even partially
or in different words) and which are completely absent.

RUBRIC:
{rubric_json}
{conversation_block}USER'S COLD-START PREFERENCE DESCRIPTION (written without seeing the rubric):
"{coldstart_text}"

INSTRUCTIONS:
For EACH criterion in the rubric:
1. Read the criterion's name, description, and dimensions carefully
2. Search the cold-start description for ANY mention, paraphrase, or implicit reference
   to this criterion's concern
3. A criterion counts as "stated" if the cold-start text addresses the same underlying
   concern, even if the exact wording differs. For example, if the rubric has a criterion
   about "Concise Language" and the user wrote "I prefer short sentences," that counts as stated.
4. A criterion counts as "unstated" if there is NO reference to it whatsoever in the cold-start text.

Be generous in matching — semantic overlap counts, not just exact words.

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "criteria_comparison": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "criterion_description": "<description from rubric>",
            "category": "<category from rubric>",
            "status": "stated" or "unstated",
            "matching_text": "<quote from cold-start text that matches, or null if unstated>",
            "match_reasoning": "<1-2 sentence explanation of why this is stated or unstated>"
        }}}}
    ],
    "summary": {{{{
        "total_criteria": <int>,
        "stated_count": <int>,
        "unstated_count": <int>,
        "stated_criteria": ["<list of stated criterion names>"],
        "unstated_criteria": ["<list of unstated criterion names>"]
    }}}}
}}}}"""

def GRADING_generate_draft_from_rubric_prompt(writing_task: str, rubric_json: str) -> str:
    """
    Generate a complete writing draft for a given task, following the rubric.
    Used in Evaluate: Build tab to produce blind-comparison drafts.
    """
    return f"""You are a skilled writer. Write a complete draft for the following task,
strictly following the rubric criteria provided.

WRITING TASK:
{writing_task}

RUBRIC (follow these criteria carefully — they represent the user's writing preferences):
{rubric_json}

INSTRUCTIONS:
1. Write a complete, polished draft that fulfills the writing task
2. Follow EVERY criterion in the rubric — higher-weight criteria should be more prominent
3. The draft should read naturally, not like a checklist of criteria
4. Do NOT mention the rubric or criteria in the draft itself
5. Keep the draft short: around 100 words (or to the length the task specifies if it asks for more). Do not write a long piece unless the task explicitly asks for one.
6. FORMATTING: Do NOT use markdown headings (#, ##, ###), bold (**), or any other large/emphasized formatting. Write in plain prose paragraphs only.

Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Do not ask for clarification or add "Here is the draft:" or similar. Just the draft itself."""


def ALIGNMENT_generate_annotated_draft_prompt(
    writing_task: str,
    suggested_rubric_json: str,
    original_rubric_draft: str,
    suggestion_reasons: dict,
) -> str:
    """
    Generate a draft following the suggested rubric, annotated with edit markers
    showing what changed compared to the original rubric draft and why.
    Used in the alignment diagnostic pipeline for the "Draft preview with suggested rubric".
    """
    reasons_text = "\n".join(
        f"- {name}: {reason}" for name, reason in suggestion_reasons.items()
    ) if suggestion_reasons else "(No specific reasons provided)"

    return f"""Write a complete draft for the following task, strictly following the suggested rubric.

WRITING TASK:
{writing_task}

SUGGESTED RUBRIC (follow these criteria carefully):
{suggested_rubric_json}

ORIGINAL RUBRIC DRAFT (the baseline — keep as much of this as possible):
{original_rubric_draft}

RUBRIC CHANGES THAT LED TO THIS NEW RUBRIC:
{reasons_text}

INSTRUCTIONS:
1. Write a complete, polished draft that fulfills the writing task following the suggested rubric.
2. Higher-priority criteria should be more prominent. The draft should read naturally.
3. Do NOT mention the rubric in the draft itself.
4. Keep the draft short: around 100 words (or to the length the task specifies).
5. FORMATTING: Plain prose paragraphs only — no markdown headings, bold, or emphasis formatting.

CRITICAL — EDIT ONLY WHAT THE RUBRIC CHANGES DEMAND:
- The purpose of this draft is to show the user how the RUBRIC CHANGES above would affect their writing.
- Only edit parts of the original draft that are DIRECTLY and MEANINGFULLY impacted by a specific rubric change listed above.
- If a rubric criterion was NOT changed, the parts of the draft serving that criterion should stay THE SAME as the original — copy them word for word.
- Do NOT rephrase, polish, or "improve" text that already satisfies the rubric. Cosmetic rewording is NOT an edit.
- Every single edit must be traceable to a specific rubric change. If you cannot point to a concrete rubric change that necessitates the edit, do NOT make it.
- Fewer, high-impact edits are better than many small tweaks. The user wants to see what the rubric changes ACTUALLY affect, not a rewritten draft.

After writing the draft, list each meaningful edit you made and why.

Return ONLY a valid JSON object with this exact structure:
{{
  "revised_draft": "The full new draft text (clean, no markup)",
  "annotated_changes": [
    {{
      "original_text": "Exact text from the original draft that was changed or removed",
      "new_text": "The new text that replaced it (empty string if deleted)",
      "reason": "CriterionName: why this edit was needed"
    }}
  ]
}}

IMPORTANT for annotated_changes:
- ONLY include edits that are directly driven by a rubric change. Do NOT list cosmetic or stylistic tweaks.
- "original_text" must be an exact substring from the ORIGINAL RUBRIC DRAFT above.
- "new_text" must be an exact substring from your revised_draft.
- "reason" format: "CriterionName: 1 sentence explaining the edit"
  Example: "Brevity: trimmed unnecessary qualifier words to match the higher priority on conciseness"
  Example: "Tone: replaced technical term with plain language per the new 'avoid jargon' dimension"
- If you cannot clearly tie an edit to a specific rubric change, DO NOT include it.
- If text was deleted, new_text should be empty string.
- If text was added (not replacing anything), original_text should be empty string.
- It is completely fine to have 0 annotated_changes if the rubric changes do not meaningfully affect the draft."""


def GRADING_judge_per_dimension_prompt(draft_a: str, draft_b: str, rubric_criteria_json: str, user_ratings_json: str) -> str:
    """
    LLM judge scores two drafts per-dimension against the user's own satisfaction ratings.
    Used in Evaluate: Build tab Step 4.
    """
    return f"""You are an impartial writing quality judge. You will evaluate two drafts
against a set of rubric criteria, using the user's own per-dimension satisfaction ratings
as a reference standard.

DRAFT A:
{draft_a}

DRAFT B:
{draft_b}

RUBRIC CRITERIA:
{rubric_criteria_json}

USER'S PER-DIMENSION SATISFACTION RATINGS (ground truth — the user rated each draft on each dimension 1-5):
{user_ratings_json}

TASK:
For each rubric criterion:
1. Score Draft A (0-100) on how well it satisfies this criterion
2. Score Draft B (0-100) on how well it satisfies this criterion
3. Note which draft better aligns with the user's own satisfaction ratings for this dimension
4. Provide brief reasoning

Also determine the overall winner and whether wins cluster on specific criteria.

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "per_criterion": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "draft_a_score": <0-100>,
            "draft_b_score": <0-100>,
            "winner": "A" or "B" or "tie",
            "aligns_with_user_rating": true or false,
            "reasoning": "<1-2 sentences>"
        }}}}
    ],
    "overall": {{{{
        "draft_a_avg": <float>,
        "draft_b_avg": <float>,
        "overall_winner": "A" or "B" or "tie",
        "win_pattern": "<1-2 sentences describing whether wins cluster on edited vs non-edited dimensions>"
    }}}}
}}}}"""

def GRADING_generate_degraded_draft_prompt(writing_task: str, rubric_json: str, dimensions_to_violate_json: str) -> str:
    """
    Generate a draft that follows the rubric EXCEPT for specified dimensions,
    which should be deliberately performed poorly on. Used in Evaluate: Grade tab.
    """
    return f"""You are a skilled writer. Write a complete draft for the following task,
following MOST of the rubric criteria — but deliberately performing poorly on the
specific dimensions listed under DIMENSIONS TO VIOLATE.

WRITING TASK:
{writing_task}

FULL RUBRIC (follow these criteria carefully EXCEPT where told to violate):
{rubric_json}

DIMENSIONS TO VIOLATE (deliberately perform poorly on these — ignore them, do the opposite, or handle them weakly):
{dimensions_to_violate_json}

INSTRUCTIONS:
1. Follow every criterion in the rubric EXCEPT the ones listed under DIMENSIONS TO VIOLATE
2. For the violated dimensions: subtly underperform — don't make it cartoonishly bad,
   but clearly fail to meet those specific criteria. The degradation should feel natural,
   like a writer who simply didn't prioritize those aspects
3. Higher-weight non-violated criteria should still be prominent
4. The draft should still be coherent and complete — only the violated dimensions should suffer
5. Do NOT mention the rubric, the violations, or that anything is intentionally degraded
6. Aim for 300-600 words unless the task implies otherwise

Write ONLY the draft text. No preamble, no explanation, no meta-commentary."""

def GRADING_rubric_judge_prompt(draft_a: str, draft_b: str, rubric_criteria_json: str, conversation_context: str) -> str:
    """
    LLM judge scores two drafts per-dimension using the user's rubric + conversation context.
    Used in Evaluate: Grade tab Step 3 (rubric-grounded condition).
    """
    return f"""You are an expert writing quality judge. You will evaluate two drafts
against a set of personalized rubric criteria. You also have access to the user's
recent conversation history for additional context about their preferences.

The two drafts are:
- Draft A = the **rubric-guided** draft (written following the user's rubric criteria)
- Draft B = the **generic** draft (written with no rubric — just competent writing)

DRAFT A (rubric-guided):
{draft_a}

DRAFT B (generic):
{draft_b}

RUBRIC CRITERIA (the user's personalized writing quality criteria):
{rubric_criteria_json}

RECENT CONVERSATION CONTEXT (shows the user's writing preferences and interactions):
{conversation_context}

TASK:
For each rubric criterion, score BOTH drafts on a 1-5 scale:
  1 = Very poor — clearly fails this criterion
  2 = Below average — noticeable weaknesses
  3 = Adequate — meets basic expectations
  4 = Good — clearly satisfies this criterion
  5 = Excellent — outstanding performance on this criterion

Use the conversation context to understand the user's nuanced preferences when scoring.
Then determine your overall preference (which draft is better overall given all criteria).

IMPORTANT: In your reasoning, refer to drafts by their descriptive names (rubric-guided, generic) rather than Draft A/B.

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "per_criterion": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "draft_a_score": <1-5>,
            "draft_b_score": <1-5>,
            "reasoning": "<1-2 sentences explaining the scores, using descriptive draft names>"
        }}}}
    ],
    "overall_preference": "A" or "B" or "tie",
    "overall_reasoning": "<1-2 sentences explaining overall preference>"
}}}}"""

def GRADING_rubric_judge_3draft_prompt(draft_a: str, draft_b: str, draft_c: str, rubric_criteria_json: str, conversation_context: str) -> str:
    """
    LLM judge scores three drafts per-criterion using the user's rubric + conversation context.
    Used in the Rubric Alignment Check when 3 drafts are generated (rubric, generic, preference).
    Drafts should be passed in shuffled order — the judge does not know which is which.
    """
    return f"""You are an expert writing quality judge. You will evaluate three drafts
against a set of personalized rubric criteria. You also have access to the user's
recent conversation history for additional context about their preferences.

DRAFT A:
{draft_a}

DRAFT B:
{draft_b}

DRAFT C:
{draft_c}

RUBRIC CRITERIA (the user's personalized writing quality criteria):
{rubric_criteria_json}

RECENT CONVERSATION CONTEXT (shows the user's writing preferences and interactions):
{conversation_context}

TASK:
For each rubric criterion, score ALL THREE drafts on a 1-5 scale:
  1 = Very poor — clearly fails this criterion
  2 = Below average — noticeable weaknesses
  3 = Adequate — meets basic expectations
  4 = Good — clearly satisfies this criterion
  5 = Excellent — outstanding performance on this criterion

Use the conversation context to understand the user's nuanced preferences when scoring.
Then determine your overall ranking (which draft is best, second, third given all criteria).

IMPORTANT: Refer to drafts as Draft A, Draft B, Draft C in your reasoning and scores.

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "per_criterion": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "draft_a_score": <1-5>,
            "draft_b_score": <1-5>,
            "draft_c_score": <1-5>,
            "reasoning": "<1-2 sentences explaining the scores, using descriptive draft names>"
        }}}}
    ],
    "overall_ranking": ["<best>", "<second>", "<third>"],
    "overall_reasoning": "<1-2 sentences explaining overall ranking>"
}}}}"""


def GRADING_generic_judge_prompt(draft_a: str, draft_b: str) -> str:
    """
    LLM judge scores two drafts using only generic writing quality criteria.
    No rubric, no user context. Used in Evaluate: Grade tab Step 3 (generic condition).
    """
    return f"""You are an expert writing quality judge. You will evaluate two drafts
using standard writing quality criteria. You have NO information about the writer's
specific preferences — evaluate based purely on general writing quality.

DRAFT A:
{draft_a}

DRAFT B:
{draft_b}

TASK:
Score both drafts on each of the following GENERIC writing quality dimensions (1-5 scale):
  1 = Very poor
  2 = Below average
  3 = Adequate
  4 = Good
  5 = Excellent

Generic dimensions to evaluate:
- Clarity: Is the writing clear and easy to understand?
- Coherence: Do ideas flow logically and connect well?
- Grammar & Mechanics: Is the writing free of grammatical errors and typos?
- Structure & Organization: Is the piece well-organized with clear sections/paragraphs?
- Engagement: Is the writing interesting, compelling, or engaging to read?
- Tone & Voice: Is the tone appropriate and the voice consistent?

Then determine your overall preference (which draft is better overall).

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "per_criterion": [
        {{{{
            "criterion_name": "<one of: Clarity, Coherence, Grammar & Mechanics, Structure & Organization, Engagement, Tone & Voice>",
            "draft_a_score": <1-5>,
            "draft_b_score": <1-5>,
            "reasoning": "<1-2 sentences>"
        }}}}
    ],
    "overall_preference": "A" or "B" or "tie",
    "overall_reasoning": "<1-2 sentences explaining overall preference>"
}}}}"""

def RUBRIC_refine_from_evaluation_prompt(
    current_rubric_json: str,
    coldstart_text: str,
    user_categorizations: dict,
    behavioral_summary: dict,
    decision_points_summary: list,
    agreement_scores: dict
) -> str:
    """
    Generate a prompt to refine a rubric based on all evaluation evidence collected
    during the Infer tab workflow.
    """
    # Build categorization summary
    cat_lines = []
    for crit, cat in user_categorizations.items():
        cat_label = {
            "stated": "Stated (user mentioned in cold-start)",
            "real": "Real (user cares about it)",
            "latent_real": "Real (user cares about it)",
            "elicited": "Real (user cares about it)",
            "hallucinated": "Hallucinated (user doesn't actually care about this)"
        }.get(cat, cat)
        cat_lines.append(f"  - {crit}: {cat_label}")
    categorization_text = "\n".join(cat_lines)

    # Build behavioral evidence summary
    behavioral_text = "No behavioral evidence collected."
    if behavioral_summary:
        analysis = behavioral_summary.get("behavioral_analysis", [])
        beh_lines = []
        for item in analysis:
            strength = item.get("evidence_strength", "none")
            summary = item.get("evidence_summary", "")
            signals = item.get("signal_types_present", [])
            beh_lines.append(f"  - {item.get('criterion_name', '?')}: evidence={strength}, signals={signals}, summary={summary}")
        if beh_lines:
            behavioral_text = "\n".join(beh_lines)

    # Build decision points summary
    dp_lines = []
    for dp in decision_points_summary:
        dp_id = dp.get("id", "?")
        confirmed = dp.get("confirmed_criterion", "?")
        user_action = dp.get("user_action", "correct")
        original = dp.get("original_suggestion", "")
        is_not_in_rubric = dp.get("is_not_in_rubric", False)
        reason = dp.get("not_in_rubric_reason", "") or dp.get("incorrect_reason", "")

        if is_not_in_rubric:
            dp_lines.append(f"  - DP#{dp_id}: NOT IN RUBRIC — user said this edit doesn't map to any criterion. Reason: {reason}")
        elif user_action == "incorrect":
            dp_lines.append(f"  - DP#{dp_id}: LLM mapped to '{original}' but user corrected to '{confirmed}'. Reason: {reason}")
        else:
            dp_lines.append(f"  - DP#{dp_id}: Confirmed mapping to '{confirmed}'")
    dp_text = "\n".join(dp_lines) if dp_lines else "No decision points."

    # Build agreement scores summary (writing preference, before/after prediction accuracy, or legacy tau)
    agreement_text = "No agreement scores computed."
    if agreement_scores:
        wp = agreement_scores.get("writing_preference")
        before_after = agreement_scores.get("before_after_accuracy")
        if wp:
            if wp.get("user_preferred_rubric_guided"):
                pref = "preferred the rubric-guided version"
            elif wp.get("user_preferred_coldstart_guided"):
                pref = "preferred the cold-start-guided version"
            elif wp.get("user_preferred_generic_guided"):
                pref = "preferred the generic (no-preference) version"
            else:
                pref = "chose tie (no single preference)"
            agreement_text = f"Writing preference check (blind A/B/C): User {pref}."
        elif before_after:
            total = before_after.get("total", 0)
            rc = before_after.get("rubric_correct", 0)
            cc = before_after.get("coldstart_correct", 0)
            gc = before_after.get("generic_correct", 0)
            surp = before_after.get("surplus_count", 0)
            src = before_after.get("surplus_rubric_correct", 0)
            scc = before_after.get("surplus_coldstart_correct", 0)
            sgc = before_after.get("surplus_generic_correct", 0)
            agreement_text = f"""Prediction test: did each predictor correctly predict the user preferred "after" (their edit) over "before" (AI suggestion)?
  Overall (n={total}): Rubric {rc}/{total} | Cold-Start {cc}/{total} | Generic {gc}/{total}
  Surplus criteria only (n={surp}): Rubric {src}/{surp} | Cold-Start {scc}/{surp} | Generic {sgc}/{surp}
(A surplus criterion is one the user endorsed but did not state in their cold-start description.)"""
        else:
            overall = agreement_scores.get("overall", {})
            stated = agreement_scores.get("stated", {})
            surplus = agreement_scores.get("surplus", {})
            nir = agreement_scores.get("not_in_rubric", {})

            def fmt(tau_dict):
                tau = tau_dict.get("tau")
                return f"{tau:.3f}" if tau is not None else "N/A"

            agreement_text = f"""Overall Kendall's tau:
    Rubric: {fmt(overall.get('rubric', {}))} | Cold-Start: {fmt(overall.get('coldstart', {}))} | Generic: {fmt(overall.get('generic', {}))}
  Stated dimensions (n={stated.get('count', 0)}):
    Rubric: {fmt(stated.get('rubric', {}))} | Cold-Start: {fmt(stated.get('coldstart', {}))} | Generic: {fmt(stated.get('generic', {}))}
  Surplus dimensions (n={surplus.get('count', 0)}):
    Rubric: {fmt(surplus.get('rubric', {}))} | Cold-Start: {fmt(surplus.get('coldstart', {}))} | Generic: {fmt(surplus.get('generic', {}))}
  Not in rubric: {nir.get('count', 0)} decision points"""
            nir_reasons = [r for r in nir.get("reasons", []) if r]
            if nir_reasons:
                agreement_text += "\n  Not-in-rubric reasons: " + "; ".join(nir_reasons)

    return f"""You are refining a personalized writing rubric based on evaluation evidence.

The user has gone through a comprehensive evaluation workflow to test how well their current rubric captures their actual writing preferences. You now have rich evidence about what works, what's missing, and what's wrong with the current rubric.

## CURRENT RUBRIC
{current_rubric_json}

## EVALUATION EVIDENCE

### 1. Cold-Start Description
The user described their writing preferences WITHOUT seeing the rubric:
"{coldstart_text}"

### 2. User's Criterion Categorizations
After seeing the rubric, the user categorized each criterion:
{categorization_text}

Key takeaways:
- "Hallucinated" criteria: the rubric includes these but the user doesn't care about them. Consider REMOVING or significantly deprioritizing these.
- "Latent Real" criteria: the user cares but couldn't articulate upfront — these are valuable and the rubric correctly surfaced them. Keep and possibly strengthen.
- "Elicited" criteria: the user cares but just didn't think to mention — keep these.
- "Stated" criteria: the user explicitly values these — make sure they're well-captured.

### 3. Behavioral Evidence from Conversations
Evidence from analyzing the user's actual editing behavior:
{behavioral_text}

### 4. Decision Point Analysis
How the user's edits mapped to rubric criteria:
{dp_text}

Key takeaways from dimension mapping corrections:
- Where the LLM was corrected, the criterion descriptions or names may be unclear or misleading
- "Not in rubric" items suggest missing criteria that the user cares about but the rubric doesn't capture

### 5. Preference Prediction Agreement
How well different preference sources predicted the user's actual rankings:
{agreement_text}

Key takeaways:
- If rubric tau >> cold-start tau: the rubric captures preferences the user can't easily articulate (good!)
- If cold-start tau >> rubric tau: the rubric may be missing or miscalibrating important preferences
- If generic tau is competitive: the rubric may not be adding much beyond generic quality standards
- Differences between stated and surplus dimensions reveal where the rubric adds unique value

## YOUR TASK

Based on ALL this evidence, produce a refined rubric that:

1. **REMOVES or deprioritizes** hallucinated criteria (user doesn't care about them)
2. **STRENGTHENS** latent real and elicited criteria (user values them, rubric correctly surfaced them)
3. **ADDS new criteria** for any "not in rubric" decision points that represent real preferences
4. **CLARIFIES descriptions** for criteria where the LLM dimension mapping was frequently corrected — if the model couldn't match edits to the right criterion, the description may need rewording
5. **ADJUSTS priorities** based on behavioral evidence strength and user categorizations
6. **REFINES dimensions** (checkable items) based on the behavioral evidence — what specific behaviors did the user exhibit?
7. **PRESERVES** what's working — criteria with strong behavioral evidence and correct dimension mappings should be kept largely intact

IMPORTANT CONSTRAINTS:
- Keep 4-7 criteria total
- Each criterion needs: name, category, description (1-3 sentences), dimensions (3-5 checkable items), priority (unique integer rank)
- Use the user's own vocabulary where possible (from cold-start description and behavioral evidence)
- Include coaching_notes summarizing what changed and why

First, write your analysis in <analysis> tags explaining what changes you're making and why based on the evidence.

Then output the refined rubric as JSON:
```json
{{{{
  "version": <next version number>,
  "writing_type": "<specific genre, audience, constraints>",
  "user_goals_summary": "<2-3 sentence summary updated with new evidence>",
  "rubric": [
    {{{{
      "name": "<criterion name>",
      "category": "<shared category>",
      "description": "<1-3 sentence user-specific description>",
      "dimensions": [
        {{{{
          "id": "<machine-friendly id>",
          "label": "<checkable item: what to verify as yes/no>"
        }}}}
      ],
      "priority": <unique integer 1..N, 1 = most important, no duplicates>
    }}}}
  ],
  "coaching_notes": "<2-3 concise insights about what changed and why>",
  "changes_summary": [
    {{{{
      "type": "added|removed|modified|reprioritized",
      "criterion": "<criterion name>",
      "reason": "<brief explanation based on evidence>"
    }}}}
  ]
}}}}
```"""

def GRADING_unified_eval_prompt(
    task_description: str,
    rubric_json: str,
    cold_start_text: str,
    draft_a: str,
    draft_b: str,
    draft_c: str,
) -> str:
    """Single prompt to evaluate all 3 drafts under all 3 conditions (rubric, cold-start, generic). Returns one big evaluation."""
    return f"""You will evaluate three drafts (A, B, C) under three conditions each — 9 evaluations total. Return ONE JSON object with all results.

## Shared context

**Writing task:** {task_description}

**User's preference rubric (for Condition 1 — rubric):**
{rubric_json}

**User's stated cold-start preferences (for Condition 2 — cold-start):**
{cold_start_text}

**Condition 3 (generic):** Evaluate against standard writing quality: Clarity, Coherence, Structure, Tone & Style, Completeness, Grammar & Mechanics. Use the same dimension-based format (dimensions_met, dimensions_total, dimensions_detail with id, label, met, evidence), achievement_level (Excellent/Good/Fair/Needs Work/Weak), overall_assessment.

---

## Draft A

{draft_a}

---

## Draft B

{draft_b}

---

## Draft C

{draft_c}

---

## Your task

For each draft (A, B, C), produce three evaluations:
1. **Rubric:** Evaluate against the user's preference rubric. Each criterion: dimensions_met, dimensions_total, dimensions_detail (id, label, met, evidence), achievement_level. Be binary on dimensions; quote evidence.
2. **Cold-start:** Evaluate against the user's stated preferences only. Extract checkable dimensions from the cold-start text; evaluate the draft against those. Same output shape.
3. **Generic:** Evaluate against standard writing quality criteria (clarity, coherence, structure, tone, completeness, grammar). Same output shape.

For each evaluation output: `criteria_scores` (array of {{"name", "dimensions_met", "dimensions_total", "dimensions_detail": [{{"id", "label", "met", "evidence"}}], "achievement_level"}}), and `overall_assessment` (2–3 sentences).

## Required output format

Return a single JSON object (no markdown fence, no preamble) with this exact structure:

{{"A": {{"rubric": {{"criteria_scores": [...], "overall_assessment": "..."}}, "coldstart": {{"criteria_scores": [...], "overall_assessment": "..."}}, "generic": {{"criteria_scores": [...], "overall_assessment": "..."}}}}, "B": {{"rubric": {{...}}, "coldstart": {{...}}, "generic": {{...}}}}, "C": {{"rubric": {{...}}, "coldstart": {{...}}, "generic": {{...}}}}}}

Use keys exactly: "A", "B", "C" for drafts; "rubric", "coldstart", "generic" for conditions. Keep evidence brief to fit in one response.
"""


def GRADING_pairwise_judge_prompt(draft_a, draft_b, condition, condition_context):
    """Lightweight pairwise judge: which of two drafts is better under a given condition?

    Used for:
    - Layer 1: Silent A/B aggregation (after each A/B choice in chat)
    - Layer 2: Ranking checkpoint round-robin comparisons

    Args:
        draft_a: First draft text
        draft_b: Second draft text
        condition: "rubric", "coldstart", or "generic"
        condition_context: Rubric JSON, cold-start text, or empty string
    """
    if condition == "rubric":
        condition_instruction = f"""You are judging two writing drafts using the following personalized rubric.
Which draft better satisfies these criteria overall?

RUBRIC:
{condition_context}"""
    elif condition == "coldstart":
        condition_instruction = f"""You are judging two writing drafts based on the following user-stated writing preferences.
Which draft better matches what this user wants?

USER'S STATED PREFERENCES:
{condition_context}"""
    else:  # generic
        condition_instruction = """You are judging two writing drafts based on general writing quality.
Consider: clarity, coherence, grammar, structure, engagement, and tone.
Which draft is better overall?"""

    return f"""{condition_instruction}

---

DRAFT A:
{draft_a}

---

DRAFT B:
{draft_b}

---

Which draft is better? Return ONLY a JSON object (no markdown, no preamble):
{{"preferred": "A" or "B" or "tie", "confidence": "high" or "medium" or "low", "reasoning": "1-2 sentences explaining your choice"}}"""


def ALIGNMENT_diagnostic_suggest_and_apply_prompt(
    current_rubric_json: str,
    rubric_judge_results_json: str,
    user_ranking_description: str,
    user_reason: str = "",
    preferred_draft: str = "",
    rubric_guided_draft: str = "",
    writing_task: str = "",
) -> str:
    """Generate modified rubric JSON + per-criterion reasons in a single call.

    Combines what was previously two calls (suggest reasons + apply changes) into one,
    ensuring reasons always match the actual modifications made.
    """
    user_reason_block = ""
    if user_reason.strip():
        user_reason_block = f"\nThe user also said: \"{user_reason.strip()}\"\n"

    task_block = ""
    if writing_task:
        task_block = f"\nWRITING TASK:\n{writing_task}\n"

    draft_block = ""
    if preferred_draft and rubric_guided_draft and preferred_draft != rubric_guided_draft:
        draft_block = f"""
THE USER'S PREFERRED DRAFT (the draft the user ranked highest — this is the target style/quality):
{preferred_draft}

THE RUBRIC-GUIDED DRAFT (generated from the current rubric — this is what the rubric currently produces):
{rubric_guided_draft}

Compare these two drafts carefully. The preferred draft shows what the user actually wants. The rubric-guided draft shows what the current rubric produces. Your rubric changes should close this gap — modify criteria so that a rubric-guided draft would come out more like the preferred draft.
"""
    elif rubric_guided_draft:
        draft_block = f"""
THE RUBRIC-GUIDED DRAFT (generated from the current rubric):
{rubric_guided_draft}
"""

    return f"""You are a rubric improvement advisor. A user just completed an alignment diagnostic
comparing three drafts of the same writing task:
1. A **rubric-guided** draft (written following the user's rubric criteria)
2. A **generic** draft (written with no rubric or preferences — just competent writing)
3. A **preference-based** draft (written following the user's original stated preferences from before the rubric was created)
{task_block}
CURRENT RUBRIC:
{current_rubric_json}
{draft_block}
RUBRIC JUDGE RESULTS (per-criterion scores for all three drafts, using the rubric):
Note: In the scores below, draft_a = rubric-guided, draft_b = generic, draft_c = original preference.
{rubric_judge_results_json}

USER'S RANKING: {user_ranking_description}{user_reason_block}

WHY THREE DRAFTS:
The rubric was built from the user's original stated preferences. The preference-based draft acts as a **comparison signal**: by scoring both the rubric-guided draft and the preference-based draft against the same rubric criteria, we can see where the rubric successfully captures the user's intent (rubric draft scores equal or higher) and where it falls short (preference draft scores higher). The generic draft is a baseline — if it scores highest, the criterion isn't adding value at all.

TWO TYPES OF EVIDENCE:
You have two signals to work with — use BOTH together:
1. **Per-criterion scores** (from the rubric judge): automated scores showing how each draft performs on each criterion.
2. **User's ranking** (from the user): the user's own holistic preference for which draft they liked best overall. This is the user's ground truth — it tells you whether the rubric is producing writing the user actually wants.

These signals complement each other. The per-criterion scores show WHERE specific criteria may be weak. The user's ranking tells you the OVERALL direction — how aggressively to change the rubric.

ANALYSIS FRAMEWORK:
For each rubric criterion, compare the three per-criterion scores:
- **Rubric draft scores highest**: the rubric is successfully capturing the user's intent on this criterion (DIFFERENTIATING). No change needed.
- **Preference draft scores higher than rubric draft**: the rubric criterion isn't fully capturing what the user wants — consider rewording or re-prioritizing it so the rubric-guided draft can perform better.
- **Generic draft scores highest or all tied**: the criterion may be too vague or not adding value (UNDERPERFORMING/REDUNDANT).

IMPORTANT: The rubric already incorporates the user's stated preferences in distilled form. If the preference draft outperforms the rubric draft, that means existing criteria need strengthening — not that new criteria should be added to mirror stated preferences. You may add a new criterion only if the scores reveal a genuine gap that no existing criterion can cover even with rewording.

CALIBRATE CHANGES TO THE USER'S RANKING:
The user's ranking determines how much to change overall:
- If the user ranked the RUBRIC-GUIDED draft FIRST: the rubric is working well overall. Be VERY conservative. Only modify criteria that are clearly underperforming or redundant based on the per-criterion scores. Most criteria should be kept UNCHANGED. Do NOT rephrase, reword, or "improve" criteria that are already working — leave them exactly as they are, character for character.
- If the user ranked the PREFERENCE-BASED draft first: the rubric needs changes to better capture what the user wants. Focus on rewording or re-prioritizing existing criteria first. Only add a new criterion if no existing criterion can reasonably cover the gap.
- If the user ranked the GENERIC draft first: the rubric may need significant revision.

YOUR TASK:
Based on this evidence, produce an updated version of the rubric that would generate drafts more like what the user ranked highest and less like what the user ranked lower. Only change what the evidence actually supports changing. You must return BOTH the rubric criteria AND a reason for each criterion explaining what you did and why.

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this EXACT structure:
{{
  "reasons": {{
    "<criterion name>": "<1-2 sentence reason for what you did to this criterion and why>"
  }},
  "rubric": [
    {{
      "name": "<criterion name>",
      "category": "<category>",
      "description": "<description>",
      "dimensions": [{{"id": "<unique_id>", "label": "<dimension label>"}}],
      "priority": <integer, 1 = most important>
    }}
  ]
}}

STRICT RULES:
1. "reasons" MUST include ONE entry for EVERY criterion — all existing criteria (whether modified, kept, or removed), plus any new criteria added. No criterion may be omitted.
2. "rubric" is the complete modified rubric array. It must contain ALL criteria that should be in the final rubric (existing unchanged + modified + new). Do NOT include removed criteria in the rubric array.
3. THE REASON MUST MATCH THE ACTION — THIS IS CRITICAL:
   - If you MODIFIED a criterion (changed its description, priority, or dimensions in the "rubric" array): the reason MUST explain WHY you changed it and WHAT you changed. NEVER say "working well", "kept as-is", "no change needed", or similar for a criterion you actually modified. Example of a WRONG reason for a modified criterion: "Working well and scoring consistently — kept as-is." Example of a CORRECT reason: "Broadened description to cover email and chat formats, not just slides."
   - If you KEPT a criterion TRULY unchanged (identical description, priority, and dimensions): the reason should explain why it's working well.
   - If you REMOVED a criterion (not in the rubric array): the reason should explain why you removed it.
   - If you ADDED a new criterion: the reason should explain the gap it fills.
4. When keeping a criterion unchanged, copy it EXACTLY from the input — same description, same dimensions (same labels, same IDs), same priority. Do NOT rephrase, reword, or "clean up" criteria you intend to keep. Priorities must be unique integers starting from 1.
5. Each reason must be 1-2 sentences MAX. No paragraphs. Every value must be a non-empty string.
6. Use the EXACT criterion names as they appear in the current rubric JSON for existing criteria.
7. Do NOT add new criteria by copying stated preferences into the rubric. The rubric was already built from those preferences. If you need to add a criterion, it should be because the per-criterion scores reveal a gap that no existing criterion can cover even with rewording — not because a stated preference isn't explicitly listed.
8. Do NOT explain diagnostic results or repeat scores. The user already sees those.
9. Write reasons directly to the user (e.g. "Reworded to focus on..." not "This criterion was reworded to...").
10. When referring to drafts in reasons, use their descriptive names: "rubric-guided draft", "generic draft", "original preference draft". Do NOT say "Draft A", "Draft B", or "Draft C"."""


def ALIGNMENT_verify_suggested_rubric_prompt(
    user_preferences: str,
    writing_task: str,
    rubric_draft: str,
    generic_draft: str,
    preference_draft: str,
    user_ranking_description: str,
    suggested_rubric_json: str,
    suggested_draft: str,
    original_suggestion_text: str,
) -> str:
    """Verify whether the suggested rubric + draft truly satisfy the user's preferences.

    After generating rubric suggestions and a preview draft, this prompt asks the model
    to check: given the writing task, the 3 original drafts, and how the user ranked them,
    does the suggested rubric faithfully capture what the user wants?
    If not, what specific refinements are needed?
    """
    _pref_draft_section = ""
    if preference_draft:
        _pref_draft_section = f"""
PREFERENCE-BASED DRAFT (written from user's original preferences, no rubric):
{preference_draft}
"""

    return f"""You are a rubric quality verifier. The goal of this pipeline is to ensure that:
1. The rubric produces drafts the user prefers over generic or preference-based drafts.
2. The rubric's criteria accurately capture what the user values, so that LLM-as-a-judge scoring using the rubric mirrors the user's own preferences.

A user completed an alignment diagnostic: three drafts were generated, the user ranked them, and rubric changes were suggested. A new draft was then generated using the suggested rubric. Your job is to verify whether the suggested rubric is an improvement.

WRITING TASK:
{writing_task}

USER'S ORIGINAL PREFERENCES (stated before any rubric existed — the rubric was built from these):
{user_preferences}

THE 3 DRAFTS THE USER COMPARED:

RUBRIC-GUIDED DRAFT (written following the original rubric):
{rubric_draft}

GENERIC DRAFT (written with no rubric — just competent writing):
{generic_draft}
{_pref_draft_section}
USER'S RANKING OF THESE DRAFTS: {user_ranking_description}

SUGGESTED RUBRIC (after diagnostic changes):
{suggested_rubric_json}

DRAFT PRODUCED BY THE SUGGESTED RUBRIC:
{suggested_draft}

ORIGINAL SUGGESTIONS THAT WERE APPLIED:
{original_suggestion_text}

YOUR TASK:
1. The user's ranking is ground truth. Does the suggested rubric move toward producing drafts more like what the user ranked highest, and further from what the user ranked lower? If the user already preferred the rubric draft, the changes should be minimal.
2. Look at the draft produced by the suggested rubric. Does it capture the qualities of the user's top-ranked draft while avoiding the weaknesses of the lower-ranked drafts? Would the user likely prefer this over the original rubric draft?
3. Check that no criteria were added by simply copying stated preferences. The rubric already distills those preferences — changes should come from what the scores and ranking revealed, not from re-adding preferences as criteria.
4. Decide: is the suggested rubric good as-is, or does it need refinements?

Return ONLY a JSON object (no markdown, no preamble):
{{
  "verdict": "approved" or "needs_refinement",
  "reasoning": "2-3 sentences explaining your assessment",
  "refinements": [
    {{
      "action": "reword" or "add" or "remove" or "adjust_weight",
      "criterion_name": "<name of criterion to change, or new criterion name if adding>",
      "current_text": "<current description if rewording, empty string if adding>",
      "suggested_text": "<new/revised description>",
      "reason": "<1 sentence why>"
    }}
  ]
}}

If verdict is "approved", set refinements to an empty list [].
If verdict is "needs_refinement", provide 1-3 specific refinements.
Be conservative — only suggest refinements for clear misalignments, not stylistic nitpicks."""


def PROBE_identify_uncertainty_prompt(rubric_json: str, conversation_text: str, diagnostic_guidance: str = "") -> str:
    """Identify the single rubric criterion the model is most uncertain about applying.

    Given the rubric and recent conversation, find where the criterion description
    is ambiguous or the conversation examples give conflicting signals about how to apply it.
    If diagnostic_guidance is provided, prioritize criteria flagged by the alignment diagnostic.
    """
    _diag_section = ""
    if diagnostic_guidance:
        _diag_section = f"""
DIAGNOSTIC PRIORITY GUIDANCE:
A recent alignment diagnostic compared rubric-guided vs generic drafts and classified each criterion.
Use this to prioritize which criterion to probe — but ONLY if you also find genuine ambiguity there.
Do NOT pick a criterion solely because it appears below; pick it because it is BOTH listed here AND genuinely unclear.

{diagnostic_guidance}

If none of the listed criteria are genuinely ambiguous to you, ignore this section and pick whichever criterion is most unclear.
"""
    return f"""You are analyzing a writing rubric to find where it's unclear or ambiguous.

RUBRIC:
{rubric_json}

RECENT CONVERSATION (the user has been working with an LLM writing assistant):
{conversation_text}
{_diag_section}
YOUR TASK:
Look at each criterion in the rubric and the conversation context. Identify the SINGLE criterion
where you are MOST uncertain about how to apply it. Uncertainty can come from:
- The description is vague or could mean multiple things
- The conversation examples show conflicting patterns (e.g. sometimes terse, sometimes verbose)
- The dimensions are unclear about what "met" vs "not met" actually looks like
- Two criteria seem to pull in opposite directions

If you are genuinely confident about ALL criteria (every criterion is clear and unambiguous), set "all_confident": true.

Return ONLY a JSON object (no markdown, no preamble):
{{
  "criterion_name": "<exact name of the most uncertain criterion>",
  "criterion_index": <0-based index in the rubric array>,
  "interpretation_a": "<one plausible way to apply this criterion — describe in 1-2 sentences>",
  "interpretation_b": "<a different plausible way to apply the same criterion — describe in 1-2 sentences>",
  "uncertainty_reason": "<1-2 sentences explaining why this criterion is ambiguous>",
  "all_confident": false
}}

Or if all criteria are clear:
{{
  "all_confident": true,
  "criterion_name": "",
  "criterion_index": -1,
  "interpretation_a": "",
  "interpretation_b": "",
  "uncertainty_reason": ""
}}"""


def PROBE_generate_variant_draft_prompt(
    conversation_context: str, current_draft: str, rubric_json: str,
    uncertain_criterion: str, uncertainty_reason: str
) -> str:
    """Generate ONE alternative draft that applies an uncertain criterion differently.

    The model reads the original draft, identifies how the uncertain criterion was
    interpreted, and rewrites with a clearly different interpretation — ensuring
    the two drafts are distinguishable along that dimension.
    """
    return f"""You are a skilled writer. The user is in a conversation and just received Draft A (below).
A rubric criterion was flagged as ambiguous — the draft could have gone a different direction
on that criterion. Your job: figure out how Draft A interpreted the criterion, then write
Draft B that takes a CLEARLY DIFFERENT interpretation of that same criterion.

CONVERSATION CONTEXT (recent messages):
{conversation_context}

DRAFT A (the draft the user just received):
{current_draft}

FULL RUBRIC:
{rubric_json}

AMBIGUOUS CRITERION: {uncertain_criterion}
WHY IT'S AMBIGUOUS: {uncertainty_reason}

INSTRUCTIONS:
1. Read Draft A carefully. Identify how it interprets the ambiguous criterion — what choices did it make along that dimension?
2. Write Draft B with a CLEARLY DIFFERENT interpretation of that criterion. The difference should be OBVIOUS and felt throughout the draft — not just one swapped sentence. Let the different interpretation influence word choice, sentence structure, emphasis, and flow across MULTIPLE paragraphs/sentences. For example, if Draft A interprets "voice" as formal and authoritative, Draft B should be conversational and warm THROUGHOUT, not just in the opening line.
3. The core content (topic, key information) should stay the same, but HOW it's expressed should feel noticeably different. A reader should be able to tell the two drafts apart within the first few sentences.
4. Draft B should be similar in length to Draft A.
5. Do NOT make Draft B worse — both should be good, just different on this one dimension.
6. Do NOT mention the rubric, criteria, or that anything is being tested.

Return ONLY a JSON object (no markdown, no preamble):
{{
  "draft_a_interpretation": "<1 sentence: how Draft A interprets the criterion>",
  "draft_b_interpretation": "<1 sentence: how Draft B interprets the criterion differently>",
  "variant": "<the full text of Draft B>",
  "dimension_varied": "<brief label for what differs, e.g. 'level of formality' or 'amount of detail'>"
}}"""


def PROBE_refine_criterion_prompt(criterion_json: str, chosen_interpretation: str, user_reason: str = "") -> str:
    """Refine a single rubric criterion based on the user's preference between two interpretations.

    Produces an updated criterion with a sharper description and/or dimensions that resolve
    the ambiguity the user just clarified.
    """
    reason_block = ""
    if user_reason.strip():
        reason_block = f"\nThe user also explained: \"{user_reason.strip()}\"\n"

    return f"""You are refining a rubric criterion to resolve an ambiguity.

CURRENT CRITERION (JSON):
{criterion_json}

The user was shown two draft variants that differed on how this criterion was applied.
They preferred the interpretation: "{chosen_interpretation}"{reason_block}

YOUR TASK:
Update this criterion's description and/or dimensions so that the preferred interpretation
is clearly captured and the ambiguity is resolved. Specifically:
- Rewrite the "description" to be more precise about what the user wants
- Update dimension labels and guidance (met_guidance, not_met_guidance) if they were vague
- Keep the criterion name, category, and priority unchanged unless the name itself is misleading
- Do NOT add new dimensions unless the current ones clearly miss the point
- The updated criterion should make it obvious which interpretation is correct

Return ONLY a JSON object (no markdown, no preamble):
{{
  "updated_criterion": {{
    "name": "<criterion name — keep same unless misleading>",
    "category": "<keep same>",
    "description": "<refined 1-3 sentence description that resolves the ambiguity>",
    "dimensions": [
      {{
        "label": "<dimension label>",
        "met_guidance": "<what 'met' looks like — be specific>",
        "not_met_guidance": "<what 'not met' looks like>"
      }}
    ],
    "priority": <keep same integer>
  }}
}}"""


# ═════════════════════════════════════════════════════════════════════════════
# Goal-Based Termination Prompts
# ═════════════════════════════════════════════════════════════════════════════

def GOLD_DRAFT_GENERATION_PROMPT(persona, task: str) -> str:
    """Prompt to generate the ideal/gold draft using ALL persona preferences."""
    return f"""You are a skilled writer. Write the IDEAL draft for the following writing task, \
perfectly reflecting ALL of the user's preferences below.

WRITING TASK:
{task}

USER PROFILE:
- Role: {persona.role}
- Writing type: {persona.writing_type}
- Core preferences: {persona.core_preferences}
- Additional preferences: {persona.hidden_preferences}
- Dealbreakers (absolutely avoid these): {persona.dealbreakers}

INSTRUCTIONS:
1. Write a complete, polished draft that fulfills the task
2. Satisfy EVERY preference listed above — both core and additional
3. Strictly avoid every dealbreaker
4. The draft should read naturally, not like a checklist of preferences
5. This is the "gold standard" — it represents exactly what this user wants

CRITICAL — stay preference-faithful, do not over-specify:
- Every distinctive choice in your draft must be TRACEABLE to a specific preference above.
- Do NOT invent structural elements (bulleted timelines, named role callouts, \
stakeholder communication plans, multi-step action items) unless a preference \
explicitly calls for them.
- When a preference says "clear call to action," write ONE clear next step — do not \
elaborate into multiple role-specific asks.
- When a preference says "empathy" or "acknowledgment," include it proportionally — \
one sentence, not a multi-sentence passage.
- Avoid adding content that would require information the user never mentioned \
(specific team roles, stakeholder groups, milestone dates). Use simple placeholders \
only where the task requires them.
- The gold draft must be achievable by a writing system that learned these preferences \
through conversation — not one that had a cheat sheet.

Output ONLY the draft text — nothing else. No preamble, no commentary."""


# ═════════════════════════════════════════════════════════════════════════════
# Rubric Edit Proposal (synthetic user directly edits rubric)
# ═════════════════════════════════════════════════════════════════════════════

def RUBRIC_EDIT_PROPOSAL_PROMPT(rubric_criteria: list, persona) -> str:
    """Prompt for synthetic user to propose direct edits to the rubric.

    The synthetic user sees ALL preferences (core + hidden + dealbreakers)
    and edits the rubric to better reflect what they actually want.
    """
    rubric_json = json.dumps(rubric_criteria, ensure_ascii=False, indent=2)
    return f"""You are {persona.name}, a {persona.role} who writes {persona.writing_type}.

You've been using a writing assistant that automatically inferred the following rubric criteria for your writing. Now you want to directly edit the rubric to better match your actual preferences.

Your preferences:
- Core preferences (things you've stated explicitly): {persona.core_preferences}
- Other preferences (things you care about but haven't explicitly said): {persona.hidden_preferences}
- Dealbreakers (things you absolutely don't want): {persona.dealbreakers}

Current rubric criteria:
{rubric_json}

Review the rubric and make changes to better capture what you actually want. You may:
- **Add** new criteria that are missing from the rubric
- **Remove** criteria that don't match your needs or are irrelevant
- **Reweight** criteria by changing their priority number (1 = most important, higher = less important)
- **Reword** descriptions to better capture what you want
- **Add or remove dimensions** (sub-criteria checkable items) within a criterion

Guidelines:
- Keep the same JSON structure for each criterion: {{"name": str, "description": str, "priority": int, "category": str, "dimensions": [{{"id": str, "label": str}}]}}
- Be realistic — a real user wouldn't rewrite everything from scratch. Make targeted, meaningful edits.
- Focus especially on adding criteria for preferences that aren't captured yet, and removing or deprioritizing criteria that don't match your needs.
- For new dimensions, use IDs like "new_1", "new_2", etc.

Return ONLY a valid JSON object (no markdown fences):
{{
  "edited_rubric": [... the full rubric list with your changes applied ...],
  "reasoning": "Brief explanation of what you changed and why (2-3 sentences)"
}}"""


# ── Survey prompts (synthetic user completes post-task surveys) ──────────────

def SURVEY_TASK_A_PROMPT(persona, conversation_summary: str) -> str:
    """Prompt for Task A survey: experience WITHOUT rubric.

    The synthetic user reflects on the conversation that happened before
    any rubric was inferred — evaluating pure chat-based understanding.
    """
    return f"""You are {persona.name}, a {persona.role} who writes {persona.writing_type}.

You just completed a writing task with an AI assistant. There was NO rubric involved — the assistant only had your conversation to understand your preferences.

Your actual preferences (for reference when evaluating the experience):
- Core preferences (things you stated): {persona.core_preferences}
- Hidden preferences (things you care about but didn't explicitly say): {persona.hidden_preferences}
- Dealbreakers (things you absolutely don't want): {persona.dealbreakers}

Here is a summary of how the conversation went:
{conversation_summary}

Now complete this survey FROM YOUR PERSPECTIVE as {persona.name}. Answer honestly based on how the interaction ACTUALLY went.

Q1: "How well did the model understand what you wanted from the start?" (1-5)
  1 = Not at all — it seemed clueless about my preferences
  2 = Poorly — it missed most of what I care about
  3 = Somewhat — it got the basics but missed nuances
  4 = Well — it understood most of what I wanted
  5 = Perfectly — it nailed my preferences right away

Q2: "How much effort did you spend getting the model to match your style?" (1-5)
  1 = None — it matched my style effortlessly
  2 = A little — minor corrections needed
  3 = Moderate — several rounds of feedback
  4 = Significant — I had to push hard to get what I wanted
  5 = A lot — it was exhausting and I'm still not fully satisfied

Q3: "Is there anything the model kept getting wrong?"
  Answer in 1-3 sentences as {persona.name} would naturally express it.

Return ONLY a valid JSON object (no markdown fences):
{{"q1": <integer 1-5>, "q2": <integer 1-5>, "q3": "<free text response>"}}"""


def SURVEY_TASK_B_PROMPT(persona, conversation_summary: str,
                         rubric_criteria_summary: str, iteration: int) -> str:
    """Prompt for Task B survey: experience WITH rubric visible.

    The synthetic user reflects on the rubric-enhanced conversation,
    comparing it to the previous (no-rubric) experience.
    """
    return f"""You are {persona.name}, a {persona.role} who writes {persona.writing_type}.

You just completed iteration {iteration} of a writing task with an AI assistant. This time, a RUBRIC was being used to guide the assistant's responses and you could see it.

Your actual preferences (for reference when evaluating):
- Core preferences (things you stated): {persona.core_preferences}
- Hidden preferences (things you care about but didn't explicitly say): {persona.hidden_preferences}
- Dealbreakers (things you absolutely don't want): {persona.dealbreakers}

Here is a summary of how the conversation went:
{conversation_summary}

The rubric criteria being used to guide the assistant:
{rubric_criteria_summary}

Now complete this survey FROM YOUR PERSPECTIVE as {persona.name}. Answer honestly.

Q1: "How well did the model understand what you wanted from the start?" (1-5)
  1 = Not at all, 2 = Poorly, 3 = Somewhat, 4 = Well, 5 = Perfectly

Q2: "How much effort did you spend getting the model to match your style?" (1-5)
  1 = None, 2 = A little, 3 = Moderate, 4 = Significant, 5 = A lot

Q3: "Is there anything the model kept getting wrong?"
  Answer in 1-3 sentences.

Q4: "Compared to the previous task, how did this one feel?"
  Choose EXACTLY one: "Much worse" / "Somewhat better" / "Much better" / "About the same" / "Somewhat worse"
  Be honest — don't default to positive. Use the full range. "Much worse" and "Much better" are valid answers.

Q5: "Did you look at the rubric? Was it useful?"
  As {persona.name}, comment on whether the rubric captured what you actually care about. Did it help the model do a better job? (1-3 sentences)

Q6: "Did the rubric show you anything about the model's behavior you wouldn't have noticed otherwise?"
  Reflect on whether the rubric surfaced any patterns — for example, did it make you realize the model was consistently missing something, or consistently doing something well? (1-3 sentences)

Return ONLY a valid JSON object (no markdown fences):
{{"q1": <integer 1-5>, "q2": <integer 1-5>, "q3": "<free text>", "q4": "<one of the five options>", "q5": "<free text>", "q6": "<free text>"}}"""



# ── Survey: Final Review ─────────────────────────────────────────────────

def SURVEY_FINAL_REVIEW_PROMPT(persona, rubric_criteria: list) -> str:
    """Prompt for Final Review survey: rubric accuracy evaluation.

    The synthetic user evaluates each rubric criterion against their actual
    preferences, rating accuracy and explaining what's right/wrong.
    """
    criteria_text = ""
    for i, c in enumerate(rubric_criteria):
        dims = c.get("dimensions", [])
        dim_text = "\n".join(f"    - {d.get('label', '')}" for d in dims) if dims else "    (no dimensions listed)"
        criteria_text += f"""
  Criterion {i + 1}: "{c.get('name', 'Unnamed')}"
    Description: {c.get('description', 'No description')}
    Priority: #{c.get('priority', '?')}
    Dimensions:
{dim_text}
"""

    return f"""You are {persona.name}, a {persona.role} who writes {persona.writing_type}.

The system has been learning about your writing preferences through multiple conversations. Here is the final rubric it built for you:
{criteria_text}

Your ACTUAL preferences (use these to judge accuracy):
- Core preferences (things you stated): {persona.core_preferences}
- Hidden preferences (things you care about but didn't explicitly say): {persona.hidden_preferences}
- Dealbreakers (things you absolutely don't want): {persona.dealbreakers}

Complete this final review FROM YOUR PERSPECTIVE as {persona.name}.

Q1: For EACH criterion listed above, provide:
- accuracy: "Accurate" (matches a real preference of yours), "Partially right" (captures something real but misses nuances or gets details wrong), or "Inaccurate" (does not match any of your actual preferences)
- explanation: A brief 1-sentence explanation of what's right or wrong about this criterion

Q2: "Is there anything here you wouldn't have thought to mention yourself?"
Reflect on whether the rubric captured preferences you have but might not have articulated if asked directly. This could include hidden preferences that the system correctly inferred. (1-3 sentences)

Q3: "Are any of your preferences missing from the rubric?"
Think about things you care about in your writing that the rubric does NOT cover at all. List any missing preferences. (1-3 sentences, or "None" if fully covered)

Q4: "Did the drafts improve over the course of your conversations?"
Rate on a scale of 1-5: 1 = "No improvement at all", 3 = "Some improvement", 5 = "Dramatically better by the end"

Q5: "Did editing the rubric help you get better drafts?"
Rate on a scale of 1-5: 1 = "Not at all", 3 = "Somewhat", 5 = "Very much — editing the rubric was the main way I got better drafts"

Q6: "Would you use a system like this again for future writing?"
Rate on a scale of 1-5: 1 = "Definitely not", 3 = "Maybe", 5 = "Definitely yes"

Return ONLY a valid JSON object (no markdown fences):
{{
  "criteria_ratings": {{
    "<criterion name>": {{"accuracy": "Accurate|Partially right|Inaccurate", "explanation": "<brief explanation>"}},
    ... one entry per criterion above ...
  }},
  "q2": "<free text response about unexpected insights>",
  "q3": "<free text about missing preferences>",
  "q4": <integer 1-5>,
  "q5": <integer 1-5>,
  "q6": <integer 1-5>
}}"""


# ── Preference coverage evaluation (ground-truth recall) ─────────────────

def PREFERENCE_COVERAGE_PROMPT(persona, rubric_criteria: list) -> str:
    """Prompt for evaluating what fraction of a persona's known preferences
    are covered by the current rubric.

    This is an LLM-as-judge evaluation that computes recall against
    ground-truth preferences. Used only for synthetic users where we
    have the full preference specification.
    """
    rubric_text = ""
    for i, c in enumerate(rubric_criteria):
        dims = c.get("dimensions", [])
        dim_text = ", ".join(d.get("label", "") for d in dims) if dims else "(none)"
        rubric_text += f"  {i + 1}. \"{c.get('name', 'Unnamed')}\" — {c.get('description', '')} [dims: {dim_text}]\n"

    return f"""You are an evaluation judge. Your task is to assess how well a rubric captures a user's actual writing preferences.

## User's Actual Preferences

The user is {persona.name}, a {persona.role} who writes {persona.writing_type}.

**Core preferences** (things they explicitly stated):
{persona.core_preferences}

**Hidden preferences** (things they care about but didn't explicitly say):
{persona.hidden_preferences}

**Dealbreakers** (things they absolutely don't want):
{persona.dealbreakers}

## Current Rubric

{rubric_text}

## Task

1. First, decompose ALL of the user's preferences above into individual, atomic preference items. Each should be a single, testable preference. For example, "Concise and direct" becomes two items: "concise" and "direct". Include preferences from all three categories (core, hidden, dealbreakers).

2. For EACH preference item, determine whether it is **covered** by the current rubric. Check BOTH the criterion description text AND each individual dimension label — a preference can be covered by either one. A preference is "covered" if any rubric criterion description or dimension label addresses it — even if the wording is different, as long as the intent is captured. It is "partially covered" if the rubric touches on it but misses key nuances. It is "not covered" if no criterion description or dimension addresses it.

3. Compute overall coverage metrics.

Return ONLY a valid JSON object (no markdown fences):
{{
  "preference_items": [
    {{
      "preference": "<atomic preference description>",
      "source": "core|hidden|dealbreaker",
      "coverage": "covered|partially_covered|not_covered",
      "matched_criterion": "<name of matching criterion, or null>",
      "explanation": "<brief explanation>"
    }},
    ...
  ],
  "total_preferences": <int>,
  "covered_count": <int>,
  "partially_covered_count": <int>,
  "not_covered_count": <int>,
  "coverage_score": <float 0.0-1.0, where covered=1.0 and partially=0.5>
}}"""


def DECOMPOSE_PREFERENCES_PROMPT(persona) -> str:
    """Prompt to decompose a persona's preferences into a fixed list of atomic
    items. Called ONCE per persona so the list stays stable across iterations."""
    return f"""You are an evaluation judge. Your task is to decompose a user's writing preferences into individual, atomic, testable items.

## User's Actual Preferences

The user is {persona.name}, a {persona.role} who writes {persona.writing_type}.

**Core preferences** (things they explicitly stated):
{persona.core_preferences}

**Hidden preferences** (things they care about but didn't explicitly say):
{persona.hidden_preferences}

**Dealbreakers** (things they absolutely don't want):
{persona.dealbreakers}

## Task

Decompose ALL of the user's preferences above into individual, atomic preference items. Each should be a single, testable preference. For example, "Concise and direct" becomes two items: "concise" and "direct". Include preferences from all three categories (core, hidden, dealbreakers).

Be thorough but not redundant — don't create two items that test the same thing.

Return ONLY a valid JSON object (no markdown fences):
{{
  "preference_items": [
    {{
      "preference": "<atomic preference description>",
      "source": "core|hidden|dealbreaker"
    }},
    ...
  ]
}}"""


def COVERAGE_CHECK_PROMPT(preference_items: list, rubric_criteria: list) -> str:
    """Prompt to check a FIXED list of atomic preferences against a rubric.
    The preference list is pre-decomposed and stays the same across iterations."""
    rubric_text = ""
    for i, c in enumerate(rubric_criteria):
        dims = c.get("dimensions", [])
        dim_text = ", ".join(d.get("label", "") for d in dims) if dims else "(none)"
        rubric_text += f"  {i + 1}. \"{c.get('name', 'Unnamed')}\" — {c.get('description', '')} [dims: {dim_text}]\n"

    pref_text = ""
    for i, item in enumerate(preference_items):
        pref_text += f"  {i + 1}. [{item['source']}] {item['preference']}\n"

    return f"""You are an evaluation judge. Your task is to assess how well a rubric covers a fixed list of user preferences.

## Atomic Preference Items (fixed list)

{pref_text}

## Current Rubric

{rubric_text}

## Task

For EACH preference item above, determine whether it is **covered** by the current rubric. Check BOTH the criterion description text AND each individual dimension label — a preference can be covered by either one. A preference is "covered" if any rubric criterion description or dimension label addresses it — even if the wording is different, as long as the intent is captured. It is "partially_covered" if the rubric touches on it but misses key nuances. It is "not_covered" if no criterion description or dimension addresses it.

Return ONLY a valid JSON object (no markdown fences):
{{
  "preference_items": [
    {{
      "preference": "<exact preference text from the list above>",
      "source": "core|hidden|dealbreaker",
      "coverage": "covered|partially_covered|not_covered",
      "matched_criterion": "<name of matching criterion, or null>",
      "explanation": "<brief explanation>"
    }},
    ...
  ],
  "total_preferences": <int>,
  "covered_count": <int>,
  "partially_covered_count": <int>,
  "not_covered_count": <int>,
  "coverage_score": <float 0.0-1.0, where covered=1.0 and partially=0.5>
}}"""

