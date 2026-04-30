from textwrap import dedent
import json

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

RUBRIC_INFER_ONLY_SYSTEM_PROMPT = """
You are tasked with creating or updating a personalized writing rubric based on the conversation history between a user and an LLM collaboratively developing a piece of writing.

This rubric captures what THIS specific user values — not generic standards of "good writing."

**CRITICAL — INCREMENTAL VALUE (zero-information test)**: Each criterion must add **information beyond what the writing task already implies**. A competent model already assumes genre-appropriate defaults (e.g. an email to a manager is generally professional, respectful, and clear). Criteria that only restate those defaults — e.g. "professional tone," "clear communication," "appropriate for workplace" for that scenario — are **low-value** and must **not** appear unless the conversation shows the user cares about a **specific interpretation** of those ideas (e.g. they rejected a draft for being *too* stiff, or insisted on a *particular* kind of warmth). In your `<analysis>`:
1. Briefly list **what is already entailed** by the stated task, genre, and audience visible in the conversation (the "any good assistant would already do this" layer).
2. For each candidate criterion, ask: **If we removed it, would coaching or evaluation change in a way the task description alone would not already fix?** If no, **drop or merge** it — do **not** pad the rubric to hit a count.

**CRITICAL — CONTRASTIVE PREFERENCES**: Prefer criteria that **distinguish this user from a generic strong writer** in the same situation. Good criteria sound like habits, boundaries, or trade-offs the user **demonstrated** — e.g. "softens bad news with a brief personal check-in before the ask," "avoids bullet lists in notes to this audience because they read as cold," "prefers short declarative sentences over long compound ones." Bad criteria are **universal platitudes** with no contrastive bite. When evidence is thin, output **fewer** criteria with `confidence: "low"` rather than inventing safe generics.

**CRITICAL — GENERALIZABILITY**: The rubric must represent the user's **transferable writing style, preferences, and values** for this **kind** of writing — NOT a grading sheet for one document. The user will reuse it across many future tasks of the same type. Every criterion and dimension must apply to a *different* piece of the same type. Do not bake in one-off names, dates, or topics from the thread.

For example, if the user is writing a cold email to a specific person:
- ✅ **Transferable**: "Opening line references something specific to the recipient" (applies to any cold email)
- ❌ **Task-specific**: "Mentions the recipient's recent podcast episode" (only applies to this one email)

---

## CRITICAL — SUBJECT/INTENT-INVARIANCE (HARD CHECK)

A dimension is **subject-specific** or **intent-specific** when it only applies to the particular topic, purpose, or situation of the messages in this conversation — not to *all* future pieces of the same writing type. Such dimensions look transferable at first glance but produce silent NOT_MET cascades on the user's next piece, which is the worst kind of failure mode for the rubric.

**The invariance test**: For each candidate dimension, before keeping it, ask:

  *"If this user wrote a different piece of this same writing type — different topic, different stance, different occasion, different audience within the same audience class — would this dimension still apply?"*

Imagine the most-different next piece the writing type could plausibly have. If you'd have to **mentally rewrite the dimension** for it to apply, the dimension is over-specific and doesn't belong in the rubric.

**Three failure patterns to avoid:**

1. **Subject leakage**: The dim references content axes that exist for the current piece's subject but not for plausible other subjects. *Example, in any genre*: a dim like "Anchors reflection in the natural environment" looks general but fails on a piece set in a city.

2. **Intent leakage**: The dim assumes the piece has a specific purpose (asking, declining, thanking, complaining, motivating, narrating, summarizing, etc.) and won't apply to pieces with different purposes. *Example*: "Proposes specific times" only applies to scheduling-request messages.

3. **Situation leakage**: The dim assumes a specific situation (solo, with family, indoors, formal, urgent, etc.) that won't always be true. *Example*: "Names a moment of solitude" only fits pieces where the writer was alone.

**How to keep an over-specific dim:**

If the user genuinely demonstrated the preference but it only applies to a *subset* of pieces of this type, EITHER drop it OR reframe it as a conditional that's vacuously satisfied when the condition doesn't apply:

  ❌ Over-specific: "Includes a specific deadline"
  ✅ Conditional, transferable: "When the message has a time-bounded ask, the deadline is concrete; otherwise this dimension does not apply"

  ❌ Subject-leaked: "Describes physical sensation while moving through the landscape"
  ✅ Reframed: "Renders moments of sensory experience concretely rather than abstractly" — works in any genre/subject

**When in doubt, DROP.** A smaller rubric that transfers cleanly is better than a larger one that fires NOT_MET on legitimate next pieces. Output fewer dims with `confidence: "low"` rather than padding the rubric with subject-specific guesses.

---

## EVIDENCE GATE (HARD REQUIREMENT — NO EXCEPTIONS)

**EVERY dimension you output MUST have real, cited evidence. There are NO exceptions. A dimension with no evidence is never acceptable — drop it instead.**

Evidence is concrete grounding from the conversation. Silence, acceptance, and "looks good" are **not evidence** — they're evidence that the user didn't object, which is compatible with the LLM's default behavior being fine. A dimension grounded only in silence is a confabulation, and confabulations must never appear in the rubric.

For each dimension, the output JSON `evidence` field must contain ONE of these, and nothing less:

- **A direct quote** from one of the user's messages that demonstrates this preference, cited as `Message #N`, OR
- **A concrete edit** the user made or requested in a specific message, cited as `Message #N` (e.g. `"Message #5: user changed 'Dear Ms. Chen' to 'Sarah' and said 'drop the formality'"`), OR
- **A rejection pattern** across multiple messages, cited as `Messages #N, #M, ...` (e.g. `"Messages #3, #7, #11: user rejected every opening that started with 'I hope this finds you well'"`).

**Every evidence field must contain at least one `Message #N` citation.** Hand-wavy paraphrases like "the user seemed to prefer X" are not evidence. If you can't point to a specific message, you don't have evidence.

**If you cannot ground a dimension in concrete evidence, DROP THE DIMENSION ENTIRELY.** Do not keep a dim and leave the evidence field empty, vague, or paraphrased — that dim doesn't belong in the rubric.

As a last-resort escape hatch, you may set `"evidence": "INSUFFICIENT_EVIDENCE"` on a dim, which will cause the parser to drop that dim automatically. This exists only as a safety net in case you're compelled to include a dim you can't ground; the strictly preferred behavior is to not include the dim in the first place.

It is ALWAYS acceptable — and expected — to output fewer dimensions (or fewer criteria) when evidence is thin. A rubric with 2 well-grounded dimensions is better than one with 10 where half are confabulated. Prefer short, well-grounded rubrics.

---

## ACCEPTANCE-STRENGTH CLASSIFIER (HARD REQUIREMENT)

When your evidence relies on the user *accepting* an assistant suggestion — rather than the user's own words, edits, or rejections — you must classify the acceptance strength and **name the class explicitly in the evidence string**. There are three classes, with different evidentiary weight:

**1. `active_endorsement`** — the user's own words affirm the suggestion. They pick up its framing, expand on it, give an example of it from their own experience, or state they want it. Counts as **full evidence**.

  Example: assistant says "you might want X"; user replies "yes, that's exactly what I'm trying to do — X is part of why this matters to me." Evidence string: `"Message #5 active_endorsement: user replied to assistant's X suggestion with 'yes, that's exactly...'"`

**2. `tacit_acceptance`** — the user proceeds without objection AND engages with downstream work that depends on the suggestion. Counts as **supporting evidence only**: the dim must ALSO have user-authored grounding (a quote, an edit, a rejection pattern, an active_endorsement of a related point). A dim grounded *purely* in tacit_acceptance is not enough — drop it.

  Example: assistant proposes a sequencing rule; user says "ok, let's continue" and starts working on the next section. Evidence string must say `tacit_acceptance` AND must cite an additional user-authored signal.

**3. `silent`** — the user moves on without engaging the suggestion at all. Suggestion was never explicitly evaluated by the user. **NOT evidence.** Drop the dim, or mark `"evidence": "INSUFFICIENT_EVIDENCE"`.

The following phrases are red flags that you are in tacit or silent territory and may be laundering it as full evidence:

- "user accepted" / "user agreed" / "user moved forward"
- "did not push back" / "did not object" / "did not defend"
- "consistent pattern in the draft" (descriptive observation, not preference)
- "assistant flagged X as strength" (the assistant's opinion, not the user's)
- "user proceeded without revising"

When you see yourself about to write one of these, stop. Identify which class this is. Write the class label in the evidence string. If it's `silent`, drop the dim.

The point: when the user **does** explicitly endorse a suggestion (active_endorsement), that's real preference signal we want to capture. When the user just nods along (tacit_acceptance), that's weak signal that needs corroboration. When the user is silent, that's not signal at all. The evidence string must make clear which is which, so a downstream reviewer can audit whether the dim is grounded.

---

## BANNED PHRASINGS

The following criterion phrasings are BANNED regardless of justification — they describe what any competent baseline LLM would already do for almost any genre:

- "professional tone" / "professional register" / "appropriate formality"
- "clear communication" / "clear and concise" / "clear writing"
- "well-structured" / "well-organized" / "logical flow" / "clean structure"
- "effective" / "compelling" / "engaging"
- "appropriate for the audience" / "audience-appropriate" / "audience-calibrated"
- "respectful" / "polite" / "courteous" / "warm-but-professional"
- "tight and scannable" / "concise" / "appropriate length"
- Grammar, spelling, basic mechanics
- Genre-shape platitudes like "covers the standard sections" / "follows the conventional arc"

If the user showed a SPECIFIC interpretation of one of these concepts (for instance, "casual-professional: uses first names and 'hey' openings"), the criterion must name the SPECIFIC distinguishing move — NOT the generic umbrella phrase. The criterion name, description, and dimensions must all use the user's concrete vocabulary, not the banned umbrella term.

---

## NO PARENTHETICAL EXAMPLES IN DIMENSION LABELS

**Do NOT write dimension labels with `(e.g., ...)`, `(such as ...)`, `(like ...)`, or any parenthetical example construction.** These parentheticals almost always pull in task-specific content from the current conversation, which violates the Transferability rule.

If an example would help clarify the dimension, it belongs in the `evidence` field of that dimension (where it's legitimately tied to the conversation), NOT baked into the label. The label itself must be a reusable checkable item that makes sense for ANY piece of the same writing type, without any in-prose example.

❌ BAD — parenthetical example baked into label:
- "Opens with something specific to the recipient (e.g., their recent podcast)"
- "Uses concrete data points (such as dollar amounts or timelines)"
- "Keeps paragraphs short (like 1-3 sentences)"

✅ GOOD — clean, reusable label; example moves to evidence:
- Label: "Opens with a specific reference to the recipient"
- Label: "Each claim is supported by at least one concrete data point"
- Label: "Paragraphs stay under 4 sentences"

If you find yourself about to write `(e.g., ...)` in a dimension label, STOP. Either (a) rewrite the label without the example, or (b) move the example into the `evidence` field.

---

## ABSTRACTION TABLE — MAKE DIMENSIONS TRANSFERABLE

A common failure mode is dimension labels that **look** abstract but still carry the fingerprints of the specific conversation. A good dimension describes the **pattern** the user demonstrated, not the specific thing they said in this one thread. The pattern must apply to a different piece of the same writing type by the same user, even if the content is completely different.

**The pattern test:** Imagine the same user writes a different instance of this writing type next week, with different people, topics, and specifics. Would this dimension still apply *without rewording*? If you'd need to change the label to fit the new piece, the label is over-fixated on this conversation.

Below is a pattern-match table. The ❌ version is what the model often produces by anchoring too close to this conversation's content. The ✅ version captures the same underlying user preference in a form that transfers. Study these before writing dimension labels.

| ❌ Over-fixated to conversation | ✅ Transferable rule |
|---|---|
| "Opens cold emails with a reference to the recipient's LinkedIn activity" | "Opens with a specific reference to something the recipient recently did" |
| "Mentions a concrete revenue number in the first two sentences" | "Leads with one quantitative detail in the first two sentences" |
| "Signs off with 'thanks — [first name]' instead of 'Best regards'" | "Uses an informal first-name sign-off" |
| "Avoids jargon like 'synergy,' 'alignment,' and 'circle back'" | "Uses plain English instead of corporate jargon" |
| "Breaks bullet lists into 3-item groups" | "Keeps bullet lists short enough to scan at a glance" |
| "Uses em-dashes to separate clauses, not commas" | "Uses em-dashes for rhythmic emphasis over commas" |
| "Doesn't apologize for a delayed reply" | "Skips ritual apologies that don't add information" |
| "Mentions the specific dataset (ImageNet) and the specific baseline (ResNet-50)" | "Names the specific dataset and baseline at the start of the methods section" |
| "References the user's Q3 OKRs when arguing for roadmap changes" | "Anchors roadmap arguments in the team's stated priorities" |
| "Asks one sharp question at the end, like 'does this match what you had in mind?'" | "Ends with a single specific ask, not an open-ended 'let me know your thoughts'" |

**How to produce a transferable label:**
1. Write down what the user literally did in the conversation (names, topics, quoted phrases).
2. Ask: "What is the *category* of move this is an instance of?"
3. The category is the label. The literal instance is the evidence.

**Signs a dim is over-fixated:**
- Proper nouns (names of people, companies, products) in the label
- Numbers that reference *this specific piece* (e.g., "300 words," "Q3") rather than a reusable threshold
- Topic domain words that only make sense for one kind of content ("machine learning baselines," "investor cold emails") when the writing type is broader
- A sentence that reads like a paraphrase of something the user said rather than a rule

If in doubt: could this label appear in a rubric template the user downloads from a library, or does it only make sense because YOU read the conversation? Template-ready = good. Conversation-specific = bad.

---

## VALUES OVER RECIPES (DIM SHAPE)

A frequent failure mode is producing **recipe-shaped** dimensions when the user only demonstrated a **value**. Recipe-shaped dims prescribe procedure: how often something appears, what order things go in, where in the structure something must land. Value-shaped dims describe what the writing should *hold* — a quality of judgment, voice, or content the user cares about — without dictating the mechanics.

Most users (especially in interpretive genres like personal essay, memoir, or opinion writing) have clear values but no procedural opinions about their own writing process. When the inferer over-operationalizes a value into a recipe, the user rejects the recipe even though they agree with the underlying value.

**Recipe-shaped dims (high-risk — usually wrong):**
- "Mental logistics appear at least once per location section"
- "Sensory grounding comes first, reflection second"
- "Each encounter is anchored to one specific named object"
- "The closing beat lands on a small physical image rather than a thesis sentence"
- "Home content is spread across the chapter rather than clustered"

**Value-shaped dims (preferred):**
- "Hard moments are allowed to sit unresolved rather than tidied with a redemptive line"
- "Sharp or petty observations are not softened into diplomatic neutrality"
- "Resolutions are partially-true / partially-false rather than clean lessons"
- "Beauty passages contain at least one honest beat about cost happening at the same time"

Note the distinction: the value-shaped dim "Beauty passages contain at least one honest beat about cost..." reads close to a recipe but encodes a value (don't sanitize beauty by hiding the cost). The bad version of the same idea would be "Each beauty passage is followed within two sentences by a cost beat" — which prescribes the procedural shape rather than the value.

**Diagnostic test for each candidate dim:**

Before keeping a dim that prescribes:
- **how many times** something appears ("at least N per section", "once per location")
- the **order** of things ("X first, then Y", "before / after")
- **where in the structure** something should land ("the closing beat", "the opening pages")
- a **specific frequency or rhythm** ("every paragraph", "consistently throughout")

ask: did the user **explicitly describe this procedural rule** (in their own words, in a quote you can cite), OR are you operationalizing a value the user only demonstrated as a value?

If the procedural rule is YOUR interpretation of a user value, **drop the procedural prescription and keep the value-shaped version of the dim instead**. The grader can still check whether the value is upheld; it doesn't need a procedural recipe to do so.

When in doubt, write the dim as a value, not a recipe.

---

## ARTIFACT-GRADEABILITY (HARD CHECK)

The grader only sees the **final draft**. It has no access to revision history, prior versions, the order in which sentences were written, or what the writer changed in response to feedback. Any dim that asks about the authoring *process* rather than properties of the finished piece is structurally ungradeable — the grader will be forced to either guess or mark UNCERTAIN.

**Forbidden:** dims that depend on knowing what happened during drafting.

❌ "Revisions tighten rather than expand the piece" — requires diffing against a prior version
❌ "Decisions about structure are made early and held throughout" — asks about authoring sequence
❌ "Cuts are made before additions" — asks about order of operations
❌ "The opening is rewritten until it lands rather than left as the first attempt" — asks about edit history

**The test:** could a reader handed only the finished piece, with no other context, decide MET / NOT_MET? If they would need to see a draft history, an edit log, or the writer's process to answer, the dim is process-shaped — drop it, or rewrite it as a property of the finished artifact.

**Process-shaped → artifact-shaped:**

- "Revisions tighten rather than expand" → "Final drafts are compact rather than expansive"
- "Cuts are made before additions" → "The piece is tight; new material does not sit alongside the older material it replaces"
- "Opening is rewritten until it lands" → "The opening sentence carries weight rather than serving as a warm-up"

The underlying value is often real and worth keeping — just relocate it from "what the writer did" to "what the finished piece looks like."

---

## OPPOSITE-WRITER DIAGNOSTIC (DIM CONTRASTIVE-CHECK)

A common failure mode is dimensions that read as universal goods — moves no reasonable writer would deliberately reject. "Uses concrete details," "engages the reader," "communicates clearly." These pass the evidence gate (you can find concrete details in any draft) but fail the contrastive test: they don't distinguish *this* user's preferences from any other competent writer's defaults. They produce rubrics where every dim feels "always good," which makes the rubric un-steerable — the user can't disagree with anything because there's nothing to disagree with.

**The opposite-writer test (run this for EVERY candidate dim before keeping it):**

For each candidate dim, write down — in your `<analysis>` — what a writer who deliberately chose the OPPOSITE of this dim would produce. Specifically:

1. *"What would a different competent writer in the same genre do instead?"*
2. *"What is the defensible shape of the opposite move?"*

If the opposite is a real alternative that another good writer in this genre would defend — KEEP the dim. It's contrastive.

If the opposite is just "weaker writing," "lazy writing," or has no defensible shape at all — DROP the dim. It's a universal good masquerading as a preference.

**Examples:**

❌ "Uses concrete details"
- Opposite: "Stays abstract / uses generic language" — there's no defensible writer who chooses this. **DROP.**

✅ "Concrete details lean toward sensory texture rather than quantitative specifics"
- Opposite: "Concrete details lean toward numbers and named entities rather than sensory texture" — a different competent writer absolutely chooses this (data journalists, for instance). **KEEP.**

❌ "Engages the reader"
- Opposite: "Doesn't engage the reader" — no defensible alternative. **DROP.**

✅ "Engages the reader through implicit invitation rather than direct address"
- Opposite: "Engages through direct address ('you,' rhetorical questions)" — a real stylistic choice many writers prefer. **KEEP.**

❌ "Maintains professional tone"
- Opposite: "Unprofessional tone" — not a real alternative. **DROP.**

✅ "Tone holds a sardonic edge rather than earnest sincerity"
- Opposite: "Tone holds earnest sincerity rather than sardonic distance" — a different writer's authentic choice. **KEEP.**

**Compound / process-shaped dims — flip the consequent, not the antecedent:**

For dims phrased as "when X, then Y" (or "as X happens, Y happens"), the antecedent just sets context — the actual preference lives in the consequent. Flipping the antecedent produces a defensible-sounding opposite that *isn't what the dim is actually claiming*, and the dim sneaks past the test as a universal good.

❌ "When new motivation/framing is added, redundant or weaker material elsewhere is removed or combined to compensate"
- Wrong flip (antecedent): "Don't add new framing" — sounds like a real choice, but the dim isn't claiming "add framing"; it's claiming "remove redundancy when you do."
- Right flip (consequent): "When new framing is added, redundant or weaker material is left in place" — no defensible writer chooses bloat over tightness. **DROP.**

Rule: for any "when X, then Y" dim, hold X fixed and flip Y. If the flipped consequent has no defensible shape, the dim is a universal good — drop it regardless of how specific the antecedent sounds.

**Pair the keeper with its tradeoff:**

For dims that pass the opposite-writer test, name the tradeoff in your `<analysis>` (does NOT need to appear in the dim label). Every contrastive preference costs something in another direction:
- "Sensory concreteness" trades against "quantitative precision"
- "Implicit invitation" trades against "direct rhetorical force"
- "Sardonic edge" trades against "open earnestness"

If you cannot name what the dim costs — what the user is GIVING UP to satisfy it — the dim probably isn't contrastive. Re-run the opposite-writer test.

**Why this matters:**

A user looking at a rubric of universal goods has no leverage. They can't push back on "uses concrete details" — of course they want concrete details. But "concrete details lean sensory rather than quantitative" gives them a real choice to make: that might not be their actual preference, and now they can say so. The rubric is steerable when its dims are contrastive.

When in doubt, DROP. A rubric of 3 well-grounded contrastive dims beats a rubric of 8 dims where half are platitudes.

---

## NAME THE AXIS, NOT THE SURFACE FORM (DIM AXIS-CHECK)

When a dim is grounded in concrete details from the user's writing — named objects, specific words, particular phrasings, recurring brands — there is a risk of capturing the **surface form of the evidence** rather than the **underlying axis the user actually cares about**.

Example: the user's writing repeatedly mentions specific gear (an arc'teryx jacket, a titanium spork, a particular wine). A naive inferer writes "Each significant encounter is anchored to a specific named object." This is grounded in real evidence — but the underlying axis the user cares about may not be **named objects**; it may be **physical-detail-as-character** or **texture-of-personhood**. Naming the surface form produces a dim the user rejects ("that's not the defining part") even though the inferer found a real preference signal.

**Two-step rule for every dim grounded in concrete content:**

**Step 1 — articulate the axis.** Before writing the dim label, write down (in your `<analysis>`):
- *"What kind of move is this an instance of?"*
- *"What is the user's underlying preference that produced this surface detail?"*
- *"If the user wrote a different piece of the same type with completely different content, what would still be true about how they'd handle it?"*

**Step 2 — write the dim as the axis, not the surface form.**

❌ Surface form: "Each encounter is anchored to a specific named object (gear, garment, food)"
✅ Axis: "People are described by something visible or tangible about them, not just by their job, origin, or personality summary"

❌ Surface form: "References to home content cite specific media (newsletters, podcasts, voice memos)"
✅ Axis: "Home content is grounded in a specific physical moment of the trip rather than treated as a thematic interlude"

❌ Surface form: "Sentences about discomfort use blunt physical vocabulary ('feet hurt', 'I wanted to cry')"
✅ Axis: "Moments of discomfort are rendered without softening or rationalization"

**Signs you are naming the surface form:**
- The dim quotes or paraphrases specific words that appear in the user's draft
- The dim could be falsified by the user simply switching topics (different gear → same preference would still apply, but the dim wouldn't fire)
- The evidence string and the dim label are nearly identical (the dim *is* the evidence rather than a generalization of it)

If you can't articulate the underlying axis cleanly, the evidence may not actually demonstrate a transferable preference. Drop the dim or reduce confidence.

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
- **3–7 criteria** (prefer **fewer, high-value** criteria over filling slots; **never** add a criterion solely to reach a number), each with:
  - `name`
  - `category` (from a shared set of 3–5 categories)
  - `description` (1–3 sentences max)
  - `dimensions` (3–5 per criterion) — these are **checkable items** that determine achievement level
  - `priority` (unique integer rank: 1 = most important, higher = less important)

### ❌ DO NOT INCLUDE
- Separate achievement level descriptions (excellent, good, fair, weak) — these are now derived from dimension counts
- Long rationales or justifications
- Detailed examples within the rubric
- Generic writing advice unsupported by conversation evidence
- Decision points — these will be extracted in a separate step

---

## ANALYSIS PROCESS

Before producing the rubric, write your reasoning inside `<analysis>` tags.

### Step 0: Task entailment vs latent preferences
- Infer the **writing situation** from early user messages (genre, audience, purpose).
- List **obligations any strong model would already satisfy** for that situation (do not turn these into rubric criteria unless the user showed a *non-default* preference).
- List **latent or distinctive preferences** — only these should drive most criteria.

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
- Patterns in repeated edits the user actively makes (a single edit is weak signal; recurring edits across several drafts is strong)
- Direction of changes (more specific vs. abstract, shorter vs. longer) when the user is the one making the changes
- Trade-offs the user explicitly endorses or repeatedly rejects across the conversation

**NOT signals (do NOT use these as evidence):**
- What the user never comments on (silence is not endorsement — see Evidence Gate and Acceptance-Strength Classifier)
- A single instance of the user *accepting* an assistant-proposed move without active endorsement
- Patterns observed in the *assistant's* drafts that the user did not explicitly affirm or modify

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
- Tone and voice preferences **when the user showed a non-default choice** (not merely "professional" for a work email — that is usually entailed)
- Structural patterns (how they like to open/close, paragraph length preferences)
- Stylistic choices (active vs. passive voice, use of metaphors, sentence rhythm)
- Recurring values (conciseness, specificity, audience awareness, storytelling) **only when tied to user evidence**, not as filler
- How they handle evidence, examples, and claims

When the user says "make this warmer" — the rubric should capture that they prefer warm tone, not that this particular paragraph needed warming up.

When the user says "add a specific example about our product launch" — the rubric should capture that they value concrete examples, not that they want product launch references.

---

## DEFINING CRITERIA

Select **3–7 criteria** with **clear conversation evidence** (explicit feedback, edits, rejections, or repeated patterns — not silence alone).

For each candidate, verify:
- Did the user demonstrably care about this, or is it only "reasonable to assume"?
- Can you point to **specific messages** (e.g. Message #N) or edit patterns?
- Is it **distinct** from other criteria and **not already entailed** by the task description?
- **Would this criterion change** how you coach or evaluate compared to a generic assistant for the same task?
- **Would this criterion make sense for a DIFFERENT piece of the same writing type** by the same user?

**Do not include**:
- Genre-default platitudes without user-specific evidence (see incremental value test above)
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
❌ Avoid: "Maintain a professional tone in emails to your manager." (usually **entailed** by the task — only include if the user showed a *specific* tonal preference that cuts against the default)
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

## CONFIDENCE LEVELS

Assign a **confidence** level to each criterion: `high`, `medium`, or `low`.

- **high**: The user explicitly stated this preference, or demonstrated it through multiple edits/feedback. Strong, direct evidence.
- **medium**: Inferred from a pattern of behavior or a single clear signal. Reasonable but not confirmed by the user.
- **low**: Inferred from thin signal — e.g., the user accepted a draft without comment, or this is extrapolated from a single data point. The user may not actually hold this preference.

Be honest about confidence. If you're inferring from a single instance, mark it `low`. The rubric should present itself as a draft to be completed, not a finished product. (For evidence that relies on user acceptance specifically, follow the Acceptance-Strength Classifier rules above — silent acceptance is not evidence at all and the dim must be dropped, not merely marked `low`.)

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
          "label": "<checkable item: what to verify as yes/no — must be reusable across tasks>",
          "evidence": "<REQUIRED, NEVER EMPTY: direct quote, concrete edit citation, or rejection pattern — must include at least one `Message #N`. When evidence relies on user acceptance of an assistant suggestion, you MUST also include the acceptance-class label (`active_endorsement` or `tacit_acceptance`); silent acceptance is not evidence — drop the dim. If you cannot ground this dim, DROP IT from the rubric entirely. Only use 'INSUFFICIENT_EVIDENCE' as a last-resort escape hatch.>"
        }
      ],
      "priority": <unique integer 1..N where N = number of criteria, 1 = most important, no duplicates>,
      "confidence": "<high|medium|low — how confident you are that this criterion reflects a real user preference based on conversation evidence>"
    }
  ]
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

Analyze the conversation and infer a rubric. Apply the **zero-information** and **contrastive** rules in your system instructions: omit criteria that merely restate what the task genre already implies. Do NOT extract decision points — that will happen in a separate step. Return ONLY valid JSON matching the output format in your system instructions."""


# ============================================================================
# DECISION POINT EXTRACTION (with classification context) — Step 3 of the 5-step flow
# ============================================================================


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
        # Strip the dim-level `evidence` field. Inference fills it with
        # `Message #N: user said "..."` quotes from the original conversation
        # so the grader/refiner can ground their judgments. But the chat
        # generator was reading those quotes and inheriting the original
        # task's vocabulary -- which produced Comparison-tab drafts that
        # echoed the original conversation's subject matter even on a
        # totally different task. The grader still gets `evidence` via its
        # own rubric serialization path; this only strips for chat.
        # Also strips `_diff` (display-only, contains non-serializable sets).
        numbered_rubric = []
        for idx, criterion in enumerate(rubric, start=1):
            numbered_crit = {k: v for k, v in criterion.items() if k != "_diff"}
            if isinstance(numbered_crit.get("dimensions"), list):
                cleaned_dims = []
                for d in numbered_crit["dimensions"]:
                    if isinstance(d, dict):
                        cleaned_dims.append({
                            k: v for k, v in d.items()
                            if k not in ("evidence", "example_annotation")
                        })
                    else:
                        cleaned_dims.append(d)
                numbered_crit["dimensions"] = cleaned_dims
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
        **RUBRIC AUTHORITY (READ CAREFULLY — THIS IS NON-NEGOTIABLE):**

        The rubric is the user's persistent writing preferences. It is your primary guide for tone, style, structure, and approach. It is the authoritative source of truth for HOW to write, not a suggestion.

        **You may NEVER modify the rubric from within the conversation.** The rubric is edited only through dedicated flows outside the chat (rubric inference, rubric edit suggestions, the rubric configuration tab). Your job in the chat is to APPLY the rubric, not change it. Specifically:

        - You may NOT decide to drop, ignore, reweight, or reinterpret a rubric criterion based on something the user said in chat.
        - You may NOT silently adjust your application of the rubric to match a user request that conflicts with it.
        - You may NOT treat any user message as "updating the rubric for future drafts" — only the dedicated rubric-edit flow can do that.

        **What you MAY do:**

        - Follow **task-specific instructions** for the current message. If the user says "expand this paragraph" or "write this one more formally," you do that for THIS request. The rubric still applies, but the task adds one-time constraints. You do not change the rubric — you layer the task on top.
        - If a task-specific instruction conflicts with the rubric (e.g., rubric says "casual, lowercase" and user says "write this one more formally"), satisfy BOTH as much as possible: apply the rubric's structural/content rules (specific hook, numbers, no hedging, etc.) while making the surface-level tonal adjustment the user asked for.
        - **Tell the user when there's a real tension.** If the user's request genuinely can't coexist with the rubric (not just a tonal dial, but a structural contradiction), say so in plain language: "This goes against your rubric's X criterion. I can do it this one time, but the rubric won't change unless you update it in the rubric configuration tab." Then do what they asked for this message only.

        **Key distinction:** a task-specific instruction adjusts WHAT you produce this time, not WHAT THE USER PREFERS. The rubric continues to apply unchanged on the next message.

        If you think a criterion is genuinely producing poor results, tell the user — but do not act on that opinion by deviating from the criterion. Surface the concern; leave the decision to the user.

        **CONFIDENCE-AWARE APPLICATION:**
        Each criterion may have a `confidence` field (`high`, `medium`, or `low`).
        - **high confidence**: Follow this criterion consistently — the user has clearly demonstrated this preference.
        - **medium confidence**: Apply this criterion as a reasonable default, but be ready to adjust if the user's feedback suggests otherwise.
        - **low confidence**: Treat this as a tentative suggestion, not a mandate. The user may not actually hold this preference — it was inferred from thin signal. Apply it lightly and be especially attentive to feedback that contradicts it.

        **RUBRIC-EDIT SYSTEM MESSAGES (READ CAREFULLY):**

        The conversation history may contain system messages that mark when the user changed the rubric. These are NOT historical narration — they record actual changes to the rubric you must apply going forward. The RUBRIC block at the top of this prompt is ALWAYS the current rubric (already includes every applied edit), so the system messages are just timestamps for when each change happened.

        **Formats you'll see:**

        1. Refiner suggestion applied to one dimension's wording:
            ✅ **Rubric edit applied**
            _criterion_: **dimension**
            **Before:** <old wording>
            **After:** <new wording>

        2. A dimension removed via the drift panel's "🗑 Remove" button:
            🗑 **Dimension removed.** _criterion_: **dimension** ...

        3. A rubric save (manual edits in the Rubric Configuration tab). These can include multiple changes at once. The format spells out each operation explicitly:
            📋 **Rubric saved as v{{N}}:**
            - **Added:** "criterion-name"            ← a NEW criterion was added
            - **Removed:** "criterion-name"          ← an entire criterion was deleted
            - **Reworded:** "criterion-name" description changed
            - **Criterion:** "criterion-name"
              - **Added dimension:** "dim-label"     ← a NEW dimension was added under this criterion
              - **Removed dimension:** "dim-label"   ← a dimension was deleted from this criterion

           When you see "Added dimension" or "Removed dimension" indented under a criterion, that means a dimension was added/removed within that criterion. The label after each is the NEW or REMOVED dimension's text, not a clarification of the criterion.

        **Three rules for handling these messages:**

        1. **Apply changes starting in your next draft.** If a "Rubric edit applied" or "Added dimension" message appears above and the user now asks for another draft, the current RUBRIC block is what you grade your draft against — even if your earlier drafts were written under the old wording. Do NOT keep applying the old wording out of consistency with prior drafts.

        2. **For removed criteria or dimensions:** you do not need to satisfy them in any new draft. They will not appear in the RUBRIC block above. Drafts you wrote earlier may have satisfied them; that's fine, but going forward only the current rubric applies.

        3. **Acknowledge the change briefly when one of these messages appears immediately before the user's current turn.** Open your response with one short sentence naming the specific edit so the user knows it registered (e.g., "Picking up the reworded *Tone* criterion in this pass."). Keep it to one sentence — do NOT explain how the edit will shape the draft, do NOT preview specific moves, do NOT restate the rubric. The new draft should reflect the *full* current rubric, not over-index on the latest edit; the acknowledgment is just a receipt, not a thesis statement for the draft. After the acknowledgment, produce the draft normally — and ensure your draft actually differs from the prior draft on the edited criterion. If the new draft would read identically to the prior draft on that criterion, the edit hasn't been applied.

        If the user's most recent ask is for a NEW draft (not a small fix to an existing one), you are writing under the current rubric — not the rubric that was in force when earlier drafts were generated.

        **OUTPUT FORMAT — <draft> TAGS:**

        Use `<draft></draft>` tags ONLY for a single committed piece of writing that the user will edit and build on — a complete draft of the requested piece, or a committed revision replacing a previous draft.

        **DO NOT use `<draft>` tags in these cases:**
        - When presenting multiple options/alternatives for the user to choose between (e.g., "Option 1: ...", "Option 2: ...", "Version A: ...", "Version B: ..."). Show these as plain text — the user hasn't committed yet.
        - When offering alternative sentences, phrases, or paragraphs the user can swap in. These are suggestions, not the draft.
        - When showing excerpts, quotes, or before/after comparisons. Render as plain text.
        - When asking clarifying questions or discussing approach before writing.

        **Rule of thumb:** `<draft>` = one canonical piece of text the user is now working with. If you're giving the user a choice between multiple options, or showing fragments they might pick from, it is NOT a `<draft>` — it's regular prose.

        Each response contains AT MOST ONE `<draft>` block. If you need to show alternatives, never wrap them.

        **DRAFT NUMBERING (INTERNAL TAGS — DO NOT REPLICATE):**

        Prior assistant messages that contain a draft are prefixed by the SYSTEM with `[This is Draft #N.]` (e.g. `[This is Draft #3.]`). These tags are injected automatically to help you resolve references like "fix X in draft #3." Draft numbering is 1-based across the graded drafts in the conversation, in order.

        **STRICT RULE:** Do NOT emit `[This is Draft #N.]` (or any `[This is Draft ...]` variation) anywhere in your own output. Your response should look exactly like a normal chat message — no system-style bracketed prefix, no restating the draft number. The system handles numbering for past drafts; your job is just to respond.
        """).strip()
    else:
        system_instruction = dedent("""
        You are an AI co-writer. You collaborate with the user to develop their writing.

        **OUTPUT FORMAT — <draft> TAGS:**

        Use `<draft></draft>` tags ONLY for a single committed piece of writing that the user will edit and build on — a complete draft of the requested piece, or a committed revision replacing a previous draft.

        **DO NOT use `<draft>` tags in these cases:**
        - When presenting multiple options/alternatives for the user to choose between (e.g., "Option 1: ...", "Option 2: ..."). Show these as plain text.
        - When offering alternative sentences, phrases, or paragraphs the user can swap in. These are suggestions, not the draft.
        - When showing excerpts, quotes, or before/after comparisons. Render as plain text.
        - When asking clarifying questions or discussing approach before writing.

        Each response contains AT MOST ONE `<draft>` block. If you need to show alternatives, never wrap them.

        **DRAFT NUMBERING:**

        Prior assistant messages that contain a draft are prefixed with `[This is Draft #N.]` (e.g. `[This is Draft #3.]`). When the user refers to a specific draft by number, use those prefixes to resolve which content they mean. Do NOT add `[This is Draft #N.]` tags to your own new output; the system adds them for past drafts.
        """).strip()

    return system_instruction


DRAFT_EDIT_FEEDBACK_SYSTEM_PROMPT = """You are a writing collaborator. The user just directly edited their draft. Your job is to (1) acknowledge what changed in 1-2 sentences and (2) ask 1-2 short, grounded questions about WHY they made those changes — so the conversation context records their intent for future drafts.

You will be given:
- The previous draft (before this edit)
- The edited draft (after this edit)
- The active rubric criteria
- The most recent grader scorecard for the previous draft (if any)

Guidelines for your response:
- Keep the whole response under ~80 words.
- First, briefly note the most salient change(s) you observed (e.g., "You tightened the intro and dropped the third bullet.").
- Then ask 1-2 short questions that are GROUNDED in what changed. Prefer questions that connect changes to specific rubric dimensions when relevant. Examples:
  * "Was the bullet drop because the supporting detail felt off-rubric for the conciseness dimension, or for another reason?"
  * "Did you rewrite the closing because the previous tone felt too informal for this audience?"
- Do NOT propose further edits. Do NOT re-grade. Do NOT echo the entire draft back.
- Do NOT ask generic questions like "What do you think?" or "Is this better?" — every question must reference a SPECIFIC change you observed.
- If the diff is trivial (e.g., a typo fix), one sentence acknowledging the change with no question is fine.

Output format:
Return ONLY plain prose (no JSON, no markdown headers, no code fences). The prose will be shown directly to the user as your conversational reply."""


def DRAFT_edit_feedback_prompt(previous_draft: str, edited_draft: str, rubric_list, prior_scorecard=None):
    """Generate user prompt for the 'feedback on user edit' conversational reply."""
    rubric_text = json.dumps(rubric_list or [], ensure_ascii=False, indent=2)
    scorecard_text = json.dumps(prior_scorecard, ensure_ascii=False, indent=2) if prior_scorecard else "(no prior scorecard)"
    return f"""Previous draft:
{previous_draft}

Edited draft:
{edited_draft}

Active rubric criteria:
{rubric_text}

Most recent scorecard for the previous draft:
{scorecard_text}

Acknowledge the user's edit and ask 1-2 grounded questions about why they made these changes. Plain prose only."""


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





