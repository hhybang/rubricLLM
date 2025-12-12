from textwrap import dedent
import json

RUBRIC_SCORING_PROMPT = """
You are tasked with evaluating a draft of writing against a personalized rubric that captures what the user values in their writing.

You will receive:
1. The draft to be evaluated
2. A rubric with specific criteria, achievement levels, and weights

## Scoring System

Each criterion is scored on werfdesw worth:
- **Exemplary**: 100% (fully meets user's vision)
- **Proficient**: 75% (meets core requirements)
- **Developing**: 50% (shows understanding, needs significant work)
- **Beginning**: 25% (misses key elements user values)

The overall score is calculated by:
1. Scoring each criterion (0-100%)
2. Multiplying by that criterion's weight
3. Summing all weighted scores for a total out of 100

## Evaluation Process

For each criterion:
1. **Carefully read the criterion description**: Understand what THIS user specifically values (not generic writing standards)
2. **Review all four achievement level descriptions**: Note the specific, observable features that distinguish each level
3. **Evaluate the draft against the descriptors**: Find evidence in the draft for or against each level
4. **Assign the appropriate level**: Choose the level whose descriptors best match what you observe in the draft
5. **Provide specific evidence**: Quote passages or cite examples that justify your rating

## Analysis Structure

Wrap your detailed analysis in <evaluation> tags:

### [Criterion Name]
**Weight**: [X%]
**Achievement Level**: [Level Name] ([percentage]%)
**Weighted Score**: [percentage × weight / 100]

**Evidence from draft**:
- [Specific quote or example 1]
- [Specific quote or example 2]
- [etc.]

**Rationale**:
[Explain why this level was chosen over adjacent levels, referencing the rubric's specific descriptors for this criterion]

**To reach next level**:
[If not Exemplary: What specific changes would move this to the next achievement level based on the rubric descriptors?]

---

[Repeat for all criteria]

## Scoring Guidelines

**Choosing between adjacent levels:**

*Beginning (25%) vs. Developing (50%):*
- Beginning: Misses or contradicts what the user has indicated matters to them
- Developing: Shows awareness of user's goals but execution falls short

*Developing (50%) vs. Proficient (75%):*
- Developing: Would require significant revision; user would need to substantially rework this
- Proficient: Would satisfy user with minor polish; the core is right

*Proficient (75%) vs. Exemplary (100%):*
- Proficient: Meets requirements but doesn't fully realize user's best articulated vision
- Exemplary: This is what the user was striving for; they'd approve with minimal or no changes

**When in doubt:**
- Default to the lower level unless strong evidence supports the higher one
- Ask: "Would the user accept this aspect as-is, or request changes?"
- Remember: Score against THIS user's values shown in the rubric, not general writing quality

## Summary Output

After your evaluation, provide this JSON:
```json
{
  "overall_score": <sum of all weighted scores, out of 100>,
  "score_interpretation": "<Exceptional/Strong/Solid/Developing/Emerging based on score bands>",
  "criteria_scores": [
    {
      "name": "<criterion name>",
      "weight": <percentage as integer>,
      "achievement_level": "<Exemplary/Proficient/Developing/Beginning>",
      "level_percentage": <25/50/75/100>,
      "weighted_score": <level_percentage × weight / 100>,
      "evidence_summary": "<1-2 sentence summary of key evidence>"
    },
  ],
  "overall_assessment": "<2-3 sentences: Does this draft align with user's demonstrated values? What does the score mean? What's the main focus for revision?>"
}
```

## Priority Calculation

To identify revision priorities, calculate potential gain:
```
Potential Gain = (Next Level % - Current Level %) × (Weight / 100)
```

Example: A criterion at Developing (50%) with 25% weight:
- Moving to Proficient (75%) would gain: (75-50) × 0.25 = 6.25 points
- Prioritize high-gain opportunities that align with user's weighted priorities

Provide your evaluation following this structure.
"""

ASSESS_RUBRIC_PROMPT = """
Please evaluate the most recent draft from our conversation against the rubric.

## Context & Purpose

You've been collaborating with the user on their writing. They've now asked for a formal assessment of how well the current draft aligns with their rubric. Review the conversation history to understand:
- The original writing goal and any constraints mentioned
- How the draft evolved through our discussion
- What the user explicitly valued or requested

Then assess the **final draft** (the most recent assistant message) against each criterion in the rubric.

## Scoring System

Each criterion uses a four-level achievement scale:
- **Exemplary (100%)**: Fully realizes the user's vision for this criterion; they'd approve with minimal or no changes
- **Proficient (75%)**: Meets the core requirements; would satisfy user with minor polish
- **Developing (50%)**: Shows awareness of user's goals but needs significant revision to meet their standards
- **Beginning (25%)**: Misses or contradicts what the user values for this criterion

**Overall Score Calculation:**
1. Assign each criterion a level (25%, 50%, 75%, or 100%)
2. Multiply by that criterion's weight
3. Sum all weighted scores for the overall score out of 100

## Evaluation Process

For each criterion in the rubric:

1. **Read the criterion carefully**: Understand what THIS specific user values (not generic writing quality)

2. **Review the achievement level descriptors**: Each criterion has four levels with specific, observable features. Note what distinguishes each level.

3. **Examine the draft**: Look for evidence of the features described in each level. Consider:
   - Direct quotes that exemplify or violate the criterion
   - Structural patterns (organization, flow, balance)
   - Tone, style, and rhetorical choices
   - What's present AND what's missing

4. **Choose the best-matching level**: Select the level whose descriptors most accurately describe what you observe in the draft. When between two levels:
   - **Beginning vs. Developing**: Does it show awareness of the user's goal?
   - **Developing vs. Proficient**: Would the user need substantial rework, or just polish?
   - **Proficient vs. Exemplary**: Does it fully realize their vision, or just meet requirements?

5. **Document your reasoning**: Provide specific evidence and explain why this level fits better than adjacent levels.

## Required Output Format

Wrap your detailed evaluation in `<evaluation>` tags, following this structure for EACH criterion:

### [Criterion Name from Rubric]
**Weight**: [X%]
**Achievement Level**: [Level Name] ([percentage]%)
**Weighted Score**: [percentage × weight / 100]

**Evidence from draft**:
- [Specific quote, example, or observation from the draft]
- [Another piece of concrete evidence]
- [Continue with all relevant evidence]

**Rationale**:
[Explain why you chose this level over adjacent levels. Reference the specific descriptors from the rubric's achievement levels. Be concrete about what features you observed or what was missing.]

**To reach next level**:
[If not Exemplary: Describe the specific, actionable changes needed to reach the next achievement level. Reference what the next level's descriptors require that the current draft lacks.]

---

[Repeat the above structure for EVERY criterion in the rubric]

## After All Criteria

After completing the evaluation for all criteria, provide a JSON summary:

```json
{
  "overall_score": <sum of all weighted scores, out of 100>,
  "score_interpretation": "<Exceptional (90+) | Strong (80-89) | Solid (70-79) | Developing (50-69) | Emerging (<50)>",
  "criteria_scores": [
    {
      "name": "<exact criterion name from rubric>",
      "weight": <weight as integer>,
      "achievement_level": "<Exemplary|Proficient|Developing|Beginning>",
      "level_percentage": <25|50|75|100>,
      "weighted_score": <level_percentage × weight / 100>,
      "evidence_summary": "<1-2 sentence summary of key evidence>"
    }
    // Include ALL criteria
  ],
  "overall_assessment": "<2-3 sentences: Does this draft align with the user's demonstrated values? What does the score mean? What's the main focus for revision?>",
  "highest_priority_improvements": [
    "<Criterion with highest potential gain if improved (calculate: (next_level% - current_level%) × weight)>",
    "<Second highest priority>",
    "<Third highest priority>"
  ]
}
```

## Scoring Principles

- **Be calibrated to THIS user's values**: The rubric reflects what THIS user cares about, not universal writing standards
- **Default to lower levels when uncertain**: Better to under-promise than over-claim
- **Quote specific evidence**: Vague assessments aren't helpful; cite actual text
- **Consider the conversation context**: If the user requested specific changes, check if they were addressed
- **Prioritize high-impact improvements**: In your JSON summary, identify which criteria offer the biggest score gains if improved (weight × level jump)

Provide your complete evaluation now, starting with the `<evaluation>` tags.
"""

# Prompt for comparing rubrics and generating contrasting revisions
COMPARE_WRITE_EDIT_PROMPT = r"""
You are an editor comparing how two rubrics influence a piece of writing.
Your job is to make the effects of each rubric clearly visible to a reader.

========================
GLOBAL RULES
========================
- The two revisions MUST both start from the exact same BASE DRAFT (do not revise A from B or vice versa).
- Change ONLY what is required to satisfy each rubric. Keep content, argument order, and structure stable unless a rubric explicitly requires otherwise.
- If a rubric requires additions or deletions, make them, but keep changes localized and intentional.
- Be explicit and consistent: when words are added, use **bold**; when words are removed relative to the base, use ~~strikethrough~~ inside the revision text.
- If a whole sentence exists only in one version, show "—" for the missing version in the comparison table.
- Keep differences attributable to rubric deltas. Avoid unrelated rewrites.

========================
INSTRUCTIONS
========================
1. Write a **base draft** that fulfills the USER TASK naturally (without following any rubric).

2. Starting from that same base draft:
- Revise once to meet Rubric A.
- Revise once to meet Rubric B.

3. Focus each change only on the differences between the rubrics.
- Keep content and meaning constant unless the rubric directly affects it.
- If a sentence remains the same, repeat it identically in all columns.

========================
OUTPUT FORMAT (STRICT)
========================
Return sections in this exact order and headings:

### Key Rubric Differences
- ...

### Stage 1 – Base Draft
<paste the full base draft here. No formatting beyond paragraphs.>

### Stage 2 – Revisions
#### Rubric A Revision (from the base)
<paste the full revised text here. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.>

#### Rubric B Revision (from the base)
<paste the full revised text here. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.>

### Summary of Impact
- In 3–5 bullets, explain how Rubric A vs Rubric B affected tone, concision, evidence, structure, or polish.
- Mention any additions/deletions and why they were necessary for the rubric.
"""

def compare_rubric_user_prompt(task, rubric_a, rubric_b):
   return f"""
    ========================
    RUBRIC A
    ========================
    {rubric_a}

    ========================
    RUBRIC B
    ========================
    {rubric_b}

    User Writing Task: {task}
    """

old_RUBRIC_INFERENCE_SYSTEM_PROMPT = """
You are tasked with creating a writing rubric based on the conversation history between a user and an LLM who are collaboratively developing a piece of writing. 
This rubric should capture what this specific user values and wants to achieve in their writing, not generic standards of "good writing."

## Your Goal

Create a rubric that:
- Makes the user's tacit rhetorical preferences and goals explicit
- Provides clear criteria that can be linked to specific achievement levels
- Serves as a coaching tool for future iterations of similar writing
- Can be used to reverse-engineer the writing process (align future writing to these discovered priorities)
- Helps produce writing that matches the user's intent, style, and goals
- **Can be reliably evaluated by an LLM** - achievement levels must be distinguishable through concrete, observable features

## Approach Based on Context

Your approach depends on whether a previous rubric exists:
- **If no previous rubric exists**: Create a new rubric based on the conversation(s)
- **If a previous rubric exists**: Update it incrementally, preserving continuity while incorporating new insights. Only modify criteria that have new evidence; don't change criteria that remain stable.

## Analysis Process

Before creating your final rubric, wrap your systematic analysis in <analysis> tags. Follow these steps. This section should be thorough and evidence-based.

### Step 1: Scenario Identification
- Determine whether you're creating a new rubric or updating an existing one
- If updating: identify what has changed in the user's priorities or approach since the last version

### Step 2: Analyze the Conversation History

**Extract explicit signals:**
- Direct statements about goals ("I want this to sound more authoritative")
- Specific requests for changes ("Can you make this more concise?")
- Rejections or approvals ("I like this part" / "This doesn't work")
- Questions that reveal values ("Is this too formal?" suggests they care about formality level)
- Stated audience, purpose, or constraints

**Extract implicit signals:**
- Patterns in what the user consistently revises (e.g., always simplifying complex sentences suggests they value clarity)
- What they never comment on (may indicate lower priority or satisfaction)
- The direction of their edits (adding detail vs. cutting, being more specific vs. more abstract)
- Emotional reactions or enthusiasm about certain elements
- Trade-offs they make when given options

**Document the writing type:**
Be specific about the genre/format (e.g., "persuasive op-ed for general audience" not just "article")

**Identify constraints and non-negotiables:**
Note any must-haves mentioned by the user (word limits, required sections, stylistic requirements)

**CRITICAL: Analyze User Feedback on Previous Assessments**

If the conversation contains user feedback on rubric assessments (messages that start with "I have some feedback on your rubric assessment:" followed by criterion-specific comments), this is the MOST VALUABLE data for understanding how the user interprets criteria:

- **Feedback reveals misalignments**: When users disagree with scores, it shows where criterion definitions don't match their actual values
- **Feedback clarifies priorities**: Comments about scores being "too harsh" or "too lenient" reveal what the user truly considers Exemplary vs. Developing
- **Feedback provides concrete examples**: User corrections often cite specific features they value that weren't in the original criterion descriptors

**For each piece of assessment feedback found:**

1. **Identify the criterion being discussed**: Match the feedback to the specific rubric criterion
2. **Determine the disagreement type**:
   - Score disagreement: User thinks achievement level should be higher/lower
   - Interpretation disagreement: User values different features than the assessment emphasized
   - Priority disagreement: User cares more/less about this criterion than the weight suggests
   - Definition disagreement: The achievement level descriptors don't match user's vision

3. **Extract corrective signals**:
   - What specific features did the user mention that the assessment missed?
   - What examples did they provide of what they actually value?
   - Did they redefine what "Exemplary" or "Proficient" means to them?
   - Did they suggest the criterion is too strict/lenient?

4. **Update criterion understanding**:
   - Revise criterion descriptions to match user's clarified values
   - Adjust achievement level descriptors based on their corrections
   - Modify weights if feedback suggests different priorities
   - Add concrete examples from their feedback to make levels more accurate

**Example of using assessment feedback:**

*User feedback:* "**Concision**: I disagree with the Developing score. The draft is meant to be thorough, not minimalist. Some repetition is intentional for emphasis."

*Inference:*
- User values thoroughness over brevity for THIS writing type
- "Concision" criterion may be misnamed - should focus on "purposeful repetition" vs. "redundancy"
- Achievement levels should distinguish intentional elaboration from empty wordiness
- Weight for this criterion may be too high if over-emphasized

### Step 3: Identify Evidence-Based Criteria

List 4-7 criteria categories based on what the conversation actually reveals, not generic writing standards. 

**For each potential criterion, ask:**
- Is there clear evidence in the conversation that the user cares about this?
- Can I point to specific moments where this mattered to the user?
- Is this distinct from other criteria, or is it overlapping?

**Avoid including criteria if:**
- The user never mentioned or demonstrated concern about it
- It's a generic "good writing" principle without user-specific evidence
- You're making assumptions about what they "should" care about

**Common HIGH-LEVEL categories** (use these to group multiple criteria):
- **Style**: Tone, voice, word choice, formality, register, concision
- **Structure**: Organization, argument flow, transitions, paragraph ordering
- **Content**: Ideas, themes, technical accuracy, depth, evidence
- **Audience**: Reader awareness, accessibility, context appropriateness
- **Mechanics**: Grammar, formatting, citations (only if user shows concern)

**IMPORTANT - Category vs. Criterion:**
- **Category** = broad grouping (there should only be 3-5 unique categories total)
- **Criterion** = specific aspect the user cares about (there can be 4-7+ criteria)
- Multiple criteria can share the same category

**Examples of proper categorization:**

✅ **GOOD - Multiple criteria share categories:**
```json
[
  {
    "name": "Academic Tone & Register",
    "category": "Style",
    ...
  },
  {
    "name": "Precision & Concision", 
    "category": "Style",
    ...
  },
  {
    "name": "Logical Argument Structure",
    "category": "Structure",
    ...
  },
  {
    "name": "Paragraph Transitions",
    "category": "Structure",
    ...
  },
  {
    "name": "Thematic Coherence",
    "category": "Content",
    ...
  }
]
```
Result: 5 criteria across 3 categories (Style, Structure, Content)

❌ **BAD - Each criterion gets unique category:**
```json
[
  {
    "name": "Academic Tone & Register",
    "category": "Academic Tone",
    ...
  },
  {
    "name": "Precision & Concision",
    "category": "Precision",
    ...
  },
  {
    "name": "Logical Argument Structure", 
    "category": "Logical Structure",
    ...
  }
]
```
Result: 3 criteria with 3 categories - categories aren't serving as groupings

**How to assign categories:**
1. For each criterion, ask: "What broad aspect of writing does this relate to?"
   - Is it about HOW the text sounds/reads? → Style
   - Is it about HOW ideas are organized? → Structure  
   - Is it about WHAT is being communicated? → Content
   - Is it about WHO the audience is? → Audience

2. Use the same category label for criteria that relate to the same broad aspect
3. It's normal and expected for 2-3 criteria to share the same category

### Step 4: Define Each Criterion with User-Specific Language

For each criterion:

**Write a precise description that:**
- Uses vocabulary and concepts from the actual conversation
- Specifies what "good" means for THIS user (not generally)
- Includes observable features when possible

**Examples of user-specific vs. generic:**
- ❌ Generic: "Uses clear and effective language"
- ✅ User-specific: "Uses short, declarative sentences with minimal jargon, prioritizing accessibility for non-expert readers"

- ❌ Generic: "Well-organized structure"
- ✅ User-specific: "Opens with a concrete anecdote before transitioning to broader analysis, using the personal-to-universal pattern the user prefers"

### Step 5: Create Achievement Levels with CONCRETE, OBSERVABLE Features

**CRITICAL REQUIREMENT**: Achievement levels must be distinguishable by an LLM evaluator without human judgment. This means:
- NO vague quantifiers like "rare," "some," "frequent," "generally," "mostly"
- YES to specific counts, percentages, or checkable features
- NO abstract qualities like "flows well" or "engaging"
- YES to concrete patterns like "every paragraph begins with transition sentence"

**Level naming:** Use consistent labels across all criteria: Exemplary → Proficient → Developing → Beginning

**For each level, you MUST provide:**

1. **Quantifiable thresholds** (choose based on what's measurable):
   - Exact counts: "0 instances," "1-2 instances per 1000 words," "3-5 instances"
   - Percentages: "100% of paragraphs," "80%+ of sentences," "less than 50%"
   - Ratios: "every paragraph," "most paragraphs (4+ out of 5)," "some paragraphs (2-3)"
   - Binary checks: "all key terms defined," "no metaphorical language," "problem and solution explicitly linked"

2. **Concrete examples** showing what to look for:
   - Positive examples (what Exemplary looks like)
   - Negative examples (what Developing/Beginning looks like)
   - Specific phrases or patterns to identify

3. **Observable patterns** an LLM can check:
   - Sentence structures to count
   - Word types to identify (e.g., subjective adjectives, transition words)
   - Organizational features to verify (e.g., "each paragraph references previous")

**Achievement Level Definition Framework:**

**Exemplary (100%):**
- Define the HIGHEST standard the user demonstrated wanting
- Use "zero," "all," "every," "100%," "no instances of [negative feature]"
- Include specific positive examples from the conversation
- Example: "Zero instances of subjective adjectives (e.g., 'hard-won,' 'daunting'). Every sentence uses precise, neutral descriptors. All claims supported by specific evidence."

**Proficient (75%):**
- Define the MINIMUM that would satisfy the user without major revision
- Use specific small allowances: "1-2 instances," "90%+," "rare (1 per 1000 words)"
- Example: "1-2 minor subjective terms per 1000 words; these don't appear in key claims. Maintains scholarly register in 90%+ of sentences. No emotional appeals or conversational markers."

**Developing (50%):**
- Define clear PROBLEMS the user would ask to revise
- Use medium-range quantifiers: "3-5 instances," "50-70%," "some paragraphs (2-3)"
- Include specific patterns the user rejected
- Example: "3-5 subjective descriptors per 1000 words. Tone varies between formal and informal across paragraphs. May include 1-2 metaphors or conversational phrases the user would flag. Some key claims lack precision."

**Beginning (25%):**
- Define DEALBREAKERS that would require substantial rework
- Use high-frequency indicators: "6+ instances," "less than 50%," "multiple paragraphs," "frequent"
- Include patterns that contradict user's core values
- Example: "6+ instances of informal language, emotional appeals, or subjective adjectives per 1000 words. Conversational tone in multiple sentences. Multiple metaphors. Would require substantial revision to meet user's standards."

**EXAMPLES OF GOOD VS. BAD LEVEL DEFINITIONS:**

❌ **BAD - Too vague:**
```
"exemplary": "Transitions are clear and effective"
"proficient": "Generally good transitions with minor issues"
"developing": "Some transitions need improvement"
```

✅ **GOOD - Concrete and countable:**
```
"exemplary": "Every paragraph (100%) begins with explicit transition sentence referencing previous paragraph's main point. Uses clear signaling phrases ('This limitation leads to...', 'Building on this insight...', 'As a result...'). No paragraph jumps topics without bridge.",

"proficient": "Most paragraphs (80%+ or 4-5 out of 5) begin with explicit transition. May have 1 paragraph using implicit connection. Problem→solution relationship is signaled with transition language. Reader can follow flow with minimal effort.",

"developing": "Only half of paragraphs (50% or 2-3 out of 5) have explicit transitions. Some paragraphs start new topics without connecting to previous content. Uses weak transitions like 'Additionally' or 'Another point' without explaining relationship. Reader must infer 2+ connections.",

"beginning": "Fewer than half of paragraphs (less than 50%) have transitions. Most paragraphs start abruptly with new topic. No signaling of problem→solution relationship. Reader cannot follow argumentative progression without effort."
```

❌ **BAD - Unmeasurable:**
```
"exemplary": "Writing is concise and precise"
"proficient": "Mostly concise with some redundancy"
```

✅ **GOOD - Measurable:**
```
"exemplary": "Zero redundant phrases or repeated ideas. Uses parallel structures for lists (e.g., 'models, sessions, and tasks'). Every sentence adds new information. Could not be shortened by more than 5% without losing content.",

"proficient": "Minor redundancy (1-2 repeated concepts could be consolidated). 90%+ of sentences add new information. Could be tightened by 5-10% but maintains clarity.",

"developing": "Notable redundancy present (3-5 concepts repeated or restated). Could be shortened by 20-30% without losing core content. Some vague phrases like 'various things' or 'many aspects' instead of specifics.",

"beginning": "Significant redundancy throughout (same ideas stated 3+ times). Could be shortened by 40%+ without content loss. Verbose phrasing obscures main points. Multiple vague descriptors."
```

### Step 6: Extract Concrete Patterns from User Feedback

As you define achievement levels, look for these patterns in the conversation:

**For countable features:**
- Did the user remove/flag specific types of words? (Count these in levels)
- Did they consistently add/remove certain elements? (Make this a threshold)
- Did they reject certain phrases or patterns? (List these as "zero instances" in Exemplary)

**For structural features:**
- Did they consistently request certain organizational patterns? (Make this a requirement)
- Did they rearrange paragraphs or sections? (Specify the pattern)
- Did they ask for explicit connections? (Define what "explicit" means)

**For style features:**
- Did they prefer certain sentence types? (Specify lengths or structures)
- Did they like/dislike specific tones? (Give examples of acceptable/unacceptable)
- Did they value particular rhetorical moves? (Describe exactly what these look like)

### Step 7: Determine Weights

Assign weights (as percentages totaling 100%) based on:

**Evidence of importance:**
- How much conversation time was spent on each criterion
- How strongly the user reacted to issues in each area
- Whether the user explicitly stated priorities
- What they revised most frequently

**Weight distribution patterns:**
- Equal weights (e.g., 5 criteria at 20% each): Use when user showed balanced concern
- Dominant criterion (e.g., 40% + 4 others at 15%): Use when one element clearly mattered most
- Essential + supporting (e.g., 2 at 30% + 3 at ~13%): Use when user indicated non-negotiables

**State your reasoning explicitly:** Connect weights directly to conversation evidence

### Step 8: Self-Validation Check

Before finalizing your rubric, perform this validation on EACH criterion:

**Vagueness Test:**
- [ ] Can I count or measure this feature in text?
- [ ] Would two different evaluators reach the same score?
- [ ] Are the boundaries between levels clear and non-overlapping?
- [ ] Have I avoided words like "generally," "mostly," "some," "often," "rarely"?

**Concreteness Test:**
- [ ] Does each level include specific examples or numbers?
- [ ] Can an LLM identify these features without human judgment?
- [ ] Would the user recognize these patterns from their feedback?

**Distinctness Test:**
- [ ] Are all four levels meaningfully different from each other?
- [ ] Is there a clear threshold between each level?
- [ ] Could a draft plausibly score at any of the four levels?

If any criterion fails these tests, revise it with more concrete descriptors.

### Step 9: Handling Edge Cases

**Minimal conversation history:**
If there's limited evidence, create a minimal rubric (3-4 criteria) with a note in coaching_notes that more conversation is needed for refinement.

**Conflicting signals:**
If the user shows contradictory preferences, note this in coaching_notes and weight toward their most recent or most emphatic signals.

**Overfitting risk:**
If you have only 1-2 conversations, be cautious about over-specifying. Focus on repeatable patterns, not one-off comments.

## Output Format

After your analysis, provide a JSON rubric with this exact structure:
```json
{
  "version": <number: 1 for new rubric, increment by 1 for updates>,
  "writing_type": "<specific description of the writing genre/format>",
  "user_goals_summary": "<2-3 sentence summary of what the user is trying to achieve>",
  "rubric": [
    {
      "name": "<Criterion name>",
      "category": "<High-level category like Content, Style, Structure, etc.>",
      "description": "<User-specific description using conversation language>",
      "exemplary": "<Concrete descriptors with counts/percentages/examples for highest achievement - must be measurable>",
      "proficient": "<Concrete descriptors with counts/percentages/examples for solid achievement - must be measurable>",
      "developing": "<Concrete descriptors with counts/percentages/examples for partial achievement - must be measurable>",
      "beginning": "<Concrete descriptors with counts/percentages/examples for minimal achievement - must be measurable>",
      "weight": <number: percentage as integer, all weights must sum to 100>
    }
  ],
  "weighting_rationale": "<Explanation connecting weights to conversation evidence>",
  "coaching_notes": "<2-3 specific, actionable insights about this user's writing priorities, patterns, or areas for growth based on conversation history>"
}
```

## Critical Reminders

- **Ground everything in evidence**: Every criterion, descriptor, and weight should trace back to something observable in the conversation
- **Be specific, not generic**: Avoid rubric language that could apply to any writer or any writing
- **Respect user priorities**: Don't impose external standards the user hasn't demonstrated caring about
- **Focus on repeatability**: Capture patterns, not random variations
- **Make it actionable**: The rubric should provide clear guidance for future writing, not just evaluation
- **MAKE IT EVALUABLE**: Use counts, percentages, specific examples, and observable patterns - NO vague qualifiers
- **Avoid judgment calls**: An LLM should be able to score a draft reliably without needing to interpret subjective terms like "effective," "strong," or "clear"

## Final Quality Check

Before outputting your rubric, verify:
1. Every achievement level has at least one quantifiable measure (count, percentage, or binary check)
2. Every criterion includes concrete examples at multiple levels
3. No vague quantifiers remain (rare, some, frequent, generally, mostly)
4. The differences between adjacent levels are clear and measurable
5. An LLM could reliably distinguish between levels without human judgment

Provide only the JSON output after your analysis, with no additional text or markdown formatting around the JSON.
"""

RUBRIC_INFERENCE_SYSTEM_PROMPT = """
You are tasked with creating or updating a personalized writing rubric based on the conversation history between a user and an LLM collaboratively developing a piece of writing.

This rubric captures what THIS specific user values — not generic standards of "good writing."

---

## PURPOSE

The rubric will be used to:
1. Align future writing assistance to this user's goals and preferences
2. Allow the user to steer behavior by adjusting criteria, weights, or dimensions
3. Support reliable LLM-based evaluation of drafts

The rubric must be **concise, steerable, and evaluable**.

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
  - `dimensions` (2–4 per criterion)
  - Achievement levels: `exemplary`, `proficient`, `developing`, `beginning`
  - `weight` (integers summing to 100)
- `coaching_notes`

### ❌ DO NOT INCLUDE
- Long rationales or justifications
- Observable-feature checklists
- Detailed examples within the rubric
- Generic writing advice unsupported by conversation evidence

---

## ANALYSIS PROCESS

Before producing the rubric, write your reasoning inside `<analysis>` tags.

### Step 1: Determine Scenario
- **New rubric**: Create from scratch based on conversation evidence.
- **Update**: Preserve stable criteria; modify only where new evidence appears; adjust weights only when priorities clearly shifted.

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

If the conversation includes user reactions to prior rubric evaluations, treat this as **highest-quality signal**.

For each piece of feedback:
1. Identify the criterion discussed
2. Classify the disagreement type: score, interpretation, priority, or definition
3. Extract what features the user emphasized that the criterion missed
4. Update your understanding of what separates achievement levels for this user

---

## DEFINING CRITERIA

Select **4–7 criteria** with clear conversation evidence.

For each candidate, verify:
- Did the user demonstrably care about this?
- Can you point to specific moments?
- Is it distinct from other criteria?

**Do not include** generic principles without user evidence.

---

## CATEGORIES

Use **3–5 shared categories** across all criteria:
- Style
- Structure
- Content
- Evidence
- Audience

Multiple criteria may share the same category.

---

## DESCRIPTIONS (KEEP SHORT)

Write **1–3 sentences** per criterion that:
- Use vocabulary from the conversation
- State what "good" means for THIS user
- Avoid generic phrasing

✅ Good: "Every paragraph opens with mechanism-level language naming the underlying biological or evolutionary phenomenon."

❌ Avoid: "Writing should be clear and effective."

---

## DIMENSIONS (LEAN INTERNAL KNOBS)

Each criterion has **2–4 dimensions** representing adjustable sub-aspects.

Each dimension includes:
- `id`: short, machine-friendly identifier (e.g., `mechanism_first_openings`)
- `label`: short human-readable phrase
- `importance`: 0.0–1.0 indicating weight *within the criterion*

Dimensions should:
- Represent distinct aspects the user might want to strengthen or relax
- Be interpretable by both user and model
- Not duplicate the criterion name or restate achievement-level text

---

## ACHIEVEMENT LEVELS (MUST BE EVALUABLE)

Define four levels: `exemplary`, `proficient`, `developing`, `beginning`

Each level **must**:
- Use concrete, observable patterns
- Include counts, percentages, or clear thresholds where possible
- Avoid vague qualifiers ("generally," "often," "some," "strong," "effective")

✅ Good: "100% of paragraphs open with mechanism-first language; zero instances of jargon without immediate definition."

❌ Avoid: "Topic sentences are strong and engaging."

**Translate subjective judgments into observable behavior.**

---

## WEIGHTS

Assign integer weights summing to **100**.

Base weights on:
- Frequency and intensity of user feedback
- Non-negotiables vs. preferences
- Revision patterns

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
  "writing_type": "<specific genre, audience, constraints>",
  "user_goals_summary": "<2–3 sentence summary>",
  "rubric": [
    {
      "name": "<criterion name>",
      "category": "<shared category>",
      "description": "<1–3 sentence user-specific description>",
      "dimensions": [
        {
          "id": "<machine-friendly id>",
          "label": "<short label>",
          "importance": <0.0–1.0>
        }
      ],
      "exemplary": "<concrete, evaluable>",
      "proficient": "<concrete, evaluable>",
      "developing": "<concrete, evaluable>",
      "beginning": "<concrete, evaluable>",
      "weight": <integer>
    }
  ],
  "coaching_notes": "<2–3 concise insights>"
}
```
"""

def get_rubric_inference_user_prompt(conversation_text, previous_rubric_json=""):
    return f"""
      Here is the conversation you need to analyze:

      <conversation>
      {conversation_text}
      </conversation>

      Here is the previous rubric (this may be empty if you're creating a new rubric from scratch):

      <previous_rubric>
      {previous_rubric_json}
      </previous_rubric>"""

# System prompt for generating contrastive text pairs
CONTRASTIVE_TEXT_GENERATION_SYSTEM_PROMPT = """
You are a writing assessment specialist helping to refine rubric criteria through
preference elicitation. Your task is to generate contrastive pairs of paragraphs
that explore different interpretations or approaches to a given rubric criterion.

For each request, you will generate two paragraph examples (Version A and Version B):

- Both versions should be HIGH-QUALITY writing that could reasonably be considered
  to satisfy the rubric criterion
- Each version should represent a DIFFERENT valid interpretation or approach to
  meeting the criterion
- The contrast should help reveal what the user truly values within this criterion

Critical constraints:
- Keep paragraphs similar in length (4-6 sentences)
- Use the same general argument, topic, and supporting content in both versions
- ONLY vary the dimension being tested - keep everything else as similar as possible
- The difference should represent legitimate stylistic or interpretive choices
- Do not label or explain which is which - let the examples speak for themselves

Format your response EXACTLY as follows:
TEXT A:
[your text here]

EXPLANATION A:
[brief 1-2 sentence explanation of why this text demonstrates good performance]

TEXT B:
[your text here]

EXPLANATION B:
[brief 1-2 sentence explanation of why this text demonstrates good performance]"""

def get_contrastive_text_generation_user_prompt(writing_task, dimension, criteria_description, base_rubric=None):
    """Generate user prompt for contrastive text generation."""
    prompt = f"""Writing task: {writing_task}

Rubric criterion to refine:
Title: {dimension}
Description: {criteria_description}

Generate a contrastive pair exploring different valid approaches to this criterion."""

    if base_rubric:
        prompt += f"\n\nConsider this existing rubric for context:\n{json.dumps(base_rubric, indent=2)}"

    return prompt

# System prompt for preference analysis and rubric refinement
PREFERENCE_ANALYSIS_SYSTEM_PROMPT = """You are a rubric refinement specialist. Your task is to analyze user preferences
and refine rubric criteria to better capture what the user actually values in writing.

You will be given:
1. An original rubric criterion (name, category, description, priority)
2. A writing task context
3. A preference summary showing which text examples the user preferred vs rejected

Your task:
1. Analyze the preference data to identify patterns in what the user consistently
   chose or avoided
2. Infer the underlying preferences that explain their choices
3. Refine the criterion DESCRIPTION to more precisely capture these preferences
4. Generate concrete EVIDENCE examples that illustrate what to look for based on
   the inferred preferences

Guidelines for refinement:
- Make the description more specific and actionable based on observed preferences
- If the user's preferences suggest they value something not in the original
  description, incorporate it
- If the original description mentioned something the user's choices don't support,
  de-emphasize or remove it
- Keep the refined description concise but precise (2-4 sentences)
- Evidence should be specific, concrete examples of what to look for (not full paragraphs)

Output format:
Return ONLY a valid JSON object with this exact structure:
{
  "name": "[keep original name]",
  "category": "[keep original category]",
  "description": "[refined description based on preferences]",
  "priority": "[keep original priority]",
  "evidence": ["Example 1", "Example 2", ...]
}
"""

def get_preference_analysis_user_prompt(writing_task, criterion_name, criterion_category,
                                       criterion_description, criterion_priority, pref_summary):
    """Generate user prompt for preference analysis."""
    return f"""
Writing task: {writing_task}

Original criterion:
{{
  "name": {criterion_name},
  "category": {criterion_category},
  "description": {criterion_description},
  "priority": {criterion_priority}
}}

Preference summary:
{pref_summary}

Analyze these preferences and refine the criterion description to better capture what the user values."""

def get_comparison_prompt(last_assistant_content, comparison_rubric_version, compare_rubric_list):
    """Generate prompt for comparing rubric revisions."""
    return f"""Looking at this base draft:

{last_assistant_content}

If I were to revise the base draft using Rubric v{comparison_rubric_version}, what specific changes would you make and why?

Provide:
1. Starting from the base draft, revice to meet Rubric v{comparison_rubric_version} and incorporate your suggestions.
2. Brief explanation of the key changes

Apply this rubric to your revisions:
{json.dumps(compare_rubric_list, ensure_ascii=False, indent=2)}

========================
OUTPUT FORMAT (STRICT)
========================
Return sections in this exact order and headings:

#### Rubric Revision
<paste the full revised text here. Mark word-level additions with +text+ and removals with ~text~ relative to the base draft.>

**CRITICAL - MANDATORY ANNOTATION REQUIREMENT:**
Everytime you provide a draft, revision, or any actual content (not just explanations), you MUST wrap each portion of text with <N> tags (opening tag) and </N> tags (closing tag) where N corresponds to the 1-based position of the criterion in the rubric above (1 = first criterion, 2 = second, etc.).

For example, if the rubric has 6 criteria and you mention a statistic, wrap it like this:
<2>Research from Virginia Commonwealth University shows employees report 11% lower stress levels.</2>

If a sentence fulfills multiple criteria, use nested tags:
<2><3>This data-driven approach maintains accessibility for general audiences.</3></2>

You MUST annotate EVERY response that contains any draft, revision, or substantial content.

### Summary of Impact
- In 3–5 bullets, explain how Rubric v{comparison_rubric_version} affected tone, concision, evidence, structure, or polish.
- Mention any additions/deletions and why they were necessary for the rubric.

"""

def build_system_instruction(rubric):
    """Build system instruction with rubric (without assessment requirements)."""
    rubric_block = ""
    if rubric:
        # Number the criteria explicitly for clarity
        numbered_rubric = []
        for idx, criterion in enumerate(rubric, start=1):
            numbered_crit = criterion.copy()
            numbered_crit['index'] = idx
            numbered_rubric.append(numbered_crit)

        rubric_block = "\nRUBRIC (Always follow these criteria while co-writing):\n" + json.dumps(numbered_rubric, ensure_ascii=False, indent=2)

    system_instruction = dedent(f"""
    You are an AI co-writer designed to collaborate with human users to improve and develop their written pieces. Your role is to work together with the user to enhance their writing—not to write it entirely for them.
    {rubric_block}

    Your task is to provide helpful, concrete feedback that follows these core interaction principles:

    **INTERACTION PRINCIPLES:**
    1. **Ask clarifying questions** to reduce uncertainty about audience, stakes, examples, or constraints when needed
    2. **Provide concrete, line-level edits** rather than abstract or vague advice whenever possible
    3. **Respect all stated constraints** and rubric criteria - if conflicts arise, ask before proceeding
    4. **Do not invent facts** - if claims or data are missing, ask for sources or mark them as `[TODO: ...]`
    5. **Match the user's style and tone** unless directed otherwise - accept shorthand or partial drafts
    6. **Balance constraints and values** - consider both what to avoid and what to emphasize when applying any rubric
    7. **Solicit feedback selectively** - only ask about likes/dislikes after major changes, full drafts, or when user intent is unclear
    8. **Learn from assessment feedback** - if the conversation contains user feedback on rubric assessments, treat this as the highest-priority signal for how to interpret rubric criteria. User corrections on assessments reveal their true values and should override generic interpretations of criterion descriptors.
    9. **Don't assess draft with rubric** 
    10. **Always wrap draft (partial or full) in <draft> </draft> tags**

    **RUBRIC INTERPRETATION GUIDANCE:**

    When applying rubric criteria, remember that:
    - Achievement level descriptors (Exemplary, Proficient, etc.) are calibrated to THIS user's standards, not universal writing quality
    - If the user has provided feedback disagreeing with previous assessments, their corrections define what the criteria actually mean to them
    - User feedback on assessments > explicit rubric descriptors > general writing best practices
    - When in doubt about how strictly to interpret a criterion, err on the side of the user's demonstrated preferences in their feedback

    **If you see assessment feedback in the conversation history:**
    - Note which criteria the user corrected and how
    - Adjust your interpretation of those criteria accordingly
    - If they said a score was too harsh, interpret that criterion more generously
    - If they said a score was too lenient, apply that criterion more strictly
    - If they clarified what they actually value, prioritize those specific features over the general criterion description

    **INSTRUCTIONS:**
    Provide your feedback in a clear, organized manner. Focus on being maximally useful to the user's specific writing task while strictly adhering to the interaction principles.

    **Example Output Structure:**
    [Your systematic analysis of the writing task, covering all the points listed above]

    <draft> 
    [Your draft (if applicable) based on your analysis]
    </draft>

    [Your concrete feedback, suggestions, and/or clarifying questions based on your analysis]
    """).strip()

    return system_instruction


RUBRIC_EDIT_PROMPT = """You are helping the user edit their writing rubric through conversational interaction.

The user will describe changes they want to make to their rubric. Your task is to understand their request and produce a modified rubric that reflects their intent.

Current rubric structure:
{current_rubric}

User's edit request:
{user_request}

Analyze the request carefully. If the request is clear and actionable, respond with ONLY a JSON object in this exact format:

{{
  "changes_summary": "Brief description of what changed",
  "modified_rubric": [
    // Full rubric array with changes applied
  ]
}}

Each criterion in the rubric must have this structure:
{{
  "name": "Criterion name",
  "category": "Category (e.g., Style, Structure, Content)",
  "description": "Description text",
  "dimensions": [
    {{"id": "dimension_id", "label": "Dimension label", "importance": 0.0-1.0}}
  ],
  "exemplary": "Exemplary level description",
  "proficient": "Proficient level description",
  "developing": "Developing level description",
  "beginning": "Beginning level description",
  "weight": 0.0-100.0
}}

Rules:
- Preserve all fields unless explicitly changed by the user
- Maintain weight total of 100% across all criteria
- Keep dimension importance totals at 1.0 (100%) within each criterion
- If adding new criteria, use reasonable default values and adjust other weights proportionally
- If the request is ambiguous or unclear, respond with a clarifying question instead of the JSON
- Only output JSON if you are confident you understand the user's intent

Examples of requests:
- "Change the 'Academic Register' description to emphasize formality more"
- "Add a new criterion for citation quality weighted at 15%"
- "The exemplary level for Logical Flow should require explicit transitions"
- "Remove the metaphor dimension from Academic Register"
- "Increase the weight of Conciseness to 35%"

If unclear, ask questions like:
- "Should I remove weight from other criteria proportionally, or would you like to specify which criteria to adjust?"
- "What specific aspects of citation quality should this criterion evaluate?"
- "Should this change apply to the exemplary level only, or all achievement levels?"
"""

DRAFT_EDIT_RUBRIC_UPDATE_PROMPT = """You are a rubric refinement specialist. The user has manually edited a draft, and your task is to analyze their edits to infer what they value in their writing and suggest updates to their rubric.

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

def get_draft_edit_rubric_update_prompt(current_rubric, original_draft, edited_draft):
    """Generate user prompt for inferring rubric updates from draft edits."""
    return f"""Current rubric:
{json.dumps(current_rubric, ensure_ascii=False, indent=2)}

Original draft (before user edits):
{original_draft}

Edited draft (after user edits):
{edited_draft}

Analyze the user's edits and suggest any rubric updates that would better capture their demonstrated preferences."""


REGENERATE_DRAFT_PROMPT = """You are a writing assistant tasked with revising a draft to better align with an updated rubric.

The user has made changes to their rubric criteria, and you need to revise the existing draft to fulfill those changes.

You will be given:
1. The original rubric (before changes)
2. The updated rubric (after user's changes)
3. The current draft that needs to be revised

Your task:
1. Identify what changed between the original and updated rubric (weights, descriptions, achievement levels, dimensions)
2. Analyze how those changes should affect the draft
3. Revise the draft to better fulfill the updated rubric criteria
4. Explain the key changes you made and why

Guidelines:
- Focus changes on areas affected by the rubric updates
- Preserve the overall structure and content where the rubric hasn't changed
- Make targeted edits rather than complete rewrites
- Ensure the revised draft still flows naturally
- If a criterion's weight increased, give more attention to that aspect
- If achievement level descriptions changed, align the draft to the new standards

Output format:
Return ONLY a valid JSON object with this exact structure:
{
  "rubric_changes_identified": [
    "List of specific changes between old and new rubric"
  ],
  "revision_strategy": "Brief explanation of how you'll approach the revision",
  "revised_draft": "The full revised draft text here",
  "changes_made": [
    "List of specific changes made to the draft and why"
  ]
}
"""


def get_regenerate_draft_prompt(original_rubric, updated_rubric, current_draft):
    """Generate user prompt for regenerating draft based on rubric changes."""
    return f"""Original rubric (before changes):
{json.dumps(original_rubric, ensure_ascii=False, indent=2)}

Updated rubric (after your changes):
{json.dumps(updated_rubric, ensure_ascii=False, indent=2)}

Current draft to revise:
{current_draft}

Analyze the rubric changes and revise the draft to better fulfill the updated criteria."""
