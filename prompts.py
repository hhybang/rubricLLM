from textwrap import dedent
import json

RUBRIC_SCORING_PROMPT = """
You are tasked with evaluating a draft of writing against a personalized rubric that captures what the user values in their writing.

You will receive:
1. The draft to be evaluated
2. A rubric with specific criteria, dimensions (checkable items), and priority rankings (1 = most important)

## Dimension-Based Evaluation

Each criterion has **dimensions** — checkable items that can be marked as met (✓) or not met (✗).

Achievement levels are determined by how many dimensions are met:
- **⭐⭐⭐ Excellent**: 100% of dimensions met (all checked)
- **⭐⭐ Good**: 75%+ of dimensions met
- **⭐ Fair**: 50-74% of dimensions met
- **☆ Weak**: Less than 50% of dimensions met

## Evaluation Process

For each criterion:
1. **Read the criterion description**: Understand what THIS user specifically values
2. **Check each dimension**: For each dimension, determine if the draft meets it (yes/no)
3. **Provide evidence**: Quote passages that support your check/uncheck decision
4. **Calculate achievement level**: Based on the percentage of dimensions met

## Analysis Structure

Wrap your detailed analysis in <evaluation> tags:

### [Criterion Name]
**Priority**: #[N]

**Dimension Checklist**:
- [✓/✗] [Dimension label]: [Brief evidence or reason]
- [✓/✗] [Dimension label]: [Brief evidence or reason]
- [etc. for all dimensions]

**Dimensions Met**: [X] of [Y] ([percentage]%)
**Achievement Level**: [Excellent/Good/Fair/Weak]

**Key Evidence**:
- [Specific quote supporting dimension checks]
- [Specific quote supporting dimension checks]

**To improve**:
[What specific changes would check off the unchecked dimensions?]

---

[Repeat for all criteria, ordered by priority (highest priority first)]

## Dimension Checking Guidelines

When checking dimensions:
- **Be binary**: Each dimension is either met or not met — no partial credit
- **Look for evidence**: If you can't find clear evidence the dimension is met, mark it as not met
- **Be consistent**: Apply the same standard to similar features throughout the draft
- **Quote specifics**: Cite actual text that demonstrates whether the dimension is met

## Summary Output

After your evaluation, provide this JSON:
```json
{
  "criteria_scores": [
    {
      "name": "<criterion name>",
      "priority": <integer rank, 1 = most important>,
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
      "achievement_level": "<Excellent/Good/Fair/Weak>",
      "evidence_summary": "<1-2 sentence summary of key evidence>",
      "improvement_explanation": "<What specific changes would check off the unchecked dimensions? Be concrete and actionable.>"
    }
  ],
  "level_counts": {
    "excellent": <count of criteria at excellent>,
    "good": <count>,
    "fair": <count>,
    "weak": <count>
  },
  "top_priorities_status": "<Summary of how the draft performs on the top 2-3 priority criteria>",
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

## Priority-Weighted Assessment

Focus your overall assessment on how the draft performs on the user's highest-priority criteria:
- If top priorities have most dimensions checked: The draft is strong where it matters most
- If top priorities have many unchecked dimensions: Key areas need attention
- Frame improvements around which specific dimensions to check off next

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

## Dimension-Based Evaluation

Each criterion has **dimensions** — checkable items that can be marked as met (✓) or not met (✗).

Achievement levels are determined by how many dimensions are met:
- **⭐⭐⭐ Excellent**: 100% of dimensions met (all checked)
- **⭐⭐ Good**: 75%+ of dimensions met
- **⭐ Fair**: 50-74% of dimensions met
- **☆ Weak**: Less than 50% of dimensions met

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
**Achievement Level**: [Excellent/Good/Fair/Weak]

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
      "priority": <integer rank, 1 = most important>,
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
      "achievement_level": "<Excellent|Good|Fair|Weak>",
      "evidence_summary": "<1-2 sentence summary of key evidence>",
      "improvement_explanation": "<What specific changes would check off the unchecked dimensions? Be concrete and actionable.>"
    }
    // Include ALL criteria, ordered by priority
  ],
  "level_counts": {
    "excellent": <count of criteria at excellent>,
    "good": <count>,
    "fair": <count>,
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

# Prompt for comparing rubrics and generating contrasting revisions
COMPARE_WRITE_EDIT_PROMPT = """You are an editor comparing how two different rubrics influence writing on the same topic.

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
2. Allow the user to steer behavior by adjusting criteria priorities or dimensions
3. Support reliable LLM-based evaluation of drafts via dimension checklists

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
- Mechanics
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
          "label": "<checkable item: what to verify as yes/no>"
        }
      ],
      "priority": <integer rank, 1 = most important>
    }
  ],
  "coaching_notes": "<2–3 concise insights>"
}
```

**NOTE**: Do NOT include `excellent`, `good`, `fair`, or `weak` fields. Achievement levels are derived from dimension counts.
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
  "priority": <keep original priority as integer>,
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

def build_system_instruction(rubric_dict_or_list):
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
        # Number the criteria explicitly for clarity
        numbered_rubric = []
        for idx, criterion in enumerate(rubric, start=1):
            numbered_crit = criterion.copy()
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

    system_instruction = dedent(f"""
    You are an AI co-writer designed to collaborate with human users to improve and develop their written pieces. Your role is to work together with the user to enhance their writing—not to write it entirely for them.
    {rubric_block}
    {template_guidance}
    **RUBRIC AUTHORITY:**
    The rubric represents the user's persistent writing preferences. It is your primary guide for tone, style, structure, and approach. Follow it consistently across all turns.

    - If the user gives a task-specific instruction in conversation (e.g., "expand this paragraph," "add an example here"), follow it for that specific request. These are local instructions, not preference changes.
    - If the user pushes back on your interpretation of a rubric criterion (e.g., "when I say casual I don't mean slangy"), follow their clarification for the remainder of this conversation. Interpretation mismatches like this will be resolved through rubric updates between sessions — you do not need to silently reinterpret criteria on your own.
    - If the user's feedback conflicts with the rubric and it is unclear whether they want a one-time exception or a persistent change, ask: "Should I treat this as a one-time adjustment, or is this a preference you'd like going forward?"
    - Never override or silently deviate from the rubric based on your own judgment about what would be better. If you think a rubric criterion is producing poor results, flag it to the user rather than ignoring it.

    **INTERACTION PRINCIPLES:**
    1. **Ask clarifying questions** when you are uncertain about audience, scope, examples, or constraints — but do not ask about preferences that are already specified in the rubric.
    2. **Provide concrete, line-level edits** rather than abstract advice whenever possible.
    3. **Respect all stated constraints** — if the rubric and a user instruction conflict, ask before proceeding.
    4. **Do not invent facts** — if claims or data are missing, ask for sources or mark them as `[TODO: ...]`.
    5. **Match the rubric's specified style and tone.** If the rubric is silent on a dimension, follow the user's lead from their own writing.
    6. **Be direct and efficient.** Offer feedback and suggestions without excessive hedging or preamble.

    **OUTPUT FORMAT:**
    Always wrap any draft content (partial or full) in <draft> </draft> tags.

    Provide your feedback in a clear, organized manner. Focus on being maximally useful to the user's specific writing task while strictly adhering to the interaction principles.
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
    {{"id": "dimension_id", "label": "Checkable dimension label (yes/no item)"}}
  ],
  "priority": 1-N (unique integer rank, 1 = most important)
}}

Note: Achievement levels (Excellent/Good/Fair/Weak) are automatically derived from how many dimensions are met:
- Excellent: 100% of dimensions met
- Good: 75%+ of dimensions met
- Fair: 50-74% of dimensions met
- Weak: Less than 50% of dimensions met

Rules:
- Preserve all fields unless explicitly changed by the user
- Maintain unique priority rankings (1 = most important, each criterion gets a unique rank)
- If adding new criteria, assign appropriate priority ranking and adjust other priorities as needed
- If the request is ambiguous or unclear, respond with a clarifying question instead of the JSON
- Only output JSON if you are confident you understand the user's intent

Examples of requests:
- "Change the 'Academic Register' description to emphasize formality more"
- "Add a new criterion for citation quality as priority 2"
- "The exemplary level for Logical Flow should require explicit transitions"
- "Remove the metaphor dimension from Academic Register"
- "Make Conciseness the top priority"

If unclear, ask questions like:
- "What priority rank should the new criterion have?"
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
1. Identify what changed between the original and updated rubric (priorities, descriptions, dimensions)
2. Analyze how those changes should affect the draft
3. Revise the draft to better fulfill the updated rubric criteria
4. Explain the key changes you made and why

Guidelines:
- Focus changes on areas affected by the rubric updates
- Preserve the overall structure and content where the rubric hasn't changed
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


def extract_decision_pts(conversation_text, rubric_json=None):
    rubric_context = ""
    if rubric_json:
        rubric_context = f"""
Here is the current rubric that has been inferred from this user's writing preferences:

{rubric_json}

When identifying decision points, PRIORITIZE moments that:
- Directly relate to criteria in this rubric (validate or contradict them)
- Could help refine or add nuance to existing rubric criteria
- Reveal preferences not yet captured in the rubric
- Show the user's priorities when multiple rubric criteria might conflict
"""

    return f"""Here is a conversation where a user collaborated with an AI to write a piece. Each message is numbered for reference:

{conversation_text}
{rubric_context}
Identify 3–4 of the MOST CRUCIAL and NON-REDUNDANT moments where the user made an explicit writing choice. Quality over quantity - each decision point should reveal a distinct preference. Prioritize:

1. **Rubric-relevant decisions**: Choices that directly relate to rubric criteria (if rubric provided)
2. **User edits**: Places where the model suggested something and the user changed it
3. **User rejections**: Places where the model offered a direction and user went a different way
4. **User selections**: Places where the model offered options and user chose one
5. **Unprompted user changes**: Places where user edited the draft without being prompted

For each moment:
- Reference the EXACT message numbers involved
- Quote the model's suggestion or the "before" state (keep quotes short, ~30 words max)
- Quote the user's change or the "after" state (keep quotes short, ~30 words max)
- Identify what dimension this choice reflects (tone, structure, detail, etc.)
- Note whether the user explained their reasoning (if visible in conversation)
- If a rubric is provided, note which rubric criterion (if any) this decision relates to

Avoid moments that are:
- Factual corrections (not style preferences)
- Trivial word changes with no clear pattern
- Ambiguous (can't tell what user preferred)
- Redundant with another decision point (showing the SAME preference twice - e.g., if you already have a "prefers concise language" example, don't include another conciseness example)

CRITICAL: Each decision point must reveal a DISTINCT preference dimension. If the user made 10 edits for conciseness, only include ONE of them. Spread your 3-4 decision points across DIFFERENT preference dimensions (e.g., tone, structure, detail level, formality, etc.).

Return your analysis as a JSON object with the following structure:
```json
{{{{
  "decision_points": [
    {{{{
      "id": 1,
      "title": "Brief descriptive title of the decision",
      "dimension": "tone/structure/detail/clarity/voice/etc.",
      "assistant_message_num": <number of the assistant message with the suggestion>,
      "user_message_num": <number of the user message with the change>,
      "before_quote": "Brief quote from assistant's suggestion",
      "after_quote": "Brief quote showing user's change",
      "user_reason": "Quote from user explaining why, or null if not stated",
      "summary": "One sentence explaining what choice the user made and why it matters",
      "related_rubric_criterion": "Name of the rubric criterion this relates to, or null if none",
      "rubric_impact": {{{{
        "type": "validates|refines|contradicts|suggests_new|none",
        "description": "1 sentence explaining the impact: e.g., 'Validates the Conciseness criterion by showing user prefers shorter sentences' or 'Suggests new criterion around technical terminology preferences'"
      }}}}
    }}}}
  ],
  "overall_patterns": "2-3 sentences describing any patterns you notice across all decision points (e.g., user consistently prefers formal tone, user values conciseness, etc.)",
  "rubric_insights": "2-3 sentences about how these decisions relate to the rubric - do they validate it, contradict it, or suggest new criteria? (only if rubric was provided)"
}}}}
```

IMPORTANT:
- Return ONLY valid JSON, no other text before or after
- Message numbers must be integers that match the [Message #X] labels in the conversation
- Include exactly 3-4 decision points (no more) - choose the most important and non-redundant ones
- If a rubric is provided, prioritize decision points that have rubric implications"""


def generate_reflection_questions_prompt(conversation_text, decision_points_json):
    """Generate reflection questions for all decision points to understand user preferences."""
    return f"""Here is a collaborative writing conversation:

{conversation_text}

Here are decision points where the user made a choice:

{decision_points_json}

For each decision point, I want to understand WHY the user made this choice—not just WHAT they chose.

For each decision point, generate reflection content that:
1. Reminds the user of the specific change they made
2. Identifies what quality/dimension the original had vs. what the user's version has
3. Suggests what underlying preference this might reveal

Return your analysis as JSON:
```json
{{{{
    "reflection_items": [
        {{{{
            "decision_point_id": 1,
            "before_text": "The original text (cleaned up for readability)",
            "after_text": "What the user changed it to (cleaned up for readability)",
            "dimension": "The writing dimension this affects (tone, structure, clarity, etc.)",
            "original_quality": "What quality the original version emphasized (e.g., 'more formal', 'more detailed')",
            "user_quality": "What quality the user's version emphasizes (e.g., 'more conversational', 'more concise')",
            "potential_preference": "A hypothesis about what this choice reveals about the user's preferences"
        }}}}
    ]
}}}}
```

IMPORTANT:
- Return ONLY valid JSON
- Include an entry for each decision point
- Keep before_text and after_text brief but clear (max 50 words each)
- Be specific about the qualities being traded off"""


def extract_preference_dimensions_prompt(user_reflections_json):
    """Extract underlying preference dimensions from user's reflection responses."""
    return f"""Here are a user's explanations for why they made certain writing choices:

{user_reflections_json}

Extract the underlying preference dimensions from these responses.

For each response:
1. Identify the core preference being expressed
2. Generalize it beyond this specific instance
3. Name the dimension (e.g., tone, structure, detail level, directness, formality, etc.)

Return your analysis as JSON:
```json
{{{{
    "decision_point_analyses": [
        {{{{
            "decision_point_id": 1,
            "user_motivation": "What the user said motivated the change",
            "user_reasoning": "What the user said was wrong/better",
            "core_preference": "The specific preference being expressed",
            "generalized_preference": "A preference statement that applies broadly to all writing",
            "dimension": "The writing dimension name"
        }}}}
    ],
    "preference_dimensions": [
        {{{{
            "dimension": "Dimension name (e.g., Tone, Structure, Detail Level)",
            "preference_statement": "Clear statement of what the user prefers",
            "evidence_count": 1,
            "confidence": "high/medium/low"
        }}}}
    ]
}}}}
```

IMPORTANT:
- Return ONLY valid JSON
- Group similar preferences into the same dimension
- Preference statements should be actionable (e.g., "Prefer concise sentences over elaborate ones")
- Only include dimensions with clear evidence from the user's responses
- Confidence is based on how explicitly the user stated this preference"""


def generate_test_comparisons_prompt(preference_dimensions_json, original_context="", writing_type="", user_goals=""):
    """Generate test comparisons to validate whether a rubric captures the user's preferences."""
    context_parts = []
    if writing_type:
        context_parts.append(f"Writing type: {writing_type}")
    if user_goals:
        context_parts.append(f"User's goals: {user_goals}")
    if original_context:
        context_parts.append(f"Original context: {original_context}")

    context_note = "\n\n**DOMAIN CONTEXT (CRITICAL - all test comparisons MUST be in this domain):**\n" + "\n".join(context_parts) if context_parts else ""

    return f"""Here are a user's writing preference dimensions:

{preference_dimensions_json}{context_note}

Generate a set of test comparisons to validate whether a rubric captures these preferences.

For each preference dimension, create:
1. A brief writing context that is DIRECTLY related to the user's writing domain (see DOMAIN CONTEXT above)
2. Two versions of that writing that differ ONLY on this dimension
3. Version A should ALIGN with the user's stated preference
4. Version B should go AGAINST the user's stated preference

**CRITICAL - Domain alignment:**
- ALL test comparisons MUST be in the SAME domain/genre as the user's rubric
- If the rubric is for emails, create email examples
- If the rubric is for technical documentation, create technical documentation examples
- If the rubric is for creative writing, create creative writing examples
- Do NOT create generic examples like "blog posts about productivity" unless that's the user's actual domain
- The test writing should feel like something the user would actually write given their stated goals

Other guidelines:
- Keep the content/meaning identical between versions
- Only vary the dimension being tested
- Both versions should be competent writing (not "good vs. bad")
- Make the contrast clear but realistic
- The writing should NOT be the exact same text they already wrote, but should be a NEW example in the same domain
- Each version should be 2-4 sentences

Return your test comparisons as JSON:
```json
{{{{
    "test_comparisons": [
        {{{{
            "test_id": 1,
            "dimension": "The dimension being tested",
            "preference_statement": "The user's stated preference",
            "context": "What this writing is for (brief description)",
            "version_aligned": {{{{
                "text": "The text version that aligns with user's preference",
                "description": "Brief description of why this aligns"
            }}}},
            "version_against": {{{{
                "text": "The text version that goes against user's preference",
                "description": "Brief description of why this goes against"
            }}}},
            "predicted_choice": "aligned"
        }}}}
    ]
}}}}
```

IMPORTANT:
- Return ONLY valid JSON
- Create one test for each preference dimension
- Make versions similar in length and quality
- The only difference should be the dimension being tested"""


def format_user_tests_prompt(test_comparisons_json):
    """Format test comparisons for neutral user-facing presentation with randomized order."""
    return f"""Here are test comparisons to show a user:

{test_comparisons_json}

Reformat these for user testing:
1. Remove all labels that indicate which version aligns with their preference
2. Randomly assign which version appears as Version 1 vs Version 2 (vary this across tests)
3. Create neutral, unbiased presentation

Return the formatted tests and answer key as JSON:
```json
{{{{
    "user_tests": [
        {{{{
            "test_id": 1,
            "context": "What this writing is for",
            "version_1": {{{{
                "text": "First version text"
            }}}},
            "version_2": {{{{
                "text": "Second version text"
            }}}},
            "questions": [
                "Which version do you prefer?",
                "What makes that version better?"
            ]
        }}}}
    ],
    "answer_key": [
        {{{{
            "test_id": 1,
            "dimension": "The dimension being tested",
            "preference_predicts": "Version 1 or Version 2",
            "aligned_version": "Which version number matches the user's preference"
        }}}}
    ]
}}}}
```

IMPORTANT:
- Return ONLY valid JSON
- Randomize the order (don't always put aligned version first)
- The user_tests should have NO indication of which version is "correct"
- The answer_key is for researcher use only"""


def score_tests_with_rubric_prompt(rubric_json, test_comparisons_json):
    """Score test writing samples using specific rubric criteria to predict user preferences."""
    return f"""Here is an inferred rubric:

{rubric_json}

Here are test comparisons, each targeting a specific preference dimension:

{test_comparisons_json}

For each test:

1. Identify which rubric criterion (if any) is most relevant to this dimension
   - Look for criteria that directly relate to the dimension being tested (e.g., if testing "Tone", find a criterion about tone, formality, voice, etc.)
   - If no criterion matches, note "No matching criterion"

2. If a criterion matches, determine what it predicts:
   - Based on that criterion's description/preference, which version should score higher on THIS criterion alone?
   - Version A is the "aligned" version (matches user's stated preference)
   - Version B is the "against" version (opposite of user's stated preference)

Return your analysis as JSON:
```json
{{{{
    "test_scores": [
        {{{{
            "test_id": 1,
            "dimension": "The dimension being tested",
            "preference_statement": "The user's stated preference for this dimension",
            "context": "The writing context",
            "matching_criterion": "The rubric criterion name that matches this dimension, or null if none",
            "matching_criterion_description": "The description/preference from the rubric criterion, or null if none",
            "has_matching_criterion": true,
            "rubric_predicts": "Version A",
            "reasoning": "Why this criterion favors that version based on its description"
        }}}}
    ],
    "summary": {{{{
        "total_tests": 5,
        "tests_with_matching_criteria": 4,
        "tests_without_matching_criteria": 1,
        "rubric_predicts_aligned": 3,
        "rubric_predicts_against": 1
    }}}}
}}}}
```

IMPORTANT:
- Return ONLY valid JSON
- Focus on finding the SINGLE most relevant criterion for each test dimension
- Only predict based on that specific criterion, NOT the overall rubric
- If no criterion matches, set has_matching_criterion to false and rubric_predicts to null
- The rubric_predicts field should be "Version A" or "Version B" based on what the matching criterion suggests"""


ALIGNMENT_SCORING_PROMPT = """
Please evaluate the provided draft against the rubric criteria.

## Purpose

You are scoring a draft to measure alignment between human and AI interpretation of rubric criteria. Your scores will be compared against a human's scores to assess whether the rubric language is clear and unambiguous.

## Scoring System

Achievement levels are determined by how many dimensions are met for each criterion:
- **Excellent (100%)**: All dimensions are met - fully realizes the user's vision
- **Good (75%+)**: Most dimensions are met - meets core requirements with minor gaps
- **Fair (50-74%)**: Some dimensions are met - shows awareness but needs significant work
- **Weak (<50%)**: Few/no dimensions are met - misses or contradicts what the user values

## Evaluation Process

For each criterion in the rubric:

1. **Read the criterion carefully**: Understand what THIS specific user values (not generic writing quality), what the criterion description says and review all four achievement level descriptors.

2. **Identify specific evidence in the draft**: Find exact quotes or passages that relate to this criterion. Mark the start and end positions of each piece of evidence.

4. **Calculate the achievement level**: Based on what percentage of dimensions are met:
   - **Weak vs. Fair**: Are at least half the dimensions met?
   - **Fair vs. Good**: Are at least 75% of dimensions met?
   - **Good vs. Excellent**: Are ALL dimensions met?
   
4. **Document your reasoning**: Explain why you chose this level, referencing the rubric's specific language.

## Required Output Format

Return a JSON object with the following structure:

```json
{{
    "criteria_scores": [
        {{
            "name": "<exact criterion name from rubric>",
            "achievement_level": "<Excellent|Good|Fair|Weak>",
            "level_percentage": <25|50|75|100>,
            "evidence": [
                {{
                    "quote": "<exact text from the draft that serves as evidence>",
                    "start_index": <character position where quote starts in draft>,
                    "end_index": <character position where quote ends in draft>,
                    "relevance": "<brief explanation of how this quote relates to the criterion>"
                }}
            ],
            "evidence_summary": "<1-2 sentence summary of key evidence>",
            "rationale": "<explanation of why this level was chosen, referencing rubric language>"
        }}
    ],
    "evidence_highlights": [
        {{
            "quote": "<exact text from draft>",
            "start_index": <start position>,
            "end_index": <end position>,
            "criteria": ["<criterion name 1>", "<criterion name 2>"]
        }}
    ]
}}
```

## Important Instructions

1. **Quote exactly**: The "quote" field must contain the EXACT text from the draft, character-for-character.

2. **Accurate positions**: The start_index and end_index must be accurate character positions in the draft (0-indexed). These will be used to highlight the text in the UI.

3. **Evidence can overlap**: A single passage may be evidence for multiple criteria. Include it in each criterion's evidence array and list all relevant criteria in the evidence_highlights array.

4. **Be thorough**: Identify ALL relevant evidence for each criterion, not just one example.

5. **Score EVERY criterion**: Include a score for every criterion in the rubric.

6. **Return ONLY valid JSON**: No additional text before or after the JSON object.

## Scoring Principles

- **Be calibrated to the rubric's language**: Score based on what the rubric says, not general writing quality
- **Be consistent**: Apply the same standards to similar passages
- **Default to lower levels when uncertain**: Better to under-score than over-score
- **Quote specific evidence**: Vague assessments aren't helpful; cite actual text with positions"""


# =============================================================================
# Evaluate: Coverage Tab - 9-Step Workflow Prompts
# =============================================================================

def generate_novel_alternatives_prompt(decision_point: dict, dimension: str) -> str:
    """
    Step 3: Generate 3 novel text alternatives for a decision point along the identified dimension.

    Args:
        decision_point: Dict with 'before_quote', 'after_quote', and context
        dimension: The dimension to vary (e.g., 'conciseness', 'formality', 'tone')

    Returns:
        Prompt string for Claude API
    """
    before_text = decision_point.get('before_quote', '')
    after_text = decision_point.get('after_quote', '')
    context = decision_point.get('context', '')

    return f"""You are generating alternative text versions for preference testing.

CONTEXT OF THE PASSAGE:
{context if context else "A collaborative writing conversation between a user and an AI assistant."}

ORIGINAL AI TEXT:
"{before_text}"

USER'S EDITED VERSION:
"{after_text}"

DIMENSION BEING VARIED: {dimension}

YOUR TASK:
Generate 3 NEW text alternatives that:
1. Communicate the same core content/meaning as both versions above
2. Vary ONLY on the "{dimension}" dimension
3. Are DISTINCTLY different from each other on this dimension
4. Are NOT identical (or nearly identical) to either the original AI text or user's edited version
5. Are all competent, reasonable writing (not obviously bad or error-filled)

IMPORTANT GUIDELINES:
- Create a spectrum: one alternative at one extreme of the dimension, one at the other extreme, one in the middle
- Examples by dimension:
  - "conciseness": very concise (minimal words), moderate length, verbose (fully detailed)
  - "formality": very formal/professional, moderate, casual/conversational
  - "tone": warm/enthusiastic, neutral, direct/matter-of-fact
  - "detail": high-level overview, balanced, granular with specifics
  - "structure": simple/flowing, moderately organized, highly structured with headers/bullets
- Each alternative should be a plausible version someone might actually write
- Keep the core message/information the same across all alternatives

Return ONLY valid JSON (no markdown code blocks):
{{
    "alternatives": [
        {{
            "id": "alt_1",
            "text": "The first alternative text here...",
            "dimension_position": "Description of where this sits on the dimension (e.g., 'very concise', 'formal', 'warm')"
        }},
        {{
            "id": "alt_2",
            "text": "The second alternative text here...",
            "dimension_position": "moderate"
        }},
        {{
            "id": "alt_3",
            "text": "The third alternative text here...",
            "dimension_position": "Description of opposite extreme (e.g., 'verbose', 'casual', 'direct')"
        }}
    ],
    "dimension_description": "Brief explanation of what the '{dimension}' dimension means and how these alternatives vary along it",
    "generation_notes": "Brief notes on how the alternatives were constructed to ensure they differ from original/user versions"
}}"""


def score_alternatives_with_rubric_prompt(alternatives: list, rubric_json: str, context: str = "") -> str:
    """
    Step 5: LM-as-judge scores 3 alternatives against the active rubric.

    Args:
        alternatives: List of dicts with 'id' and 'text' keys
        rubric_json: JSON string of the full rubric
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    alternatives_formatted = json.dumps(alternatives, indent=2)

    return f"""You are evaluating text alternatives against a user's personalized writing rubric.

RUBRIC:
{rubric_json}

CONTEXT: {context if context else "A collaborative writing task."}

ALTERNATIVES TO EVALUATE:
{alternatives_formatted}

TASK:
1. For EACH alternative, score it against EACH criterion in the rubric
2. For each criterion, evaluate how well the alternative meets the described dimensions/qualities
3. Calculate a total weighted score (higher priority criteria matter more)
4. Produce a final ranking from best (most aligned with rubric) to worst

SCORING RULES:
- Score each criterion 0-100 based on how well the text fulfills that criterion
- Weight by priority: priority 1 = 3x weight, priority 2 = 2x weight, others = 1x weight
- Provide brief reasoning for each criterion score
- The ranking should reflect rubric alignment, not generic quality

Return ONLY valid JSON (no markdown code blocks):
{{
    "scores": [
        {{
            "alternative_id": "alt_X",
            "total_score": <weighted average 0-100>,
            "criterion_scores": [
                {{
                    "criterion_name": "<exact name from rubric>",
                    "criterion_priority": <1-N>,
                    "score": <0-100>,
                    "reasoning": "Brief explanation of why this score"
                }}
            ]
        }}
    ],
    "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
    "ranking_reasoning": "Explanation of why this ranking based on rubric criteria alignment"
}}"""


def score_alternatives_with_freetext_prompt(alternatives: list, user_preferences_text: str, context: str = "") -> str:
    """
    Step 6: LM-as-judge scores 3 alternatives against user's free-text preference description.

    Args:
        alternatives: List of dicts with 'id' and 'text' keys
        user_preferences_text: User's self-authored preference description (from Task A Survey Q4)
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    alternatives_formatted = json.dumps(alternatives, indent=2)

    return f"""You are evaluating text alternatives against a user's stated writing preferences.

USER'S STATED PREFERENCES (written by the user themselves):
"{user_preferences_text}"

CONTEXT: {context if context else "A collaborative writing task."}

ALTERNATIVES TO EVALUATE:
{alternatives_formatted}

TASK:
1. First, identify the key preferences mentioned in the user's description (extract 3-6 distinct preferences)
2. For EACH alternative, score it on how well it aligns with EACH identified preference
3. Calculate a total alignment score (average across all preferences)
4. Produce a final ranking from best (most aligned with stated preferences) to worst

SCORING RULES:
- Extract clear, distinct preferences from the user's text
- Score each preference 0-100 for each alternative
- Give equal weight to all identified preferences
- Provide brief reasoning for each score
- Be honest about ambiguity - if the preference is unclear, note that

Return ONLY valid JSON (no markdown code blocks):
{{
    "identified_preferences": [
        {{
            "preference": "Brief description of the preference",
            "source_quote": "The part of user's text this came from"
        }}
    ],
    "scores": [
        {{
            "alternative_id": "alt_X",
            "total_score": <average 0-100>,
            "preference_scores": [
                {{
                    "preference": "<preference description>",
                    "score": <0-100>,
                    "reasoning": "Brief explanation"
                }}
            ]
        }}
    ],
    "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
    "ranking_reasoning": "Explanation of ranking based on stated preferences"
}}"""


def score_alternatives_generic_prompt(alternatives: list, context: str = "") -> str:
    """
    Step 7: LM-as-judge scores 3 alternatives with NO preference info - generic quality only.

    Args:
        alternatives: List of dicts with 'id' and 'text' keys
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    alternatives_formatted = json.dumps(alternatives, indent=2)

    return f"""You are evaluating text alternatives for general writing quality.

IMPORTANT: You have NO information about the user's specific preferences.
Evaluate purely on generic, universal writing quality standards.

CONTEXT: {context if context else "A writing task."}

ALTERNATIVES TO EVALUATE:
{alternatives_formatted}

TASK:
1. Score each alternative on these GENERIC writing quality dimensions:
   - Clarity: Is the meaning clear and unambiguous?
   - Correctness: Grammar, spelling, punctuation accuracy
   - Coherence: Does it flow logically? Are ideas connected well?
   - Appropriateness: Does it seem suitable for the context?
2. Calculate a total quality score (average of the 4 dimensions)
3. Produce a ranking from best generic quality to worst

CRITICAL RULES - DO NOT assume any preference for:
- Formality level (formal and casual are equally valid if clear)
- Length (concise and detailed are equally valid if appropriate)
- Tone (warm and neutral are equally valid)
- Style choices (bullets vs prose, etc.)

You are measuring ONLY: Is it clear? Is it correct? Does it make sense? Does it fit the context?

Return ONLY valid JSON (no markdown code blocks):
{{
    "scores": [
        {{
            "alternative_id": "alt_X",
            "total_score": <average 0-100>,
            "quality_scores": [
                {{"dimension": "Clarity", "score": <0-100>, "reasoning": "..."}},
                {{"dimension": "Correctness", "score": <0-100>, "reasoning": "..."}},
                {{"dimension": "Coherence", "score": <0-100>, "reasoning": "..."}},
                {{"dimension": "Appropriateness", "score": <0-100>, "reasoning": "..."}}
            ]
        }}
    ],
    "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
    "ranking_reasoning": "Explanation based ONLY on generic quality (not style preferences)"
}}"""