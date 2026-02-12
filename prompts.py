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
- **⭐⭐⭐ Excellent**: 90%+ of dimensions met
- **⭐⭐ Good**: 75-89% of dimensions met
- **⭐ Fair**: 50-74% of dimensions met
- **◇ Needs Work**: 25-49% of dimensions met
- **☆ Weak**: Less than 25% of dimensions met

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
**Achievement Level**: [Excellent/Good/Fair/Needs Work/Weak]

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
      "achievement_level": "<Excellent/Good/Fair/Needs Work/Weak>",
      "evidence_summary": "<1-2 sentence summary of key evidence>",
      "improvement_explanation": "<What specific changes would check off the unchecked dimensions? Be concrete and actionable.>"
    }
  ],
  "level_counts": {
    "excellent": <count of criteria at excellent>,
    "good": <count>,
    "fair": <count>,
    "needs_work": <count>,
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
- Focus on: explicit rejections, significant rewrites, strong feedback ("I don't like this", "make it more X"), and choices that reveal core preferences. Skip minor wording tweaks, trivial edits, or factual corrections.
- Use the **exact message numbers** from the conversation (e.g. the [Message #N] labels). Each decision point must have `assistant_message_num` and `user_message_num` pointing to real messages.
- Each decision point should map to a `related_rubric_criterion` (one of the criterion names in your rubric). This links the evidence to what you inferred.
- Quality over quantity: 5–10 strong decision points are better than 15+ weak ones.

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

Note: Achievement levels (Excellent/Good/Fair/Needs Work/Weak) are automatically derived from how many dimensions are met:
- Excellent: 90%+ of dimensions met
- Good: 75-89% of dimensions met
- Fair: 50-74% of dimensions met
- Needs Work: 25-49% of dimensions met
- Weak: Less than 25% of dimensions met

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
4. Provide an annotated version showing exactly what changed and why

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
  "change_summary": "2-4 sentence summary for the user: what rubric changes led to what draft changes and why. Write for the user who just clicked 'Log Changes'.",
  "revised_draft": "The full revised draft text here (clean, no annotations)",
  "revised_draft_annotated": "Same content as revised_draft but with inline annotations. Use ONLY these tags: <ins>added text</ins> for insertions, <del>removed text</del> for deletions. Immediately after any changed phrase insert a rubric marker [1], [2], etc. (1-based index into annotated_changes). Example: 'The <del>long-winded</del><ins>concise</ins> version [1] works better.' Keep the draft readable; use ins/del only where something actually changed.",
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
- revised_draft_annotated: Must be valid: every <ins> and <del> must be closed; every [N] must correspond to an index in annotated_changes (1-based). Prefer this over revised_draft_with_markers when you can show clear add/remove inline.
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


def get_rubric_suggestions_from_edit_feedback_prompt(active_rubric_json: str, edited_rubric_json: str, edits_with_feedback: list):
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


def get_apply_suggestion_to_rubric_prompt(active_rubric_json: str, edited_rubric_json: str, suggestion_text: str):
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


def extract_decision_pts(conversation_text, rubric_json=None):
    rubric_context = ""
    if rubric_json:
        rubric_context = f"""
Here is the current rubric that has been inferred from this user's writing preferences:

{rubric_json}

CRITICAL: Every decision point you extract MUST map to a specific criterion in this rubric. The rubric was inferred from this same conversation, so every user writing choice should correspond to something the rubric captures. If a moment doesn't clearly relate to any rubric criterion, skip it — do NOT include decision points with no rubric match.

When identifying decision points, focus on moments that:
- Directly relate to criteria in this rubric (validate or refine them)
- Show the user's priorities when multiple rubric criteria might conflict
- Add nuance or evidence to existing rubric criteria

When listing decision points, include ALL that qualify (see below). Do not limit to a small number; extract every moment where the user made an explicit writing choice that maps to a rubric criterion.
"""

    return f"""Here is a conversation where a user collaborated with an AI to write a piece. Each message is numbered for reference:

{conversation_text}
{rubric_context}
Identify **ALL** moments where the user made an explicit writing choice — do not limit the number. Include every qualifying moment. Prioritize and include:

1. **Rubric-relevant decisions**: Choices that directly relate to rubric criteria (if rubric provided)
2. **User edits**: Places where the model suggested something and the user changed it
3. **User rejections**: Places where the model offered a direction and user went a different way
4. **User selections**: Places where the model offered options and user chose one
5. **Unprompted user changes**: Places where user edited the draft without being prompted

For each moment:
- Reference the EXACT message numbers involved
- before_quote: Copy an EXACT substring from the assistant message (the text the user reacted to). Keep it short (~30 words max) but it must appear verbatim in the conversation so we can highlight it.
- after_quote: Copy an EXACT substring from the user message (their edit or response). Keep it short (~30 words max) but it must appear verbatim in the conversation so we can highlight it.
- Identify what dimension this choice reflects (tone, structure, detail, etc.)
- Note whether the user explained their reasoning (if visible in conversation)
- If a rubric is provided, note which rubric criterion (if any) this decision relates to

Exclude only moments that are:
- Factual corrections (not style preferences)
- Trivial word changes with no clear pattern
- Ambiguous (can't tell what user preferred)

Include every moment where the user made a discernible writing choice (edit, rejection, or selection), even if multiple choices relate to the same dimension (e.g., include each conciseness edit as its own decision point if it's a distinct moment in the conversation). Do not cap or limit the number of decision points.

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
      "related_rubric_criterion": "Name of the rubric criterion this relates to (REQUIRED — every DP must map to a criterion)",
      "rubric_impact": {{{{
        "type": "validates|refines|contradicts",
        "description": "1 sentence explaining the impact: e.g., 'Validates the Conciseness criterion by showing user prefers shorter sentences'"
      }}}}
    }}}}
  ],
  "overall_patterns": "2-3 sentences describing any patterns you notice across all decision points (e.g., user consistently prefers formal tone, user values conciseness, etc.)",
  "rubric_insights": "2-3 sentences about how these decisions relate to the rubric — which criteria are most strongly validated by user behavior, and which have weaker evidence? (only if rubric was provided)"
}}}}
```

IMPORTANT:
- Return ONLY valid JSON, no other text before or after
- Message numbers must be integers that match the [Message #X] labels in the conversation
- Include **ALL** qualifying decision points — do not limit to 3-4; extract every moment where the user made an explicit writing choice
- If a rubric is provided, EVERY decision point MUST have a related_rubric_criterion — do NOT include DPs that don't map to any rubric criterion. The rubric was inferred from this conversation, so all user choices should map to something in it."""


def generate_writing_task_from_conversation_prompt(conversation_text: str) -> str:
    """Prompt for generating a single writing task in the same domain as the conversation.
    The task must be specific to the conversation's domain so the inferred rubric still applies.
    The LLM should return only the writing instruction (1-3 sentences), no JSON or extra text.
    """
    return f"""Below is a collaborative writing conversation between a user and an AI. The user was working on a specific kind of writing in a specific domain (e.g., professional email, academic essay, technical report, short story, memo, grant paragraph).

{conversation_text}

Your job: Write ONE short writing instruction (1-3 sentences) that will be used to generate drafts for a preference check. The task MUST:

1. Be in the EXACT SAME domain and type of writing as this conversation. If the conversation is about emails, the task must be to write an email (e.g. "Write a brief professional email declining a meeting and proposing an alternative."). If it's about a report, the task must be a report excerpt. If it's about a story, the task must be a story passage. Do NOT give a generic "write a short passage" — be as specific as the conversation's domain.
2. Ask for a short but concrete product (e.g., one email, one short paragraph, 2-4 sentences of the same type) so that the rubric inferred from this conversation clearly applies.
3. You may add "Match the tone and style the user prefers" or similar only at the end, but the main instruction must be domain-specific (e.g. "Write a 2-3 sentence opening for a grant proposal that states the problem." not "Write a short passage.").

Output ONLY the writing instruction itself. No preamble, no "Here is the task:", no JSON. Just the instruction.
"""


def generate_writing_task_similar_but_different_prompt(conversation_text: str, project_task_examples: str = "") -> str:
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


def generate_draft_from_coldstart_prompt(writing_task: str, coldstart_text: str) -> str:
    """Generate a draft for the task using ONLY the user's cold-start preference description (no rubric)."""
    return f"""You are a skilled writer. Write a complete draft for the following task, following ONLY the user's stated preferences below. Do not use any rubric or other criteria.

WRITING TASK:
{writing_task}

USER'S PREFERENCES (follow these — they describe how the user wants the writing to sound and what they care about):
{coldstart_text}

Keep the draft to around 100 words (or to the length the task specifies). Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Just the draft itself."""


def generate_draft_generic_prompt(writing_task: str) -> str:
    """Generate a draft for the task with no rubric or user preferences — generic writing."""
    return f"""You are a skilled writer. Write a complete draft for the following task.

WRITING TASK:
{writing_task}

Keep the draft to around 100 words (or to the length the task specifies). Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Just the draft itself. Fulfill the task in a clear, competent way."""


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
            "achievement_level": "<Excellent|Good|Fair|Needs Work|Weak>",
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

def generate_novel_alternatives_prompt(decision_point: dict, dimension: str, writing_type: str = "", user_goals: str = "", rubric_json: str = "", confirmed_criterion: str = "") -> str:
    """
    Step 3: Generate 3 novel text alternatives for a decision point along the identified dimension.

    Args:
        decision_point: Dict with 'before_quote', 'after_quote', and context
        dimension: The dimension to vary (e.g., 'conciseness', 'formality', 'tone')
        writing_type: The type of writing from the rubric (e.g., "professional emails", "academic essays")
        user_goals: The user's goals summary from the rubric
        rubric_json: Full rubric JSON string for context
        confirmed_criterion: The specific rubric criterion this DP maps to

    Returns:
        Prompt string for Claude API
    """
    before_text = decision_point.get('before_quote', '')
    after_text = decision_point.get('after_quote', '')
    summary = decision_point.get('summary', '')
    title = decision_point.get('title', '')

    # Build domain context from rubric
    domain_context = ""
    if writing_type or user_goals:
        domain_context = f"""
WRITING DOMAIN/CONTEXT (from user's rubric):
- Type of writing: {writing_type if writing_type else "Not specified"}
- User's goals: {user_goals if user_goals else "Not specified"}

The alternatives you generate must be appropriate for this domain. Do not generate content that would be out of place for this type of writing.
"""

    # Build rubric context so the model knows the full set of dimensions
    rubric_section = ""
    if rubric_json:
        criterion_note = ""
        if confirmed_criterion:
            criterion_note = f"""
The specific rubric criterion being tested is: "{confirmed_criterion}"
You are varying the alternatives along THIS criterion. Study its description in the rubric to understand exactly what it measures."""
        rubric_section = f"""
THE FULL RUBRIC:
{rubric_json}
{criterion_note}
Use this rubric to understand:
1. What "{dimension}" (the dimension being tested) actually means for this user's writing — look at the criterion description, not just the name
2. What ALL the other criteria are — you must hold ALL of these constant across your 3 alternatives. For example, if the rubric has criteria for "tone," "detail level," and "structure," and you are varying "tone," then detail level and structure must be identical across all 3 versions.
"""

    return f"""You are generating alternative text versions for a controlled preference test.

The goal: determine the user's preference along ONE specific dimension by giving them 3 versions that ONLY differ on that dimension. If the alternatives differ on other qualities too (e.g., one is clearly better-written, more complete, or more natural), the test is ruined because the user will pick based on overall quality rather than the dimension being tested.
{domain_context}{rubric_section}
CONTEXT:
The user was working on a writing task. Here is what was happening at this point:
- Title: {title if title else "N/A"}
- Summary: {summary if summary else "A piece of text that the user edited to better match their preferences."}

ORIGINAL AI TEXT:
"{before_text}"

USER'S EDITED VERSION:
"{after_text}"

DIMENSION BEING TESTED: {dimension}

YOUR TASK:
1. First, identify exactly what this passage is doing — what information is being conveyed, what purpose it serves in context, who the audience is. Write this as a clear, specific description (not generic).

2. Then generate 3 NEW text alternatives that:
   - Convey the EXACT same information and serve the EXACT same purpose
   - Are roughly the SAME length (within ~20% of each other)
   - Are ALL equally well-written, natural, and competent — no version should feel awkward, incomplete, or obviously worse
   - Differ ONLY in how they handle "{dimension}"
   - Represent three plausible, natural approaches a skilled writer might take along this dimension

CRITICAL RULES FOR ISOLATION:
- Hold ALL other dimensions constant. If varying "tone," keep the same level of detail, same structure, same formality, same vocabulary complexity. If varying "conciseness," keep the same tone, same formality, same structure.
- Do NOT let quality vary. All 3 must be equally polished. A reader should think "these are all good, I just prefer this style."
- Do NOT make any version obviously incomplete, awkward, or error-filled. The differences should be stylistic choices, not quality differences.
- Each version should feel like something a competent writer would naturally produce — not a forced or artificial style.
- Avoid extreme caricatures. Instead of "absurdly formal" vs "extremely casual," aim for "professional" vs "conversational" vs "warm but direct" — all within the realistic range for this domain.

Return ONLY valid JSON (no markdown code blocks):
{{
    "content_objective": "A specific description of what this text does in context (e.g., 'email declining a meeting and suggesting an alternative time', 'opening paragraph introducing the team's quarterly results'). Be specific enough that someone could understand what they're reading without seeing the original. 10-20 words.",
    "alternatives": [
        {{
            "id": "alt_1",
            "text": "The first alternative text here...",
            "dimension_position": "Where this sits on the {dimension} dimension (e.g., 'more concise', 'more formal', 'warmer')"
        }},
        {{
            "id": "alt_2",
            "text": "The second alternative text here...",
            "dimension_position": "Where this sits (e.g., 'balanced', 'moderate')"
        }},
        {{
            "id": "alt_3",
            "text": "The third alternative text here...",
            "dimension_position": "Where this sits (e.g., 'more detailed', 'more casual', 'more direct')"
        }}
    ],
    "dimension_description": "One sentence explaining what '{dimension}' means here and how these 3 versions differ along it"
}}"""


def score_alternatives_with_rubric_prompt(all_dp_alternatives: dict, rubric_json: str, context: str = "") -> str:
    """
    Rank alternatives for ALL decision points in a single call.
    Each DP specifies which criterion it tests — only evaluate on that criterion's dimensions.

    Args:
        all_dp_alternatives: Dict mapping dp_id -> {"criterion": str, "alternatives": [{"id", "text"}]}
        rubric_json: JSON string of the full rubric
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    formatted = json.dumps(all_dp_alternatives, indent=2)

    return f"""You are ranking text alternatives based on specific criteria from a user's personalized writing rubric.

RUBRIC:
{rubric_json}

CONTEXT: {context if context else "A collaborative writing task."}

DECISION POINTS WITH ALTERNATIVES:
{formatted}

Each decision point (DP) has 3 alternatives and specifies which rubric **criterion** it is testing. The alternatives were designed to vary on that specific criterion while holding everything else constant.

## Dimension-Based Evaluation

Each criterion has **dimensions** — checkable items that are either met (✓) or not met (✗). No partial credit.

## Task

For EACH decision point:
1. Identify the criterion being tested (specified in the DP data)
2. For each of the 3 alternatives, check each **dimension of that criterion only** as met or not met
3. Rank the alternatives from best (1st) to worst (3rd) based on how many dimensions of that criterion are met

## Rules
- Only evaluate on the criterion specified for each DP — ignore other criteria
- Be binary: each dimension is either met or not met
- If you can't find clear evidence a dimension is met, mark it as not met
- Apply the same standard consistently across all 3 alternatives
- Evaluate each DP independently

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "criterion_name": "<the criterion being evaluated>",
            "alternatives": [
                {{
                    "alternative_id": "alt_X",
                    "dimensions_detail": [
                        {{
                            "id": "<dimension id>",
                            "label": "<dimension label>",
                            "met": true/false,
                            "evidence": "<brief quote or reason>"
                        }}
                    ],
                    "dimensions_met": <count met>,
                    "dimensions_total": <total>
                }}
            ],
            "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
            "ranking_reasoning": "Why this ranking based on dimension checks"
        }}
    }}
}}"""


def score_alternatives_with_freetext_prompt(all_dp_alternatives: dict, user_preferences_text: str, context: str = "") -> str:
    """
    Rank alternatives for ALL decision points in a single call based on user's free-text preferences.
    Each DP specifies which dimension it tests — only evaluate on the most relevant preference.

    Args:
        all_dp_alternatives: Dict mapping dp_id -> {"criterion": str, "alternatives": [{"id", "text"}]}
        user_preferences_text: User's self-authored preference description
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    formatted = json.dumps(all_dp_alternatives, indent=2)

    return f"""You are ranking text alternatives based on a user's stated writing preferences.

USER'S STATED PREFERENCES (written by the user themselves):
"{user_preferences_text}"

CONTEXT: {context if context else "A collaborative writing task."}

DECISION POINTS WITH ALTERNATIVES:
{formatted}

Each decision point (DP) has 3 alternatives and specifies which **criterion/dimension** it is testing. The alternatives were designed to vary on that specific dimension while holding everything else constant.

## Task

For EACH decision point:
1. Identify which part of the user's stated preferences (if any) is relevant to the criterion being tested
2. Rank the 3 alternatives from best (1st) to worst (3rd) based on how well they align with that relevant preference
3. If the user's preferences don't mention anything relevant to this criterion, note that and rank as tied or based on any implicit preference you can infer

## Rules
- Focus only on the dimension being tested for each DP
- Be honest if the user's preferences don't cover this dimension
- Evaluate each DP independently

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "criterion_tested": "<the criterion/dimension being evaluated>",
            "relevant_preference": "<quote or paraphrase from user's text that relates to this dimension, or 'Not mentioned'>",
            "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
            "ranking_reasoning": "Why this ranking based on the user's stated preferences"
        }}
    }}
}}"""


def score_alternatives_generic_prompt(all_dp_alternatives: dict, context: str = "") -> str:
    """
    Rank alternatives for ALL decision points in a single call based on generic writing quality only.

    Args:
        all_dp_alternatives: Dict mapping dp_id -> {"criterion": str, "alternatives": [{"id", "text"}]}
        context: Optional context about what the writing is for

    Returns:
        Prompt string for Claude API
    """
    import json
    formatted = json.dumps(all_dp_alternatives, indent=2)

    return f"""You are ranking text alternatives based on general writing quality.

IMPORTANT: You have NO information about the user's specific preferences or rubric.
Evaluate purely on generic, universal writing quality standards.

CONTEXT: {context if context else "A writing task."}

DECISION POINTS WITH ALTERNATIVES:
{formatted}

Each decision point (DP) has 3 alternatives. Evaluate each DP independently.

## Task

For each DP, rank the 3 alternatives from best (1st) to worst (3rd) based on generic writing quality:
- Clarity: Is the meaning clear and unambiguous?
- Correctness: Grammar, spelling, punctuation accuracy
- Coherence: Does it flow logically?
- Appropriateness: Does it seem suitable for the context?

## Rules
- DO NOT assume any preference for formality, length, tone, or style
- You are ranking ONLY on: Is it clear? Is it correct? Does it make sense?
- Evaluate each DP independently

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "ranking": ["<best_alt_id>", "<middle_alt_id>", "<worst_alt_id>"],
            "ranking_reasoning": "Why this ranking based on generic writing quality"
        }}
    }}
}}"""


# =============================================================================
# Evaluate: Infer — Predict user's actual edit (before vs. after)
# Ground truth: user preferred "after" (they edited that way). No synthetic alternatives.
# =============================================================================

def predict_before_after_rubric_prompt(decision_points_data: dict, rubric_json: str) -> str:
    """
    For each DP, score "before" and "after" on the criterion's dimensions only.
    Predict "after" if after meets more dimensions, "before" if before does, "tie" if equal.
    """
    import json
    formatted = json.dumps(decision_points_data, indent=2)
    return f"""You are predicting whether a user would prefer the "before" (AI suggestion) or "after" (user's edit) text at each decision point, using ONLY the user's rubric.

RUBRIC:
{rubric_json}

DECISION POINTS (each has before_quote, after_quote, criterion_name):
{formatted}

TASK:
For EACH decision point, evaluate ONLY the criterion named for that DP:
1. Score "before_quote" on that criterion's dimensions (how many dimensions are met?)
2. Score "after_quote" on that same criterion's dimensions
3. Predict "after" if after meets more dimensions than before, "before" if before meets more, "tie" if equal

Rules:
- Evaluate ONLY the specified criterion for each DP; ignore other criteria
- Be binary per dimension: met or not met
- If the criterion was inferred from this user's behavior, apply it as written — we are testing whether the rubric captures what the user preferred when they made this edit

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "prediction": "after" | "before" | "tie",
            "before_dimensions_met": <int>,
            "after_dimensions_met": <int>,
            "reasoning": "One sentence: why this prediction based on dimension counts"
        }}
    }}
}}"""


def predict_before_after_coldstart_prompt(decision_points_data: dict, coldstart_text: str) -> str:
    """
    For each DP, given ONLY the user's cold-start description, which would this user prefer: before or after?
    """
    import json
    formatted = json.dumps(decision_points_data, indent=2)
    return f"""You are predicting whether a user would prefer the "before" (AI suggestion) or "after" (user's edit) text at each decision point, using ONLY what the user wrote in their cold-start preference description.

USER'S COLD-START PREFERENCE DESCRIPTION (written without seeing the rubric):
"{coldstart_text}"

You have NO access to the rubric or any other information. Use ONLY the text above.

DECISION POINTS (each has before_quote, after_quote, criterion_name):
{formatted}

TASK:
For EACH decision point, decide which version (before or after) better matches the user's STATED preferences above.
- If the cold-start text clearly implies a preference that favors "after", predict "after"
- If it favors "before", predict "before"
- If the cold-start says nothing relevant to this dimension or both are equally consistent, predict "tie"

Be strict: if the user never mentioned anything that would distinguish this choice, say "tie". Do not infer preferences they did not state.

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "prediction": "after" | "before" | "tie",
            "relevant_preference": "Quote or paraphrase from cold-start that applies, or 'Not mentioned'",
            "reasoning": "One sentence: why this prediction"
        }}
    }}
}}"""


def predict_before_after_generic_prompt(decision_points_data: dict) -> str:
    """
    For each DP, which text is better by generic writing quality (clarity, correctness, coherence)?
    No user preferences — purely universal writing quality.
    """
    import json
    formatted = json.dumps(decision_points_data, indent=2)
    return f"""You are predicting which text is better by generic writing quality only: clarity, correctness, coherence, appropriateness. You have NO information about the user's preferences.

DECISION POINTS (each has before_quote, after_quote, criterion_name):
{formatted}

TASK:
For EACH decision point, decide which version is better writing by universal standards:
- Clarity, grammar, flow, suitability for context
- Do NOT prefer one just because it's shorter, more formal, or more casual — only quality
- Predict "after" if after is clearly better, "before" if before is, "tie" if roughly equal

Return ONLY valid JSON (no markdown code blocks):
{{
    "decision_points": {{
        "<dp_id>": {{
            "prediction": "after" | "before" | "tie",
            "reasoning": "One sentence: why this prediction based on generic quality"
        }}
    }}
}}"""


def compare_rubric_to_coldstart_prompt(rubric_json: str, coldstart_text: str) -> str:
    """
    Step 2 of Evaluate: Infer tab.
    Compare rubric criteria against a user's cold-start preference description
    to determine which criteria the user stated vs. which are absent.
    """
    return f"""You are analyzing a user's writing preferences.

TASK: Compare a structured rubric (inferred from the user's editing behavior) against
a free-text "cold-start" description the user wrote BEFORE seeing the rubric.

Determine which rubric criteria the user's cold-start description covers (even partially
or in different words) and which are completely absent.

RUBRIC:
{rubric_json}

USER'S COLD-START PREFERENCE DESCRIPTION (written without seeing the rubric):
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


def classify_behavioral_evidence_prompt(conversation_text: str, rubric_json: str, coldstart_text: str, user_categorizations: str) -> str:
    """
    Step 4 of Evaluate: Infer tab.
    Classify whether user editing behavior provides independent evidence
    for each rubric criterion.
    """
    return f"""You are analyzing a collaborative writing conversation to find behavioral evidence
for writing preferences.

CONTEXT:
A user worked with an AI writing assistant. Separately, a rubric was inferred from their
editing behavior. The user also wrote a "cold-start" preference description BEFORE seeing
the rubric. For each criterion NOT in the cold-start description, the user categorized it as:
- "latent_real": They care about it but couldn't have stated it upfront
- "elicited": They care about it, could have stated it, but didn't think to
- "hallucinated": They do NOT care about it — the model fabricated this criterion

Your job: examine the conversation for BEHAVIORAL EVIDENCE that independently confirms or
disconfirms each rubric criterion. Behavioral evidence means the user ACTED on this preference
through natural interaction patterns — WITHOUT being prompted by the rubric.

SIGNAL TYPES:
Classify each behavioral instance by the type of interaction signal it came from:
- "edit": The user directly modified, rewrote, or restructured text produced by the assistant
- "rejection": The user rejected, declined, or asked to redo an assistant suggestion or draft
- "correction": The user corrected specific wording, tone, structure, or content in the assistant's output
- "explicit_feedback": The user gave direct verbal feedback about preferences (e.g., "I prefer shorter sentences")
- "choice": The user chose between alternatives or consistently accepted/rejected a pattern across multiple turns
- "instruction": The user proactively gave instructions that reflect this preference before seeing output

RUBRIC:
{rubric_json}

USER'S COLD-START DESCRIPTION:
"{coldstart_text}"

USER'S CATEGORIZATIONS OF UNSTATED CRITERIA:
{user_categorizations}

CONVERSATION:
{conversation_text}

INSTRUCTIONS:
For EACH criterion in the rubric (both stated and unstated):
1. Search the conversation for moments where the user's behavior relates to this criterion
2. For each instance, identify the SIGNAL TYPE — what kind of interaction produced this evidence
3. Determine if there is "strong" evidence (clear, repeated behavioral signal),
   "weak" evidence (some signal but ambiguous), or "none" (no relevant behavior found)
4. Quote specific conversation moments as evidence

For criteria the user marked as "hallucinated," behavioral evidence to the contrary is
especially important — it might indicate the criterion IS real but the user is unaware of it.

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "behavioral_analysis": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "user_category": "stated" or "latent_real" or "elicited" or "hallucinated",
            "evidence_strength": "strong" or "weak" or "none",
            "behavioral_instances": [
                {{{{
                    "signal_type": "edit" or "rejection" or "correction" or "explicit_feedback" or "choice" or "instruction",
                    "message_numbers": [<int>, <int>],
                    "description": "<what the user did>",
                    "quote": "<brief relevant quote from conversation>",
                    "supports_criterion": true or false
                }}}}
            ],
            "signal_types_present": ["<list of distinct signal types found for this criterion>"],
            "evidence_summary": "<1-2 sentence summary citing the signal types, e.g. 'Inferred from 3 rejections of formal phrasing and 1 direct edit replacing passive voice'>",
            "behavioral_confirms_category": true or false,
            "confirmation_reasoning": "<1 sentence: does the behavior match the user's self-categorization?>"
        }}}}
    ],
    "overall_summary": {{{{
        "criteria_with_strong_evidence": <int>,
        "criteria_with_weak_evidence": <int>,
        "criteria_with_no_evidence": <int>,
        "signal_type_counts": {{{{
            "edit": <int — total instances across all criteria>,
            "rejection": <int>,
            "correction": <int>,
            "explicit_feedback": <int>,
            "choice": <int>,
            "instruction": <int>
        }}}},
        "hallucinated_with_evidence": ["<list of criterion names user marked hallucinated but behavior suggests otherwise>"],
        "latent_confirmed": ["<list of latent_real criteria confirmed by behavior>"],
        "elicited_confirmed": ["<list of elicited criteria confirmed by behavior>"]
    }}}}
}}}}"""


def generate_draft_from_rubric_prompt(writing_task: str, rubric_json: str) -> str:
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

Output ONLY the draft text — nothing else. No preamble, no questions, no notes, no commentary before or after. Do not ask for clarification or add "Here is the draft:" or similar. Just the draft itself."""


def judge_drafts_per_dimension_prompt(draft_a: str, draft_b: str, rubric_criteria_json: str, user_ratings_json: str) -> str:
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


def generate_degraded_draft_prompt(writing_task: str, rubric_json: str, dimensions_to_violate_json: str) -> str:
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


def grade_rubric_judge_prompt(draft_a: str, draft_b: str, rubric_criteria_json: str, conversation_context: str) -> str:
    """
    LLM judge scores two drafts per-dimension using the user's rubric + conversation context.
    Used in Evaluate: Grade tab Step 3 (rubric-grounded condition).
    """
    return f"""You are an expert writing quality judge. You will evaluate two drafts
against a set of personalized rubric criteria. You also have access to the user's
recent conversation history for additional context about their preferences.

DRAFT A:
{draft_a}

DRAFT B:
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

Return ONLY valid JSON (no markdown code blocks):
{{{{
    "per_criterion": [
        {{{{
            "criterion_name": "<exact name from rubric>",
            "draft_a_score": <1-5>,
            "draft_b_score": <1-5>,
            "reasoning": "<1-2 sentences explaining the scores>"
        }}}}
    ],
    "overall_preference": "A" or "B" or "tie",
    "overall_reasoning": "<1-2 sentences explaining overall preference>"
}}}}"""


def grade_generic_judge_prompt(draft_a: str, draft_b: str) -> str:
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


def refine_rubric_from_evaluation_prompt(
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
      "priority": <integer rank, 1 = most important>
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


# --- Claim 3 Evaluation (Step 6 grading: rubric / cold-start / generic) ---

def claim3_rubric_eval_prompt(task_description: str, rubric_json: str, draft: str) -> str:
    """LLM evaluates draft against user's rubric. Returns prompt for Condition 1."""
    return f"""Please evaluate the following draft against the user's preference rubric.

## Context

Writing task: {task_description}

## User's Preference Rubric

{rubric_json}

## Draft to Evaluate

{draft}

## Dimension-Based Evaluation

Each criterion has dimensions — checkable items that can be marked as met (✓) or not met (✗).

Achievement levels are determined by how many dimensions are met:
- ⭐⭐⭐ Excellent: 100% of dimensions met (all checked)
- ⭐⭐ Good: 75%+ of dimensions met
- ⭐ Fair: 50-74% of dimensions met
- ☆ Weak: Less than 50% of dimensions met

## Evaluation Process

For each criterion in the rubric (ordered by priority, highest first):

1. Read the criterion carefully: Understand what THIS specific user values (not generic writing quality)
2. Check each dimension: For each dimension listed under the criterion, determine if the draft meets it (yes/no)
3. Examine the draft: Look for evidence — direct quotes, structural patterns, tone choices
4. Calculate achievement level: Based on the percentage of dimensions met
5. Document your reasoning: Provide specific evidence for each dimension check

## Assessment Principles

- Be binary on dimensions: Each dimension is either met or not — no partial credit
- Priority matters most: A draft that checks off dimensions on priority #1-3 criteria is stronger
- Be calibrated to THIS user's values
- Default to "not met" when uncertain
- Quote specific evidence

## Required Output Format

For EACH criterion:

### [Criterion Name]
Priority: #[N]

Dimension Checklist:
- [✓/✗] [Dimension label]: [Quote or evidence]

Dimensions Met: [X] of [Y] ([percentage]%)
Achievement Level: [Excellent/Good/Fair/Needs Work/Weak]

Key Evidence:
- [Specific quote from draft]

---

After all criteria, provide JSON summary (no markdown code fence):

{{"criteria_scores": [{{"name": "<criterion name>", "priority": <integer>, "dimensions_met": <count>, "dimensions_total": <total>, "dimensions_detail": [{{"id": "<dimension id>", "label": "<dimension label>", "met": true/false, "evidence": "<brief quote or reason>"}}], "achievement_level": "<Excellent|Good|Fair|Needs Work|Weak>"}}], "overall_assessment": "<2-3 sentence summary>"}}
"""


def claim3_coldstart_eval_prompt(task_description: str, cold_start_text: str, draft: str) -> str:
    """LLM evaluates draft against user's cold-start preference description. Returns prompt for Condition 2."""
    return f"""Please evaluate the following draft against the user's stated writing preferences.

## Context

Writing task: {task_description}

## User's Stated Preferences

The user provided the following description of their writing preferences before any interaction with the system:

---
{cold_start_text}
---

## Draft to Evaluate

{draft}

## Step 1: Extract Dimensions

Read the user's preference description carefully. Extract concrete, checkable dimensions organized into criteria. For each preference the user expressed, create a criterion with specific dimensions that can be checked yes/no.

IMPORTANT: Only extract dimensions from what the user explicitly stated. Do NOT add criteria the user did not mention.

## Step 2: Evaluate

For each extracted criterion, check each dimension against the draft.

Achievement levels: ⭐⭐⭐ Excellent (90%+), ⭐⭐ Good (75-89%), ⭐ Fair (50-74%), ◇ Needs Work (25-49%), ☆ Weak (<25%).

- Be binary on dimensions
- Only evaluate against what the user stated
- Default to "not met" when uncertain
- Quote specific evidence

## Required Output Format

First list Extracted Criteria. Then for EACH criterion give Dimension Checklist, Dimensions Met, Achievement Level, Key Evidence.

After all criteria, provide JSON summary (no markdown code fence):

{{"criteria_scores": [{{"name": "<criterion name>", "dimensions_met": <count>, "dimensions_total": <total>, "dimensions_detail": [{{"id": "<dimension id>", "label": "<dimension label>", "met": true/false, "evidence": "<brief quote or reason>"}}], "achievement_level": "<Excellent|Good|Fair|Needs Work|Weak>"}}], "overall_assessment": "<2-3 sentence summary>"}}
"""


def claim3_generic_eval_prompt(task_description: str, draft: str) -> str:
    """LLM evaluates draft against standard writing quality criteria. Returns prompt for Condition 3."""
    return f"""Please evaluate the following draft against standard writing quality criteria.

## Context

Writing task: {task_description}

## Draft to Evaluate

{draft}

## Standard Writing Quality Criteria

Evaluate the draft against each of the following generic dimensions (not any specific user's personal preferences).

### Clarity
- [ ] Main point is immediately identifiable
- [ ] Sentences are unambiguous and easy to parse
- [ ] Technical terms or jargon are used appropriately for the audience

### Coherence
- [ ] Ideas flow logically from one to the next
- [ ] Transitions between paragraphs are smooth
- [ ] The piece maintains a consistent thread throughout

### Structure
- [ ] Organization is appropriate for the task type
- [ ] Paragraphs are well-scoped (one idea per paragraph)
- [ ] Opening and closing are effective

### Tone & Style
- [ ] Tone is appropriate for the task and likely audience
- [ ] Register is consistent throughout
- [ ] Voice is engaging rather than flat or generic

### Completeness
- [ ] The task is fully addressed
- [ ] No obvious gaps in reasoning or content
- [ ] Appropriate level of detail

### Grammar & Mechanics
- [ ] Free of grammatical errors
- [ ] Punctuation is correct
- [ ] Word choice is precise

Achievement levels: ⭐⭐⭐ Excellent (90%+), ⭐⭐ Good (75-89%), ⭐ Fair (50-74%), ◇ Needs Work (25-49%), ☆ Weak (<25%).

- Be binary on dimensions
- Evaluate against general writing quality only
- Default to "not met" when uncertain
- Quote specific evidence

## Required Output Format

For EACH criterion give Dimension Checklist, Dimensions Met, Achievement Level, Key Evidence.

After all criteria, provide JSON summary (no markdown code fence):

{{"criteria_scores": [{{"name": "<criterion name>", "dimensions_met": <count>, "dimensions_total": <total>, "dimensions_detail": [{{"id": "<dimension id>", "label": "<dimension label>", "met": true/false, "evidence": "<brief quote or reason>"}}], "achievement_level": "<Excellent|Good|Fair|Needs Work|Weak>"}}], "overall_assessment": "<2-3 sentence summary>"}}
"""


def claim3_unified_eval_prompt(
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


def infer_rubric_from_evaluation_prompt(
    rubric_json: str,
    writing_task: str,
    user_preferred_draft: str,
    preferred_was_condition: str,
    blind_scores: dict,
    agreement_6a: dict,
    agreement_6b: dict,
    agreement_6c: dict,
) -> str:
    """Prompt to infer a refined rubric from Claim 3 evaluation results."""
    return f"""You have run a writing evaluation with three drafts (A, B, C) under three conditions: rubric, cold-start preferences, and generic quality. The user preferred one draft and rated all three blindly. LLM grading under each condition was compared to the user.

## Current rubric

{rubric_json}

## Writing task used

{writing_task}

## User's preferred draft

The user preferred **Draft {user_preferred_draft}**, which was written using the **{preferred_was_condition}** condition.

## User's blind satisfaction (1–5 per draft)

{json.dumps(blind_scores, indent=2)}

## Agreement results

- **6a (overall satisfaction):** Kendall τ for rubric / cold-start / generic vs user blind ratings: {agreement_6a.get('corr_rubric')}, {agreement_6a.get('corr_coldstart')}, {agreement_6a.get('corr_generic')}
- **6b (dimension-level):** Rubric-condition LLM vs user dimension checks: {agreement_6b.get('total_agree')}/{agreement_6b.get('total_count')} agreement
- **6c (top-ranked):** Did each condition pick the user's preferred draft? Rubric: {agreement_6c.get('top_rubric')}, Cold-start: {agreement_6c.get('top_coldstart')}, Generic: {agreement_6c.get('top_generic')}

## Your task

Produce a **refined rubric** that incorporates what this evaluation reveals. Output a complete rubric in the same JSON structure as the current one: a single JSON object with key "rubric" whose value is an array of criteria. Each criterion: name, category, description, dimensions (list of {{"id", "label"}}), priority. Return ONLY valid JSON. No markdown code fence or preamble.
"""