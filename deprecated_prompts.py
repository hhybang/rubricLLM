"""
Deprecated prompts moved from prompts.py.
These are not actively used by app.py but preserved for reference.
"""

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

RUBRIC_EDIT_SYSTEM_PROMPT = """You are helping the user edit their writing rubric through conversational interaction.

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

# --- Deprecated: Separate DP extraction (now combined with rubric inference) ---

DP_EXTRACT_SYSTEM_PROMPT = """
You are analyzing a conversation between a user and an AI writing assistant to identify **decision points** — moments where the user corrected, redirected, or refined the AI's output, revealing their writing preferences.

## What is a Decision Point?

A decision point is a moment in the conversation where:
- The user explicitly rejected or modified something the AI wrote
- The user gave feedback that caused a change in direction
- The user made a choice that reveals what they value in writing

## Your Task

Analyze the numbered conversation below and extract the most important decision points.

## OUTPUT FORMAT

Return a JSON object with this structure:
```json
{
  "decision_points": [
    {
      "id": 1,
      "title": "<brief descriptive title>",
      "dimension": "<tone/structure/detail/clarity/voice/formality/length/etc.>",
      "assistant_message_num": <message number where AI suggestion appeared>,
      "user_message_num": <message number where user's change appeared>,
      "before_quote": "<short quote from assistant's suggestion, ~30 words max>",
      "after_quote": "<short quote showing user's change/feedback, ~30 words max>",
      "summary": "<one sentence: what choice the user made and why it matters>",
      "suggested_criterion_name": "<suggested rubric criterion name this maps to>"
    }
  ],
  "overall_patterns": "<2-3 sentences describing patterns across decision points>",
  "preference_summary": "<2-3 sentences summarizing the user's key writing preferences>"
}
```

## GUIDELINES

- Extract the **most important** decision points only — moments where the user's writing preferences are **strongly and clearly** demonstrated. Maximum **10 decision points**.
- Focus on: explicit rejections, significant rewrites, strong feedback ("I don't like this", "make it more X"), and choices that reveal core preferences. Skip minor wording tweaks, trivial edits, or factual corrections.
- Use the **exact message numbers** from the conversation (e.g. the [Message #N] labels). Each decision point must have `assistant_message_num` and `user_message_num` pointing to real messages.
- Quality over quantity: 5–10 strong decision points are better than 15+ weak ones.
"""

def DP_extract_user_prompt(conversation_text):
    """Build user prompt for Phase 1 DP extraction."""
    return f"""Analyze this conversation and extract the decision points.

<conversation>
{conversation_text}
</conversation>

Return ONLY valid JSON matching the format specified in your instructions. No markdown code fences or preamble."""

def DP_infer_rubric_user_prompt(confirmed_dps_json, previous_rubric_json=""):
    """Build Phase 2 user prompt: infer rubric using confirmed decision points."""
    prev_section = ""
    if previous_rubric_json:
        prev_section = f"""
## Previous Rubric (to update)

{previous_rubric_json}

When updating, preserve stable criteria and only modify where the confirmed decision points provide new evidence.
"""

    return f"""The user has reviewed and confirmed the decision points extracted from their conversation. Use these confirmed preferences to create a personalized writing rubric.

## Confirmed Decision Points

{confirmed_dps_json}

Decision points marked as "correct" confirm the suggested criterion mapping.
Decision points marked as "incorrect" have been remapped by the user to a different criterion — use the user's corrected mapping.
Decision points marked as "not_in_rubric" represent preferences NOT captured by any existing criterion — you MUST create new criteria to capture these.
{prev_section}
Create a rubric following the output format in your system instructions. Return ONLY valid JSON."""
