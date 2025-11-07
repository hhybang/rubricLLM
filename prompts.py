"""
Prompts for the rubricLLM application.
This module contains all prompts used throughout the application.
"""

from textwrap import dedent
import json


# Prompt for comparing rubrics and generating contrasting revisions
COMPARE_WRITE_EDIT_PROMPT = r"""
You are an editor comparing how two rubrics influence a piece of writing.
Your job is to make the effects of each rubric clearly visible to a reader.

========================
RUBRIC A
========================
{rubric_a}

========================
RUBRIC B
========================
{rubric_b}

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

User Writing Task: {task}
"""


# System prompt for inferring rubrics from conversations
RUBRIC_INFERENCE_SYSTEM_PROMPT = """You are an expert writing coach and rubric designer.
Your task is to analyze conversations between users and AI writing assistants, then either create a new writing rubric or update an existing one based on the writing preferences and signals you discover.

Your approach will depend on whether a previous rubric exists:
- **If no previous rubric exists**: Create a completely new rubric based on the conversation
- **If a previous rubric exists**: Update the existing rubric incrementally, preserving continuity while incorporating new insights

Before creating your final rubric, wrap your systematic analysis in <analysis> tags. Follow these steps in your analysis. It's OK for this section to be quite long.

**Step 1: Scenario Identification**
Determine whether you're creating a new rubric from scratch or updating an existing one.

**Step 2: Evidence Gathering**
Extract and quote the most relevant parts of the conversation that demonstrate writing preferences. For each piece of evidence, quote it directly and explain what it reveals:
- Direct quotes showing explicit preferences (clear statements about what the user likes/dislikes)
- Examples of implicit approval signals (acceptance, positive responses, moving forward without critique)
- Preferences for style, tone, structure, and content
- Any constraints or requirements mentioned
- Patterns in what the user accepts versus rejects

**Step 3: Existing Rubric Review** (if updating)
For each criterion in the existing rubric, systematically evaluate it by:
- Stating the current criterion name and description
- Quoting conversation evidence that supports keeping it unchanged
- Quoting conversation evidence that suggests modifications are needed
- Quoting conversation evidence that suggests it should be removed
- Making a final decision: keep unchanged, modify (explain how), or remove (explain why)

**Step 4: New Criteria Identification**
Based on conversation evidence, identify potential new criteria that aren't covered by existing ones. For each potential new criterion, quote the specific conversation evidence that supports it.

**Step 5: Category Planning**
Group your planned criteria into logical categories (e.g., "Structure", "Style", "Content", "Tone", etc.) that will help organize the rubric.

**Step 6: Criteria Planning**
For each final criterion you plan to include:
- State the criterion name and description
- Identify which category it belongs to
- Quote specific conversation evidence that supports it
- Assign a point value between -10 and 10

**Step 7: Final Validation**
Count the number of criterias with positive points and negative points. Make sure there is a balance. 

**Point Value Guidelines:**
- Positive values (1-10): Desirable behaviors, with higher values for more important criteria
- Negative values (-1 to -10): Undesirable behaviors, with lower values for behaviors that should be strongly avoided
- Consider the relative importance when assigning point values

After completing your analysis, create a JSON rubric following this exact structure:

```json
{
  "version": 2,
  "rubric": [
    {
      "name": "Clarity in Language",
      "category": "style",
      "description": "Writing should use clear, straightforward language that avoids ambiguity and complex jargon.",
      "evidence": "User explicitly stated preference for simple explanations and criticized overly technical language in the conversation.",
      "points": 8
    }
  ]
}
```

**JSON Requirements:**
- Set version to 1 if creating new, or increment from previous version if updating
- Each criterion must have: "name", "category", "description", "evidence", and "points"
- Names should be short, informative, and topic-neutral
- Categories should be logical groupings like "structure", "style", "content", "tone", etc.
- Descriptions should be 1-2 sentences explaining what constitutes good writing for this criterion
- Evidence should be 1-2 sentences referencing specific conversation signals that support this criterion
- Points must be an integer between -10 and 10 inclusive

Provide only the JSON output after your analysis, with no additional text."""


def get_rubric_inference_user_prompt(conversation_text, previous_rubric_json=""):
    """Generate user prompt for rubric inference."""
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
1. An original rubric criterion (name, category, description, points)
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
  "points": "[keep original points]",
  "evidence": ["Example 1", "Example 2", ...]
}
"""


def get_preference_analysis_user_prompt(writing_task, criterion_name, criterion_category,
                                       criterion_description, criterion_points, pref_summary):
    """Generate user prompt for preference analysis."""
    return f"""
Writing task: {writing_task}

Original criterion:
{{
  "name": {criterion_name},
  "category": {criterion_category},
  "description": {criterion_description},
  "points": {criterion_points}
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


def build_system_instruction(rubric, include_assessment=True):
    """Build system instruction with rubric and assessment requirements."""
    rubric_block = ""
    if rubric:
        # Number the criteria explicitly for clarity
        numbered_rubric = []
        for idx, criterion in enumerate(rubric, start=1):
            numbered_crit = criterion.copy()
            numbered_crit['index'] = idx
            numbered_rubric.append(numbered_crit)

        rubric_block = "\nRUBRIC (Always follow these criteria while co-writing):\n" + json.dumps(numbered_rubric, ensure_ascii=False, indent=2)

    # Conditionally build the rubric assessment requirement section
    rubric_assessment_block = ""
    if include_assessment and rubric:
        rubric_assessment_block = dedent("""
        **RUBRIC ASSESSMENT REQUIREMENT:**
        ⚠️ MANDATORY: Whenever you provide written content (ANY paragraphs, drafts, revisions, or substantial text), you MUST conclude your response with a <rubric_assessment> section.

        Assessment requirements:
        - Include one assessment entry for EACH criterion in the rubric above
        - For each criterion, determine if it was "met" (true/false):
          * For criteria with POSITIVE points: "met" = true means the draft showed signs of the desirable behavior
          * For criteria with NEGATIVE points: "met" = true means the draft showed signs of the undesirable behavior that should be avoided
        - Provide concrete evidence from your output explaining whether the criterion was met
        - Note specific improvement areas for each criterion

        **REQUIRED Output Format:**

        <analysis>
        [Your systematic analysis of the writing task]
        </analysis>

        [Your written content, feedback, draft, or revision goes here]

        <rubric_assessment>
        ```json
        {{
        "rubric_assessment": [
            {{
            "name": "[Exact criterion name from the rubric]",
            "met": true,
            "evidence": "[Quote or reference specific parts of your output explaining why this criterion was or wasn't met]",
            "areas_for_improvement": "[Concrete actionable suggestions]"
            }},
            [... one entry for each rubric criterion ...]
        ]
        }}
        ```
        </rubric_assessment>

        ⚠️ DO NOT SKIP THE RUBRIC ASSESSMENT - it is required for all written output.
        """).strip()
    else:
        rubric_assessment_block = dedent("""
        **Example Output Structure:**

        <analysis>
        [Your systematic analysis of the writing task, covering all the points listed above]
        </analysis>

        [Your concrete feedback, draft, suggestions, and/or clarifying questions based on your analysis]
        """).strip()

    # Add a closing reminder if assessment is required
    assessment_reminder = ""
    if include_assessment and rubric:
        assessment_reminder = "\n\n**CRITICAL REMINDER:** Your response MUST end with the <rubric_assessment> section when you provide written content. Do not forget this requirement."

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

    **INSTRUCTIONS:**
    Provide your feedback in a clear, organized manner. Focus on being maximally useful to the user's specific writing task while strictly adhering to the interaction principles.

    {rubric_assessment_block}{assessment_reminder}

    **TOPIC**
    """).strip()

    return system_instruction
