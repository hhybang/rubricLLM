"""SyntheticUser — LLM acting as a user with consistent writing preferences.

Supports multiple providers: Anthropic (Claude), OpenAI (GPT-5.2), Google (Gemini 3).
"""

import json
import logging
import random
import time
from typing import Optional

from sim_config import Persona, PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_GOOGLE

log = logging.getLogger("sim.user")


# ═════════════════════════════════════════════════════════════════════════════
# Multi-provider LLM client
# ═════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Unified interface for calling different LLM providers."""

    def __init__(self, provider: str, model: str, temperature: float = 0.8):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == PROVIDER_ANTHROPIC:
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider == PROVIDER_OPENAI:
            from openai import OpenAI
            self._client = OpenAI()
        elif self.provider == PROVIDER_GOOGLE:
            from google import genai
            self._client = genai.Client()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 1000, max_retries: int = 3) -> str:
        """Generate text from the LLM (single-turn). Returns the response string."""
        return self.generate_multiturn(
            system_prompt,
            [{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    def generate_multiturn(self, system_prompt: str, messages: list,
                           max_tokens: int = 1000, max_retries: int = 3) -> str:
        """Generate text from the LLM with full conversation history.

        Args:
            system_prompt: System instructions
            messages: List of {"role": "user"|"assistant", "content": str}
            max_tokens: Max output tokens
            max_retries: Retry count for rate limits
        """
        for attempt in range(max_retries + 1):
            try:
                return self._call_multiturn(system_prompt, messages, max_tokens)
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ("rate", "overloaded", "429", "529", "quota")) and attempt < max_retries:
                    delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"[{self.provider}] Rate limited, retrying in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise

    def _call_multiturn(self, system_prompt: str, messages: list, max_tokens: int) -> str:
        if self.provider == PROVIDER_ANTHROPIC:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=self.temperature,
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        elif self.provider == PROVIDER_OPENAI:
            # Build OpenAI messages format
            oai_messages = [{"role": "developer", "content": system_prompt}]
            oai_messages.extend(messages)
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                temperature=self.temperature,
                max_completion_tokens=max_tokens,
            )
            return resp.choices[0].message.content

        elif self.provider == PROVIDER_GOOGLE:
            from google.genai import types
            # Build Google contents format
            contents = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text

        raise ValueError(f"Unknown provider: {self.provider}")


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic User
# ═════════════════════════════════════════════════════════════════════════════

PERSONA_SYSTEM_PROMPT = """You are simulating a real human user in a writing co-authoring session. \
You have specific, consistent writing preferences defined by your persona.

PERSONA:
- Name: {name}
- Role: {role}
- Writing type: {writing_type}
- Core preferences: {core_preferences}
- Hidden preferences: {hidden_preferences}
- Dealbreakers: {dealbreakers}

BEHAVIORAL RULES:
1. Stay in character. Your preferences are FIXED — do not change them based on what the system suggests.
2. You have both "stated" preferences (core_preferences) and "hidden" preferences (hidden_preferences). \
When asked about preferences upfront, mention only stated ones. Hidden preferences emerge through your feedback on drafts.
3. Be realistic: real users are sometimes vague, sometimes specific, occasionally contradictory in minor ways.
4. Give natural feedback, not rubric-like evaluations.
5. When ranking drafts, genuinely evaluate them against your preferences (both stated and hidden). \
Do not always pick the rubric-guided draft.
6. Always return valid JSON when asked for structured output."""


class SyntheticUser:
    """An LLM-powered synthetic user with a fixed persona."""

    def __init__(self, persona: Persona, provider: str = PROVIDER_ANTHROPIC,
                 model: str = "claude-sonnet-4-6", temperature: float = 0.8):
        self.persona = persona
        self.llm = LLMClient(provider, model, temperature)
        self._system_prompt = PERSONA_SYSTEM_PROMPT.format(
            name=persona.name,
            role=persona.role,
            writing_type=persona.writing_type,
            core_preferences=persona.core_preferences,
            hidden_preferences=persona.hidden_preferences,
            dealbreakers=persona.dealbreakers,
        )
        self._score_history: list[float] = []  # per-iteration score tracking
        log.info(f"SyntheticUser '{persona.name}' initialized ({provider}/{model})")

    def reset_score_history(self):
        """Reset score history at the start of each iteration."""
        self._score_history = []

    # ── Cold-start preferences ───────────────────────────────────────────

    def generate_coldstart_preferences(self) -> str:
        """Generate cold-start preference text (mentions only stated preferences)."""
        prompt = (
            f"You are about to start a co-writing session for {self.persona.writing_type}. "
            f"Someone asks: 'What do you care about in your writing?'\n\n"
            f"Write a 2-4 sentence natural description of your writing preferences. "
            f"Mention ONLY your stated/core preferences. Do NOT mention hidden preferences. "
            f"Write naturally, as if talking to a person — no rubric terminology, no structured lists."
        )
        text = self.llm.generate(self._system_prompt, prompt, max_tokens=300)
        log.info(f"[{self.persona.name}] Generated cold-start preferences ({len(text)} chars)")
        return text.strip()

    # ── Chat messages ────────────────────────────────────────────────────

    def generate_chat_message(self, messages: list, rubric: Optional[dict] = None,
                              phase: str = "initial_request", iteration: int = 0) -> str:
        """Generate a realistic chat message.

        Args:
            messages: conversation history
            rubric: current rubric dict (may be None)
            phase: "initial_request", "feedback", or "new_task"
            iteration: current iteration index (0-based)
        """
        # Use pre-defined tasks from persona if available
        tasks = getattr(self.persona, 'tasks', None) or []

        if phase == "initial_request":
            task_description = tasks[0] if tasks else self.persona.initial_task
            log.info(f"[{self.persona.name}] Using pre-defined task 1/{len(tasks) if tasks else 1}")
            return task_description
        elif phase == "new_task":
            # Return pre-defined task directly — no LLM call needed
            if iteration < len(tasks):
                task_description = tasks[iteration]
                log.info(f"[{self.persona.name}] Using pre-defined task {iteration + 1}/{len(tasks)}")
                return task_description
            else:
                # Fall back to LLM-generated task when we've exhausted pre-defined ones
                log.info(f"[{self.persona.name}] No pre-defined task for iteration {iteration + 1}, generating dynamically")
                # Extract previous writing tasks to avoid repetition
                prev_tasks = []
                for m in messages:
                    if m.get("role") == "user" and not m.get("content", "").startswith("---"):
                        content = m["content"][:150]
                        if any(kw in content.lower() for kw in ("draft", "write", "help me", "need to", "can you")):
                            prev_tasks.append(content)
                seen = set()
                unique_tasks = []
                for t in prev_tasks:
                    key = t[:80]
                    if key not in seen:
                        seen.add(key)
                        unique_tasks.append(t)

                prev_tasks_text = "\n".join(f"- {t}" for t in unique_tasks) if unique_tasks else "(none)"

                rubric_nudge = ""
                if rubric and rubric.get("rubric"):
                    criteria_summary = "\n".join(
                        f"  - {c.get('name', '?')}: {c.get('description', '')[:120]}"
                        for c in rubric["rubric"]
                    )
                    rubric_nudge = (
                        f"\nYou have a rubric that captures your writing preferences:\n"
                        f"{criteria_summary}\n\n"
                        f"Pick a writing task where this rubric would be useful — something that's "
                        f"part of your regular work as a {self.persona.role}. "
                        f"Vary the scenario from your previous tasks (different audience, purpose, or stakes) "
                        f"since your job involves a range of {self.persona.writing_type}.\n"
                    )

                prompt = (
                    f"You've been working with an AI assistant on {self.persona.writing_type}.\n\n"
                    f"PREVIOUS TASKS you've already done (DO NOT repeat these scenarios):\n"
                    f"{prev_tasks_text}\n\n"
                    f"{rubric_nudge}"
                    f"Now start a DIFFERENT writing task. IMPORTANT CONSTRAINTS:\n"
                    f"- It MUST still be {self.persona.writing_type} — same genre, same general format. "
                    f"Do NOT switch to a different type of writing.\n"
                    f"- But vary the scenario: different audience or reader, different purpose, "
                    f"or different emotional stakes than before.\n"
                    f"- DO NOT write a follow-up or continuation of any previous task. "
                    f"Invent a new scenario that a {self.persona.role} would realistically face.\n"
                    f"- Keep the scope VERY SMALL — ask for 2-4 short paragraphs at most. "
                    f"Do NOT ask for a full page, full section, or anything longer. "
                    f"The AI must be able to produce the COMPLETE draft in a single response without truncation.\n\n"
                    f"Write a natural 2-4 sentence message requesting help with this new task. "
                    f"Be specific about the scenario and context."
                )
        elif phase == "feedback":
            # Count feedback messages in the CURRENT iteration only
            # (after the last iteration boundary marker)
            last_boundary = 0
            for idx, m in enumerate(messages):
                if m.get("role") == "system" and "Iteration" in m.get("content", ""):
                    last_boundary = idx
            feedback_count = sum(
                1 for m in messages[last_boundary:]
                if m.get("role") == "user"
                and not m.get("content", "").startswith("---")
            ) - 1  # subtract 1 for the initial task request
            feedback_count = max(0, feedback_count)

            # Feedback strategy shifts across rounds:
            # Rounds 1-2: React to problems (what feels wrong)
            # Rounds 3-5: Get more specific, explain what you want instead
            # Rounds 6+: Be direct and constructive — tell the assistant what to do
            if feedback_count < 2:
                feedback_strategy = (
                    f"This is early feedback (round {feedback_count + 1}). React naturally:\n"
                    f"- Point out 1-2 things that bother you about THIS draft\n"
                    f"- Describe what feels wrong, not exactly how to fix it\n"
                    f"- Example: 'The middle section feels like it's talking past me' rather than "
                    f"'Rewrite the middle from the reader's perspective'\n"
                    f"- You can mention what you LIKE too, not just problems"
                )
            elif feedback_count < 5:
                feedback_strategy = (
                    f"This is round {feedback_count + 1}. The assistant has had a few tries. "
                    f"Be more specific about what you want:\n"
                    f"- If something was removed that shouldn't have been, say so: "
                    f"'You cut the part about X but I actually need that — just rewrite it to be more Y'\n"
                    f"- If something is MISSING, say what you want added: "
                    f"'This needs a sentence about what this means for THEIR timelines'\n"
                    f"- Balance criticism with direction — don't just say 'this feels off', "
                    f"say 'this feels off because it's missing X'\n"
                    f"- IMPORTANT: If the draft is getting too short or losing substance, "
                    f"push back: 'You're cutting too much — I need the [content] back, just framed differently'"
                )
            else:
                feedback_strategy = (
                    f"This is round {feedback_count + 1}. You've given a lot of feedback. "
                    f"CRITICAL: The draft probably has some things that are WORKING. Protect them.\n"
                    f"- Start by naming 1-2 things that are RIGHT and must NOT change\n"
                    f"- Then identify the ONE remaining issue that bothers you most\n"
                    f"- Frame it as an ADDITIVE fix: 'Keep everything, but change X to Y' or "
                    f"'This is almost there — the only thing missing is Z'\n"
                    f"- NEVER ask for a restructure or reset at this stage — that undoes progress\n"
                    f"- Express your hidden preferences as concrete requests a real user would make "
                    f"(e.g., 'I need this section to start with WHY, not HOW' or "
                    f"'Give me one clear owner, not a group ask')\n"
                    f"- If the draft got WORSE since last round, say: 'The previous version was better — "
                    f"go back to that and only change [specific thing]'"
                )

            instruction = (
                f"Provide natural feedback on the assistant's latest response.\n\n"
                f"Before giving feedback, mentally check each of your hidden preferences against "
                f"the current draft:\n"
                f"{self.persona.hidden_preferences}\n\n"
                f"If a hidden preference is NOT met in the draft, your feedback MUST address it "
                f"(even if the draft looks good overall). Do NOT say the draft is done or acceptable "
                f"if any hidden preference is clearly missing.\n\n"
                f"{feedback_strategy}\n\n"
                f"IMPORTANT GROUND RULES:\n"
                f"- Keep it to 1-4 sentences\n"
                f"- If something was REMOVED that you wanted, tell the assistant to put it back (reframed)\n"
                f"- Feedback should lead toward IMPROVEMENT, not just removal of content\n"
                f"- ALWAYS mention what's working before what needs to change — this prevents the assistant "
                f"from accidentally undoing good parts\n"
                f"- Focus on ONE issue per round, not multiple — fixing everything at once causes regression\n"
                f"- Do NOT say 'let\'s reset' or ask for a complete rewrite — iterate on what exists\n"
                f"- Do NOT repeat feedback you already gave in earlier rounds\n"
                f"- Do NOT say 'we\'re done' or 'this is good to go' unless your hidden preferences are met\n"
                f"- Do NOT give rubric-style feedback. Talk like a normal person.\n"
                f"- Do NOT expand scope or introduce new writing tasks mid-conversation. "
                f"Stick to the ORIGINAL task you requested. Do not ask for additional sections, "
                f"paragraphs, or pieces beyond what you initially asked for."
            )

            # Build the conversation history as a formatted string for context.
            # Include the full iteration so the model knows what it already said.
            conv_lines = []
            for m in messages[last_boundary:]:
                if m.get("role") == "system":
                    continue
                role_label = "You (the user)" if m["role"] == "user" else "Writing assistant"
                # Truncate very long messages but keep enough for context
                content = m["content"][:500]
                conv_lines.append(f"[{role_label}]: {content}")
            conversation_history = "\n\n".join(conv_lines)

            full_prompt = (
                f"Here is the full conversation so far in this session "
                f"(you are 'You (the user)'):\n\n{conversation_history}\n\n"
                f"---\n\n"
                f"Now write your next message as the user. {instruction}"
            )

            text = self.llm.generate(self._system_prompt, full_prompt, max_tokens=500)
            log.info(f"[{self.persona.name}] Generated {phase} message ({len(text)} chars)")
            return text.strip()
        else:
            raise ValueError(f"Unknown phase: {phase}")

        text = self.llm.generate(self._system_prompt, prompt, max_tokens=500)
        log.info(f"[{self.persona.name}] Generated {phase} message ({len(text)} chars)")
        return text.strip()

    # ── Respond to assistant questions (no draft in response) ───────────

    def respond_to_assistant_question(self, messages: list) -> str:
        """Generate a contextual reply when the assistant asks a question instead of producing a draft.

        Instead of blindly saying "write a full draft", the synthetic user answers
        the assistant's question with realistic details (inventing plausible names,
        dates, or other content as needed) and redirects toward getting a draft.

        Returns:
            A natural user message answering the question.
        """
        # Build conversation history from current iteration
        last_boundary = 0
        for idx, m in enumerate(messages):
            if m.get("role") == "system" and "Iteration" in m.get("content", ""):
                last_boundary = idx

        conv_lines = []
        for m in messages[last_boundary:]:
            if m.get("role") == "system":
                continue
            role_label = "You (the user)" if m["role"] == "user" else "Writing assistant"
            content = m["content"][:500]
            conv_lines.append(f"[{role_label}]: {content}")
        conversation_history = "\n\n".join(conv_lines)

        prompt = (
            f"Here is the conversation so far (you are 'You (the user)'):\n\n"
            f"{conversation_history}\n\n---\n\n"
            f"The assistant just asked you a question or requested information instead of "
            f"producing a draft. You need to ANSWER their question so they can move forward.\n\n"
            f"RULES:\n"
            f"- Answer the question directly with realistic, plausible details\n"
            f"- If they're asking for names, dates, team details, or other specifics, "
            f"MAKE UP realistic ones (e.g., 'Sarah Torres is the lead engineer', "
            f"'milestones will be ready by Friday the 21st')\n"
            f"- You are a real person with a real job — you HAVE these details, "
            f"just provide them naturally\n"
            f"- After answering, ask them to incorporate the details into the draft\n"
            f"- Keep it to 1-3 sentences — be direct, not chatty\n"
            f"- Do NOT just say 'write a draft' — actually answer what they asked"
        )

        text = self.llm.generate(self._system_prompt, prompt, max_tokens=300)
        log.info(f"[{self.persona.name}] Generated contextual response to assistant question ({len(text)} chars)")
        return text.strip()

    # ── Combined feedback + satisfaction (user decides when to stop) ────

    def respond_to_draft(self, current_draft: str, messages: list,
                         rubric: dict = None, iteration: int = 0,
                         draft_count: int = 0) -> dict:
        """The synthetic user decides: accept the draft or give more feedback.

        This is a SINGLE call that checks preferences and either accepts or
        provides feedback. The user drives termination, not an external score.

        Returns:
            {
                "accepted": bool,
                "score": float (0.0-1.0),
                "feedback": str (the user's message — either acceptance or feedback),
                "reasoning": str (per-preference check),
            }
        """
        # Find the current iteration boundary
        last_boundary = 0
        for idx, m in enumerate(messages):
            if m.get("role") == "system" and "Iteration" in m.get("content", ""):
                last_boundary = idx

        # Count feedback rounds in current iteration
        feedback_count = sum(
            1 for m in messages[last_boundary:]
            if m.get("role") == "user"
            and not m.get("content", "").startswith("---")
        ) - 1
        feedback_count = max(0, feedback_count)

        # Build conversation history
        conv_lines = []
        for m in messages[last_boundary:]:
            if m.get("role") == "system":
                continue
            role_label = "You (the user)" if m["role"] == "user" else "Writing assistant"
            content = m["content"][:500]
            conv_lines.append(f"[{role_label}]: {content}")
        conversation_history = "\n\n".join(conv_lines)

        # Rubric note for later feedback rounds
        rubric_note = ""

        if feedback_count < 2:
            feedback_strategy = (
                f"This is early feedback (round {feedback_count + 1}). React like a real person:\n"
                f"- Express your REACTION to the draft — what felt off, what didn't land, what made you pause\n"
                f"- Describe the FEELING or EFFECT, not the technical fix — 'this part feels flat' not 'add a payoff sentence'\n"
                f"- Do NOT name specific paragraphs by number or point to exact locations\n"
                f"- Do NOT prescribe the solution — say what's wrong, not how to fix it\n"
                f"- Focus on ONE thing that bothers you most\n"
                f"- A real user says 'the ending just kind of trails off' not 'paragraph 3 needs a concluding sentence that synthesizes the argument'"
            )
        elif feedback_count < 5:
            feedback_strategy = (
                f"This is round {feedback_count + 1}. You can be slightly more specific, but still talk like a person:\n"
                f"- You can reference WHAT part of the draft you mean ('the opening', 'near the end', 'where you talk about X')\n"
                f"- Still describe the PROBLEM, not the exact solution — 'it reads like a list' not 'add transition sentences'\n"
                f"- You may give a general direction ('I want it to feel more like a narrative') but NOT a specific instruction\n"
                f"- Do NOT use technical writing terminology the persona wouldn't know\n"
                f"- Focus on ONE remaining issue"
                f"{rubric_note}"
            )
        else:
            feedback_strategy = (
                f"This is round {feedback_count + 1}. You've been going back and forth — be direct but still human:\n"
                f"- Acknowledge what's working so the assistant doesn't break it\n"
                f"- For the ONE remaining issue, you can now be more concrete about what you want\n"
                f"- But still frame it as your preference, not a rubric instruction — 'I really want a moment where it just speaks plainly' not 'insert one informal aside'\n"
                f"- Do NOT repeat the same feedback verbatim from earlier rounds — rephrase or approach from a different angle"
                f"{rubric_note}"
            )

        # Detect score plateau — if score hasn't improved in recent drafts
        plateau_note = ""
        if len(self._score_history) >= 5:
            recent = self._score_history[-5:]
            best_recent = max(recent)
            # Check if all recent scores are within a tight band (no improvement)
            if best_recent >= 0.8 and (best_recent - min(recent)) <= 0.15:
                plateau_note = (
                    f"- PLATEAU RULE: Your scores over the last 5 drafts have been "
                    f"{[round(s,2) for s in recent]} — no meaningful improvement. "
                    f"A real user would stop here. ACCEPT the draft if final score ≥ 0.80, "
                    f"even if some preferences are only partially met. "
                    f"Continuing to iterate on marginal gaps is unrealistic.\n"
                )
                log.info(f"Plateau detected: last 5 scores = {[round(s,2) for s in recent]}")

        # Parse hidden preferences into numbered list for rigorous per-item checking
        hidden_raw = self.persona.hidden_preferences
        # Split on sentence-ending periods followed by a capital letter (new preference)
        import re as _re
        hidden_items = [s.strip() for s in _re.split(r'(?<=\.)\s+(?=[A-Z])', hidden_raw) if s.strip()]
        # If splitting produced only 1 item, try splitting on semicolons
        if len(hidden_items) <= 1:
            hidden_items = [s.strip() for s in hidden_raw.split(';') if s.strip()]
        numbered_hidden = "\n".join(f"  H{i+1}. {item}" for i, item in enumerate(hidden_items))
        n_hidden = len(hidden_items)

        # Parse core preferences into per-item list for rigorous scoring
        core_raw = str(self.persona.core_preferences)
        core_items = [s.strip() for s in _re.split(r'(?<=\.)\s+(?=[A-Z])', core_raw) if s.strip()]
        if len(core_items) <= 1:
            core_items = [s.strip() for s in core_raw.split(';') if s.strip()]
        if len(core_items) <= 1:
            core_items = [s.strip() for s in core_raw.split(',') if s.strip() and len(s.strip()) > 10]
        numbered_core = "\n".join(f"  C{i+1}. {item}" for i, item in enumerate(core_items))
        n_core = len(core_items)

        prompt = (
            f"Here is the conversation so far (you are 'You (the user)'):\n\n"
            f"{conversation_history}\n\n---\n\n"
            f"The assistant's latest draft:\n\n{current_draft}\n\n"
            f"═══ YOUR PREFERENCES (ground truth) ═══\n\n"
            f"HIDDEN preferences (check EACH one individually):\n"
            f"{numbered_hidden}\n\n"
            f"CORE preferences (check EACH one individually):\n"
            f"{numbered_core}\n\n"
            f"DEALBREAKERS (any violation = score 0): {self.persona.dealbreakers}\n\n"
            f"═══ EVALUATION STEPS ═══\n\n"
            f"STEP 1: For EACH hidden preference (H1-H{n_hidden}), do ALL three:\n"
            f"  a) STATE what the preference requires (one sentence)\n"
            f"  b) QUOTE the specific draft text that satisfies it, or write \"NOT FOUND\" or \"N/A\"\n"
            f"  c) SCORE: 0 (not found / wrong), 0.5 (partially there but missing key aspect), 1.0 (fully met), "
            f"or \"N/A\" ONLY if the preference is structurally impossible to apply to this writing task\n\n"
            f"STRICT N/A RULES (N/A is rare — most preferences apply):\n"
            f"- N/A is ONLY for structural mismatch: the preference describes a specific document element "
            f"(e.g., 'methods section rhythm', 'code examples', 'call-to-action') that literally does not exist "
            f"in the current writing task's genre or format\n"
            f"- N/A is NOT for: preferences that are hard to achieve, preferences that require effort, "
            f"or preferences where the draft hasn't attempted to address them — those are 0 or 0.5\n"
            f"- If the preference is about STYLE, TONE, STRUCTURE, or RHETORIC, it almost certainly applies — score 0/0.5/1.0\n"
            f"- When in doubt, score 0 or 0.5 — do NOT use N/A as an escape hatch\n"
            f"- If you cannot quote specific text → score 0\n"
            f"- If the draft does the OPPOSITE of what the preference asks → score 0\n"
            f"- If a preference has multiple parts and ANY part fails → cap at 0.5\n"
            f"- Generic/boilerplate text that technically matches but lacks specificity → score 0.5 max\n"
            f"- PLACEHOLDER RULE: Placeholders like [Name], [Date], [Owner] for factual details "
            f"(names, dates, team info) that the user would fill in themselves are FINE — "
            f"score based on the STRUCTURE, not whether the placeholder is filled. "
            f"'[Name] will send a revised schedule by [date]' fully meets a preference for "
            f"'one clear owner with a deadline' — score 1.0. The user knows their own team; "
            f"placeholders for personal details are normal in drafting. "
            f"Only score 0 for placeholders if the STRUCTURE itself is wrong.\n\n"
            f"STEP 2: For EACH core preference (C1-C{n_core}), quote specific text or write \"NOT FOUND\". Score 0/0.5/1.0.\n"
            f"  core_avg = (C1 + C2 + ... + C{n_core}) / {n_core}\n"
            f"  Score dealbreakers (1.0 if none violated, 0.0 if any).\n\n"
            f"STEP 3: Compute (exclude any N/A scores from the average):\n"
            f"  hidden_avg = sum of non-N/A hidden scores / count of non-N/A hidden scores\n"
            f"  final = (hidden_avg * 0.70) + (core_avg * 0.20) + (dealbreaker_score * 0.10)\n"
            f"  Show the actual numbers.\n\n"
            f"STEP 4: DECIDE based on per-preference results:\n"
            f"- ACCEPT if: ALL non-N/A hidden preferences scored 1.0 AND no dealbreaker violated\n"
            f"- ALSO ACCEPT if: final score ≥ 0.95 AND you have given feedback on the same issue "
            f"3+ times already without improvement — a real user would accept a near-perfect draft "
            f"rather than endlessly iterating on a marginal gap\n"
            f"{plateau_note}"
            f"- Otherwise → REJECT and give feedback on the lowest-scoring unmet hidden preference\n"
            f"{'- IMPORTANT: This is the FIRST draft you are seeing. Be critical — even if the draft is good, look for what could be better. Real users almost always have at least one round of feedback before accepting.' + chr(10) if feedback_count == 0 else ''}\n"
            f"═══ FEEDBACK VOICE ═══\n\n"
            f"{feedback_strategy}\n\n"
            f"GROUND RULES FOR FEEDBACK:\n"
            f"- Focus on ONE issue per round (the lowest-scoring unmet preference)\n"
            f"- Do NOT say 'let's reset' or ask for a complete rewrite\n"
            f"- Stay in character as {self.persona.name} ({self.persona.role}) — use vocabulary and specificity natural to this person\n"
            f"- Describe what BOTHERS you about the draft, not the underlying preference rule that's being violated\n"
            f"- Lean toward describing the PROBLEM over prescribing the SOLUTION — but if your persona would naturally suggest a fix, that's fine\n"
            f"- Do NOT use the words 'preference', 'criteria', 'rubric', or 'hidden'\n"
            f"- Do NOT give feedback that reads like a checklist or grading rubric\n\n"
            f"═══ OUTPUT ═══\n\n"
            f"Output ONLY a JSON object (no other text before or after):\n"
            f'{{\n'
            f'  "per_preference": {{\n'
            f'    "H1": {{"quote": "...", "score": 0/0.5/1.0/"N/A"}},\n'
            f'    "H2": {{"quote": "...", "score": 0/0.5/1.0/"N/A"}},\n'
            f'    ...\n'
            f'  }},\n'
            f'  "per_core": {{\n'
            f'    "C1": {{"quote": "...", "score": 0/0.5/1.0}},\n'
            f'    "C2": {{"quote": "...", "score": 0/0.5/1.0}},\n'
            f'    ...\n'
            f'  }},\n'
            f'  "hidden_avg": <float>,\n'
            f'  "core_score": <float>,\n'
            f'  "dealbreaker_score": <1.0 or 0.0>,\n'
            f'  "score": <final computed score>,\n'
            f'  "accepted": true/false,\n'
            f'  "reasoning": "<1-2 sentence summary of what met and what didn\'t>",\n'
            f'  "feedback": "<your message to the assistant — acceptance or feedback>"\n'
            f'}}'
        )

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=1500)
            try:
                result = self._extract_json(text)
                result.setdefault("accepted", False)
                result.setdefault("score", 0.5)
                result.setdefault("feedback", "")
                result.setdefault("reasoning", "")

                # Log per-preference scores for debugging (no overrides — trust the LLM)
                per_pref = result.get("per_preference", {})
                if per_pref:
                    na_count = 0
                    numeric_scores = []
                    for v in per_pref.values():
                        if isinstance(v, dict):
                            s = v.get("score", 0)
                            if isinstance(s, str) and s.strip().upper() == "N/A":
                                na_count += 1
                            else:
                                try:
                                    numeric_scores.append(float(s) if s is not None else 0)
                                except (ValueError, TypeError):
                                    numeric_scores.append(0)
                    if na_count > 0:
                        log.info(f"N/A scores: {na_count}/{len(per_pref)} hidden preferences marked N/A")
                    log.info(f"Hidden scores: {numeric_scores}, accepted={result['accepted']}")

                # Track score for stagnation detection
                self._score_history.append(result["score"])

                # Ensure feedback is never empty
                if result["accepted"] and not result["feedback"]:
                    result["feedback"] = "This looks good — I'm happy with this draft."
                elif not result["accepted"] and not result["feedback"]:
                    result["feedback"] = "This isn't quite there yet — can you revise it?"
                return result
            except Exception as e:
                last_error = e
                log.warning(f"respond_to_draft parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"respond_to_draft failed after {max_retries} retries: {last_error}\nLast raw response (first 500 chars): {text[:500]}")
        return {
            "accepted": False,
            "score": 0.3,
            "feedback": "Can you revise this? Something still feels off.",
            "reasoning": f"parse error after {max_retries} retries: {last_error}",
        }

    # ── Criteria classification ──────────────────────────────────────────

    def classify_criteria(self, rubric_criteria: list, llm_classification: dict) -> dict:
        """Classify rubric criteria as stated/real/hallucinated.

        Returns: {"criterion_name": "stated"|"real"|"hallucinated", ...}
        """
        criteria_list = "\n".join(
            f"- {c.get('name', '?')}: {c.get('description', '')}"
            for c in rubric_criteria
        )
        llm_summary = json.dumps(llm_classification, indent=2) if llm_classification else "{}"
        prompt = (
            f"The system inferred these rubric criteria from your writing session:\n\n"
            f"{criteria_list}\n\n"
            f"The system's initial classification:\n{llm_summary}\n\n"
            f"For each criterion, classify it as:\n"
            f'- "stated": You explicitly mentioned this preference in your cold-start description\n'
            f'- "real": You didn\'t mention it upfront, but it IS a real preference of yours\n'
            f'- "hallucinated": This does NOT match your actual preferences\n\n'
            f"Remember:\n"
            f"- Your stated preferences are: {self.persona.core_preferences}\n"
            f"- Your hidden preferences are: {self.persona.hidden_preferences}\n"
            f'- Mark criteria matching hidden preferences as "real"\n'
            f'- Mark criteria that don\'t match ANY of your preferences as "hallucinated"\n'
            f'- Be GENEROUS with "real" — if a criterion captures the SPIRIT of one of your '
            f'preferences even if the wording differs, mark it "real". Only mark "hallucinated" '
            f'if the criterion describes something you genuinely do NOT care about.\n\n'
            f'Return ONLY a JSON object: {{"criterion_name": "stated|real|hallucinated", ...}}'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=1000)
            try:
                result = self._extract_json(text)
                # Flatten if nested under a key
                if isinstance(result, dict):
                    if "classifications" in result:
                        return result["classifications"]
                    # Check if values are valid classification strings
                    valid = {"stated", "real", "hallucinated"}
                    if all(isinstance(v, str) and v in valid for v in result.values()):
                        return result
                log.warning(f"Unexpected classification format, returning as-is")
                return result if isinstance(result, dict) else {}
            except Exception as e:
                last_error = e
                log.warning(f"classify_criteria parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"classify_criteria failed after {max_retries} retries: {last_error}")
        # Fallback: mark everything as "real"
        return {c.get("name", "?"): "real" for c in rubric_criteria}

    # ── Draft ranking ────────────────────────────────────────────────────

    def rank_drafts(self, blind_drafts: dict, writing_task: str) -> tuple:
        """Rank blind drafts A/B/C.

        Args:
            blind_drafts: {"A": draft_text, "B": draft_text, "C": draft_text}
            writing_task: the writing task description
        Returns:
            (ranking_labels, reason) e.g. (["A", "C", "B"], "A felt most natural...")
        """
        drafts_text = "\n\n".join(
            f"--- Draft {label} ---\n{text[:1500]}"
            for label, text in sorted(blind_drafts.items())
        )
        prompt = (
            f"Three drafts were written for this task: {writing_task}\n\n"
            f"{drafts_text}\n\n"
            f"Rank these drafts from best to worst based on your writing preferences "
            f"(both stated and hidden). Evaluate honestly — which one would you actually "
            f"want to send/publish?\n\n"
            f'Return ONLY a JSON object: {{"ranking": ["A", "B", "C"], "reason": "brief explanation"}}'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=500)
            try:
                result = self._extract_json(text)
                ranking = result.get("ranking", ["A", "B", "C"])
                reason = result.get("reason", "")
                return ranking, reason
            except Exception as e:
                last_error = e
                log.warning(f"rank_drafts parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"rank_drafts failed after {max_retries} retries: {last_error}")
        labels = sorted(blind_drafts.keys())
        random.shuffle(labels)
        return labels, "Could not determine preference"

    # ── Decision point feedback ──────────────────────────────────────────

    def provide_dp_feedback(self, dp_result: dict) -> dict:
        """Confirm or correct extracted decision points.

        Returns: {"confirmed": [ids], "corrected": [...], "had_corrections": bool}
        """
        if not dp_result or not dp_result.get("parsed_data", {}).get("decision_points"):
            return {"confirmed": [], "corrected": [], "had_corrections": False}

        dps = dp_result["parsed_data"]["decision_points"]
        dp_list = "\n".join(
            f"- DP #{dp.get('id', i+1)}: \"{dp.get('summary', '')}\" → mapped to criterion \"{dp.get('suggested_criterion_name', dp.get('related_rubric_criterion', 'unknown'))}\""
            for i, dp in enumerate(dps)
        )
        prompt = (
            f"The system extracted these decision points from your conversation — moments "
            f"where your preferences were demonstrated:\n\n{dp_list}\n\n"
            f"For each decision point, does it accurately describe what happened and is the "
            f"criterion mapping correct?\n\n"
            f'Return ONLY a JSON object: {{"confirmed": [1, 2, ...], "corrected": [{{"id": N, "correction": "..."}}], "had_corrections": true/false}}'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=500)
            try:
                result = self._extract_json(text)
                result.setdefault("confirmed", [dp.get("id", i+1) for i, dp in enumerate(dps)])
                result.setdefault("corrected", [])
                result.setdefault("had_corrections", bool(result.get("corrected")))
                return result
            except Exception as e:
                last_error = e
                log.warning(f"provide_dp_feedback parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"provide_dp_feedback failed after {max_retries} retries: {last_error}")
        return {
            "confirmed": [dp.get("id", i+1) for i, dp in enumerate(dps)],
            "corrected": [],
            "had_corrections": False,
        }

    # ── Rubric suggestion decision ───────────────────────────────────────

    def decide_on_rubric_suggestion(self, current_rubric: dict, suggested_rubric: list,
                                     suggestion_reasons: dict,
                                     preview_draft: str = "") -> str:
        """Decide whether to apply suggested rubric changes + preview draft, or revert.

        Returns "apply_all" to accept rubric suggestion + preview draft,
        or "revert" to discard all edits and go back to original.
        """
        current_names = [c.get("name", "?") for c in current_rubric.get("rubric", [])]
        suggested_names = [c.get("name", "?") for c in suggested_rubric]
        reasons_text = "\n".join(f"- {k}: {v}" for k, v in suggestion_reasons.items()) if suggestion_reasons else "No specific reasons given."

        draft_section = ""
        if preview_draft:
            draft_section = (
                f"\n\nHere is a preview of how the draft would look with the suggested rubric:\n"
                f"<draft>{preview_draft[:3000]}</draft>\n"
            )

        prompt = (
            f"The system analyzed your edit feedback and suggests these rubric changes:\n\n"
            f"Current criteria: {', '.join(current_names)}\n"
            f"Suggested criteria: {', '.join(suggested_names)}\n\n"
            f"Reasons:\n{reasons_text}\n"
            f"{draft_section}\n"
            f"You have two choices:\n"
            f"1. **Apply All** — accept the suggested rubric changes and the preview draft\n"
            f"2. **Revert** — discard ALL edits (both your manual rubric edits and the suggestions) "
            f"and go back to the original draft and rubric\n\n"
            f"Based on your actual writing preferences, which do you choose?\n\n"
            f'Return ONLY a JSON object: {{"decision": "apply_all" or "revert", "reason": "brief explanation"}}'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=300)
            try:
                result = self._extract_json(text)
                decision = result.get("decision", "apply_all")
                if decision not in ("apply_all", "revert"):
                    decision = "apply_all"
                log.info(f"[{self.persona.name}] Rubric suggestion decision: {decision}")
                return decision
            except Exception as e:
                last_error = e
                log.warning(f"decide_on_rubric_suggestion parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"decide_on_rubric_suggestion failed after {max_retries} retries: {last_error}")
        return "revert"

    def decide_on_revised_draft(self, original_draft: str, revised_draft: str,
                                 edited_rubric: list, original_rubric: list,
                                 messages: list) -> bool:
        """Decide whether to accept the revised draft from manual rubric edits, or revert.

        Used when the user made rubric edits but had no feedback on the annotated changes
        (agreed with all edits). They can still choose to revert if the draft doesn't look right.

        Returns True to accept (keep edited rubric + revised draft),
        False to revert (go back to original rubric + original draft).
        """
        edited_names = [c.get("name", "?") for c in edited_rubric]
        original_names = [c.get("name", "?") for c in original_rubric]

        prompt = (
            f"You manually edited the rubric and the system regenerated the draft based on your changes.\n\n"
            f"Original rubric criteria: {', '.join(original_names)}\n"
            f"Your edited rubric criteria: {', '.join(edited_names)}\n\n"
            f"Original draft (before your edits):\n<draft>{original_draft[:2000]}</draft>\n\n"
            f"Revised draft (after your rubric edits):\n<draft>{revised_draft[:2000]}</draft>\n\n"
            f"You have two choices:\n"
            f"1. **Accept** — keep your rubric edits and the revised draft (save as new rubric version)\n"
            f"2. **Revert** — discard your rubric edits and go back to the original draft and rubric\n\n"
            f"Based on your actual writing preferences, does the revised draft better match what you want?\n\n"
            f'Return ONLY a JSON object: {{"accept": true/false, "reason": "brief explanation"}}'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=300)
            try:
                result = self._extract_json(text)
                accept = result.get("accept", True)
                log.info(f"[{self.persona.name}] Revised draft decision: {'accepted' if accept else 'reverted'}")
                return accept
            except Exception as e:
                last_error = e
                log.warning(f"decide_on_revised_draft parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"decide_on_revised_draft failed after {max_retries} retries: {last_error}")
        return True

    # ── Importance ranking ───────────────────────────────────────────────

    def rank_importance(self, rubric_criteria: list, classifications: dict) -> list:
        """Rank criteria by importance. Returns list of criterion names, most important first."""
        criteria_list = "\n".join(
            f"- {c.get('name', '?')} ({classifications.get(c.get('name', '?'), 'unknown')}): {c.get('description', '')}"
            for c in rubric_criteria
        )
        prompt = (
            f"Here are the rubric criteria with their classifications:\n\n{criteria_list}\n\n"
            f"Rank these criteria from MOST important to LEAST important based on your "
            f"actual writing preferences.\n\n"
            f'Return ONLY a JSON array of criterion names in order: ["most important", "second", ...]'
        )
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=500)
            try:
                import re
                match = re.search(r'\[[\s\S]*\]', text)
                if match:
                    return json.loads(match.group())
                return json.loads(text)
            except Exception as e:
                last_error = e
                log.warning(f"rank_importance parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"rank_importance failed after {max_retries} retries: {last_error}")
        return [c.get("name", "?") for c in rubric_criteria]

    # ── Rubric editing (Log Changes) ────────────────────────────────────

    def should_edit_rubric(self, messages: list, draft_count: int,
                           max_drafts: int, edit_count: int) -> bool:
        """Decide whether the synthetic user wants to directly edit the rubric now.

        The user can edit the rubric multiple times per iteration. They must do it
        at least once. They're more likely to do it after seeing a pattern of
        issues across multiple drafts.

        Args:
            messages: conversation history
            draft_count: how many drafts have been produced so far
            max_drafts: maximum drafts allowed this iteration
            edit_count: how many times log_changes has fired this iteration

        Returns:
            True if the user wants to edit the rubric now.
        """
        # Need at least 2 drafts of context to have a pattern worth editing for
        if draft_count < 2:
            return False

        # Force at least one edit: if we haven't edited yet and we're on the
        # last draft, force it now (this is the last chance)
        if edit_count == 0 and draft_count >= max_drafts - 1:
            return True

        # Ask the LLM to decide based on conversation context
        recent_msgs = messages[-8:] if len(messages) > 8 else messages
        context = "\n".join(
            f"[{m['role']}]: {m['content'][:200]}"
            for m in recent_msgs if m.get("role") in ("user", "assistant")
        )

        already_note = ""
        if edit_count > 0:
            already_note = (
                f"\n\nYou have already edited the rubric {edit_count} time(s) this "
                f"session. Only edit again if you've noticed NEW recurring issues "
                f"since your last edit."
            )

        prompt = (
            f"You've been giving feedback on drafts from an AI writing assistant.\n\n"
            f"Here's the recent conversation:\n{context}\n\n"
            f"You've seen {draft_count} drafts so far. You have access to a rubric "
            f"that guides the AI's writing. You can directly edit the rubric to fix "
            f"recurring issues you've noticed across drafts.\n\n"
            f"Based on the conversation so far, do you feel like there's a PATTERN "
            f"of issues that would be better fixed by editing the rubric rather than "
            f"giving another round of feedback? Only say yes if you've seen the same "
            f"kind of problem come up multiple times.{already_note}\n\n"
            f"Return ONLY a JSON object: {{\"edit_rubric\": true/false, \"reason\": \"brief explanation\"}}"
        )

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=200)
            try:
                result = self._extract_json(text)
                should = result.get("edit_rubric", False)
                reason = result.get("reason", "")
                log.info(f"[{self.persona.name}] Should edit rubric? {should} — {reason}")
                return should
            except Exception as e:
                last_error = e
                log.warning(f"should_edit_rubric parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"should_edit_rubric failed after {max_retries} retries: {last_error}")
        # If we can't parse, use a heuristic: edit after draft 5 if not yet done
        return edit_count == 0 and draft_count >= 5

    def propose_rubric_edits(self, rubric_criteria: list) -> dict:
        """Synthetic user proposes direct edits to the rubric.

        Returns dict with 'edited_rubric' (list) and 'reasoning' (str).
        """
        from prompts import RUBRIC_EDIT_PROPOSAL_PROMPT
        prompt = RUBRIC_EDIT_PROPOSAL_PROMPT(rubric_criteria, self.persona)
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=4000)
            try:
                result = self._extract_json(text)
                if "edited_rubric" not in result:
                    raise ValueError("No edited_rubric key in response")
                return result
            except Exception as e:
                last_error = e
                log.warning(f"propose_rubric_edits parse attempt {attempt + 1}/{max_retries} failed: {e}")
                log.debug(f"propose_rubric_edits raw response (attempt {attempt + 1}): {text[:500]}")

        log.error(f"propose_rubric_edits failed after {max_retries} retries: {last_error}")
        return {"edited_rubric": rubric_criteria, "reasoning": f"parse error after {max_retries} retries: {last_error}"}

    def review_annotated_changes(self, annotated_changes: list,
                                 original_draft: str, revised_draft: str) -> list:
        """Review each annotated change from a draft regeneration and give feedback.

        In the real app, the user sees each [N] marker with the edit reason and
        can leave feedback if they disagree. This simulates that.

        Args:
            annotated_changes: list of dicts with 'reason', 'original_text', 'new_text'
            original_draft: the draft before rubric changes
            revised_draft: the draft after rubric changes

        Returns:
            list of annotated_changes with 'user_feedback' added to each entry.
            Empty string means the user agrees with the edit.
        """
        if not annotated_changes:
            return []

        changes_text = "\n\n".join(
            f"[{i+1}] Reason: {ac.get('reason', '')}\n"
            f"    Original: \"{(ac.get('original_text') or '')[:100]}\"\n"
            f"    New: \"{(ac.get('new_text') or '')[:100]}\""
            for i, ac in enumerate(annotated_changes)
        )

        prompt = (
            f"The AI writing assistant just regenerated your draft based on rubric "
            f"changes you made. Here are the specific edits it made:\n\n"
            f"{changes_text}\n\n"
            f"For each edit, decide if you AGREE or DISAGREE. If you disagree, "
            f"explain briefly what you would have done differently — talk like a "
            f"normal person giving feedback, not a rubric analyst.\n\n"
            f"You should disagree with edits that don't match your actual "
            f"preferences, even if they technically follow the rubric wording.\n\n"
            f"Return ONLY a JSON array with one entry per edit:\n"
            f'[{{"edit_number": 1, "agree": true/false, "feedback": "..."}}]\n'
            f"If you agree, set feedback to empty string."
        )

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=1000)
            try:
                import re
                match = re.search(r'\[[\s\S]*\]', text)
                if match:
                    reviews = json.loads(match.group())
                else:
                    reviews = json.loads(text)

                # Merge feedback back into annotated_changes
                result = []
                for i, ac in enumerate(annotated_changes):
                    ac_copy = dict(ac)
                    review = reviews[i] if i < len(reviews) else {}
                    ac_copy["user_feedback"] = review.get("feedback", "")
                    result.append(ac_copy)

                log.info(
                    f"[{self.persona.name}] Reviewed {len(annotated_changes)} edits, "
                    f"feedback on {sum(1 for r in result if r.get('user_feedback', '').strip())} edits"
                )
                return result
            except Exception as e:
                last_error = e
                log.warning(f"review_annotated_changes parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"review_annotated_changes failed after {max_retries} retries: {last_error}")
        return [dict(ac, user_feedback="") for ac in annotated_changes]

    # ── Surveys ──────────────────────────────────────────────────────────

    def complete_survey_task_a(self, messages: list) -> dict:
        """Complete Task A survey (without rubric) from persona's perspective.

        Returns: {"q1": int, "q2": int, "q3": str, "completed": True, "timestamp": str}
        """
        from prompts import SURVEY_TASK_A_PROMPT
        from datetime import datetime

        # Build conversation summary from recent messages
        summary_msgs = messages[-12:] if len(messages) > 12 else messages
        conversation_summary = "\n".join(
            f"[{m['role']}]: {m['content'][:300]}"
            for m in summary_msgs if m.get("role") in ("user", "assistant")
        )

        prompt = SURVEY_TASK_A_PROMPT(self.persona, conversation_summary)

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=500)
            try:
                result = self._extract_json(text)
                result.setdefault("q1", None)
                result.setdefault("q2", None)
                result.setdefault("q3", "")
                if result["q1"] is not None:
                    result["q1"] = max(1, min(5, int(result["q1"])))
                if result["q2"] is not None:
                    result["q2"] = max(1, min(5, int(result["q2"])))
                result["completed"] = True
                result["timestamp"] = datetime.now().isoformat()
                return result
            except Exception as e:
                last_error = e
                log.warning(f"complete_survey_task_a parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"complete_survey_task_a failed after {max_retries} retries: {last_error}")
        return {"q1": None, "q2": None, "q3": "parse error",
                "completed": True, "timestamp": datetime.now().isoformat()}

    def complete_survey_task_b(self, messages: list, rubric_data: dict,
                               iteration: int) -> dict:
        """Complete Task B survey (with rubric) from persona's perspective.

        Returns: {"q1"-"q6": ..., "completed": True, "iteration": int, "timestamp": str}
        """
        from prompts import SURVEY_TASK_B_PROMPT
        from datetime import datetime

        summary_msgs = messages[-12:] if len(messages) > 12 else messages
        conversation_summary = "\n".join(
            f"[{m['role']}]: {m['content'][:300]}"
            for m in summary_msgs if m.get("role") in ("user", "assistant")
        )

        rubric_criteria = rubric_data.get("rubric", []) if rubric_data else []
        rubric_criteria_summary = "\n".join(
            f"- {c.get('name', '?')} (priority #{c.get('priority', '?')}): {c.get('description', '')}"
            for c in rubric_criteria
        )

        prompt = SURVEY_TASK_B_PROMPT(
            self.persona, conversation_summary, rubric_criteria_summary, iteration
        )
        valid_q4 = {"Much better", "Somewhat better", "About the same",
                     "Somewhat worse", "Much worse"}

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            text = self.llm.generate(self._system_prompt, prompt, max_tokens=800)
            try:
                result = self._extract_json(text)
                result.setdefault("q1", None)
                result.setdefault("q2", None)
                result.setdefault("q3", "")
                result.setdefault("q4", None)
                result.setdefault("q5", "")
                result.setdefault("q6", "")
                if result["q1"] is not None:
                    result["q1"] = max(1, min(5, int(result["q1"])))
                if result["q2"] is not None:
                    result["q2"] = max(1, min(5, int(result["q2"])))
                if result["q4"] is not None and result["q4"] not in valid_q4:
                    result["q4"] = None
                result["completed"] = True
                result["iteration"] = iteration
                result["timestamp"] = datetime.now().isoformat()
                return result
            except Exception as e:
                last_error = e
                log.warning(f"complete_survey_task_b parse attempt {attempt + 1}/{max_retries} failed: {e}")

        log.error(f"complete_survey_task_b failed after {max_retries} retries: {last_error}")
        return {"q1": None, "q2": None, "q3": "parse error",
                "q4": None, "q5": "parse error", "q6": "parse error",
                "completed": True, "iteration": iteration,
                "timestamp": datetime.now().isoformat()}

    # ── Survey: Final Review ────────────────────────────────────────────

    def complete_survey_final(self, rubric_data: dict) -> dict:
        """Complete Final Review survey from persona's perspective.

        Returns: {"criteria_ratings": {name: {"accuracy": str, "explanation": str}},
                  "q2": str, "q3": str, "q4": int, "q5": int, "q6": int,
                  "completed": True, "timestamp": str}
        """
        from prompts import SURVEY_FINAL_REVIEW_PROMPT
        from datetime import datetime

        rubric_criteria = rubric_data.get("rubric", []) if rubric_data else []
        if not rubric_criteria:
            return {"criteria_ratings": {}, "q2": "No rubric available",
                    "q3": "", "q4": 3, "q5": 3, "q6": 3,
                    "completed": True, "timestamp": datetime.now().isoformat()}

        prompt = SURVEY_FINAL_REVIEW_PROMPT(self.persona, rubric_criteria)
        text = self.llm.generate(self._system_prompt, prompt, max_tokens=2000)

        valid_accuracy = {"Accurate", "Partially right", "Inaccurate"}
        try:
            result = self._extract_json(text)
            result.setdefault("criteria_ratings", {})
            result.setdefault("q2", "")
            result.setdefault("q3", "")
            for qk in ("q4", "q5", "q6"):
                val = result.get(qk)
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    result[qk] = int(val)
                else:
                    result[qk] = 3  # default to neutral
            for crit_name, rating in result.get("criteria_ratings", {}).items():
                if isinstance(rating, dict):
                    if rating.get("accuracy") not in valid_accuracy:
                        rating["accuracy"] = "Partially right"
                    rating.setdefault("explanation", "")
            result["completed"] = True
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception as e:
            log.error(f"Failed to parse Final Review survey: {e}")
            return {
                "criteria_ratings": {
                    c.get("name", "?"): {"accuracy": "Partially right", "explanation": "parse error"}
                    for c in rubric_criteria
                },
                "q2": "parse error", "q3": "parse error",
                "q4": 3, "q5": 3, "q6": 3,
                "completed": True,
                "timestamp": datetime.now().isoformat(),
            }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str):
        """Extract JSON object from LLM response text."""
        import re
        text = text.strip()

        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        fence_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find the LAST complete JSON object (models often put analysis before it)
        # Use a non-greedy approach: find all { positions and try parsing from each
        last_valid = None
        for i in range(len(text) - 1, -1, -1):
            if text[i] == '}':
                # Find matching opening brace by trying substrings
                for j in range(i, -1, -1):
                    if text[j] == '{':
                        candidate = text[j:i + 1]
                        try:
                            last_valid = json.loads(candidate)
                            return last_valid
                        except json.JSONDecodeError:
                            continue
                break  # only try the last '}'

        # Fallback: greedy match (original behavior)
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        return json.loads(text)
