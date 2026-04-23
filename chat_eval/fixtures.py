"""Fixtures for the chat system prompt eval.

Each fixture is (rubric, task, checks) where:
  - `rubric`: a realistic rubric dict matching the app's schema (with criteria,
    dimensions, priorities, and confidence levels). These mirror what the
    rubric inference prompt would produce for the same conversation.
  - `task`: a concrete writing task the user might ask the chat to handle.
  - `checks`: a list of dicts with {criterion, dim_label, priority, confidence,
    question} where `question` is a yes/no test a judge can run against the
    draft. Questions are phrased so YES = dim honored, NO = dim violated.

Fixtures span 5 genres with varying rubric sizes. Each fixture includes at
least one deliberately counter-default dim (a preference that goes AGAINST
what a baseline LLM would naturally produce) so we can detect whether the
rubric actually steered the output vs. the LLM just writing genre-default
prose and coincidentally satisfying soft checks."""

FIXTURES: list[dict] = [
    {
        "name": "cold_email_founder",
        "task": "Write a cold email to Priya Desai at Accel. I'm raising a seed round for my dev-tools startup. We automate GitHub code review with AI. $800K ARR, 40 enterprise customers signed this quarter.",
        "rubric": {
            "writing_type": "cold outreach emails (founder to investor)",
            "rubric": [
                {
                    "name": "Recipient-specific hook",
                    "category": "Structure",
                    "description": "The first line references something specific to the recipient — a recent post, a portfolio company, a stated thesis. No 'hope this finds you well'.",
                    "dimensions": [
                        {"id": "hook_specific", "label": "Opens with a specific reference to the recipient, not a greeting"},
                        {"id": "no_boilerplate_opener", "label": "Does not open with 'hope this finds you well' or 'I hope you're doing well'"},
                    ],
                    "priority": 1,
                    "confidence": "high",
                },
                {
                    "name": "Hard number up front",
                    "category": "Content",
                    "description": "At least one concrete quantitative detail (ARR, customers, growth rate) appears in the first three lines.",
                    "dimensions": [
                        {"id": "number_in_first_three", "label": "At least one specific number (revenue, customer count, growth rate) appears in the first three lines"},
                    ],
                    "priority": 2,
                    "confidence": "high",
                },
                {
                    "name": "Plain-English founder voice",
                    "category": "Style",
                    "description": "No banker/corporate jargon. Reads like a human talking, not a pitch deck.",
                    "dimensions": [
                        {"id": "no_jargon", "label": "Uses plain English rather than investor/banker jargon"},
                        {"id": "no_hedging", "label": "No hedging language like 'we believe' or 'we think we may be'"},
                    ],
                    "priority": 3,
                    "confidence": "high",
                },
                {
                    "name": "Compact ask",
                    "category": "Structure",
                    "description": "Single short ask at the end. Names a specific duration.",
                    "dimensions": [
                        {"id": "ask_specific_duration", "label": "Closing ask names a specific time duration (e.g. '10 min', '15 min')"},
                    ],
                    "priority": 4,
                    "confidence": "medium",
                },
            ],
        },
        "checks": [
            {
                "criterion": "Recipient-specific hook",
                "dim_label": "Opens with a specific reference to the recipient, not a greeting",
                "priority": 1, "confidence": "high",
                "question": "Does the first sentence reference something specific to Priya or Accel (a portfolio company, a recent post, a stated thesis), rather than a generic greeting?",
            },
            {
                "criterion": "Recipient-specific hook",
                "dim_label": "Does not open with 'hope this finds you well' or similar",
                "priority": 1, "confidence": "high",
                "question": "Does the draft AVOID opening with 'I hope this finds you well', 'Hope you're doing well', or any similar boilerplate greeting?",
            },
            {
                "criterion": "Hard number up front",
                "dim_label": "Number in first three lines",
                "priority": 2, "confidence": "high",
                "question": "Does at least one specific number (revenue, customer count, growth rate) appear in the first three lines of the email body?",
            },
            {
                "criterion": "Plain-English founder voice",
                "dim_label": "No investor/banker jargon",
                "priority": 3, "confidence": "high",
                "question": "Is the draft free of investor/banker jargon like 'thesis-aligned', 'go-to-market motion', 'tackling the compliance side', 'velocity', etc.?",
            },
            {
                "criterion": "Plain-English founder voice",
                "dim_label": "No hedging language",
                "priority": 3, "confidence": "high",
                "question": "Is the draft free of hedging phrases like 'we believe', 'we think we may be', 'potentially', 'hopefully'?",
            },
            {
                "criterion": "Compact ask",
                "dim_label": "Closing ask names specific duration",
                "priority": 4, "confidence": "medium",
                "question": "Does the closing include a specific time duration (like '10 min', '15 min', 'a quick call next Tuesday')?",
            },
        ],
    },

    {
        "name": "eng_manager_slack_update",
        "task": "Write a Slack update for my eng team. We're behind on the data-pipeline migration; a race condition in the batch worker caused the delay; we've got a fix in review; new ship date is December 2nd.",
        "rubric": {
            "writing_type": "engineering manager Slack updates",
            "rubric": [
                {
                    "name": "Lowercase, Slack-native register",
                    "category": "Style",
                    "description": "Lowercase voice. No corporate headers like 'Team,'. Reads like a real Slack message.",
                    "dimensions": [
                        {"id": "lowercase", "label": "The message is lowercase throughout (including first words of sentences)"},
                        {"id": "no_formal_salutation", "label": "Does not open with 'Team,', 'Dear team,', or similar formal salutation"},
                    ],
                    "priority": 1,
                    "confidence": "high",
                },
                {
                    "name": "Name the technical root cause",
                    "category": "Content",
                    "description": "The specific technical cause of the delay is named, not paraphrased as 'an issue' or 'unforeseen challenges'.",
                    "dimensions": [
                        {"id": "root_cause_named", "label": "Names the specific technical cause (e.g. a race condition, a deadlock, a schema mismatch)"},
                    ],
                    "priority": 2,
                    "confidence": "high",
                },
                {
                    "name": "End on the number/date that matters",
                    "category": "Structure",
                    "description": "The last line is the single piece of info the team needs to take away — typically a date, a ship target, or a concrete number.",
                    "dimensions": [
                        {"id": "ends_on_key_info", "label": "Final line is a specific date, number, or target — not 'ping me with questions'"},
                    ],
                    "priority": 3,
                    "confidence": "high",
                },
            ],
        },
        "checks": [
            {
                "criterion": "Lowercase, Slack-native register",
                "dim_label": "Lowercase throughout",
                "priority": 1, "confidence": "high",
                "question": "Is the message written in lowercase throughout, INCLUDING the first word of each sentence? (Proper nouns like names of technologies may be capitalized.)",
            },
            {
                "criterion": "Lowercase, Slack-native register",
                "dim_label": "No formal salutation",
                "priority": 1, "confidence": "high",
                "question": "Does the message AVOID opening with a formal salutation like 'Team,', 'Dear team,', 'Hi all,', etc.?",
            },
            {
                "criterion": "Name the technical root cause",
                "dim_label": "Specific technical cause named",
                "priority": 2, "confidence": "high",
                "question": "Does the message name a specific technical cause (like 'race condition in the batch worker', 'deadlock', 'schema mismatch'), rather than vague phrases like 'unforeseen challenges' or 'an issue'?",
            },
            {
                "criterion": "End on the number/date that matters",
                "dim_label": "Final line is a date/number/target",
                "priority": 3, "confidence": "high",
                "question": "Does the message END on a specific date, number, or ship target (like 'new target ship date: December 2nd'), rather than ending with 'ping me with questions' or similar?",
            },
        ],
    },

    {
        "name": "ml_blog_intro",
        "task": "Write the intro to my blog post announcing our new method for reducing classifier-free guidance saturation in diffusion models. Audience: ML practitioners.",
        "rubric": {
            "writing_type": "ML research blog post intros",
            "rubric": [
                {
                    "name": "Mechanism-first opening",
                    "category": "Structure",
                    "description": "Opens by naming the specific phenomenon or mechanism the post is about — not by explaining why diffusion models matter.",
                    "dimensions": [
                        {"id": "phenomenon_first", "label": "First sentence names a specific phenomenon, pathology, or mechanism"},
                        {"id": "no_hype_framing", "label": "Does not open with 'X has revolutionized Y' or similar hype framing"},
                    ],
                    "priority": 1,
                    "confidence": "high",
                },
                {
                    "name": "No marketing voice",
                    "category": "Style",
                    "description": "Reads like a researcher wrote it, not marketing. No 'you'll be amazed', no 'stunning results', no 'cutting-edge'.",
                    "dimensions": [
                        {"id": "no_marketing_superlatives", "label": "Avoids marketing superlatives ('stunning', 'amazing', 'revolutionary', 'cutting-edge', 'state-of-the-art')"},
                        {"id": "no_you_will_language", "label": "Does not use 'you'll learn', 'you'll be amazed', 'you'll discover' constructions"},
                    ],
                    "priority": 2,
                    "confidence": "high",
                },
                {
                    "name": "Explicit payoff sign-posting",
                    "category": "Structure",
                    "description": "Ends the intro by telling the reader what tables/figures/results to expect later.",
                    "dimensions": [
                        {"id": "signposts_results", "label": "Intro closes with an explicit sign-post (e.g. 'Tables at the end', 'Ablations in section 3')"},
                    ],
                    "priority": 3,
                    "confidence": "medium",
                },
            ],
        },
        "checks": [
            {
                "criterion": "Mechanism-first opening",
                "dim_label": "Phenomenon/mechanism in first sentence",
                "priority": 1, "confidence": "high",
                "question": "Does the first sentence name a specific phenomenon, pathology, or mechanism (like 'Classifier-free guidance has a known saturation pathology'), rather than explaining why diffusion models matter?",
            },
            {
                "criterion": "Mechanism-first opening",
                "dim_label": "No hype framing",
                "priority": 1, "confidence": "high",
                "question": "Does the intro AVOID hype framings like 'X has revolutionized Y', 'transformative advances in...', 'cutting-edge research in...'?",
            },
            {
                "criterion": "No marketing voice",
                "dim_label": "No marketing superlatives",
                "priority": 2, "confidence": "high",
                "question": "Is the draft free of marketing superlatives like 'stunning', 'amazing', 'revolutionary', 'cutting-edge', 'state-of-the-art'?",
            },
            {
                "criterion": "No marketing voice",
                "dim_label": "No 'you'll learn' / 'you'll be amazed'",
                "priority": 2, "confidence": "high",
                "question": "Does the draft AVOID 'you'll learn', 'you'll be amazed', 'you'll discover' style sentences?",
            },
            {
                "criterion": "Explicit payoff sign-posting",
                "dim_label": "Closes with result sign-post",
                "priority": 3, "confidence": "medium",
                "question": "Does the intro end with an explicit sign-post telling the reader what tables, figures, or specific results to expect (like 'Tables at the end', 'See ablations in Section 3')?",
            },
        ],
    },

    {
        "name": "condolence_note",
        "task": "One of my direct reports' mother just passed. She's been on my team four years, professional-friendly rapport but not personal friends. Write a short note.",
        "rubric": {
            "writing_type": "manager-to-report condolence notes",
            "rubric": [
                {
                    "name": "Zero boilerplate condolence language",
                    "category": "Style",
                    "description": "No 'deepest sympathy', 'sincere condolences', 'heartfelt', 'thoughts and prayers'. Reads like a real person.",
                    "dimensions": [
                        {"id": "no_deepest_sympathy", "label": "Does not use phrases like 'deepest sympathy', 'sincere condolences', 'heartfelt', 'thoughts and prayers'"},
                    ],
                    "priority": 1,
                    "confidence": "high",
                },
                {
                    "name": "Support as already-done action",
                    "category": "Content",
                    "description": "Offers concrete support by naming what's already been handled, not by making a vague offer like 'let me know if you need anything'.",
                    "dimensions": [
                        {"id": "support_concrete", "label": "States a specific concrete action already taken on the recipient's behalf (e.g. 'I've let the team know you'll be out')"},
                        {"id": "no_generic_offer", "label": "Avoids vague offers like 'let me know if there's anything I can do' or 'here if you need anything'"},
                    ],
                    "priority": 2,
                    "confidence": "high",
                },
                {
                    "name": "Stripped-down brevity",
                    "category": "Structure",
                    "description": "Short. Four sentences or fewer. No introduction, no framing.",
                    "dimensions": [
                        {"id": "brief", "label": "Body is four sentences or fewer"},
                    ],
                    "priority": 3,
                    "confidence": "medium",
                },
            ],
        },
        "checks": [
            {
                "criterion": "Zero boilerplate condolence language",
                "dim_label": "No boilerplate condolence phrases",
                "priority": 1, "confidence": "high",
                "question": "Is the note free of stock phrases like 'deepest sympathy', 'sincere condolences', 'heartfelt', 'thoughts and prayers'?",
            },
            {
                "criterion": "Support as already-done action",
                "dim_label": "Names a concrete action already taken",
                "priority": 2, "confidence": "high",
                "question": "Does the note state at least one specific concrete action already taken on the recipient's behalf (like 'I've already let the team know', 'nothing needs a decision from you right now')?",
            },
            {
                "criterion": "Support as already-done action",
                "dim_label": "No vague 'let me know' offer",
                "priority": 2, "confidence": "high",
                "question": "Does the note AVOID vague offers like 'let me know if there's anything I can do' or 'here for you if you need anything'?",
            },
            {
                "criterion": "Stripped-down brevity",
                "dim_label": "Four sentences or fewer",
                "priority": 3, "confidence": "medium",
                "question": "Is the body of the note four sentences or fewer?",
            },
        ],
    },

    {
        "name": "physics_abstract",
        "task": "Draft an abstract for my paper reporting anomalous thermal conductivity measurements in twisted bilayer graphene at the magic angle.",
        "rubric": {
            "writing_type": "physics paper abstracts",
            "rubric": [
                {
                    "name": "Quantitative result in-line",
                    "category": "Content",
                    "description": "Names a specific quantitative finding (numerical value + conditions) rather than describing the result qualitatively.",
                    "dimensions": [
                        {"id": "quantitative_result", "label": "Abstract contains at least one specific numerical result with conditions (e.g. factor of 4.3 suppression at T=100K)"},
                    ],
                    "priority": 1,
                    "confidence": "high",
                },
                {
                    "name": "Measurement → result → mechanism structure",
                    "category": "Structure",
                    "description": "Abstract proceeds: what was measured, what was found, what mechanism is proposed. In that order.",
                    "dimensions": [
                        {"id": "measurement_first", "label": "First sentence describes what was measured, not what was found"},
                        {"id": "mechanism_claim", "label": "A mechanism-level claim appears (not just phenomenology)"},
                    ],
                    "priority": 2,
                    "confidence": "high",
                },
                {
                    "name": "No hand-wavy implications closer",
                    "category": "Style",
                    "description": "Does not end with 'These findings have significant implications for...' or similar hand-waving toward applications.",
                    "dimensions": [
                        {"id": "no_implications_closer", "label": "Does not end with a hand-wavy 'implications for next-generation X' sentence"},
                    ],
                    "priority": 3,
                    "confidence": "high",
                },
            ],
        },
        "checks": [
            {
                "criterion": "Quantitative result in-line",
                "dim_label": "Specific numerical result with conditions",
                "priority": 1, "confidence": "high",
                "question": "Does the abstract include at least one specific numerical result with conditions (like 'κ drops by a factor of 4.3 at T=100K', '30% reduction at 1.08°')?",
            },
            {
                "criterion": "Measurement → result → mechanism structure",
                "dim_label": "First sentence: what was measured",
                "priority": 2, "confidence": "high",
                "question": "Does the first sentence describe WHAT WAS MEASURED (the observable, the conditions), rather than jumping to the RESULT?",
            },
            {
                "criterion": "Measurement → result → mechanism structure",
                "dim_label": "Mechanism-level claim present",
                "priority": 2, "confidence": "high",
                "question": "Does the abstract propose a mechanism-level claim (like 'suggesting coupling between X and Y'), not just phenomenological observations?",
            },
            {
                "criterion": "No hand-wavy implications closer",
                "dim_label": "No 'implications for next-gen X' closer",
                "priority": 3, "confidence": "high",
                "question": "Does the abstract AVOID ending with a vague 'implications for next-generation electronic devices' / 'transformative applications' style closer?",
            },
        ],
    },
]



# ---------------------------------------------------------------------------
# CONFLICT FIXTURES
# ---------------------------------------------------------------------------
# Tests the "rubric authority" guidance: when the user's task contradicts
# the rubric, the chat prompt says "follow the task for that request, but
# don't silently deviate from the rubric." We want drafts that satisfy the
# task's explicit content AND the rubric's style/structure constraints --
# NOT drafts that abandon the rubric whenever the task surface-disagrees.

CONFLICT_FIXTURES: list[dict] = [
    {
        "name": "conflict_cold_email_formal_ask",
        "task": "Write a more formal cold email to Priya Desai at Accel -- I want this one to be traditional and polished since she's a senior partner. I'm raising a seed for my dev-tools AI startup. $800K ARR.",
        # User explicitly asks for "more formal" and "traditional" which
        # surface-conflicts with the rubric's "plain-English founder voice"
        # and "no hedging". The chat prompt should adjust tone slightly
        # (one-time adjustment) without abandoning the rubric core.
        "rubric": {
            "writing_type": "cold outreach emails (founder to investor)",
            "rubric": [
                {
                    "name": "Recipient-specific hook",
                    "description": "First line references something specific to the recipient.",
                    "dimensions": [
                        {"id": "hook_specific", "label": "Opens with a specific reference to the recipient, not a greeting"},
                        {"id": "no_boilerplate_opener", "label": "Does not open with 'hope this finds you well'"},
                    ],
                    "priority": 1, "confidence": "high",
                },
                {
                    "name": "Hard number up front",
                    "description": "At least one concrete number in the first three lines.",
                    "dimensions": [
                        {"id": "number_in_first_three", "label": "At least one specific number appears in the first three lines"},
                    ],
                    "priority": 2, "confidence": "high",
                },
                {
                    "name": "No hedging",
                    "description": "No 'we believe', 'we think we may be', 'potentially'.",
                    "dimensions": [
                        {"id": "no_hedging", "label": "No hedging phrases like 'we believe', 'potentially'"},
                    ],
                    "priority": 3, "confidence": "high",
                },
            ],
        },
        "checks": [
            # Rubric dims should still be satisfied even under the "make it
            # more formal" task -- a one-time tonal adjustment shouldn't mean
            # abandoning specific-hook, numbers, or no-hedging.
            {
                "criterion": "Recipient-specific hook",
                "dim_label": "Still opens with recipient-specific reference",
                "priority": 1, "confidence": "high",
                "question": "Does the first sentence reference something specific to Priya or Accel, rather than starting with a generic formal opener like 'Dear Ms. Desai, I hope this message finds you well'?",
            },
            {
                "criterion": "Recipient-specific hook",
                "dim_label": "No boilerplate opener despite formality request",
                "priority": 1, "confidence": "high",
                "question": "Does the draft AVOID opening with 'Dear Ms. Desai, I hope this email finds you well' or similar formal boilerplate greetings?",
            },
            {
                "criterion": "Hard number up front",
                "dim_label": "Number still in first three lines",
                "priority": 2, "confidence": "high",
                "question": "Does a specific number (like '$800K ARR') still appear in the first three lines of the email body?",
            },
            {
                "criterion": "No hedging",
                "dim_label": "Still no hedging despite formality",
                "priority": 3, "confidence": "high",
                "question": "Is the draft still free of hedging phrases like 'we believe', 'we think we may be', 'potentially' -- despite being more formal in tone?",
            },
            # Task-side check: the draft SHOULD be somewhat more formal than
            # the standard rubric-driven style. This tests whether the model
            # adjusted at all for the one-time request.
            {
                "criterion": "[task] Tonal adjustment honored",
                "dim_label": "More formal than default",
                "priority": 99, "confidence": "high",
                "question": "Does the draft read somewhat MORE formally than a casual founder voice (e.g. uses 'Priya,' or 'Ms. Desai,' instead of just the first name as a dash, slightly more complete sentences)?",
            },
        ],
    },

    {
        "name": "conflict_slack_update_leadership_ask",
        "task": "The VP of Engineering specifically asked me to write this Slack update as a more professional announcement since it'll also go to the broader engineering org. Migration is ~2 weeks behind; race condition in batch worker caused it; fix in review; new target is December 2.",
        # User asks for "more professional announcement" -- surface conflict
        # with the rubric's lowercase Slack-native register. The rubric
        # prefers lowercase and no formal salutation; a broader audience
        # arguably calls for slight elevation. The chat prompt should NOT
        # silently abandon the rubric -- it should ask about permanent
        # preference shift, OR apply the rubric while slightly adjusting.
        "rubric": {
            "writing_type": "engineering manager Slack updates",
            "rubric": [
                {
                    "name": "Lowercase, Slack-native register",
                    "description": "Lowercase throughout. No 'Team,' salutations.",
                    "dimensions": [
                        {"id": "lowercase", "label": "The message is lowercase throughout"},
                        {"id": "no_formal_salutation", "label": "Does not open with 'Team,' or similar formal salutation"},
                    ],
                    "priority": 1, "confidence": "high",
                },
                {
                    "name": "Name the technical root cause",
                    "description": "Names the specific technical cause.",
                    "dimensions": [
                        {"id": "root_cause_named", "label": "Names the specific technical cause"},
                    ],
                    "priority": 2, "confidence": "high",
                },
                {
                    "name": "End on date/number",
                    "description": "Ends on the date or number that matters.",
                    "dimensions": [
                        {"id": "ends_on_key_info", "label": "Final line is a specific date, number, or target"},
                    ],
                    "priority": 3, "confidence": "high",
                },
            ],
        },
        "checks": [
            # This is a harder test -- the broader-audience ask partially
            # justifies some tonal adjustment. We're looking for: the rubric
            # isn't silently discarded. At minimum the technical root cause
            # and the ending-on-a-date rules should be intact.
            {
                "criterion": "Name the technical root cause",
                "dim_label": "Specific technical cause still named",
                "priority": 2, "confidence": "high",
                "question": "Does the message name the specific technical cause ('race condition in the batch worker'), rather than vague phrases like 'unforeseen challenges'?",
            },
            {
                "criterion": "End on date/number",
                "dim_label": "Still ends on a date/number",
                "priority": 3, "confidence": "high",
                "question": "Does the message end on a specific date/number/target (like 'new target ship date: December 2'), rather than ending with a generic 'please reach out with questions'?",
            },
            # The lowercase rule is the one the task most directly pressures.
            # We expect at MOST mild case adjustment, not full abandonment.
            {
                "criterion": "Lowercase, Slack-native register",
                "dim_label": "Did NOT silently abandon the rubric",
                "priority": 1, "confidence": "high",
                "question": "Is the message EITHER (a) still lowercase throughout per the rubric, OR (b) written in a hybrid style that acknowledges the rubric's informal voice (short sentences, direct phrasing) even if sentence-initial words are now capitalized? In other words, did the model avoid silently defaulting to a fully corporate 'Dear Team,' announcement?",
            },
            # Task-side check: did the model acknowledge the tension at all?
            # The chat prompt explicitly says: "If the user's feedback seems
            # to contradict the rubric, ask: 'Should I treat this as a one-
            # time adjustment, or a preference you'd like going forward?'"
            {
                "criterion": "[task] Flagged the rubric conflict",
                "dim_label": "Did the model surface the tension?",
                "priority": 99, "confidence": "medium",
                "question": "Did the model EITHER (a) ask the user whether this should be a one-time adjustment vs. a permanent preference change, OR (b) explicitly note that it's deviating from the established rubric for this one request?",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# LOW-CONFIDENCE FIXTURES
# ---------------------------------------------------------------------------
# Tests the "confidence-aware application" guidance: low-confidence dims
# should be applied lightly. The chat prompt should NOT treat low-confidence
# rubric dims as mandates.

LOW_CONF_FIXTURES: list[dict] = [
    {
        "name": "low_conf_blog_conversational",
        "task": "Write the opening paragraph of a blog post about the recent wave of LLM evaluation research. Audience: ML practitioners.",
        # Notice: the low-confidence dim ("uses second-person 'you' generously")
        # is a weak signal. The chat prompt should NOT force it into this
        # genre (technical blog for practitioners) if it makes the draft
        # feel off. High-confidence dims should be honored fully.
        "rubric": {
            "writing_type": "ML research blog posts",
            "rubric": [
                {
                    "name": "Mechanism-first opening",
                    "description": "First sentence names a specific phenomenon or mechanism.",
                    "dimensions": [
                        {"id": "mechanism_first", "label": "First sentence names a specific phenomenon or mechanism"},
                        {"id": "no_hype_framing", "label": "Does not open with 'X has revolutionized Y' or similar hype"},
                    ],
                    "priority": 1, "confidence": "high",
                },
                {
                    "name": "Conversational direct address",
                    "description": "Talks directly to the reader using 'you' generously.",
                    "dimensions": [
                        {"id": "you_address", "label": "Uses 'you' to directly address the reader within the opening paragraph"},
                    ],
                    "priority": 2, "confidence": "low",
                },
            ],
        },
        "checks": [
            # High-confidence dim must be satisfied.
            {
                "criterion": "Mechanism-first opening",
                "dim_label": "Phenomenon/mechanism in first sentence",
                "priority": 1, "confidence": "high",
                "question": "Does the first sentence name a specific phenomenon or mechanism related to LLM evaluation research, rather than explaining why LLM evaluation matters?",
            },
            {
                "criterion": "Mechanism-first opening",
                "dim_label": "No hype framing",
                "priority": 1, "confidence": "high",
                "question": "Does the draft AVOID hype framings like 'X has revolutionized Y' or 'transformative advances in...'?",
            },
            # Low-confidence dim is a soft signal. It's FINE if the model
            # doesn't force 'you' address -- the genre doesn't naturally
            # call for it. We check whether the model at least didn't mangle
            # the draft trying to satisfy this dim.
            {
                "criterion": "[diagnostic] Low-conf dim handled lightly",
                "dim_label": "Draft feels natural, not forced to 'you'",
                "priority": 99, "confidence": "medium",
                "question": "Does the opening paragraph read naturally for an ML practitioner audience? In particular, if the model did use 'you' address, did it do so naturally rather than shoehorning it in awkwardly?",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# SPARSE RUBRIC FIXTURES
# ---------------------------------------------------------------------------
# Tests fallback behavior. The chat prompt has a branch for when there's no
# rubric at all (or a very short one). The model should produce a reasonable
# draft without trying to bullshit a "rubric" it doesn't have.

SPARSE_FIXTURES: list[dict] = [
    {
        "name": "sparse_no_rubric",
        "task": "Write a short tweet announcing that my startup just raised our seed round.",
        "rubric": {
            "writing_type": "tweets",
            "rubric": [],  # No criteria at all
        },
        "checks": [
            {
                "criterion": "[baseline] Produces a reasonable draft",
                "dim_label": "Draft exists and is tweet-length",
                "priority": 1, "confidence": "high",
                "question": "Is the draft a plausible tweet (short, single-topic, roughly 280 characters or less) rather than a multi-paragraph essay?",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

ALL_FIXTURES = FIXTURES + CONFLICT_FIXTURES + LOW_CONF_FIXTURES + SPARSE_FIXTURES


def get_fixture(name: str) -> dict:
    for f in ALL_FIXTURES:
        if f["name"] == name:
            return f
    raise KeyError(name)
