"""Fixtures for the grading prompt eval.

Each fixture defines a rubric and three draft variants with known ground truth:

  - `all_met`: a draft deliberately crafted to satisfy EVERY dimension.
    Ground truth: every dim is MET.

  - `all_violated`: a draft deliberately crafted to violate EVERY dimension.
    Ground truth: every dim is NOT_MET.

  - `mixed`: a draft that satisfies some dims and violates others. Ground
    truth is per-dim and explicit in `mixed_ground_truth`.

These variants let us measure:
  - false positive rate (grader says MET when ground truth is NOT_MET)
  - false negative rate (grader says NOT_MET when ground truth is MET)
  - accuracy on mixed drafts (can it distinguish partial violations?)
  - confidence calibration (high-conf graded accurately more often?)

The rubrics are deliberately short and the dim wordings deliberately
unambiguous. If the grader fails on these, it's a grader-quality issue,
not an ambiguous-rubric issue."""


FIXTURES: list[dict] = [
    # -----------------------------------------------------------------------
    # Cold email
    # -----------------------------------------------------------------------
    {
        "name": "cold_email",
        "rubric": {
            "writing_type": "cold outreach emails (founder to investor)",
            "rubric": [
                {
                    "name": "Opening",
                    "dimensions": [
                        {"id": "recipient_hook", "label": "First sentence references something specific to the recipient (not a generic greeting)"},
                        {"id": "no_boilerplate", "label": "Does not contain 'hope this finds you well' or 'hope you're doing well'"},
                    ],
                    "priority": 1,
                },
                {
                    "name": "Substance",
                    "dimensions": [
                        {"id": "number_early", "label": "At least one specific number appears in the first three lines of the body"},
                        {"id": "no_hedging", "label": "Does not contain hedging phrases like 'we believe', 'we think', 'potentially', 'hopefully'"},
                    ],
                    "priority": 2,
                },
            ],
        },
        "drafts": {
            "all_met": (
                "Sarah,\n\n"
                "You recently backed CompetitorX in voluntary carbon markets. "
                "We're building the regulated version — $2M ARR, 3 utility contracts "
                "signed last quarter.\n\n"
                "10 min next Tuesday?\n\n"
                "— Alex"
            ),
            "all_violated": (
                "Dear Ms. Chen,\n\n"
                "I hope this email finds you well. I'm reaching out to introduce my startup "
                "in the climate space. We believe our approach is potentially very impactful, "
                "and we think we may be a good fit for Ribbit's portfolio.\n\n"
                "Would you be open to a conversation?\n\n"
                "Best regards,\n"
                "Alex"
            ),
            "mixed": (
                "Sarah,\n\n"
                "I hope this email finds you well. You recently backed CompetitorX in voluntary "
                "carbon markets and we think we may be doing something complementary. "
                "$2M ARR, 3 utility contracts.\n\n"
                "10 min next Tuesday?\n\n"
                "— Alex"
            ),
            "mixed_ground_truth": {
                # For the "mixed" draft, explicit per-dim ground truth.
                "recipient_hook": "MET",   # "You recently backed CompetitorX..."
                "no_boilerplate": "NOT_MET",  # "I hope this email finds you well"
                "number_early": "MET",     # "$2M ARR, 3 utility contracts" in first 3 lines
                "no_hedging": "NOT_MET",   # "we think we may be..."
            },
        },
    },

    # -----------------------------------------------------------------------
    # Slack update
    # -----------------------------------------------------------------------
    {
        "name": "slack_update",
        "rubric": {
            "writing_type": "engineering manager Slack updates",
            "rubric": [
                {
                    "name": "Register",
                    "dimensions": [
                        {"id": "lowercase", "label": "All sentence-initial words are lowercase (proper nouns may be capitalized)"},
                        {"id": "no_formal_salutation", "label": "Does not open with a formal salutation like 'Team,', 'Dear team,', 'Hi all,'"},
                    ],
                    "priority": 1,
                },
                {
                    "name": "Content",
                    "dimensions": [
                        {"id": "root_cause", "label": "Names a specific technical root cause (e.g. race condition, deadlock) rather than vague phrases like 'unforeseen challenges'"},
                        {"id": "ends_on_date", "label": "Final line is a specific date, number, or target"},
                    ],
                    "priority": 2,
                },
            ],
        },
        "drafts": {
            "all_met": (
                "quick update on the migration —\n\n"
                "we're ~2 weeks behind. root cause: a race condition in the batch worker "
                "where concurrent jobs stomped on shared state. fix is in review now.\n\n"
                "new target ship date: december 2"
            ),
            "all_violated": (
                "Team,\n\n"
                "I wanted to give everyone a comprehensive update on our ongoing migration "
                "project. As you are aware, we have encountered a number of unforeseen "
                "challenges that have impacted our original timeline. Leadership remains "
                "fully committed.\n\n"
                "Please reach out with any questions or concerns."
            ),
            "mixed": (
                "Hi team,\n\n"
                "quick update on the migration: we hit a race condition in the batch worker "
                "that broke concurrent writes. fix is in review now.\n\n"
                "new target ship date: december 2"
            ),
            "mixed_ground_truth": {
                "lowercase": "NOT_MET",         # "Hi team,"
                "no_formal_salutation": "NOT_MET",  # "Hi team,"
                "root_cause": "MET",            # "race condition in the batch worker"
                "ends_on_date": "MET",          # "new target ship date: december 2"
            },
        },
    },

    # -----------------------------------------------------------------------
    # Physics abstract
    # -----------------------------------------------------------------------
    {
        "name": "physics_abstract",
        "rubric": {
            "writing_type": "physics paper abstracts",
            "rubric": [
                {
                    "name": "Quantitative content",
                    "dimensions": [
                        {"id": "numerical_result", "label": "Contains at least one specific numerical result with its measurement conditions (e.g. 'factor of 4.3 suppression at T=100K')"},
                    ],
                    "priority": 1,
                },
                {
                    "name": "Structure",
                    "dimensions": [
                        {"id": "measurement_first", "label": "First sentence describes what was measured, not what was found"},
                        {"id": "mechanism_claim", "label": "Proposes a mechanism-level claim (not only phenomenological observation)"},
                    ],
                    "priority": 2,
                },
                {
                    "name": "Closing",
                    "dimensions": [
                        {"id": "no_implications", "label": "Does not end with a vague 'implications for next-generation X' or 'paves the way for' closer"},
                    ],
                    "priority": 3,
                },
            ],
        },
        "drafts": {
            "all_met": (
                "The in-plane thermal conductivity κ of twisted bilayer graphene is measured at twist angles 1.05°–1.15°. "
                "At the magic angle (1.08°), κ drops by a factor of 4.3 relative to monolayer graphene at T = 100 K, "
                "well beyond phonon-boundary-scattering predictions. "
                "The suppression tracks the flat-band bandwidth extracted from separate ARPES measurements, "
                "suggesting a coupling between electronic correlations and heat transport."
            ),
            "all_violated": (
                "We find anomalous thermal behavior in twisted bilayer graphene. "
                "Our results demonstrate significant deviations from conventional phonon transport models. "
                "These findings have significant implications for the design of next-generation electronic devices "
                "and pave the way for novel thermal management strategies."
            ),
            "mixed": (
                "Thermal conductivity suppression of a factor of 4.3 is observed in twisted bilayer graphene at T = 100 K, "
                "beyond phonon-scattering predictions. "
                "The suppression tracks the flat-band bandwidth, suggesting electronic correlations couple to heat transport. "
                "These findings have significant implications for next-generation thermal management."
            ),
            "mixed_ground_truth": {
                "numerical_result": "MET",       # "factor of 4.3 at T=100K"
                "measurement_first": "NOT_MET",  # First sentence names the result, not what was measured
                "mechanism_claim": "MET",        # "suggesting electronic correlations couple..."
                "no_implications": "NOT_MET",    # "significant implications for next-generation..."
            },
        },
    },
]



# ---------------------------------------------------------------------------
# EDGE-CASE FIXTURES
# ---------------------------------------------------------------------------
# Designed to expose weak spots in the grader:
#
#   - "subtle_violations": dims that LOOK satisfied on cursory read but aren't.
#     Tests whether the grader actually inspects vs. pattern-matches keywords.
#
#   - "ambiguous_dim_wording": dims whose wording allows multiple reasonable
#     interpretations. Tests whether the grader correctly lowers confidence.
#
#   - "borderline": drafts that are on the boundary — meet the spirit but not
#     the literal rule, or vice versa. Tests whether the grader grades "what
#     the dimension literally says" (per the prompt).
#
#   - "adversarial_format": drafts that try to game the grader — e.g. use
#     the exact words from the dim label while not actually satisfying it.
#
# Ground truth encodes the *strict* reading per the grader prompt's rules.
# If the grader is being too lenient, it will over-assign MET here.

EDGE_CASE_FIXTURES: list[dict] = [

    # -----------------------------------------------------------------------
    # subtle_violations: draft looks fine on skim but fails on inspection
    # -----------------------------------------------------------------------
    {
        "name": "subtle_violations_cold_email",
        "category": "subtle_violations",
        "rubric": {
            "writing_type": "cold outreach emails",
            "rubric": [
                {
                    "name": "Opening",
                    "dimensions": [
                        {"id": "no_boilerplate_opener", "label": "Does not open with 'I hope this finds you well', 'I hope you're doing well', or similar pleasantries"},
                        {"id": "number_first_para", "label": "At least one specific quantitative number appears in the first paragraph of the body"},
                        {"id": "direct_ask_end", "label": "Ends with a single direct ask (call, meeting, reply) that specifies a concrete next step"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            # Single draft with subtle violations — reads smoothly but fails:
            "subtle": (
                "Hi Priya,\n\n"
                "Hope you're well! I wanted to reach out because I think Accel would be "
                "interested in what we're building. Our traction has been strong, with meaningful "
                "enterprise adoption over the past few quarters.\n\n"
                "Would love to chat if you have time — let me know your thoughts.\n\n"
                "— Alex"
            ),
            "subtle_ground_truth": {
                # "Hope you're well!" is a boilerplate pleasantry, just shortened
                "no_boilerplate_opener": "NOT_MET",
                # "meaningful enterprise adoption" has no specific number
                "number_first_para": "NOT_MET",
                # "let me know your thoughts" is not a direct ask with a concrete next step
                "direct_ask_end": "NOT_MET",
            },
        },
    },

    {
        "name": "subtle_violations_physics",
        "category": "subtle_violations",
        "rubric": {
            "writing_type": "physics paper abstracts",
            "rubric": [
                {
                    "name": "Quantitative",
                    "dimensions": [
                        {"id": "numerical_result_with_conditions", "label": "Contains at least one specific numerical result WITH its measurement conditions (e.g. temperature, angle, sample). A bare number without conditions does not satisfy this."},
                        {"id": "mechanism_not_just_phenomenology", "label": "Proposes a mechanism-level claim (names a physical mechanism or coupling). Describing only what was observed, without proposing why, does not satisfy this."},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            "subtle": (
                "We report anomalous thermal transport in twisted bilayer graphene. "
                "The thermal conductivity is suppressed by a factor of 4.3 compared to "
                "monolayer graphene. This suppression is reproducible across multiple samples "
                "and is robust to measurement noise. The finding represents a significant "
                "departure from conventional transport theory."
            ),
            "subtle_ground_truth": {
                # "factor of 4.3" exists but NO measurement conditions (temperature, twist angle)
                "numerical_result_with_conditions": "NOT_MET",
                # Only describes the observation. "Departure from conventional theory"
                # is a negative claim, not a mechanism.
                "mechanism_not_just_phenomenology": "NOT_MET",
            },
        },
    },

    # -----------------------------------------------------------------------
    # borderline: draft meets spirit but violates letter, or vice versa
    # -----------------------------------------------------------------------
    {
        "name": "borderline_literal_reading",
        "category": "borderline",
        "rubric": {
            "writing_type": "cold outreach emails",
            "rubric": [
                {
                    "name": "Opening",
                    "dimensions": [
                        {"id": "first_name_only_greeting", "label": "Opens with the recipient's FIRST NAME only (not 'Dear [full name]', not 'Hi [first name]', just the first name followed by a comma or newline)"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            "spirit_not_letter": (
                "Hi Sarah,\n\n"
                "Quick note — we're working on something Accel's portfolio might find interesting. "
                "$2M ARR, 40 enterprise customers.\n\n"
                "15 min next week?\n\n"
                "— Alex"
            ),
            "spirit_not_letter_ground_truth": {
                # "Hi Sarah," has the first name but adds "Hi" which the dim
                # explicitly disallows. The grader is told "grade what the
                # dimension literally says." Strict reading: NOT_MET.
                "first_name_only_greeting": "NOT_MET",
            },
        },
    },

    {
        "name": "borderline_letter_not_spirit",
        "category": "borderline",
        "rubric": {
            "writing_type": "Slack engineering updates",
            "rubric": [
                {
                    "name": "Register",
                    "dimensions": [
                        {"id": "lowercase_throughout", "label": "All sentence-initial words are lowercase (proper nouns may be capitalized)"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            # Technically satisfies the rule (lowercase sentence-initial) but
            # reads stilted. Grader should still call MET because the prompt
            # says "grade what the dimension literally says."
            "letter_not_spirit": (
                "quick update. the migration is ~2 weeks behind. "
                "root cause: a race condition in the batch worker. "
                "fix is in review. new target: december 2."
            ),
            "letter_not_spirit_ground_truth": {
                "lowercase_throughout": "MET",
            },
        },
    },

    # -----------------------------------------------------------------------
    # ambiguous_dim_wording: dim is vague, should yield low confidence
    # -----------------------------------------------------------------------
    {
        "name": "ambiguous_dim_wording",
        "category": "ambiguous",
        "rubric": {
            "writing_type": "cold outreach emails",
            "rubric": [
                {
                    "name": "Tone",
                    "dimensions": [
                        # Deliberately vague — "warm" and "professional" are
                        # both fuzzy. A good grader should assign low confidence.
                        {"id": "warm_professional", "label": "Tone is warm yet professional"},
                        # Also vague — "concrete" is fuzzy.
                        {"id": "concrete_language", "label": "Uses concrete language"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            "ambiguous": (
                "Sarah,\n\n"
                "You recently backed CompetitorX in voluntary carbon markets. We're building "
                "the regulated version — $2M ARR, 3 utility contracts signed last quarter.\n\n"
                "10 min next Tuesday?\n\n"
                "— Alex"
            ),
            # We don't have strong ground truth on MET/NOT_MET for vague dims.
            # The important test here is CONFIDENCE -- these should be low or
            # medium, not high. We encode ground truth as MET (the draft does
            # seem warm+professional and use concrete language) but the eval
            # separately inspects confidence.
            "ambiguous_ground_truth": {
                "warm_professional": "MET",
                "concrete_language": "MET",
            },
            # Additional check: confidence on these ambiguous dims should NOT
            # be "high" — the grader prompt says "low confidence when wording
            # is open to interpretation." We check this in the harness.
            "expected_confidence_not_high": ["warm_professional", "concrete_language"],
        },
    },

    # -----------------------------------------------------------------------
    # adversarial_format: draft tries to game the grader by using dim keywords
    # -----------------------------------------------------------------------
    {
        "name": "adversarial_keyword_match",
        "category": "adversarial",
        "rubric": {
            "writing_type": "ML blog post intros",
            "rubric": [
                {
                    "name": "Opening",
                    "dimensions": [
                        {"id": "mechanism_first", "label": "First sentence names a specific phenomenon, pathology, or mechanism (not an abstract topic intro)"},
                        {"id": "no_marketing_superlatives", "label": "Does not use marketing superlatives like 'stunning', 'amazing', 'revolutionary', 'cutting-edge', 'state-of-the-art'"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            "adversarial": (
                "In this post we'll discuss mechanisms, pathologies, and phenomena in "
                "diffusion models — a topic that has seen stunning and cutting-edge progress "
                "in recent years. We hope you find this overview useful."
            ),
            "adversarial_ground_truth": {
                # The first sentence uses the literal words "mechanisms",
                # "pathologies", "phenomena" but does NOT name a specific
                # mechanism/phenomenon — it announces a topic category.
                "mechanism_first": "NOT_MET",
                # Literal keyword trigger: "stunning" and "cutting-edge" are
                # present, so this should clearly fail.
                "no_marketing_superlatives": "NOT_MET",
            },
        },
    },

    # -----------------------------------------------------------------------
    # negation_dims: dims phrased as "does NOT do X" — common failure mode
    # for LLM graders because they can flip the logic.
    # -----------------------------------------------------------------------
    {
        "name": "negation_handling",
        "category": "negation",
        "rubric": {
            "writing_type": "professional email",
            "rubric": [
                {
                    "name": "Avoidance",
                    "dimensions": [
                        {"id": "no_apology_for_delay", "label": "Does not apologize for a delayed reply"},
                        {"id": "no_hope_well", "label": "Does not include 'I hope this email finds you well' or similar"},
                        {"id": "no_please_let_me_know", "label": "Does not end with 'please let me know if you have any questions' or similar boilerplate"},
                    ],
                    "priority": 1,
                },
            ],
        },
        "drafts": {
            # Deliberately violates all three "do NOT" rules
            "violates_all_negations": (
                "Hi Maria,\n\n"
                "Sorry for the delayed reply — and I hope this email finds you well!\n\n"
                "Attached is the report you asked for. Let me know if you have any questions.\n\n"
                "Best,\n"
                "Alex"
            ),
            "violates_all_negations_ground_truth": {
                "no_apology_for_delay": "NOT_MET",  # "Sorry for the delayed reply"
                "no_hope_well": "NOT_MET",           # "I hope this email finds you well"
                "no_please_let_me_know": "NOT_MET",  # "Let me know if you have any questions"
            },
            # Complementary draft that satisfies all three
            "satisfies_all_negations": (
                "Hi Maria,\n\n"
                "Attached is the report you requested. I've included context on assumptions in the intro.\n\n"
                "Happy to walk through anything specific on a call.\n\n"
                "— Alex"
            ),
            "satisfies_all_negations_ground_truth": {
                "no_apology_for_delay": "MET",
                "no_hope_well": "MET",
                "no_please_let_me_know": "MET",
            },
        },
    },
]


def get_fixture(name: str) -> dict:
    for f in FIXTURES + EDGE_CASE_FIXTURES:
        if f["name"] == name:
            return f
    raise KeyError(name)
