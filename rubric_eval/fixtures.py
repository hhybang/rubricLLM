"""Hand-authored conversation fixtures for rubric inference eval.

Each fixture is a tuple of (name, genre, messages) where messages is a list
of {role, content} dicts mirroring st.session_state.messages. Conversations
are intentionally short (5-9 turns) but carry enough signal that a careful
reader could identify 2-4 specific user preferences that distinguish this
user from a generic competent writer in the same genre.

The conversations deliberately mix STRONG signals (explicit feedback,
rejections, exact phrasings) with WEAK signals (vague likes, silence) so
the eval can measure whether the inference prompt conservatively sticks
to the strong signals or pads the rubric with near-generic criteria."""

from __future__ import annotations

FIXTURES: list[dict] = [
    {
        "name": "cold_email_founder",
        "genre": "cold outreach email (founder to investor)",
        "messages": [
            {"role": "user", "content": "I need to write a cold email to Sarah Chen at Ribbit Capital. I'm raising a seed round for my climate tech startup."},
            {"role": "assistant", "content": "Here's a first draft:\n\nSubject: Seed opportunity — [Company] tackling grid-scale carbon removal\n\nDear Ms. Chen,\n\nI hope this email finds you well. I'm reaching out to introduce [Company], a seed-stage climate tech startup focused on grid-scale carbon removal. We are raising our seed round and I believe Ribbit's thesis aligns well with our mission.\n\nWe would welcome the opportunity to share more about our technology and traction. Please let me know if you would be available for a 30-minute call next week.\n\nBest regards,\n[Your name]"},
            {"role": "user", "content": "No way. This sounds like every other cold email. 'Hope this email finds you well' — delete that. Also, don't call her Ms. Chen, just Sarah. And drop 'I believe Ribbit's thesis aligns well with our mission' — it's vague and she'll tune out."},
            {"role": "assistant", "content": "Here's a sharper version:\n\nSubject: Seed — grid-scale carbon removal, $2M ARR\n\nSarah,\n\nYou recently backed [CompetitorX] in voluntary carbon markets. We're tackling the compliance side of the same thesis — grid-scale removal pre-sold to utilities. $2M ARR, 3 utility contracts signed last quarter.\n\n10 min next Tuesday?\n\n[Your name]"},
            {"role": "user", "content": "Better. I like that it opens with something specific to her. But 'tackling the compliance side' is jargon — a founder should sound like a human, not a banker. Rewrite that line in plain English."},
            {"role": "assistant", "content": "Updated:\n\nSarah,\n\nYou recently backed [CompetitorX] in voluntary carbon markets. We're building the regulated version — utilities have to remove carbon by 2030, and we sell them the removal. $2M ARR, 3 utility contracts signed last quarter.\n\n10 min next Tuesday?\n\n[Your name]"},
            {"role": "user", "content": "Perfect. That's the one. One rule for future drafts: always lead with a specific hook about the recipient, and always include one hard number in the first three lines. No hedging language."},
        ],
    },
    {
        "name": "blog_intro_ml_researcher",
        "genre": "technical blog post introduction (ML research)",
        "messages": [
            {"role": "user", "content": "Writing a blog post intro for my new paper on diffusion models. The audience is ML practitioners, not the general public."},
            {"role": "assistant", "content": "Here's a draft intro:\n\nDiffusion models have revolutionized generative AI, producing stunning images that rival those of professional artists. In this post, we dive deep into our latest research, which makes diffusion models faster and more accurate than ever before. You'll be amazed at what we discovered."},
            {"role": "user", "content": "This is marketing fluff. The audience already knows diffusion models matter — don't explain them. Start by naming the specific mechanism we improved, not with 'revolutionized generative AI'. Also, never say 'you'll be amazed'. That's a tell for a blog written by marketing, not a researcher."},
            {"role": "assistant", "content": "Rewritten:\n\nClassifier-free guidance has a known pathology: at high guidance scales, samples become saturated and lose diversity. We trace this to the implicit temperature drop in the conditional score, and show a three-line fix that restores diversity without losing fidelity. Tables at the end."},
            {"role": "user", "content": "Much better. This reads like it was written by someone who actually did the work. One more thing: the 'tables at the end' is good — it tells the reader what payoff to expect. Keep that kind of explicit sign-posting. But also, lead with the phenomenon before the fix: 'Classifier-free guidance has a known pathology' is the phenomenon, 'we trace this to...' is the explanation. That's the right order."},
            {"role": "assistant", "content": "Got it. Want me to draft a similar intro for a follow-up post on DDIM sampling?"},
            {"role": "user", "content": "Yes. Same structure: phenomenon first, explanation second, and tell the reader what tables or figures to expect."},
        ],
    },
    {
        "name": "slack_update_eng_manager",
        "genre": "Slack status update (engineering manager to team)",
        "messages": [
            {"role": "user", "content": "Write a Slack update for my team about the migration we've been running. It's been slower than planned."},
            {"role": "assistant", "content": "Team,\n\nI wanted to give you a comprehensive update on the status of our ongoing migration project. As you are aware, we have encountered a number of unforeseen challenges that have impacted our original timeline. I want to assure everyone that leadership remains fully committed to seeing this through, and we are actively working on solutions.\n\nPlease reach out with any questions or concerns.\n\nThanks,\n[Your name]"},
            {"role": "user", "content": "Too corporate and I'm not delivering this at a town hall. It's Slack. Cut everything that sounds like a press release. Also don't call it 'the migration project' — everyone on the team knows what the migration is."},
            {"role": "assistant", "content": "Shorter:\n\nquick update on the migration —\n\nwe're ~2 weeks behind the original plan. root cause: the feature-flag rollout hit a race condition in the batch worker that we didn't catch in staging. fix is in review, should land by friday.\n\nnew target ship date: nov 14. ping me with questions."},
            {"role": "user", "content": "That's what I wanted. Lowercase, specific, names the actual technical cause. One tweak — instead of 'ping me with questions' at the end, end on the new date. That's the one piece of info the team actually needs to take away."},
            {"role": "assistant", "content": "Updated:\n\nquick update on the migration —\n\nwe're ~2 weeks behind the original plan. root cause: the feature-flag rollout hit a race condition in the batch worker that we didn't catch in staging. fix is in review, should land by friday.\n\nnew target ship date: nov 14."},
            {"role": "user", "content": "Yes. Rule for these: lowercase, name the technical root cause, end on the number or date that matters."},
        ],
    },
    {
        "name": "paper_abstract_physicist",
        "genre": "physics paper abstract",
        "messages": [
            {"role": "user", "content": "Draft an abstract for my paper on anomalous thermal transport in twisted bilayer graphene."},
            {"role": "assistant", "content": "We report experimental measurements of thermal transport in twisted bilayer graphene at magic-angle configurations. Our results demonstrate anomalous behavior that cannot be explained by conventional phonon transport models. These findings have significant implications for the design of next-generation electronic devices."},
            {"role": "user", "content": "Two problems. First, 'significant implications for next-generation electronic devices' is the kind of vague hand-wavy closer every abstract has — physicists skip it. Cut it. Second, you need to name the actual quantitative result in the abstract. If the reader doesn't know what number we measured, the abstract failed."},
            {"role": "assistant", "content": "Revised:\n\nWe measure the in-plane thermal conductivity κ of twisted bilayer graphene at twist angles 1.05°–1.15°. At the magic angle (1.08°), κ drops by a factor of 4.3 relative to monolayer graphene at T = 100 K, well beyond phonon-boundary-scattering predictions. The suppression tracks the flat-band bandwidth extracted from separate ARPES measurements, suggesting a coupling between electronic correlations and heat transport."},
            {"role": "user", "content": "Much better. You named the number (4.3x suppression, 100 K, 1.08°) and the mechanism claim (coupling to electronic correlations). Good structure: measurement, quantitative result, mechanism. Keep that order for every abstract. And you can drop 'We measure' — passive voice is standard in physics abstracts."},
            {"role": "assistant", "content": "Updated first line: 'The in-plane thermal conductivity κ of twisted bilayer graphene is measured at twist angles 1.05°–1.15°.'"},
            {"role": "user", "content": "Perfect."},
        ],
    },
    {
        "name": "condolence_note_manager",
        "genre": "condolence note (manager to direct report)",
        "messages": [
            {"role": "user", "content": "One of my direct reports' father passed away last week. I want to send her a short note. She's been at the company 4 years, we have a good working relationship but we're not close friends."},
            {"role": "assistant", "content": "Dear [Name],\n\nI was deeply saddened to hear of your father's passing. Please accept my sincere condolences during this difficult time. My thoughts are with you and your family.\n\nWe are here to support you in any way we can as you navigate this loss. Please do not hesitate to reach out if there is anything you need. Take all the time you need.\n\nWith deepest sympathy,\n[Your name]"},
            {"role": "user", "content": "This sounds like a Hallmark card. 'Deeply saddened', 'sincere condolences', 'deepest sympathy' — every phrase is a cliche. I want to say something real without being over-familiar. Try again — shorter, no boilerplate phrases, and don't end with a generic offer to help."},
            {"role": "assistant", "content": "[Name] —\n\nI heard about your father. I'm so sorry. Take whatever time you need — I'll cover your 1:1s and can reschedule the review. If it helps to be away from work for a while, we'll make that work.\n\n[Your name]"},
            {"role": "user", "content": "Closer. But 'I'll cover your 1:1s and can reschedule the review' is too transactional — it reads like I'm managing her calendar. Cut that sentence. Replace it with one concrete way I'm already handling work on her behalf so she doesn't have to think about it."},
            {"role": "assistant", "content": "[Name] —\n\nI heard about your father. I'm so sorry. I've already let the team know you'll be out indefinitely, and nothing needs a decision from you right now. Take whatever time you need.\n\n[Your name]"},
            {"role": "user", "content": "Yes. 'Nothing needs a decision from you right now' is exactly the reassurance I want to give. Rule going forward: no boilerplate condolence phrases, and whenever I'm offering support, state a concrete thing I've already done rather than a generic offer."},
        ],
    },
]



# ---------------------------------------------------------------------------
# WEAK-SIGNAL FIXTURES
# ---------------------------------------------------------------------------
# These fixtures simulate realistic "thin signal" conversations where the user
# mostly accepts drafts, gives short vague feedback ("looks good", "tweak this
# a bit"), and rarely articulates rules. If the inference prompt pads rubrics
# with generic platitudes when signal is thin, we should see the sniff-test
# pass rate drop sharply on these vs the strong-signal fixtures above.

WEAK_FIXTURES: list[dict] = [
    {
        "name": "weak_linkedin_post",
        "genre": "LinkedIn post announcing a job change",
        "messages": [
            {"role": "user", "content": "Help me write a LinkedIn post announcing I'm leaving my current role and joining a new startup as head of product."},
            {"role": "assistant", "content": "Here's a draft:\n\nAfter four incredible years at [Current Company], I'm excited to share that I'm joining [New Startup] as Head of Product. Huge thanks to everyone I've had the privilege of working with — I've learned so much from this team. Onwards to the next chapter. Can't wait to share what we're building."},
            {"role": "user", "content": "Looks good. Maybe tweak the ending a bit?"},
            {"role": "assistant", "content": "Updated ending:\n\n...Onwards to the next chapter. If you're interested in product thinking at the earliest stages, I'd love to hear from you."},
            {"role": "user", "content": "Yeah that works better."},
            {"role": "assistant", "content": "Want me to polish anything else, or is this ready to post?"},
            {"role": "user", "content": "I think it's ready."},
        ],
    },
    {
        "name": "weak_meeting_recap",
        "genre": "internal meeting recap email",
        "messages": [
            {"role": "user", "content": "Write a recap email after a planning meeting with the design team. We discussed Q1 priorities, decided to focus on the onboarding redesign, and agreed to reconvene in two weeks."},
            {"role": "assistant", "content": "Team,\n\nQuick recap of today's planning discussion:\n\n- Q1 priorities: reviewed the backlog and aligned on top focus areas.\n- Decision: onboarding redesign is the priority for this quarter.\n- Next step: reconvene in two weeks to review initial scoping.\n\nLet me know if I missed anything.\n\nThanks,\n[Your name]"},
            {"role": "user", "content": "Good."},
            {"role": "assistant", "content": "Want to add a follow-up task list or is this enough?"},
            {"role": "user", "content": "This is enough."},
        ],
    },
    {
        "name": "weak_product_blurb",
        "genre": "short product description for a landing page",
        "messages": [
            {"role": "user", "content": "I need a short product blurb for our landing page. The product is an AI tool that auto-generates test cases from code."},
            {"role": "assistant", "content": "Draft:\n\nAutomate your test coverage. [Product] generates test cases directly from your code — so your engineers can focus on shipping, not writing boilerplate tests. Integrates with your existing CI pipeline. Start generating tests in minutes."},
            {"role": "user", "content": "Nice, like it."},
            {"role": "assistant", "content": "Want me to try a variant that leads with a specific result or customer outcome?"},
            {"role": "user", "content": "Sure, try one."},
            {"role": "assistant", "content": "Alternative:\n\nShip faster with 90% less time spent writing tests. [Product] reads your code and generates test cases that catch real bugs — integrated with your CI, no config required. Used by teams at [Customer], [Customer], and [Customer]."},
            {"role": "user", "content": "Hmm, I like parts of both. Can you just pick one?"},
            {"role": "assistant", "content": "Going with the second variant — the concrete '90% less time' opener is stronger than 'automate your test coverage'. Final version above."},
            {"role": "user", "content": "Cool."},
        ],
    },
    {
        "name": "weak_birthday_card",
        "genre": "birthday card message for a coworker",
        "messages": [
            {"role": "user", "content": "Need a short birthday note for a coworker I've worked with for two years but don't know super well outside of work."},
            {"role": "assistant", "content": "Here's a draft:\n\nHappy birthday! Hope you have a great day and a fantastic year ahead. It's been a pleasure working with you.\n\n[Your name]"},
            {"role": "user", "content": "Sounds fine."},
            {"role": "assistant", "content": "Want me to add anything personal or keep it brief?"},
            {"role": "user", "content": "Brief is fine."},
        ],
    },
    {
        "name": "weak_bug_report",
        "genre": "internal bug report in a ticket tracker",
        "messages": [
            {"role": "user", "content": "Write a bug report. Users are seeing the checkout button disappear on mobile after they apply a coupon code. Happens on Safari iOS 17+."},
            {"role": "assistant", "content": "Draft:\n\n**Title:** Checkout button disappears after coupon application on iOS Safari\n\n**Summary:** Users on iOS Safari (17+) report that the checkout button vanishes after they apply a coupon code at the cart step.\n\n**Steps to reproduce:**\n1. Open the cart on iOS Safari 17+\n2. Apply a valid coupon code\n3. Observe that the checkout button is no longer visible\n\n**Expected:** Checkout button remains visible after coupon applies.\n**Actual:** Checkout button disappears; user cannot proceed.\n\n**Severity:** High — blocks checkout for affected users.\n\n**Env:** iOS Safari 17+"},
            {"role": "user", "content": "Good."},
            {"role": "assistant", "content": "Want to add a screenshot placeholder or any tags like priority?"},
            {"role": "user", "content": "No, this is enough."},
        ],
    },
]


def get_fixture(name: str) -> dict:
    for f in FIXTURES + WEAK_FIXTURES:
        if f["name"] == name:
            return f
    raise KeyError(name)
