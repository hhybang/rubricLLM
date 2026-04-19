# Grader agreement report (Opus vs Sonnet 4.6)

## Summary

- **Overall dimension agreement:** 96.8%
- **Cohen's κ (overall):** 0.9364
- **Within-model consistency (Opus):** 98.6%
- **Within-model consistency (Sonnet):** 98.1%
- **Dimensions compared:** 346
- **Disagreements:** 11
- **Avg latency Opus:** 22288 ms
- **Avg latency Sonnet:** 22520 ms
- **Est. cost Opus:** $9.9241
- **Est. cost Sonnet:** $1.8518

*Costs use placeholder $/MTok defaults; set `PRICE_*` env vars to match your billing.*

## Per-criterion agreement

| Criterion | Agreement | Cohen's κ |
|-----------|-----------|-----------|
| Transparent intent | 100.0% | 1.0000 |
| Traction as narrative momentum | 100.0% | 1.0000 |
| Double-duty personalization | 100.0% | 1.0000 |
| Peer-to-peer framing | 100.0% | 1.0000 |
| Signal-dense subject line | 100.0% | 1.0000 |
| Unflinching honesty about wrong turns | 100.0% | 1.0000 |
| Investigation pacing with delayed reveals | 100.0% | 1.0000 |
| Concrete, non-preachy takeaways | 100.0% | 1.0000 |
| Code snippets as narrative anchors | 100.0% | 1.0000 |
| Recipient-Calibrated Register | 100.0% | 1.0000 |
| Anti-cliché discipline | 100.0% | 1.0000 |
| Composed-but-human tone | 100.0% | 1.0000 |
| Engagement-driven closing | 100.0% | 1.0000 |
| Experience-grounded opening | 100.0% | 1.0000 |
| Critical decisions front-loaded in summaries | 100.0% | 1.0000 |
| Concrete debugging artifacts in operational sections | 100.0% | 1.0000 |
| Tradeoff transparency with escape hatches | 100.0% | 1.0000 |
| Bottom-Line-Up-Front Summary | 93.8% | 0.8750 |
| Calibrated Risk Framing | 93.8% | 0.8621 |
| Contextualized proof points | 93.8% | 0.8710 |
| Code examples as primary entry points | 93.8% | 0.8750 |
| Immersive scene-setting with specific details | 91.7% | 0.8235 |
| Phone-Scannable Brevity | 81.2% | 0.5385 |
| Scan-optimized formatting | 81.2% | 0.6364 |

## Per-dimension agreement (rolled up by name + id)

| Dimension | Agreement | n |
|-----------|-----------|---|
| Transparent intent :: no_false_softeners | 100.0% | 3 |
| Transparent intent :: explicit_ask | 100.0% | 3 |
| Transparent intent :: honest_framing | 100.0% | 3 |
| Transparent intent :: no_trick_feel | 100.0% | 3 |
| Traction as narrative momentum :: temporal_arc | 100.0% | 3 |
| Traction as narrative momentum :: no_bullet_stats | 100.0% | 3 |
| Traction as narrative momentum :: implied_trajectory | 100.0% | 3 |
| Traction as narrative momentum :: organic_demand_signal | 100.0% | 3 |
| Double-duty personalization :: specific_source | 100.0% | 3 |
| Double-duty personalization :: thesis_connection | 100.0% | 3 |
| Double-duty personalization :: proof_embedded | 100.0% | 3 |
| Double-duty personalization :: natural_integration | 100.0% | 3 |
| Peer-to-peer framing :: no_deference_markers | 100.0% | 3 |
| Peer-to-peer framing :: mutual_value | 100.0% | 3 |
| Peer-to-peer framing :: direct_scheduling | 100.0% | 3 |
| Signal-dense subject line :: credibility_upfront | 100.0% | 3 |
| Signal-dense subject line :: ask_visible | 100.0% | 3 |
| Signal-dense subject line :: no_cleverness_tricks | 100.0% | 3 |
| Unflinching honesty about wrong turns :: wrong_leads_shown | 100.0% | 4 |
| Unflinching honesty about wrong turns :: personal_accountability | 100.0% | 4 |
| Unflinching honesty about wrong turns :: time_cost_visible | 100.0% | 4 |
| Unflinching honesty about wrong turns :: self_deprecating_not_flagellating | 100.0% | 4 |
| Investigation pacing with delayed reveals :: confusion_before_clarity | 100.0% | 4 |
| Investigation pacing with delayed reveals :: progressive_narrowing | 100.0% | 4 |
| Investigation pacing with delayed reveals :: reveal_earned | 100.0% | 4 |
| Investigation pacing with delayed reveals :: emotional_beats_present | 100.0% | 4 |
| Concrete, non-preachy takeaways :: actionable_practice | 100.0% | 4 |
| Concrete, non-preachy takeaways :: generic_advice_rejected | 100.0% | 4 |
| Concrete, non-preachy takeaways :: grounded_in_story | 100.0% | 4 |
| Concrete, non-preachy takeaways :: peer_tone | 100.0% | 4 |
| Code snippets as narrative anchors :: minimal_snippet_count | 100.0% | 4 |
| Code snippets as narrative anchors :: annotated_key_lines | 100.0% | 4 |
| Code snippets as narrative anchors :: before_after_contrast | 100.0% | 4 |
| Code snippets as narrative anchors :: prose_frames_code | 100.0% | 4 |
| Immersive scene-setting with specific details :: sensory_anchors | 100.0% | 4 |
| Immersive scene-setting with specific details :: named_tools | 100.0% | 4 |
| Phone-Scannable Brevity :: bullets_only | 100.0% | 4 |
| Phone-Scannable Brevity :: one_two_line_bullets | 100.0% | 4 |
| Bottom-Line-Up-Front Summary :: opens_with_summary | 100.0% | 4 |
| Bottom-Line-Up-Front Summary :: covers_key_status | 100.0% | 4 |
| Bottom-Line-Up-Front Summary :: signals_escalation_need | 100.0% | 4 |
| Calibrated Risk Framing :: timeline_impact_quantified | 100.0% | 4 |
| Calibrated Risk Framing :: resolution_timeline | 100.0% | 4 |
| Calibrated Risk Framing :: escalation_signal | 100.0% | 4 |
| Recipient-Calibrated Register :: no_casual_shorthand | 100.0% | 4 |
| Recipient-Calibrated Register :: direct_not_stiff | 100.0% | 4 |
| Recipient-Calibrated Register :: recipient_modeled | 100.0% | 4 |
| Anti-cliché discipline :: no_motivational_filler | 100.0% | 4 |
| Anti-cliché discipline :: no_generic_self_description | 100.0% | 4 |
| Anti-cliché discipline :: claims_backed_by_evidence | 100.0% | 4 |
| Anti-cliché discipline :: unique_to_writer | 100.0% | 4 |
| Composed-but-human tone :: no_overly_casual_phrasing | 100.0% | 4 |
| Composed-but-human tone :: no_stiff_formality | 100.0% | 4 |
| Composed-but-human tone :: natural_sentence_rhythm | 100.0% | 4 |
| Composed-but-human tone :: confidence_without_arrogance | 100.0% | 4 |
| Contextualized proof points :: specific_metrics | 100.0% | 4 |
| Contextualized proof points :: method_visible | 100.0% | 4 |
| Contextualized proof points :: constraints_shown | 100.0% | 4 |
| Engagement-driven closing :: specific_reference | 100.0% | 4 |
| Engagement-driven closing :: curiosity_hook | 100.0% | 4 |
| Engagement-driven closing :: not_gimmicky | 100.0% | 4 |
| Engagement-driven closing :: avoids_generic_availability | 100.0% | 4 |
| Experience-grounded opening :: opens_with_experience | 100.0% | 4 |
| Experience-grounded opening :: doesnt_try_too_hard | 100.0% | 4 |
| Experience-grounded opening :: establishes_credibility_early | 100.0% | 4 |
| Critical decisions front-loaded in summaries :: decisions_in_tldr | 100.0% | 4 |
| Critical decisions front-loaded in summaries :: non_obvious_defaults_highlighted | 100.0% | 4 |
| Critical decisions front-loaded in summaries :: non_goals_stated | 100.0% | 4 |
| Critical decisions front-loaded in summaries :: decisions_before_implementation | 100.0% | 4 |
| Code examples as primary entry points :: realistic_production_patterns | 100.0% | 4 |
| Code examples as primary entry points :: error_handling_shown | 100.0% | 4 |
| Code examples as primary entry points :: inline_annotations | 100.0% | 4 |
| Concrete debugging artifacts in operational sections :: exact_error_output | 100.0% | 4 |
| Concrete debugging artifacts in operational sections :: caller_perspective | 100.0% | 4 |
| Concrete debugging artifacts in operational sections :: observability_signals_named | 100.0% | 4 |
| Concrete debugging artifacts in operational sections :: auth_rejection_examples | 100.0% | 4 |
| Tradeoff transparency with escape hatches :: tradeoff_flagged | 100.0% | 4 |
| Tradeoff transparency with escape hatches :: affected_parties_named | 100.0% | 4 |
| Tradeoff transparency with escape hatches :: alternative_provided | 100.0% | 4 |
| Tradeoff transparency with escape hatches :: visually_distinct_callout | 100.0% | 4 |
| Scan-optimized formatting :: bold_key_terms | 100.0% | 4 |
| Scan-optimized formatting :: admonition_boxes_for_warnings | 100.0% | 4 |
| Immersive scene-setting with specific details :: scene_not_summary | 75.0% | 4 |
| Phone-Scannable Brevity :: minimal_sections | 75.0% | 4 |
| Bottom-Line-Up-Front Summary :: plain_label | 75.0% | 4 |
| Calibrated Risk Framing :: neutral_language | 75.0% | 4 |
| Contextualized proof points :: narrative_not_list | 75.0% | 4 |
| Code examples as primary entry points :: caller_side_examples | 75.0% | 4 |
| Scan-optimized formatting :: sections_self_contained | 75.0% | 4 |
| Phone-Scannable Brevity :: no_redundancy | 50.0% | 4 |
| Scan-optimized formatting :: tables_for_comparisons | 50.0% | 4 |

## Disagreements (with classification)

### blog_post_01 / blog_post_01_targeted_fail — Immersive scene-setting with specific details / `scene_not_summary`
- **Label:** Important moments are dramatized as scenes with action, not summarized in retrospective narration
- **Opus (majority):** MET — Key moments are dramatized as scenes with action: "I was halfway through writing a Jira ticket... when the PagerDuty notification lit up my phone" and "Priya finally asked in standup what I'd found, and when I described my Redis theory out loud, I heard how thin it sounded."
- **Sonnet (majority):** NOT_MET — Key investigative moments like the standup exchange with Priya and the Wednesday grep session are narrated in retrospective summary rather than dramatized as present-tense scenes with action and dialogue.
- **Classification:** `judgment_call` — The draft contains scene-like elements (present-tense action, dialogue) but also retrospective narration, making it defensible to argue either that key moments are sufficiently dramatized or that they remain too summarized.

<details><summary>Draft excerpt</summary>

```
# The Ghost in the Heap: Hunting a Memory Leak That Wasn't Where I Thought

## The Alert Nobody Wanted at 4:47 PM on a Friday

I was halfway through writing a Jira ticket — the kind of meaningless ticket you write when you're trying to look productive before the weekend — when the PagerDuty notification lit up my phone. Our order-processing microservice, the one we affectionately called `order-muncher`, had tripped its memory threshold. Again.

The Datadog dashboard told a familiar story: a sawtooth pattern of memory usage that had been climbing all week, each peak a little higher than the last, each garbage collection trough a little less effective. RSS was sitting at 1.2 GB on a container with a 1.5 GB limit. We had maybe a few hours before Kubernetes started killing pods.

I Slacked my team: "order-muncher memory thing is back. I'll take a look." Three thumbs-up reacts. Nobody volunteered to help. I don't blame them.

Here's the thing I need to admit upfront: this was the *third time* this alert had fired in two weeks, and the previous two times I'd "fixed" it by just bumping the container memory limit and restarting the pods. I told myself I was buying time to investigate properly. I was actually just hoping it would go away.

It did not go away.

## The Obvious Suspect (That I Spent a Day and a Half On)

Monday morning, I decided to actually dig in. My first theory was confident and wrong: it had to be the Redis connection pool.

We'd recently upgraded our `ioredis` client and changed the connection pool configuration. The timing lined up — sort of. If you squinted. I spent the entire morning reading ioredis source code on GitHub, convinced I'd find that the new version was holding onto connection objects differently. I even found a closed issue from 2021 that described something vaguely similar, and I felt that little dopamine hit of *I've found it*.

I wrote a test harness. I simulated thousands of connections. I monitored the heap. The connections were perfe
```
</details>

### manager_update_01 / manager_update_01_good — Phone-Scannable Brevity / `no_redundancy`
- **Label:** Each piece of information appears in exactly one section — no overlap between sections
- **Opus (majority):** MET — The analytics API issue appears in 'In Progress' with details and timeline impact, and 'Next Week' only references the follow-up action ('Finalize analytics API approach'), with no duplicated information.
- **Sonnet (majority):** NOT_MET — The analytics dashboard timeline shift (~1 week, July 10 to ~July 17) is mentioned in both the opening summary line and again in the 'In Progress' section bullet.
- **Classification:** `judgment_call` — Whether a brief summary mention in the opening and a detailed elaboration in a section counts as 'overlap' or acceptable layering is a defensible interpretive choice about what 'exactly one section' means for executive summaries.

<details><summary>Draft excerpt</summary>

```
Subject: Weekly Status — June 12

Hi Maria,

Overall: Auth migration is on track for the June 30 milestone; we're working through one API dependency that may shift the analytics dashboard delivery by ~1 week. No action needed from you right now.

**Progress**
- Auth migration: core service cutover complete, running parallel validation through Friday
- User onboarding flow: QA signed off, deploying to staging Monday
- Performance benchmarks: latency targets met across all three endpoints

**In Progress**
- Resolving a rate-limit constraint on the third-party analytics API; evaluating two workarounds with the vendor — expect a clear path by Wednesday
- If the workaround requires schema changes, analytics dashboard delivery shifts from July 10 to ~July 17; will confirm by end of week
- This is a routine integration issue — team is handling it, no escalation needed

**Next Week**
- Complete auth parallel validation and begin production rollout
- Finalize analytics API approach and update the dashboard timeline if needed
- Kick off scoping for the Q3 notification service work

Let me know if you want to dig into anything.

Jake
```
</details>

### manager_update_01 / manager_update_01_good — Phone-Scannable Brevity / `minimal_sections`
- **Label:** Sections are consolidated to the fewest needed; no section exists that could be folded into another
- **Opus (majority):** MET — Only three sections are used (Progress, In Progress, Next Week), which are distinct in purpose and none could be reasonably folded into another.
- **Sonnet (majority):** NOT_MET — The 'Progress' and 'In Progress' sections both cover current work and could reasonably be consolidated into a single section, as the distinction between completed steps and ongoing steps is thin.
- **Classification:** `judgment_call` — Both interpretations are defensible: 'Progress' (completed items) and 'In Progress' (ongoing/blocked items) serve meaningfully distinct purposes, but they also both describe current-week work and could reasonably be merged into one section.

<details><summary>Draft excerpt</summary>

```
Subject: Weekly Status — June 12

Hi Maria,

Overall: Auth migration is on track for the June 30 milestone; we're working through one API dependency that may shift the analytics dashboard delivery by ~1 week. No action needed from you right now.

**Progress**
- Auth migration: core service cutover complete, running parallel validation through Friday
- User onboarding flow: QA signed off, deploying to staging Monday
- Performance benchmarks: latency targets met across all three endpoints

**In Progress**
- Resolving a rate-limit constraint on the third-party analytics API; evaluating two workarounds with the vendor — expect a clear path by Wednesday
- If the workaround requires schema changes, analytics dashboard delivery shifts from July 10 to ~July 17; will confirm by end of week
- This is a routine integration issue — team is handling it, no escalation needed

**Next Week**
- Complete auth parallel validation and begin production rollout
- Finalize analytics API approach and update the dashboard timeline if needed
- Kick off scoping for the Q3 notification service work

Let me know if you want to dig into anything.

Jake
```
</details>

### manager_update_01 / manager_update_01_targeted_fail — Phone-Scannable Brevity / `no_redundancy`
- **Label:** Each piece of information appears in exactly one section — no overlap between sections
- **Opus (majority):** NOT_MET — The auth service dependency is mentioned in the Summary, discussed at length in Blockers, and referenced again in Next Steps ('Sync with auth service team Wednesday'), with overlapping information about the situation.
- **Sonnet (majority):** MET — No information appears to be repeated across sections; the auth service issue is mentioned in the Summary and detailed only in Blockers, with the Wednesday sync appearing only in Next Steps.
- **Classification:** `judgment_call` — Whether a brief summary mention plus detailed blocker discussion plus a related next-step action constitutes 'overlap' versus appropriate cross-section referencing of the same topic is a defensible interpretive call either way.

<details><summary>Draft excerpt</summary>

```
Subject: Weekly Status — May 19

Hi Dana,

**Summary:** Platform migration is on track for the June 6 milestone, and the auth service dependency we flagged last week is being handled — no action needed from you.

**Progress**
- Migration pipeline: 4 of 5 modules complete; final module (billing reconciliation) started Monday
- Load testing on modules 1–3 passed with <2% error rate, clearing our go/no-go threshold
- Design review for the Q3 dashboard signed off by product Thursday

**Blockers**
- Auth service team pushed their API freeze back, which is creating some complications for our integration timeline. We're not entirely sure yet how long the delay will be, but the team is looking into workarounds and we're hoping to have a better picture soon. It's possible this could shift things, though it might not — we'll keep monitoring the situation and circle back when we know more.

**Next Steps**
- Complete billing reconciliation module by Friday
- Run full end-to-end regression suite early next week
- Sync with auth service team Wednesday to confirm revised API freeze date

Let me know if you want to dig into anything.

Best,
Jordan

<!-- fail: timeline_impact_quantified, resolution_timeline -->
```
</details>

### manager_update_01 / manager_update_01_targeted_fail — Calibrated Risk Framing / `neutral_language`
- **Label:** Blockers and challenges use neutral language (e.g., 'resolving,' 'handling') rather than alarm words ('setback,' 'failure,' 'crisis')
- **Opus (majority):** MET — The summary uses 'being handled' and the blockers section uses 'creating some complications' and 'looking into workarounds' — neutral language without alarm words.
- **Sonnet (majority):** NOT_MET — 'creating some complications' and 'we're not entirely sure' and 'it's possible this could shift things, though it might not' use vague, hedging language rather than neutral, matter-of-fact framing.
- **Classification:** `dimension_ambiguity` — The dimension only specifies avoiding 'alarm words' but doesn't address whether vague hedging language counts as a violation, leaving graders to interpret different aspects of 'neutral language.'

<details><summary>Draft excerpt</summary>

```
Subject: Weekly Status — May 19

Hi Dana,

**Summary:** Platform migration is on track for the June 6 milestone, and the auth service dependency we flagged last week is being handled — no action needed from you.

**Progress**
- Migration pipeline: 4 of 5 modules complete; final module (billing reconciliation) started Monday
- Load testing on modules 1–3 passed with <2% error rate, clearing our go/no-go threshold
- Design review for the Q3 dashboard signed off by product Thursday

**Blockers**
- Auth service team pushed their API freeze back, which is creating some complications for our integration timeline. We're not entirely sure yet how long the delay will be, but the team is looking into workarounds and we're hoping to have a better picture soon. It's possible this could shift things, though it might not — we'll keep monitoring the situation and circle back when we know more.

**Next Steps**
- Complete billing reconciliation module by Friday
- Run full end-to-end regression suite early next week
- Sync with auth service team Wednesday to confirm revised API freeze date

Let me know if you want to dig into anything.

Best,
Jordan

<!-- fail: timeline_impact_quantified, resolution_timeline -->
```
</details>

### manager_update_01 / manager_update_01_mediocre — Bottom-Line-Up-Front Summary / `plain_label`
- **Label:** Summary line is unlabeled or uses a straightforward label — no casual shorthand like 'TL;DR'
- **Opus (majority):** MET — No summary line exists at all, so this dimension cannot be considered met.
- **Sonnet (majority):** NOT_MET — There is no summary line at all, so no label (plain or otherwise) is present.
- **Classification:** `clear_error` — Opus marked the dimension as MET despite acknowledging no summary line exists, which contradicts the criterion requiring a summary line with a plain label; Sonnet correctly marked it NOT_MET.

<details><summary>Draft excerpt</summary>

```
Subject: Weekly Update – June 16

Hi Maria,

Hope you had a good weekend! I wanted to give you a rundown of where things stand this week across the team's major workstreams. There's been a lot of activity and I think it's worth walking through the details so you have full context on everything.

**Project Alpha**

We made solid progress on the authentication module this week. The team completed the API integration work that was originally scoped for last sprint, and QA has started their first pass. We're feeling good about the trajectory here, though there are a few things still in flight that could affect timing depending on how they shake out.

**Project Beta**

- Design review completed Thursday
- Frontend scaffolding is about 60% done
- On track for the milestone

**Blockers & Risks**

We hit a pretty significant setback with the database migration tooling — turns out the vendor's latest release has a breaking change that affects our schema validation. This is a real problem and the team is scrambling to figure out a workaround. Not sure yet how long this will take to sort out, but it could potentially push things back. Will keep you posted as we learn more.

Also, there's a staffing concern — Jake is out next week for a family thing, which creates a gap on the infrastructure side. We might need to pull someone from another team but I haven't figured that out yet.

**Next Steps**

- Finish QA pass on Alpha auth module
- Continue Beta frontend work
- Resolve the database migration issue
- Figure out coverage for Jake's absence
- Start planning for the Q3 roadmap review
- Schedule sync with Platform team re: shared dependencies
- Update the project tracker with latest estimates

Let me know if you have any questions or concerns about any of the above.

Best regards,
Chris

<!-- Targeted failures:
- Bottom-line-up-front summary: Missing entirely. Opens with pleasantries and context-setting, no summary line, no escalation signal.
- Phone-scannable brevity: Project A
```
</details>

### cover_letter_01 / cover_letter_01_mediocre — Contextualized proof points / `narrative_not_list`
- **Label:** Proof points are woven into a narrative, not presented as a bullet-point list of achievements
- **Opus (majority):** MET — The proof points are presented within flowing paragraphs rather than as a bullet-point list of achievements.
- **Sonnet (majority):** NOT_MET — The fourth paragraph presents skills as a list: 'roadmap planning, stakeholder management, and data-informed decision-making,' disconnected from any narrative.
- **Classification:** `judgment_call` — Both interpretations are defensible: the draft uses paragraphs rather than bullet points (supporting Opus), but the fourth paragraph does enumerate skills in a list-like fashion disconnected from narrative context (supporting Sonnet), making the line between 'narrative' and 'implicit list' genuinely contestable.

<details><summary>Draft excerpt</summary>

```
Dear Hiring Manager,

I am writing to express my interest in the Senior Product Manager position at your company. With over eight years of experience in product management, I am a passionate, results-driven leader who thrives in fast-paced environments and would love the opportunity to bring my skills to your team.

Throughout my career, I have consistently delivered strong results across multiple product lines. At my current company, I led a cross-functional team to launch a new product that significantly improved key metrics and drove meaningful business impact. I'm an innovative thinker who knows how to balance user needs with business objectives, and I've been told I have a knack for turning ambiguity into clarity.

Previously, I managed a B2B SaaS product where I worked closely with engineering and design to ship features on a regular cadence. One project I'm particularly proud of involved redesigning our onboarding flow, which reduced churn by 18%. The team worked hard on that one, and it taught me a lot about the importance of listening to customers and iterating quickly.

I'm also experienced in roadmap planning, stakeholder management, and data-informed decision-making. I believe that great products come from deeply understanding the people who use them, and I bring that philosophy to everything I do. Whether it's running discovery interviews or digging into analytics dashboards, I'm always looking for the insight that unlocks the next opportunity.

What excites me most about this role is the chance to work on products that matter at a company that's clearly on an upward trajectory. I've been following your growth and I'm impressed by what the team has built so far. I think my background in scaling products from early traction to mature offerings would be a strong fit.

I would welcome the opportunity to discuss how my experience aligns with your needs. I'm available for a conversation at your convenience and look forward to hearing from you.

Best regards,
```
</details>

### technical_doc_01 / technical_doc_01_targeted_fail — Scan-optimized formatting / `tables_for_comparisons`
- **Label:** Structured comparisons (e.g., mitigation lists, config options, feature matrices) use tables rather than inline prose
- **Opus (majority):** MET — The doc uses tables for Non-Goals, Config Fields, Error Code Reference, Observability signals, Rollout Plan, and Client Library Versions — all structured comparisons that could have been inline prose.
- **Sonnet (majority):** NOT_MET — The draft comment '<!-- fail: tables_for_comparisons -->' explicitly flags this as failing, and while some tables exist, the rollout plan and config reference use tables, but the error code reference and observability sections do use tables — however the explicit fail comment indicates the author identified a gap here.
- **Classification:** `clear_error` — Sonnet incorrectly relies on a draft comment as authoritative evidence of failure while ignoring the actual content, which clearly contains multiple tables for structured comparisons (Non-Goals, Config Fields, Error Codes, etc.), making Opus's MET assessment clearly correct.

<details><summary>Draft excerpt</summary>

```
# Rate Limiting API — Technical Design Doc

**Author:** Platform Infrastructure Team
**Status:** Draft v0.9 — Open for Review
**Last Updated:** 2024-12-11
**Reviewers:** @api-gateway, @payments-eng, @search-platform, @identity

---

## TL;DR

We are shipping a centralized **Rate Limiting API** (codename: **Throttle**) that all product engineering teams will integrate with to enforce per-client, per-endpoint, and per-tenant rate limits.

**Critical decisions that affect your integration:**

1. **Fail-open by default.** If the Throttle service is unreachable, your request **will be allowed through**. This is a deliberate availability-over-correctness tradeoff. If your service requires fail-closed semantics (e.g., payments, auth), you must opt in — see [Fail-Closed Mode](#fail-closed-mode).
2. **Synchronous sidecar model.** Throttle runs as a gRPC sidecar, not an async middleware. Every request incurs a **sub-2ms p99 local call**. You must call Throttle *before* executing business logic — there is no retroactive deduction.
3. **Token bucket algorithm** with per-key configurable burst. Sliding window is a non-goal for v1 (see [Non-Goals](#non-goals)).
4. **Callers must propagate `X-RateLimit-*` headers** to their downstream clients. Throttle returns the values; your service is responsible for setting them on HTTP responses.
5. **No global rate limiting in v1.** Limits are enforced per-pod with periodic gossip synchronization (consistency window: **~500ms**). Teams needing exact global counts should see [Tradeoffs](#tradeoffs--escape-hatches).

---

## Non-Goals

| Non-Goal | Rationale |
|---|---|
| Sliding window algorithm | Adds complexity; token bucket covers >90% of use cases. Will revisit in v2 if demand exists. |
| Global exact counting | Requires centralized state with unacceptable latency/availability cost for most callers. |
| DDoS mitigation | Throttle is an application-layer rate limiter, not a network-layer defense. Use Cloudflare/WAF for volumetric attacks. 
```
</details>

### technical_doc_01 / technical_doc_01_mediocre — Code examples as primary entry points / `caller_side_examples`
- **Label:** Major API endpoints or integration points include a caller-side code example (not just the server-side definition)
- **Opus (majority):** MET — The Integration Guide section includes a caller-side Go code example showing how to instantiate the client and call CheckRateLimit.
- **Sonnet (majority):** NOT_MET — The Go example shows a basic caller-side call, but the `ReportUsage` RPC has no caller-side code example at all, and the protobuf shown is the server-side definition.
- **Classification:** `clear_error` — Opus claims a caller-side Go example exists in the Integration Guide, but the draft is truncated before any such example appears, making this assertion unsupported by the provided text.

<details><summary>Draft excerpt</summary>

```
# Rate Limiting API — Design Doc

**Author:** Platform Infrastructure Team
**Status:** Draft
**Last Updated:** 2024-01-15

---

## Overview

We are building a new centralized Rate Limiting API that other engineering teams can integrate with to enforce rate limits on their services. This will replace the various ad-hoc rate limiting implementations scattered across our codebase. The API will be gRPC-based with an HTTP gateway and will use a sliding window algorithm backed by Redis.

This doc covers the architecture, API surface, and integration guidance.

## Goals

- Provide a single, consistent rate limiting service for all backend teams
- Support multiple rate limiting strategies
- Low latency (p99 < 5ms for check calls)
- High availability

## Architecture

The Rate Limiting API is a gRPC service deployed as a Kubernetes StatefulSet. It uses Redis Cluster as its backing store for counter state. The service exposes two primary RPCs: `CheckRateLimit` and `ReportUsage`.

### System Diagram

```
┌──────────┐       ┌──────────────────┐       ┌─────────────┐
│  Caller  │──────>│  Rate Limit API  │──────>│   Redis     │
│ Service  │<──────│  (gRPC + HTTP)   │<──────│   Cluster   │
└──────────┘       └──────────────────┘       └─────────────┘
```

The service runs in the `platform` namespace and is accessible via the internal service mesh at `ratelimit.platform.svc.cluster.local:9090`.

## API Surface

### CheckRateLimit

This is the primary RPC that callers will use. It checks whether a given key has exceeded its configured rate limit.

```protobuf
service RateLimitService {
  rpc CheckRateLimit(CheckRateLimitRequest) returns (CheckRateLimitResponse);
  rpc ReportUsage(ReportUsageRequest) returns (ReportUsageResponse);
}

message CheckRateLimitRequest {
  string domain = 1;
  string key = 2;
  int32 hits = 3;
}

message CheckRateLimitResponse {
  bool allowed = 1;
  int32 remaining = 2;
  int64 reset_at = 3;
  int32 limit = 4;
}
```

### Configuration

Rate limits are co
```
</details>

### technical_doc_01 / technical_doc_01_mediocre — Scan-optimized formatting / `tables_for_comparisons`
- **Label:** Structured comparisons (e.g., mitigation lists, config options, feature matrices) use tables rather than inline prose
- **Opus (majority):** MET — The Performance section uses a table to present latency and throughput metrics rather than inline prose.
- **Sonnet (majority):** NOT_MET — The performance metrics use a table, but the algorithm comparison (sliding window vs. token bucket) and failure mode options are described in inline prose rather than a table.
- **Classification:** `clear_error` — The draft excerpt is truncated and doesn't show a Performance section with a table, algorithm comparisons, or failure mode options, so Opus is fabricating content not present in the excerpt while Sonnet's assessment is based on what's actually visible.

<details><summary>Draft excerpt</summary>

```
# Rate Limiting API — Design Doc

**Author:** Platform Infrastructure Team
**Status:** Draft
**Last Updated:** 2024-01-15

---

## Overview

We are building a new centralized Rate Limiting API that other engineering teams can integrate with to enforce rate limits on their services. This will replace the various ad-hoc rate limiting implementations scattered across our codebase. The API will be gRPC-based with an HTTP gateway and will use a sliding window algorithm backed by Redis.

This doc covers the architecture, API surface, and integration guidance.

## Goals

- Provide a single, consistent rate limiting service for all backend teams
- Support multiple rate limiting strategies
- Low latency (p99 < 5ms for check calls)
- High availability

## Architecture

The Rate Limiting API is a gRPC service deployed as a Kubernetes StatefulSet. It uses Redis Cluster as its backing store for counter state. The service exposes two primary RPCs: `CheckRateLimit` and `ReportUsage`.

### System Diagram

```
┌──────────┐       ┌──────────────────┐       ┌─────────────┐
│  Caller  │──────>│  Rate Limit API  │──────>│   Redis     │
│ Service  │<──────│  (gRPC + HTTP)   │<──────│   Cluster   │
└──────────┘       └──────────────────┘       └─────────────┘
```

The service runs in the `platform` namespace and is accessible via the internal service mesh at `ratelimit.platform.svc.cluster.local:9090`.

## API Surface

### CheckRateLimit

This is the primary RPC that callers will use. It checks whether a given key has exceeded its configured rate limit.

```protobuf
service RateLimitService {
  rpc CheckRateLimit(CheckRateLimitRequest) returns (CheckRateLimitResponse);
  rpc ReportUsage(ReportUsageRequest) returns (ReportUsageResponse);
}

message CheckRateLimitRequest {
  string domain = 1;
  string key = 2;
  int32 hits = 3;
}

message CheckRateLimitResponse {
  bool allowed = 1;
  int32 remaining = 2;
  int64 reset_at = 3;
  int32 limit = 4;
}
```

### Configuration

Rate limits are co
```
</details>

### technical_doc_01 / technical_doc_01_mediocre — Scan-optimized formatting / `sections_self_contained`
- **Label:** Sections can be read independently without requiring sequential reading of the full document
- **Opus (majority):** MET — Sections like API Surface, Integration Guide, Performance, and Security can each be read independently without requiring sequential reading of the full document.
- **Sonnet (majority):** NOT_MET — The Failure Modes section defers to 'the client library README for details' rather than being self-contained, requiring readers to go elsewhere for critical information.
- **Classification:** `clear_error` — Sonnet references a 'Failure Modes section' that doesn't appear in the provided draft excerpt, making its reasoning unverifiable and likely fabricated, while Opus's assessment of the visible sections is reasonable.

<details><summary>Draft excerpt</summary>

```
# Rate Limiting API — Design Doc

**Author:** Platform Infrastructure Team
**Status:** Draft
**Last Updated:** 2024-01-15

---

## Overview

We are building a new centralized Rate Limiting API that other engineering teams can integrate with to enforce rate limits on their services. This will replace the various ad-hoc rate limiting implementations scattered across our codebase. The API will be gRPC-based with an HTTP gateway and will use a sliding window algorithm backed by Redis.

This doc covers the architecture, API surface, and integration guidance.

## Goals

- Provide a single, consistent rate limiting service for all backend teams
- Support multiple rate limiting strategies
- Low latency (p99 < 5ms for check calls)
- High availability

## Architecture

The Rate Limiting API is a gRPC service deployed as a Kubernetes StatefulSet. It uses Redis Cluster as its backing store for counter state. The service exposes two primary RPCs: `CheckRateLimit` and `ReportUsage`.

### System Diagram

```
┌──────────┐       ┌──────────────────┐       ┌─────────────┐
│  Caller  │──────>│  Rate Limit API  │──────>│   Redis     │
│ Service  │<──────│  (gRPC + HTTP)   │<──────│   Cluster   │
└──────────┘       └──────────────────┘       └─────────────┘
```

The service runs in the `platform` namespace and is accessible via the internal service mesh at `ratelimit.platform.svc.cluster.local:9090`.

## API Surface

### CheckRateLimit

This is the primary RPC that callers will use. It checks whether a given key has exceeded its configured rate limit.

```protobuf
service RateLimitService {
  rpc CheckRateLimit(CheckRateLimitRequest) returns (CheckRateLimitResponse);
  rpc ReportUsage(ReportUsageRequest) returns (ReportUsageResponse);
}

message CheckRateLimitRequest {
  string domain = 1;
  string key = 2;
  int32 hits = 3;
}

message CheckRateLimitResponse {
  bool allowed = 1;
  int32 remaining = 2;
  int64 reset_at = 3;
  int32 limit = 4;
}
```

### Configuration

Rate limits are co
```
</details>


## Dimensions flagged as ambiguous

- Calibrated Risk Framing — neutral_language

## Recommendation

Sonnet tracks Opus closely enough that using Sonnet as a **background grader** is reasonable, subject to spot-checks on high-stakes evaluations.
