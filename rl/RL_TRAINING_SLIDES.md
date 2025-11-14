---
marp: true
theme: default
paginate: true
---

# Rubric-Based RL Training Pipeline

**Teaching Language Models to Write Better Through Iterative Improvement**

---

## Overview: What Are We Doing?

**Goal**: Train a language model to write better grant proposals by learning from rubric-based feedback

**Method**: Reinforcement Learning (RL) with iterative revision

**Key Idea**:
- Model writes a draft → Gets scored → Revises → Gets rewarded for improvement
- Over many episodes, model learns what changes lead to higher scores

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Environment Setup                                        │
│     • Load rubric + prompts                                  │
│     • Initialize model + evaluator (Claude 4.5)              │
│                                                               │
│  2. Episode Loop (For each prompt)                           │
│     • Generate draft → Score → Revise → Score → Reward       │
│     • Repeat until no improvement or max turns               │
│                                                               │
│  3. Learning                                                 │
│     • Use PPO (Proximal Policy Optimization)                 │
│     • Update model weights based on rewards                  │
│     • Model learns: "what changes → better scores"           │
│                                                               │
│  4. Repeat                                                   │
│     • Process many episodes across different prompts         │
│     • Model gradually learns rubric-aligned writing          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. **Policy Model** (The Learner)
- **What**: Qwen/Qwen3-235B-A22B-Instruct-2507
- **Role**: Generates drafts and revisions
- **Training**: Learns through LoRA fine-tuning
- **Updates**: Weights adjusted based on rewards

### 2. **Evaluator Model** (The Teacher)
- **What**: Claude Sonnet 4.5
- **Role**: Scores drafts using rubric criteria
- **Fixed**: Does NOT get trained
- **Output**: Scores 0-100 based on rubric achievement levels

### 3. **Rubric** (The Standard)
- **What**: JSON with 4 criteria + achievement levels
- **Role**: Defines what "good writing" means
- **Used by**: Evaluator to score drafts

---

## The Rubric: Defining "Good"

```json
{
  "rubric": [
    {
      "name": "Evidence-Based Impact",
      "weight": 35,
      "exemplary": "Every claim backed by specific data...",
      "proficient": "90%+ of claims backed by data...",
      "developing": "50-70% of claims have data...",
      "beginning": "Less than 50% claims supported..."
    },
    {
      "name": "Professional Tone & Accessibility",
      "weight": 25,
      ...
    },
    ...
  ]
}
```

**Overall Score** = Σ(achievement_level% × weight / 100)

---

## Episode Structure: The RL Loop

### An "Episode" = One Complete Revision Cycle

```
Turn 0: Initial Draft
  ├─ User: "Write a grant proposal for community health..."
  ├─ Model: Generates initial draft
  ├─ Evaluator: Scores draft → 55/100
  └─ Reward: +5.5 (first turn reward = score/10)

Turn 1: First Revision
  ├─ User: "Your draft scored 55/100. Revise to improve..."
  ├─ Model: Generates revision
  ├─ Evaluator: Scores revision → 62/100
  └─ Reward: +0.70 (improvement: (62-55)/100 × 10 = +0.70)

Turn 2: Second Revision
  ├─ User: "Scored 62/100 (up from 55). Revise further..."
  ├─ Model: Generates revision
  ├─ Evaluator: Scores revision → 58/100 (worse!)
  └─ Reward: -0.40 (regression: (58-62)/100 × 10 = -0.40)

Episode ENDS (reward ≤ 0 or max turns reached)
```

---

## Reward Function: Learning Signal

### How Rewards Work

**Turn 0** (Initial draft):
```python
reward = score / 10.0
# Example: score=55 → reward=5.5
```

**Turn 1+** (Revisions):
```python
reward = 10.0 × (new_score - old_score) / 100.0
# Example: 62 - 55 = +7 points → reward = +0.70
# Example: 58 - 62 = -4 points → reward = -0.40
```

### Termination Conditions
Episode ends when:
1. ✅ **Reward ≤ 0** (no improvement or got worse)
2. ✅ **Max turns reached** (default: 5)
3. ✅ **Near-perfect score** (≥ 95/100)

---

## Reward Function: Why This Works

### Intuition
- **Positive reward** (+) → Model learns: "This revision was good, do more of this"
- **Negative reward** (-) → Model learns: "This made it worse, avoid this"
- **Magnitude** → Bigger improvement = bigger reward

### Examples
| Score Change | Reward | Learning Signal |
|-------------|--------|-----------------|
| 55 → 65 | +1.0 | Strong positive (good revision!) |
| 60 → 62 | +0.2 | Weak positive (small improvement) |
| 70 → 70 | 0.0 | No change (episode ends) |
| 65 → 60 | -0.5 | Negative (bad revision) |

### Over Time
- Model learns patterns that consistently lead to positive rewards
- Avoids patterns that lead to negative rewards
- Result: Better at making effective revisions

---

## Training Loop: The Full Process

```
┌──────────────────────────────────────────────────────────────┐
│ For each training step:                                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Sample Batch                                              │
│     • Select 8 groups of 4 episodes (32 episodes total)       │
│     • Each group uses same prompt, different random seeds     │
│                                                                │
│  2. Run Episodes                                              │
│     For each episode:                                         │
│       Turn 0: Generate draft → Score → Reward                 │
│       Turn 1: Revise → Score → Reward                         │
│       Turn 2: Revise → Score → Reward                         │
│       ... (until termination)                                 │
│                                                                │
│  3. Collect Trajectories                                      │
│     • Store: [state, action, reward] for each turn            │
│     • Episode data = full sequence of decisions + rewards     │
│                                                                │
│  4. Compute Advantages                                        │
│     • Compare: "How good was this action vs. expected?"       │
│     • Used by PPO to weight updates                           │
│                                                                │
│  5. Update Model (PPO)                                        │
│     • Gradient descent on policy loss                         │
│     • LoRA: Only update ~1% of parameters                     │
│     • KL penalty: Don't drift too far from original model     │
│                                                                │
│  6. Save Checkpoint (every 10 steps)                          │
│     • Save model weights for later evaluation                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Step by Step

```
┌─────────────────┐
│  User Prompt    │
│  "Write grant   │
│   proposal..."  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Policy Model (Qwen)                                     │
│  Generates draft based on rubric + prompt               │
└────────┬────────────────────────────────────────────────┘
         │
         │ Generated Draft
         ▼
┌─────────────────────────────────────────────────────────┐
│  Evaluator Model (Claude 4.5)                           │
│  • Reads draft                                          │
│  • Applies rubric criteria                              │
│  • Returns score (0-100) + detailed breakdown           │
└────────┬────────────────────────────────────────────────┘
         │
         │ Score = 55/100
         ▼
┌─────────────────────────────────────────────────────────┐
│  Reward Calculation                                     │
│  reward = score / 10 = 5.5                              │
└────────┬────────────────────────────────────────────────┘
         │
         │ reward = 5.5
         ▼
┌─────────────────────────────────────────────────────────┐
│  Revision Prompt                                        │
│  "Your draft scored 55/100. Revise to improve..."      │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
    [REPEAT CYCLE]
```

---

## The RL Environment: WritingRevisionEnv

### State (Observation)
```
System Instruction:
  "You are writing a grant proposal. Follow this rubric:
   1. Evidence-Based Impact (35% weight)
   2. Professional Tone (25% weight)
   ..."

User Message:
  Turn 0: "Write about community health initiative"
  Turn 1: "Your draft scored 62/100 (up from 55). Revise..."
```

### Action
```
Model's generated text (draft or revision)
```

### Reward
```python
if turn == 0:
    reward = score / 10.0
else:
    reward = 10.0 * (new_score - old_score) / 100.0
```

---

## PPO (Proximal Policy Optimization)

### What is PPO?

**Goal**: Update the model to generate higher-reward actions

**How**: Gradient descent on policy loss with constraints

### Key Ideas

1. **Policy** = Model's behavior (what text it generates)
2. **Optimize** = Adjust weights to increase probability of high-reward actions
3. **Proximal** = Don't change too much at once (stability)

### The Update Rule (Simplified)

```
For each (state, action, reward):
  1. How likely was this action? → π_old(action|state)
  2. How likely is it now? → π_new(action|state)
  3. Ratio = π_new / π_old
  4. Clipped objective = clip(ratio × advantage, 0.8, 1.2)
  5. Loss = -mean(clipped_objective)
  6. Gradient descent on loss
```

**Advantage** = "How good was this action vs. expected?"

---

## KL Penalty: Staying Grounded

### The Problem
Without constraints, RL can make the model:
- Forget its original knowledge
- Generate nonsense that "games" the evaluator
- Become unstable

### The Solution: KL Divergence Penalty

```python
loss = policy_loss + kl_coef × kl_divergence(new_model, original_model)
#                      ↑
#                   0.01 (default)
```

**KL Divergence** = How different are the models' output distributions?

**Effect**:
- Model can improve, but not drift too far from original
- Retains general language abilities
- More stable training

---

## LoRA: Efficient Fine-Tuning

### Why LoRA?

**Problem**: Full fine-tuning of 235B model = expensive & slow

**Solution**: Low-Rank Adaptation (LoRA)

### How It Works

```
Original weight matrix: W (large, e.g., 4096 × 4096)

LoRA adds:
  ΔW = A × B
  where A is 4096 × 32 and B is 32 × 4096

Final weight: W' = W + ΔW

Trainable params:
  - Without LoRA: 4096² = 16.8M params
  - With LoRA (rank=32): 4096×32 + 32×4096 = 262K params

Reduction: ~98% fewer parameters to train!
```

**Rank** = Bottleneck dimension (default: 32)

---

## Training Hyperparameters

```python
# Model configuration
model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
lora_rank = 32                    # LoRA compression

# Episode configuration
max_turns = 5                     # Max revisions per episode
groups_per_batch = 8              # Prompts per batch
group_size = 4                    # Episodes per prompt

# Learning configuration
learning_rate = 3e-5              # Step size for updates
kl_penalty_coef = 0.01            # KL divergence weight
max_tokens = 2048                 # Max generation length

# Evaluation & checkpointing
eval_every = 5                    # Evaluate every N steps
save_every = 10                   # Save checkpoint every N steps
```

**Batch size** = groups_per_batch × group_size = 32 episodes/step

---

## What Happens During Training?

### Step 1: Initialize
- Load base model (Qwen)
- Add LoRA adapters (trainable)
- Freeze base weights
- Initialize evaluator (Claude 4.5)

### Steps 2-N: Training Loop

**Each step** (takes ~10-30 minutes):
1. Sample 32 episodes (8 prompts × 4 episodes each)
2. Run all episodes to completion
3. Collect rewards and trajectories
4. Compute PPO loss
5. Update LoRA weights via gradient descent
6. Log metrics (scores, rewards, etc.)

**Every 5 steps**: Evaluation
**Every 10 steps**: Save checkpoint

---

## Metrics Tracked During Training

### Per Episode
- `turn`: Which turn in the episode (0-5)
- `score`: Current draft score (0-100)
- `score_delta`: Change from previous turn
- `avg_score`: Average across all turns so far
- `max_score`: Best score in this episode

### Per Step (Logged)
- `mean_reward`: Average reward across batch
- `mean_score`: Average final score
- `episode_length`: Average turns per episode
- `policy_loss`: PPO loss value
- `kl_divergence`: Distance from original model

### Saved to WandB or local logs

---

## Example Training Session

```
Step 1:
  Prompt 1 (4 episodes): avg_score=56.2, avg_reward=5.1
  Prompt 2 (4 episodes): avg_score=61.3, avg_reward=6.0
  ...
  Batch mean_score: 58.7, mean_reward: 5.5
  → Update model

Step 2:
  Batch mean_score: 59.4, mean_reward: 5.8
  → Update model

...

Step 5: EVALUATION
  Test on held-out prompts → mean_score: 61.2

Step 10: CHECKPOINT SAVED
  /tmp/tinker-examples/.../step_10

...

Step 100: CHECKPOINT SAVED
  /tmp/tinker-examples/.../step_100
```

---

## Why This Works: The Learning Process

### What the Model Learns

1. **Initial Drafts** (Turn 0)
   - Start with evidence and data
   - Use professional tone
   - Structure logically
   - Be concise

2. **Effective Revisions** (Turn 1+)
   - Add missing quantitative data (+reward)
   - Remove informal language (+reward)
   - Improve transitions (+reward)
   - Cut redundancy (+reward)

3. **What NOT to Do**
   - Remove evidence (-reward)
   - Add informal language (-reward)
   - Make it verbose (-reward)

### Emergent Behavior
Over many episodes, model learns **generalizable patterns**:
- "Vague claims → add specific numbers"
- "Informal tone → use professional language"
- "Unclear flow → add explicit transitions"

---

## Challenges & Solutions

### Challenge 1: Reward Sparsity
**Problem**: Only get reward at end of turn
**Solution**: Turn-by-turn rewards (every revision scored)

### Challenge 2: Evaluator Noise
**Problem**: Claude 4.5 might score inconsistently
**Solution**:
- Multiple episodes per prompt (group_size=4)
- Average rewards across group

### Challenge 3: Overfitting
**Problem**: Model might memorize training prompts
**Solution**:
- Diverse prompts (10 different topics)
- KL penalty (stay close to original)
- Evaluation on held-out data

### Challenge 4: Training Stability
**Problem**: RL can be unstable
**Solution**:
- PPO (clipped updates)
- Low learning rate (3e-5)
- KL penalty
- LoRA (smaller update space)

---

## Comparison to Other Approaches

### Traditional Fine-Tuning
```
Input: Prompt → Output: Target text
Loss: Cross-entropy on target tokens
```
**Limitation**: Need labeled examples of "perfect" writing

### Our RL Approach
```
Input: Prompt → Output: Generated text → Evaluator → Reward
Loss: Policy gradient based on rewards
```
**Advantage**: Only need rubric (defines "good"), not examples

### Why RL Here?
- No need for expensive human-written examples
- Can explore different approaches
- Learns from automated feedback
- Optimizes for rubric directly

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Architecture                     │
└─────────────────────────────────────────────────────────────┘

           ┌─────────────────────────────────────┐
           │       Dataset Builder               │
           │  • 10 prompts                       │
           │  • Rubric with 4 criteria           │
           │  • Creates episode configurations   │
           └──────────────┬──────────────────────┘
                          │
                          ▼
           ┌─────────────────────────────────────┐
           │     Environment Builder             │
           │  • Creates WritingRevisionEnv       │
           │  • Initializes Claude 4.5 evaluator │
           └──────────────┬──────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Episode 1    │  │ Episode 2    │  │  Episode N   │
│ (group=1)    │  │ (group=1)    │  │  (group=8)   │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ • Generate   │  │ • Generate   │  │ • Generate   │
│ • Score      │  │ • Score      │  │ • Score      │
│ • Revise     │  │ • Revise     │  │ • Revise     │
│ • Collect    │  │ • Collect    │  │ • Collect    │
│   rewards    │  │   rewards    │  │   rewards    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
           ┌─────────────────────────────────────┐
           │      PPO Training Update            │
           │  • Compute advantages               │
           │  • Calculate policy loss            │
           │  • Add KL penalty                   │
           │  • Gradient descent on LoRA         │
           └──────────────┬──────────────────────┘
                          │
                          ▼
           ┌─────────────────────────────────────┐
           │    Updated Model Checkpoint         │
           │  • Save every 10 steps              │
           │  • Evaluate every 5 steps           │
           └─────────────────────────────────────┘
```

---

## Episode Timeline Example

```
Episode for: "Community health initiative grant"

00:00 ─────────────────────────────────────────────────────────
       [Turn 0: Initial Draft]
       User: "Write a grant proposal for community health..."
       Model: Generates 500-word draft

00:45 ─────────────────────────────────────────────────────────
       [Evaluator Scoring]
       Claude 4.5 evaluates against rubric

01:20 ─────────────────────────────────────────────────────────
       [Score: 55/100]
       • Evidence-Based Impact: 15/35
       • Professional Tone: 18/25
       • Logical Structure: 12/20
       • Conciseness: 10/20
       REWARD: +5.5

01:25 ─────────────────────────────────────────────────────────
       [Turn 1: Revision Prompt]
       User: "Scored 55/100. Add specific data and improve flow"
       Model: Generates revised draft

02:10 ─────────────────────────────────────────────────────────
       [Evaluator Scoring]

02:45 ─────────────────────────────────────────────────────────
       [Score: 62/100]
       • Evidence-Based Impact: 20/35 (+5)
       • Professional Tone: 18/25 (0)
       • Logical Structure: 14/20 (+2)
       • Conciseness: 10/20 (0)
       REWARD: +0.7

02:50 ─────────────────────────────────────────────────────────
       [Turn 2: Revision Prompt]
       Model: Generates second revision

03:35 ─────────────────────────────────────────────────────────
       [Score: 61/100]
       REWARD: -0.1 (worse!)

       EPISODE ENDS (negative reward)
       Total episode reward: 5.5 + 0.7 - 0.1 = 6.1
```

---

## Code Walkthrough: Key Files

### `train.py` - Main Training Script
```python
# Loads rubric and prompts
rubric = load_rubric("rubrics/example_rubric.json")
prompts = load_prompts("prompts/writing_prompts.txt")

# Creates dataset builder
dataset_builder = WritingRLDatasetBuilder(
    prompts=prompts,
    rubric=rubric,
    groups_per_batch=8,
    group_size=4,
    max_turns=5,
)

# Configures RL training
cfg = train.Config(
    learning_rate=3e-5,
    dataset_builder=dataset_builder,
    model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
    lora_rank=32,
    kl_penalty_coef=0.01,
)

# Runs training
await train.main(cfg)
```

---

## Code Walkthrough: The Environment

### `writing_env.py` - RL Environment

**Key Methods**:

1. `initial_observation()` - Start episode
   ```python
   conversation_history = [
       {"role": "system", "content": system_instruction},
       {"role": "user", "content": prompt}
   ]
   return renderer.build_generation_prompt(conversation_history)
   ```

2. `step(action)` - Process one turn
   ```python
   # Parse model's generated text
   generated_text = renderer.parse_response(action)

   # Score the draft
   new_score = await self._score_draft(generated_text)

   # Compute reward
   reward = 10.0 * (new_score - self.current_score) / 100.0

   # Determine if done
   episode_done = (reward <= 0 or turns >= max_turns)

   return StepResult(reward, episode_done, next_observation)
   ```

---

## Code Walkthrough: Scoring

### `_score_draft()` - Evaluator Call

```python
async def _score_draft(self, draft: str) -> float:
    # Format rubric + draft for evaluator
    eval_prompt = f"""
    Evaluate this draft against the rubric:

    Draft: {draft}
    Rubric: {json.dumps(self.rubric)}

    {RUBRIC_SCORING_PROMPT}
    """

    # Call Claude 4.5
    response = self.evaluator_client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": eval_prompt}]
    )

    # Extract score from JSON response
    score = self._extract_score_from_evaluation(response.content)
    return score
```

---

## Prompt Engineering: The Rubric Scoring Prompt

The `RUBRIC_SCORING_PROMPT` (in `prompts.py`) instructs Claude 4.5 to:

1. **Understand achievement levels**
   - Exemplary (100%), Proficient (75%), Developing (50%), Beginning (25%)

2. **Evaluate each criterion**
   - Match draft to achievement level descriptors
   - Provide specific evidence from text

3. **Calculate weighted score**
   ```
   Overall = Σ(achievement_level% × weight / 100)
   ```

4. **Return structured JSON**
   ```json
   {
     "overall_score": 62,
     "criteria_scores": [...],
     "overall_assessment": "..."
   }
   ```

---

## Training Hardware & Time

### Requirements
- **GPU**: 1-4 A100/H100 GPUs (for 235B model)
- **RAM**: 128GB+ recommended
- **Storage**: 500GB+ for checkpoints

### Training Time
- **Per step**: ~10-30 minutes
  - Depends on: model size, batch size, generation length
- **100 steps**: ~24-48 hours
- **Full training**: Variable (monitor eval scores)

### Costs
- **Compute**: GPU time (cloud or local)
- **API**: Claude 4.5 calls for scoring
  - ~32 calls per step (batch size)
  - ~3,200 calls for 100 steps
  - Estimate: $50-100 for 100 steps

---

## Monitoring Training Progress

### What to Watch

1. **Mean Score** (should increase)
   ```
   Step 10: mean_score=58.2
   Step 20: mean_score=59.8
   Step 30: mean_score=61.5  ← Improving ✓
   ```

2. **Mean Reward** (should stabilize or increase)
   ```
   Step 10: mean_reward=5.1
   Step 20: mean_reward=5.8
   Step 30: mean_reward=6.2  ← Good ✓
   ```

3. **Episode Length** (may decrease as model learns)
   ```
   Step 10: avg_turns=3.2
   Step 30: avg_turns=2.1  ← Reaches good scores faster ✓
   ```

4. **KL Divergence** (should stay low)
   ```
   Step 10: kl_div=0.02
   Step 30: kl_div=0.08  ← Still reasonable ✓
   ```

---

## Signs of Good Training

### ✅ Healthy Training
- Mean score increases steadily
- Mean reward positive and increasing
- KL divergence < 0.5
- Episode length stabilizes
- Eval scores track training scores

### ⚠️ Warning Signs
- Mean score plateaus early (< 65)
  → May need more steps or higher LR

- Mean score decreases
  → Overtraining, use earlier checkpoint

- KL divergence > 1.0
  → Model drifting too far, lower LR or increase KL penalty

- High variance in scores
  → Need more episodes per batch

---

## After Training: What Next?

1. **Evaluate Checkpoints**
   ```bash
   python evaluate.py \
       --checkpoint_path .../step_100 \
       --compare_baseline
   ```

2. **Pick Best Checkpoint**
   - Highest mean score
   - Best improvement over baseline
   - Low variance

3. **Deploy**
   - Load best checkpoint
   - Use for inference on new prompts
   - Monitor quality

4. **Iterate**
   - Collect feedback on outputs
   - Refine rubric if needed
   - Re-train with updated rubric

---

## Customization: Adapting to Your Use Case

### Different Writing Tasks
1. **Update prompts** (`prompts/writing_prompts.txt`)
   - Academic papers → research prompts
   - Marketing copy → product descriptions
   - Technical docs → feature specifications

2. **Update rubric** (`rubrics/example_rubric.json`)
   - Define criteria for your domain
   - Set appropriate weights
   - Write achievement level descriptors

3. **Run training** with new data
   ```bash
   python train.py \
       --prompts_file prompts/my_prompts.txt \
       --rubric_file rubrics/my_rubric.json
   ```

---

## Advanced: Multi-Turn Learning Dynamics

### Early Training (Steps 1-20)
- Model explores different approaches
- High variance in scores
- Learning basic rubric patterns

### Mid Training (Steps 20-60)
- Model refines effective strategies
- Scores increase steadily
- Variance decreases

### Late Training (Steps 60-100+)
- Model fine-tunes details
- Scores plateau or increase slowly
- Risk of overtraining

### Key Insight
**Don't just train to 100 steps!**
- Evaluate at 50, 70, 90, 100
- Best checkpoint might be step 70

---

## Comparison: Before vs. After Training

### Before (Baseline Model)
```
Prompt: "Community health grant proposal"

Output: "This proposal seeks funding for a community
health initiative. We believe this program will help
many people in our area. Our organization has experience
in health programs..."

Score: 54/100
- Vague impact claims
- No specific data
- Informal tone ("we believe")
```

### After (Trained Model - Step 100)
```
Output: "This proposal requests $150,000 for a community
health screening program serving 2,500 residents annually.
Based on pilot data from 2023, participants showed 23%
improvement in early disease detection. The program
employs evidence-based screening protocols..."

Score: 71/100
- Specific numbers ($150K, 2,500 residents)
- Quantified impact (23% improvement)
- Professional tone
```

**Improvement: +17 points (+31%)**

---

## Key Takeaways

1. **RL enables learning from rubric feedback**
   - No need for hand-written examples
   - Model learns what improves scores

2. **Iterative revision is the core mechanism**
   - Model sees draft → score → revises
   - Learns effective revision patterns

3. **Reward function drives behavior**
   - Positive reward for improvement
   - Model optimizes for higher scores

4. **Training is guided but exploratory**
   - Rubric defines "good"
   - Model finds its own path to "good"

5. **Evaluation is critical**
   - Training scores ≠ generalization
   - Always evaluate on held-out data

---

## Resources

### Code Files
- `train.py` - Main training script
- `writing_env.py` - RL environment implementation
- `evaluate.py` - Post-training evaluation
- `prompts.py` - Prompts and system instructions

### Documentation
- `EVALUATION_GUIDE.md` - How to evaluate trained models
- `README.md` - Project overview

### Data Files
- `prompts/writing_prompts.txt` - Training prompts
- `rubrics/example_rubric.json` - Example rubric

### External Resources
- Tinker Cookbook: RL training framework
- Anthropic API: Claude 4.5 evaluator
- PPO paper: Schulman et al., 2017
- LoRA paper: Hu et al., 2021

---

## Questions?

**Common Questions**:

1. **Why not just fine-tune on examples?**
   → RL lets model explore and optimize for rubric directly

2. **Why Claude 4.5 as evaluator?**
   → Strong reasoning, reliable rubric interpretation

3. **Why LoRA instead of full fine-tuning?**
   → 98% fewer parameters, much faster, more stable

4. **How do I know if my model is good?**
   → Use evaluation script to compare against baseline

5. **Can I use different models?**
   → Yes! Change `model_name` in config

6. **Can I use different rubrics?**
   → Yes! Just create new rubric JSON file

---

## Thank You!

### Ready to Train?

```bash
# 1. Set up your rubric and prompts
# 2. Run training
python train.py

# 3. Evaluate results
python evaluate.py --checkpoint_path .../step_100 --compare_baseline

# 4. Deploy best checkpoint
```

**Happy Training!** 🚀

---

## Appendix: Math Details

### PPO Loss Function (Detailed)

```
L^CLIP(θ) = E_t[min(
    r_t(θ) × A_t,
    clip(r_t(θ), 1-ε, 1+ε) × A_t
)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
  A_t = advantage (how good was action vs. expected)
  ε = 0.2 (clip range)

Total Loss = -L^CLIP + c_1 × L^VF + c_2 × S[π_θ] + c_3 × KL[π_θ || π_ref]

where:
  L^VF = value function loss
  S = entropy bonus (exploration)
  KL = KL divergence penalty
```

---

## Appendix: Reward Calculation Examples

### Example 1: Steady Improvement
```
Turn 0: score=55 → reward=5.5
Turn 1: score=62 → reward=0.7   (improvement: 7 points)
Turn 2: score=68 → reward=0.6   (improvement: 6 points)
Turn 3: score=71 → reward=0.3   (improvement: 3 points)
Turn 4: score=71 → reward=0.0   (no change)

Episode ends (reward ≤ 0)
Total episode reward: 5.5 + 0.7 + 0.6 + 0.3 = 7.1
```

### Example 2: Regression
```
Turn 0: score=58 → reward=5.8
Turn 1: score=65 → reward=0.7   (improvement: 7 points)
Turn 2: score=61 → reward=-0.4  (regression: -4 points)

Episode ends (negative reward)
Total episode reward: 5.8 + 0.7 - 0.4 = 6.1
```

---

## Appendix: Hyperparameter Tuning

### Learning Rate
- **Too high** (>1e-4): Unstable, wild score swings
- **Good** (3e-5 to 1e-4): Steady improvement
- **Too low** (<1e-5): Very slow learning

### LoRA Rank
- **Too low** (<16): Not enough capacity
- **Good** (32-64): Balanced
- **Too high** (>128): Overfitting, slower training

### KL Penalty
- **Too low** (<0.005): Model drifts too far
- **Good** (0.01-0.05): Stays grounded
- **Too high** (>0.1): Can't improve enough

### Batch Size (groups × group_size)
- **Too small** (<16): High variance
- **Good** (32-64): Stable gradients
- **Too large** (>128): Slower iteration

