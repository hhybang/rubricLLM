# Post-Training Model Evaluation Guide

This guide explains how to evaluate your RL fine-tuned model after training to determine if it's any good.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the Evaluation System](#understanding-the-evaluation-system)
3. [Usage Examples](#usage-examples)
4. [Interpreting Results](#interpreting-results)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Evaluation (Trained Model Only)

Evaluate your trained checkpoint:

```bash
cd rl
python evaluate.py --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/YOUR_RUN_NAME/step_100
```

### Evaluation with Baseline Comparison

Compare your trained model against the baseline (pre-training) model:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/YOUR_RUN_NAME/step_100 \
    --compare_baseline
```

This is the **recommended approach** as it shows whether your model actually improved!

---

## Understanding the Evaluation System

### What Gets Evaluated?

The evaluation script:

1. **Loads your trained checkpoint** (or baseline model)
2. **Generates responses** for each evaluation prompt
3. **Scores each response** using Claude Sonnet 4.5 + your rubric
4. **Computes aggregate metrics** across all prompts
5. **Compares against baseline** (if requested)
6. **Generates comprehensive reports** with detailed analysis

### Evaluation Metrics

#### Overall Metrics
- **Mean Score**: Average score across all prompts (0-100)
- **Median Score**: Middle score when sorted
- **Standard Deviation**: How much scores vary
- **Min/Max Score**: Score range
- **Improvement**: Trained vs. baseline difference

#### Per-Criterion Metrics
For each rubric criterion (e.g., "Evidence-Based Impact"):
- Mean weighted score
- Standard deviation
- Min/max range
- Improvement over baseline

### Scoring System

Scores are based on your rubric's achievement levels:
- **Exemplary (100%)**: Fully meets your vision
- **Proficient (75%)**: Meets core requirements
- **Developing (50%)**: Shows understanding, needs work
- **Beginning (25%)**: Misses key elements

**Overall Score = Σ(achievement_level_percentage × criterion_weight / 100)**

---

## Usage Examples

### Example 1: Basic Evaluation

Evaluate a checkpoint on all prompts:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/my_model/step_100
```

**Output**:
```
Mean Score: 67.3/100 (± 8.2)
Score Range: [52.0, 81.5]

Full report saved to: evaluation_results/step_100_20250114_143022
```

### Example 2: Baseline Comparison

The most informative evaluation - compare trained vs. baseline:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/my_model/step_100 \
    --compare_baseline
```

**Output**:
```
Mean Score: 67.3/100 (± 8.2)
Score Range: [52.0, 81.5]
Baseline Mean: 58.4/100
Improvement: +8.9 points (+15.2%)

✅ Model shows improvement over baseline
```

### Example 3: Quick Evaluation (Fewer Prompts)

Evaluate on just 3 prompts for faster feedback:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/my_model/step_100 \
    --num_prompts 3
```

### Example 4: Multiple Samples Per Prompt

Generate 3 samples per prompt to measure consistency:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/my_model/step_100 \
    --num_samples_per_prompt 3
```

This helps assess:
- **Low std deviation**: Model is consistent
- **High std deviation**: Model is unpredictable

### Example 5: Custom Evaluation Data

Use different prompts/rubric than training:

```bash
python evaluate.py \
    --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/my_model/step_100 \
    --prompts_file prompts/held_out_test_set.txt \
    --rubric_file rubrics/eval_rubric.json
```

### Example 6: Baseline-Only Evaluation

Evaluate the baseline model without any checkpoint:

```bash
python evaluate.py \
    --model_name Qwen/Qwen3-235B-A22B-Instruct-2507
```

Useful for establishing baseline performance before training.

### Example 7: Compare Multiple Checkpoints

Evaluate different training steps to find the best checkpoint:

```bash
# Step 50
python evaluate.py --checkpoint_path .../step_50 --output_dir results/step_50

# Step 100
python evaluate.py --checkpoint_path .../step_100 --output_dir results/step_100

# Step 150
python evaluate.py --checkpoint_path .../step_150 --output_dir results/step_150
```

Then compare `results/*/evaluation_summary.json` to find the best step.

---

## Interpreting Results

### Output Files

After evaluation, you'll get these files in the output directory:

```
evaluation_results/
└── step_100_20250114_143022/
    ├── evaluation_summary.json      # Aggregate metrics (JSON)
    ├── evaluation_details.json      # Full results with generations
    └── README.md                    # Human-readable report
```

### Reading the Summary Report

The `README.md` contains:

1. **Overall Performance**
   ```markdown
   ## Overall Performance
   - Mean Score: 67.30/100
   - Median Score: 66.50/100
   - Std Dev: 8.21
   - Min Score: 52.00/100
   - Max Score: 81.50/100
   ```

2. **Baseline Comparison** (if enabled)
   ```markdown
   ## Baseline Comparison
   - Baseline Mean Score: 58.40/100
   - Trained Mean Score: 67.30/100
   - Improvement: +8.90 points (+15.2%)

   ✅ Model shows improvement over baseline
   ```

3. **Per-Criterion Performance**
   ```markdown
   ### Evidence-Based Impact
   - Mean: 23.5
   - Improvement over baseline: +3.2

   ### Professional Tone & Accessibility
   - Mean: 16.8
   - Improvement over baseline: +2.1
   ```

4. **Sample Results** (best and worst examples)

### How to Determine if Your Model is Good

Ask these questions:

#### 1. **Did it improve over baseline?**

✅ **Good signs**:
- Improvement > 5 points: Meaningful improvement
- Improvement > 10 points: Strong improvement
- All criteria show improvement

⚠️ **Warning signs**:
- Improvement < 2 points: Marginal improvement
- Negative improvement: Model got worse (overtraining?)
- Only some criteria improved: Unbalanced learning

#### 2. **Are the scores consistent?**

✅ **Good signs**:
- Low std deviation (< 10): Consistent performance
- Narrow score range: Reliable across different prompts

⚠️ **Warning signs**:
- High std deviation (> 15): Unpredictable
- Wide score range: Works well on some prompts, poorly on others

#### 3. **Are the scores high enough?**

Score interpretation bands:
- **80-100**: Exceptional - ready for production
- **70-79**: Strong - good for most use cases
- **60-69**: Solid - acceptable with human review
- **50-59**: Developing - needs more training
- **< 50**: Emerging - significant issues

#### 4. **Did the right criteria improve?**

Check per-criterion improvements:
- High-weight criteria should show strong improvement
- All criteria should show some improvement
- If a criterion got worse, investigate why

#### 5. **Do the generations look good?**

Read the sample outputs in `README.md`:
- Does the best output meet your standards?
- Is the worst output acceptable?
- Look for failure patterns

### Example: Good Model

```
Mean Score: 73.2/100 (± 6.5)
Baseline Mean: 59.1/100
Improvement: +14.1 points (+23.9%)

Per-Criterion Improvements:
- Evidence-Based Impact: +5.2 ✅
- Professional Tone: +3.8 ✅
- Logical Structure: +2.9 ✅
- Conciseness: +2.2 ✅
```

**Assessment**: Strong improvement across all criteria, consistent performance, scores in "Strong" range. ✅ Model is good!

### Example: Problematic Model

```
Mean Score: 61.3/100 (± 18.2)
Baseline Mean: 58.4/100
Improvement: +2.9 points (+5.0%)

Per-Criterion Improvements:
- Evidence-Based Impact: -1.2 ⚠️
- Professional Tone: +6.5 ✅
- Logical Structure: -0.3 ⚠️
- Conciseness: -2.1 ⚠️
```

**Assessment**: Marginal overall improvement, high variance, most criteria declined. ⚠️ Model needs more training or different hyperparameters.

---

## Advanced Usage

### Full Configuration Options

```bash
python evaluate.py \
    # Model configuration
    --model_name Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --checkpoint_path /path/to/checkpoint \
    --renderer_name qwen \

    # Evaluation data
    --rubric_file rubrics/example_rubric.json \
    --prompts_file prompts/writing_prompts.txt \
    --num_prompts 10 \

    # Generation settings
    --max_tokens 2048 \
    --temperature 0.7 \
    --num_samples_per_prompt 1 \

    # Evaluator settings
    --evaluator_model claude-sonnet-4-5 \
    --evaluator_api_key your_api_key \

    # Baseline comparison
    --compare_baseline \
    --baseline_model_name Qwen/Qwen3-235B-A22B-Instruct-2507 \

    # Output settings
    --output_dir custom_results \
    --save_generations true \
    --verbose true
```

### Creating Test/Validation Splits

To properly evaluate on held-out data:

1. **Split your prompts**:
   ```bash
   # Create train/test split
   head -8 prompts/writing_prompts.txt > prompts/train_prompts.txt
   tail -2 prompts/writing_prompts.txt > prompts/test_prompts.txt
   ```

2. **Train on train set**:
   ```bash
   python train.py --prompts_file prompts/train_prompts.txt
   ```

3. **Evaluate on test set**:
   ```bash
   python evaluate.py \
       --checkpoint_path .../step_100 \
       --prompts_file prompts/test_prompts.txt \
       --compare_baseline
   ```

This gives you a **true out-of-distribution evaluation**.

### Tracking Evaluation Across Training

Evaluate at multiple checkpoints to track learning:

```bash
#!/bin/bash
# evaluate_all_checkpoints.sh

CHECKPOINT_DIR="/tmp/tinker-examples/rubric_writing_rl/my_run"

for step in 10 20 30 40 50 60 70 80 90 100; do
    echo "Evaluating step $step..."
    python evaluate.py \
        --checkpoint_path "$CHECKPOINT_DIR/step_$step" \
        --output_dir "eval_tracking/step_$step" \
        --num_prompts 5  # Quick eval
done

# Compare results
python -c "
import json
from pathlib import Path

for step in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    summary = json.load(open(f'eval_tracking/step_{step}/evaluation_summary.json'))
    print(f'Step {step:3d}: {summary[\"mean_score\"]:.2f}')
"
```

This helps you:
- Identify the best checkpoint
- Detect overtraining (scores going down)
- Understand learning dynamics

### Using Different Temperature

Temperature affects generation randomness:

```bash
# More deterministic (recommended for evaluation)
python evaluate.py --checkpoint_path ... --temperature 0.3

# More creative (less consistent)
python evaluate.py --checkpoint_path ... --temperature 0.9
```

Lower temperature (0.3-0.5) is better for evaluation consistency.

---

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY must be set"

**Solution**: Set your API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
python evaluate.py ...
```

Or pass it directly:
```bash
python evaluate.py --evaluator_api_key "your-api-key-here" ...
```

### Issue: Checkpoint not found

**Error**: `FileNotFoundError: /path/to/checkpoint`

**Solution**:
1. Check your checkpoint path:
   ```bash
   ls /tmp/tinker-examples/rubric_writing_rl/
   ```

2. Use the correct run name and step:
   ```bash
   ls /tmp/tinker-examples/rubric_writing_rl/YOUR_RUN_NAME/
   # Should show: step_10, step_20, step_30, etc.
   ```

3. Use full absolute path:
   ```bash
   python evaluate.py --checkpoint_path /tmp/tinker-examples/rubric_writing_rl/rubric_writing_rl_Qwen-Qwen3-235B-A22B-Instruct-2507_gp8_gs4_lr3e-05_rank32_2025-01-14-10-30/step_100
   ```

### Issue: Evaluation is too slow

**Solutions**:

1. **Reduce number of prompts**:
   ```bash
   python evaluate.py --num_prompts 3 ...
   ```

2. **Disable baseline comparison** (cuts time in half):
   ```bash
   python evaluate.py --checkpoint_path ... # No --compare_baseline
   ```

3. **Use fewer samples per prompt**:
   ```bash
   python evaluate.py --num_samples_per_prompt 1 ...
   ```

### Issue: Scores seem wrong

**Debugging steps**:

1. **Check the generated outputs**:
   - Open `evaluation_details.json`
   - Look at `generated_text` fields
   - Are they reasonable responses to the prompts?

2. **Check the evaluator scores**:
   - Look at `score_details` in `evaluation_details.json`
   - Read the `overall_assessment` and `evidence_summary`
   - Does the evaluator's reasoning make sense?

3. **Verify your rubric**:
   - Open your rubric file
   - Are the achievement levels well-defined?
   - Are the weights appropriate?

4. **Test the baseline**:
   ```bash
   python evaluate.py --model_name YOUR_MODEL  # No checkpoint
   ```
   - Does the baseline score seem reasonable?
   - If baseline is very high/low, rubric might need adjustment

### Issue: Out of memory

**Solutions**:

1. **Use smaller max_tokens**:
   ```bash
   python evaluate.py --max_tokens 1024 ...
   ```

2. **Evaluate fewer prompts at a time**:
   ```bash
   python evaluate.py --num_prompts 1 ...
   ```

3. **Use a smaller model** (if checkpoint allows)

---

## Best Practices

### 1. Always Compare Against Baseline

Don't just evaluate the trained model - compare it!

```bash
# Good
python evaluate.py --checkpoint_path ... --compare_baseline

# Not recommended
python evaluate.py --checkpoint_path ...
```

### 2. Use Held-Out Test Data

Don't evaluate on training prompts:

```bash
# Good: Use separate test prompts
python evaluate.py --prompts_file prompts/test_prompts.txt ...

# Risky: Evaluate on training data (may overestimate performance)
python evaluate.py --prompts_file prompts/writing_prompts.txt ...
```

### 3. Evaluate Multiple Checkpoints

Don't just evaluate the final checkpoint:

```bash
# Evaluate several steps
python evaluate.py --checkpoint_path .../step_50 ...
python evaluate.py --checkpoint_path .../step_100 ...
python evaluate.py --checkpoint_path .../step_150 ...
```

The final step might not be the best due to overtraining.

### 4. Check Consistency

Use multiple samples to measure reliability:

```bash
python evaluate.py --num_samples_per_prompt 3 ...
```

Low variance = consistent model.

### 5. Read the Actual Outputs

Don't just look at numbers - read the generated text:

```bash
# Open and read the human-readable report
cat evaluation_results/*/README.md
```

Numbers can be misleading; actual text quality matters.

---

## Quick Reference

### Common Commands

```bash
# Basic evaluation
python evaluate.py --checkpoint_path PATH

# With baseline comparison (recommended)
python evaluate.py --checkpoint_path PATH --compare_baseline

# Quick eval (3 prompts)
python evaluate.py --checkpoint_path PATH --num_prompts 3

# Multiple samples for consistency check
python evaluate.py --checkpoint_path PATH --num_samples_per_prompt 3

# Baseline only
python evaluate.py --model_name MODEL_NAME

# Custom output location
python evaluate.py --checkpoint_path PATH --output_dir custom_results
```

### File Locations

- **Training checkpoints**: `/tmp/tinker-examples/rubric_writing_rl/RUN_NAME/step_N`
- **Evaluation results**: `evaluation_results/CHECKPOINT_NAME_TIMESTAMP/`
- **Rubrics**: `rubrics/example_rubric.json`
- **Prompts**: `prompts/writing_prompts.txt`

### Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key"  # Required for scoring
```

---

## Next Steps

After evaluation:

1. **If model is good** (improvement > 5 points, scores > 70):
   - Use this checkpoint for production
   - Consider fine-tuning for more epochs
   - Evaluate on more diverse prompts

2. **If model is marginal** (improvement < 5 points):
   - Train for more steps
   - Adjust hyperparameters (learning rate, LoRA rank)
   - Check if rubric needs refinement
   - Verify training data quality

3. **If model got worse**:
   - Use an earlier checkpoint (e.g., step 50 instead of 100)
   - Reduce learning rate
   - Check for overtraining

4. **If model is inconsistent** (high std deviation):
   - Generate more training episodes
   - Use temperature annealing
   - Increase LoRA rank for more capacity

---

## Support

For issues with this evaluation script:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review the [Training Guide](train.py) for context
3. Check the example rubric and prompts for formatting

Happy evaluating! 🎯
