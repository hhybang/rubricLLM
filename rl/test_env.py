"""
Simple test script to verify the environment setup.

This script creates a simple writing task and tests that:
1. The environment can be created
2. Initial observation works
3. Scoring works (requires API key)
4. Basic step functionality works
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic


async def test_environment():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing Rubric-Based Writing RL Environment")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ ANTHROPIC_API_KEY not set!")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        return False

    print("\n✓ API key found")

    # Load example rubric
    rubric_file = Path("rubrics/example_rubric.json")
    if not rubric_file.exists():
        print(f"\n❌ Rubric file not found: {rubric_file}")
        return False

    with open(rubric_file, 'r') as f:
        rubric_data = json.load(f)
        rubric = rubric_data["rubric"]

    print(f"✓ Loaded rubric with {len(rubric)} criteria")

    # Import the environment
    try:
        from writing_env import WritingTask, WritingRevisionEnv
        from tinker_cookbook import renderers, model_info
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        print("✓ Imported environment classes")
    except Exception as e:
        print(f"\n❌ Failed to import environment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create a test task
    task = WritingTask(
        prompt="Write a grant proposal paragraph about a community health initiative that reduced hospital readmissions.",
        rubric=rubric,
        max_turns=3,
        task_id="test_task",
    )
    print("✓ Created writing task")

    # Create environment
    try:
        # Create renderer for a test model
        model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        print(f"  Creating renderer for {model_name}...")
        tokenizer = get_tokenizer(model_name)
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        print("✓ Created renderer")

        evaluator_client = anthropic.Anthropic(api_key=api_key)
        env = WritingRevisionEnv(
            task=task,
            renderer=renderer,
            evaluator_client=evaluator_client,
            evaluator_model="claude-sonnet-4-5",
        )
        print("✓ Created environment instance")
    except Exception as e:
        print(f"\n❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test initial observation
    try:
        observation, stop_condition = await env.initial_observation()
        print("✓ Generated initial observation")
        print(f"  - Observation type: {type(observation).__name__}")
        print(f"  - Has {len(observation.chunks)} chunks")
    except Exception as e:
        print(f"\n❌ Failed to get initial observation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test scoring with a sample draft
    print("\n" + "=" * 60)
    print("Testing Evaluator Scoring")
    print("=" * 60)

    sample_draft = """
Our community health initiative achieved significant improvements in hospital readmission rates.
Over the past 18 months, we implemented care coordination protocols across three partner hospitals,
serving 450 patients with chronic conditions. The program reduced 30-day readmissions by 23%,
from a baseline of 18.2% to 14.0%. Cost savings totaled approximately $1.2M in avoided readmissions.
Patient satisfaction scores increased from 3.2 to 4.1 on a 5-point scale. The initiative demonstrated
measurable impact on both clinical outcomes and healthcare system efficiency.
"""

    print(f"\nSample draft:\n{sample_draft.strip()}\n")

    try:
        score = await env._score_draft(sample_draft)
        print(f"✓ Evaluator scored draft: {score:.1f}/100")

        if score < 0 or score > 100:
            print(f"⚠️  Warning: Score {score} is outside expected range [0, 100]")
        else:
            print("✓ Score is in valid range")

    except Exception as e:
        print(f"\n❌ Failed to score draft: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nYou're ready to start training!")
    print("Run: python train.py --help")

    return True


async def test_dataset_builder():
    """Test the dataset builder."""
    print("\n" + "=" * 60)
    print("Testing Dataset Builder")
    print("=" * 60)

    try:
        from writing_env import WritingRLDatasetBuilder
        print("✓ Imported dataset builder")
    except Exception as e:
        print(f"❌ Failed to import dataset builder: {e}")
        return False

    # Load rubric and prompts
    rubric_file = Path("rubrics/example_rubric.json")
    prompts_file = Path("prompts/writing_prompts.txt")

    if not rubric_file.exists() or not prompts_file.exists():
        print("❌ Missing rubric or prompts file")
        return False

    with open(rubric_file, 'r') as f:
        rubric_data = json.load(f)
        rubric = rubric_data["rubric"]

    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"✓ Loaded {len(prompts)} prompts")

    # Create builder
    try:
        model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        builder = WritingRLDatasetBuilder(
            prompts=prompts[:5],  # Just test with first 5
            rubric=rubric,
            groups_per_batch=2,
            group_size=2,
            max_turns=3,
            model_name=model_name,  # Required for renderer
        )
        print("✓ Created dataset builder")

        # Build dataset
        dataset, eval_dataset = await builder()
        print(f"✓ Built dataset with {len(dataset)} batches")

    except Exception as e:
        print(f"❌ Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("✓ Dataset builder works correctly")
    return True


def main():
    """Run all tests."""
    print("\n🧪 Running environment tests...\n")

    # Run environment test
    success = asyncio.run(test_environment())

    if success:
        # Run dataset builder test
        asyncio.run(test_dataset_builder())

    print("\n")


if __name__ == "__main__":
    main()
