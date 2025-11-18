import asyncio
import json
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
from writing_env import WritingRLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Configuration for training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    lora_rank: int = 32

    # Environment configuration
    rubric_file: str = "rubrics/example_rubric.json"
    prompts_file: str = "prompts/writing_prompts.txt"
    max_turns: int = 5

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 8
    num_substeps: int = 1
    learning_rate: float = 3e-5
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.005  # Reduced to allow more exploration

    # Evaluator configuration
    evaluator_model: str = "claude-sonnet-4-5"
    evaluator_api_key: str | None = None

    # Logging configuration
    eval_every: int = 5
    save_every: int = 10
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def load_rubric(rubric_file: str) -> list[dict]:
    """Load rubric from JSON file."""
    with open(rubric_file, 'r') as f:
        data = json.load(f)

    # Handle different rubric formats
    if isinstance(data, dict) and "rubric" in data:
        return data["rubric"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid rubric format in {rubric_file}")


def load_prompts(prompts_file: str) -> list[str]:
    """
    Load writing prompts from file.
    Each line is a separate prompt.
    """
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")

    return prompts


async def cli_main(cli_config: CLIConfig):
    """Main training function."""

    # Generate run name
    model_name_short = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rubric_writing_rl_{model_name_short}_gp{cli_config.groups_per_batch}_"
        f"gs{cli_config.group_size}_lr{cli_config.learning_rate}_"
        f"rank{cli_config.lora_rank}_{date_and_time}"
    )

    # Determine log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rubric_writing_rl/{run_name}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Load rubric and prompts
    logger.info(f"Loading rubric from {cli_config.rubric_file}")
    rubric = load_rubric(cli_config.rubric_file)
    logger.info(f"Loaded rubric with {len(rubric)} criteria")

    logger.info(f"Loading prompts from {cli_config.prompts_file}")
    prompts = load_prompts(cli_config.prompts_file)
    logger.info(f"Loaded {len(prompts)} writing prompts")

    # Create dataset builder
    dataset_builder = WritingRLDatasetBuilder(
        prompts=prompts,
        rubric=rubric,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        max_turns=cli_config.max_turns,
        model_name=cli_config.model_name,  # Pass model name for renderer
        evaluator_api_key=cli_config.evaluator_api_key,
        evaluator_model=cli_config.evaluator_model,
    )

    # Create training configuration
    cfg = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        lora_rank=cli_config.lora_rank,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name or run_name,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        stream_minibatch_config=None,
    )

    logger.info("Starting training...")
    await train.main(cfg)
    logger.info("Training complete!")


def main():
    """Entry point for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    main()
