"""
Post-Training Evaluation Script for RL Fine-Tuned Models

This script evaluates a trained model checkpoint by:
1. Loading the checkpoint (or baseline model)
2. Generating responses for evaluation prompts
3. Scoring outputs using Claude 4.5 + rubric
4. Comparing against baseline performance
5. Generating comprehensive evaluation reports

Usage:
    # Evaluate a trained checkpoint
    python evaluate.py --checkpoint_path /path/to/checkpoint

    # Evaluate baseline model (no checkpoint)
    python evaluate.py --model_name Qwen/Qwen3-235B-A22B-Instruct-2507

    # Compare trained model against baseline
    python evaluate.py --checkpoint_path /path/to/checkpoint --compare_baseline

    # Use custom evaluation prompts/rubric
    python evaluate.py --checkpoint_path /path/to/checkpoint \
                       --prompts_file custom_prompts.txt \
                       --rubric_file custom_rubric.json
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Import prompts and rubric loading utilities
from prompts import RUBRIC_SCORING_PROMPT
from train import load_prompts, load_rubric

logger = logging.getLogger(__name__)


@chz.chz
class EvalConfig:
    """Configuration for model evaluation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    checkpoint_path: str | None = None  # Path to trained checkpoint to evaluate
    renderer_name: str | None = None  # Override renderer if needed

    # Evaluation data
    rubric_file: str = "rubrics/example_rubric.json"
    prompts_file: str = "prompts/writing_prompts.txt"
    num_prompts: int | None = None  # Limit number of prompts (None = use all)

    # Generation configuration
    max_tokens: int = 2048
    temperature: float = 0.7
    num_samples_per_prompt: int = 1  # Generate N samples per prompt

    # Evaluator configuration
    evaluator_model: str = "claude-sonnet-4-5"
    evaluator_api_key: str | None = None

    # Baseline comparison
    compare_baseline: bool = False  # Compare against baseline model
    baseline_model_name: str | None = None  # Override baseline model name

    # Output configuration
    output_dir: str | None = None  # Where to save results
    save_generations: bool = True  # Save generated texts to file
    verbose: bool = True  # Print detailed progress


@dataclass
class GenerationResult:
    """Result from generating text for a single prompt."""

    prompt: str
    generated_text: str
    score: float
    score_details: dict[str, Any]
    generation_time: float  # seconds


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    model_name: str
    checkpoint_path: str | None
    timestamp: str
    num_prompts: int
    num_samples: int

    # Aggregate metrics
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float

    # Per-criterion metrics
    criterion_scores: dict[str, dict[str, float]]  # criterion_name -> {mean, std, etc.}

    # Individual results
    results: list[GenerationResult]

    # Baseline comparison (if available)
    baseline_comparison: dict[str, Any] | None = None


class ModelEvaluator:
    """Evaluates a model checkpoint using rubric-based scoring."""

    def __init__(
        self,
        model_name: str,
        rubric: list[dict],
        evaluator_client: anthropic.Anthropic,
        evaluator_model: str = "claude-sonnet-4-5",
        checkpoint_path: str | None = None,
        renderer: renderers.Renderer | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.rubric = rubric
        self.evaluator_client = evaluator_client
        self.evaluator_model = evaluator_model
        self.checkpoint_path = checkpoint_path
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create renderer if not provided
        if renderer is None:
            tokenizer = get_tokenizer(model_name)
            renderer_name = model_info.get_recommended_renderer_name(model_name)
            self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        else:
            self.renderer = renderer

        # Initialize model (will be loaded lazily)
        self._model = None

    async def _load_model(self):
        """Load the model (checkpoint or baseline)."""
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        if self.checkpoint_path:
            logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            # Load checkpoint using tinker
            self._model = await tinker.load_model_from_checkpoint(
                self.checkpoint_path,
                model_name=self.model_name,
            )
        else:
            logger.info("Loading baseline model (no checkpoint)")
            # Load base model
            self._model = await tinker.load_model(self.model_name)

        logger.info("Model loaded successfully")

    async def generate(self, prompt: str) -> str:
        """Generate text for a given prompt."""
        await self._load_model()

        # Build messages for generation
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Use renderer to build generation prompt
        model_input = self.renderer.build_generation_prompt(messages)

        # Generate using the model
        import time
        start_time = time.time()

        response = await self._model.generate(
            model_input,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop_sequences=self.renderer.get_stop_sequences(),
        )

        generation_time = time.time() - start_time

        # Parse response
        message, parse_success = self.renderer.parse_response(response)
        generated_text = message["content"]

        return generated_text, generation_time

    async def score_text(self, text: str) -> tuple[float, dict[str, Any]]:
        """Score a generated text using the rubric."""
        # Format the rubric for the evaluator
        rubric_text = json.dumps(self.rubric, indent=2)

        # Create evaluation prompt
        eval_prompt = f"""
Please evaluate the following draft against the provided rubric.

## Draft to Evaluate:
{text}

## Rubric:
{rubric_text}

{RUBRIC_SCORING_PROMPT}
"""

        try:
            # Call Claude 4.5 to score the draft
            response = self.evaluator_client.messages.create(
                model=self.evaluator_model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": eval_prompt}
                ],
            )

            # Extract score from response
            content = response.content[0].text
            score, details = self._parse_evaluation(content)
            return score, details

        except Exception as e:
            logger.error(f"Error scoring text: {e}")
            # Return neutral score on error
            return 50.0, {"error": str(e)}

    def _parse_evaluation(self, evaluation_text: str) -> tuple[float, dict[str, Any]]:
        """
        Parse the evaluation response to extract score and details.
        Returns (overall_score, score_details_dict)
        """
        import re

        try:
            # Find JSON block in the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', evaluation_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)

                overall_score = float(data.get("overall_score", 50.0))

                # Extract per-criterion scores
                criterion_scores = {}
                for criterion in data.get("criteria_scores", []):
                    criterion_scores[criterion["name"]] = {
                        "achievement_level": criterion.get("achievement_level", "Unknown"),
                        "level_percentage": criterion.get("level_percentage", 50),
                        "weighted_score": criterion.get("weighted_score", 0),
                        "evidence": criterion.get("evidence_summary", ""),
                    }

                return overall_score, {
                    "overall_score": overall_score,
                    "score_interpretation": data.get("score_interpretation", ""),
                    "criterion_scores": criterion_scores,
                    "overall_assessment": data.get("overall_assessment", ""),
                }

            # Fallback: look for "overall_score" directly
            score_match = re.search(r'"overall_score"\s*:\s*([0-9.]+)', evaluation_text)
            if score_match:
                score = float(score_match.group(1))
                return score, {"overall_score": score, "raw_response": evaluation_text}

            # If no score found, return neutral
            logger.warning("Could not parse evaluation, using neutral score")
            return 50.0, {"raw_response": evaluation_text}

        except Exception as e:
            logger.error(f"Error parsing evaluation: {e}")
            return 50.0, {"error": str(e)}

    async def evaluate_prompts(
        self,
        prompts: list[str],
        num_samples_per_prompt: int = 1,
        verbose: bool = True,
    ) -> list[GenerationResult]:
        """Evaluate model on a list of prompts."""
        results = []

        for i, prompt in enumerate(prompts):
            if verbose:
                logger.info(f"Evaluating prompt {i+1}/{len(prompts)}")
                logger.info(f"Prompt: {prompt[:100]}...")

            for sample_idx in range(num_samples_per_prompt):
                if num_samples_per_prompt > 1 and verbose:
                    logger.info(f"  Sample {sample_idx+1}/{num_samples_per_prompt}")

                # Generate text
                generated_text, gen_time = await self.generate(prompt)

                if verbose:
                    logger.info(f"  Generated {len(generated_text)} characters in {gen_time:.2f}s")

                # Score the generated text
                score, details = await self.score_text(generated_text)

                if verbose:
                    logger.info(f"  Score: {score:.1f}/100")

                result = GenerationResult(
                    prompt=prompt,
                    generated_text=generated_text,
                    score=score,
                    score_details=details,
                    generation_time=gen_time,
                )
                results.append(result)

        return results


def compute_evaluation_metrics(
    results: list[GenerationResult],
    rubric: list[dict],
) -> dict[str, Any]:
    """Compute aggregate metrics from evaluation results."""
    import statistics

    scores = [r.score for r in results]

    # Overall metrics
    metrics = {
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
    }

    # Per-criterion metrics
    criterion_metrics = {}
    for criterion in rubric:
        criterion_name = criterion["name"]
        criterion_scores = []

        for result in results:
            if "criterion_scores" in result.score_details:
                if criterion_name in result.score_details["criterion_scores"]:
                    weighted_score = result.score_details["criterion_scores"][criterion_name]["weighted_score"]
                    criterion_scores.append(weighted_score)

        if criterion_scores:
            criterion_metrics[criterion_name] = {
                "mean": statistics.mean(criterion_scores),
                "std": statistics.stdev(criterion_scores) if len(criterion_scores) > 1 else 0.0,
                "min": min(criterion_scores),
                "max": max(criterion_scores),
            }

    metrics["criterion_scores"] = criterion_metrics

    return metrics


def create_evaluation_report(
    model_name: str,
    checkpoint_path: str | None,
    results: list[GenerationResult],
    rubric: list[dict],
    baseline_results: list[GenerationResult] | None = None,
) -> EvaluationReport:
    """Create a comprehensive evaluation report."""
    metrics = compute_evaluation_metrics(results, rubric)

    # Baseline comparison
    baseline_comparison = None
    if baseline_results:
        baseline_metrics = compute_evaluation_metrics(baseline_results, rubric)

        baseline_comparison = {
            "baseline_mean_score": baseline_metrics["mean_score"],
            "trained_mean_score": metrics["mean_score"],
            "improvement": metrics["mean_score"] - baseline_metrics["mean_score"],
            "improvement_percentage": (
                (metrics["mean_score"] - baseline_metrics["mean_score"]) / baseline_metrics["mean_score"] * 100
            ),
            "baseline_criterion_scores": baseline_metrics["criterion_scores"],
            "trained_criterion_scores": metrics["criterion_scores"],
        }

    report = EvaluationReport(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        timestamp=datetime.now().isoformat(),
        num_prompts=len(set(r.prompt for r in results)),
        num_samples=len(results),
        mean_score=metrics["mean_score"],
        median_score=metrics["median_score"],
        std_score=metrics["std_score"],
        min_score=metrics["min_score"],
        max_score=metrics["max_score"],
        criterion_scores=metrics["criterion_scores"],
        results=results,
        baseline_comparison=baseline_comparison,
    )

    return report


def save_evaluation_report(report: EvaluationReport, output_dir: str, save_generations: bool = True):
    """Save evaluation report to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary report
    summary_file = output_path / "evaluation_summary.json"
    summary_data = {
        "model_name": report.model_name,
        "checkpoint_path": report.checkpoint_path,
        "timestamp": report.timestamp,
        "num_prompts": report.num_prompts,
        "num_samples": report.num_samples,
        "mean_score": report.mean_score,
        "median_score": report.median_score,
        "std_score": report.std_score,
        "min_score": report.min_score,
        "max_score": report.max_score,
        "criterion_scores": report.criterion_scores,
        "baseline_comparison": report.baseline_comparison,
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved evaluation summary to: {summary_file}")

    # Save detailed results with generations
    if save_generations:
        details_file = output_path / "evaluation_details.json"
        details_data = []

        for result in report.results:
            details_data.append({
                "prompt": result.prompt,
                "generated_text": result.generated_text,
                "score": result.score,
                "score_details": result.score_details,
                "generation_time": result.generation_time,
            })

        with open(details_file, 'w') as f:
            json.dump(details_data, f, indent=2)

        logger.info(f"Saved detailed results to: {details_file}")

    # Save human-readable report
    readme_file = output_path / "README.md"
    with open(readme_file, 'w') as f:
        f.write(format_report_markdown(report))

    logger.info(f"Saved human-readable report to: {readme_file}")


def format_report_markdown(report: EvaluationReport) -> str:
    """Format evaluation report as markdown."""
    lines = []

    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append(f"**Model**: {report.model_name}")
    lines.append(f"**Checkpoint**: {report.checkpoint_path or 'Baseline (no checkpoint)'}")
    lines.append(f"**Timestamp**: {report.timestamp}")
    lines.append(f"**Prompts Evaluated**: {report.num_prompts}")
    lines.append(f"**Total Samples**: {report.num_samples}")
    lines.append("")

    lines.append("## Overall Performance")
    lines.append("")
    lines.append(f"- **Mean Score**: {report.mean_score:.2f}/100")
    lines.append(f"- **Median Score**: {report.median_score:.2f}/100")
    lines.append(f"- **Std Dev**: {report.std_score:.2f}")
    lines.append(f"- **Min Score**: {report.min_score:.2f}/100")
    lines.append(f"- **Max Score**: {report.max_score:.2f}/100")
    lines.append("")

    # Baseline comparison
    if report.baseline_comparison:
        comp = report.baseline_comparison
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(f"- **Baseline Mean Score**: {comp['baseline_mean_score']:.2f}/100")
        lines.append(f"- **Trained Mean Score**: {comp['trained_mean_score']:.2f}/100")
        lines.append(f"- **Improvement**: {comp['improvement']:.2f} points ({comp['improvement_percentage']:.1f}%)")
        lines.append("")

        if comp['improvement'] > 0:
            lines.append("✅ **Model shows improvement over baseline**")
        else:
            lines.append("⚠️ **Model did not improve over baseline**")
        lines.append("")

    # Per-criterion performance
    if report.criterion_scores:
        lines.append("## Per-Criterion Performance")
        lines.append("")

        for criterion_name, scores in report.criterion_scores.items():
            lines.append(f"### {criterion_name}")
            lines.append(f"- Mean: {scores['mean']:.2f}")
            lines.append(f"- Std Dev: {scores['std']:.2f}")
            lines.append(f"- Range: [{scores['min']:.2f}, {scores['max']:.2f}]")

            # Show baseline comparison if available
            if report.baseline_comparison:
                baseline_criterion = report.baseline_comparison["baseline_criterion_scores"].get(criterion_name)
                if baseline_criterion:
                    improvement = scores['mean'] - baseline_criterion['mean']
                    lines.append(f"- Improvement over baseline: {improvement:+.2f}")

            lines.append("")

    # Sample results
    lines.append("## Sample Results")
    lines.append("")

    # Show best and worst
    sorted_results = sorted(report.results, key=lambda r: r.score, reverse=True)

    lines.append("### Best Performance")
    best = sorted_results[0]
    lines.append(f"**Score**: {best.score:.2f}/100")
    lines.append(f"**Prompt**: {best.prompt}")
    lines.append(f"**Generated Text** (first 500 chars):")
    lines.append("```")
    lines.append(best.generated_text[:500] + ("..." if len(best.generated_text) > 500 else ""))
    lines.append("```")
    lines.append("")

    lines.append("### Worst Performance")
    worst = sorted_results[-1]
    lines.append(f"**Score**: {worst.score:.2f}/100")
    lines.append(f"**Prompt**: {worst.prompt}")
    lines.append(f"**Generated Text** (first 500 chars):")
    lines.append("```")
    lines.append(worst.generated_text[:500] + ("..." if len(worst.generated_text) > 500 else ""))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


async def cli_main(config: EvalConfig):
    """Main evaluation function."""
    logger.info("Starting model evaluation...")

    # Load rubric and prompts
    logger.info(f"Loading rubric from {config.rubric_file}")
    rubric = load_rubric(config.rubric_file)
    logger.info(f"Loaded rubric with {len(rubric)} criteria")

    logger.info(f"Loading prompts from {config.prompts_file}")
    prompts = load_prompts(config.prompts_file)

    if config.num_prompts:
        prompts = prompts[:config.num_prompts]

    logger.info(f"Loaded {len(prompts)} prompts for evaluation")

    # Initialize evaluator client
    api_key = config.evaluator_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set or provided")

    evaluator_client = anthropic.Anthropic(api_key=api_key)

    # Create model evaluator
    logger.info("Initializing model evaluator...")
    evaluator = ModelEvaluator(
        model_name=config.model_name,
        rubric=rubric,
        evaluator_client=evaluator_client,
        evaluator_model=config.evaluator_model,
        checkpoint_path=config.checkpoint_path,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Evaluate trained model
    logger.info("=" * 80)
    logger.info("EVALUATING TRAINED MODEL")
    logger.info("=" * 80)
    results = await evaluator.evaluate_prompts(
        prompts,
        num_samples_per_prompt=config.num_samples_per_prompt,
        verbose=config.verbose,
    )

    # Baseline comparison (if requested)
    baseline_results = None
    if config.compare_baseline:
        logger.info("=" * 80)
        logger.info("EVALUATING BASELINE MODEL FOR COMPARISON")
        logger.info("=" * 80)

        baseline_model_name = config.baseline_model_name or config.model_name

        baseline_evaluator = ModelEvaluator(
            model_name=baseline_model_name,
            rubric=rubric,
            evaluator_client=evaluator_client,
            evaluator_model=config.evaluator_model,
            checkpoint_path=None,  # No checkpoint = baseline
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        baseline_results = await baseline_evaluator.evaluate_prompts(
            prompts,
            num_samples_per_prompt=config.num_samples_per_prompt,
            verbose=config.verbose,
        )

    # Create evaluation report
    logger.info("Creating evaluation report...")
    report = create_evaluation_report(
        model_name=config.model_name,
        checkpoint_path=config.checkpoint_path,
        results=results,
        rubric=rubric,
        baseline_results=baseline_results,
    )

    # Determine output directory
    if config.output_dir:
        output_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = Path(config.checkpoint_path).name if config.checkpoint_path else "baseline"
        output_dir = f"evaluation_results/{checkpoint_name}_{timestamp}"

    # Save report
    save_evaluation_report(report, output_dir, save_generations=config.save_generations)

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Mean Score: {report.mean_score:.2f}/100 (± {report.std_score:.2f})")
    logger.info(f"Score Range: [{report.min_score:.2f}, {report.max_score:.2f}]")

    if report.baseline_comparison:
        comp = report.baseline_comparison
        logger.info(f"Baseline Mean: {comp['baseline_mean_score']:.2f}/100")
        logger.info(f"Improvement: {comp['improvement']:.2f} points ({comp['improvement_percentage']:.1f}%)")

    logger.info(f"\nFull report saved to: {output_dir}")


def main():
    """Entry point for the evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = chz.entrypoint(EvalConfig)
    asyncio.run(cli_main(config))


if __name__ == "__main__":
    main()
