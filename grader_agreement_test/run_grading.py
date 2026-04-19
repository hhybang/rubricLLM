#!/usr/bin/env python3
"""
Grade each test-case draft with Opus and Sonnet (3 runs each). Semaphore-limited concurrency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from _anthropic_util import call_anthropic, extract_json_object

DIR = Path(__file__).resolve().parent
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"

GRADING_SYSTEM = "You grade drafts against rubrics. Follow instructions exactly."

GRADING_USER_TEMPLATE = """You are grading a writing draft against a personalized rubric.

For each criterion in the rubric, evaluate each dimension as MET or NOT_MET.
A dimension is MET if the draft clearly satisfies what the dimension describes.
A dimension is NOT_MET if the draft does not satisfy it or if it's ambiguous.

Be strict and literal — grade what the dimension says, not what you think "good writing" means in general.

<rubric>
__RUBRIC_JSON__
</rubric>

<draft>
__DRAFT_TEXT__
</draft>

Respond with ONLY this JSON — no explanation, no preamble:
{
  "grades": [
    {
      "criterion_name": "<name from rubric>",
      "dimension_grades": [
        {
          "dimension_id": "<id from rubric>",
          "grade": "MET" | "NOT_MET",
          "evidence": "<1 sentence: quote or cite the specific part of the draft that justifies this grade>"
        }
      ]
    }
  ]
}
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_grading_response(raw: str) -> dict[str, Any] | None:
    data = extract_json_object(raw.strip())
    if not data or "grades" not in data:
        return None
    if not isinstance(data["grades"], list):
        return None
    return data


def grade_once(
    client: Anthropic,
    *,
    model: str,
    rubric: dict,
    draft_text: str,
) -> tuple[dict[str, Any] | None, str, int | None, int | None, str | None, int]:
    """Returns (parsed_grades_dict_or_none, raw_text, in_tok, out_tok, error, latency_ms)."""
    rubric_for_prompt = {
        "writing_type": rubric.get("writing_type"),
        "rubric": rubric.get("rubric", []),
        "user_goals_summary": rubric.get("user_goals_summary"),
    }
    rubric_json = json.dumps(rubric_for_prompt, ensure_ascii=False, indent=2)
    user = GRADING_USER_TEMPLATE.replace("__RUBRIC_JSON__", rubric_json).replace(
        "__DRAFT_TEXT__", draft_text
    )
    err: str | None = None
    parsed: dict[str, Any] | None = None
    raw_out = ""
    in_tok = out_tok = None
    latency_ms = 0
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            raw_out, in_tok, out_tok = call_anthropic(
                client,
                model=model,
                system=GRADING_SYSTEM,
                user=user,
                max_tokens=16000,
                temperature=0.0,
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            err = f"api_error: {e}"
            return None, raw_out, in_tok, out_tok, err, latency_ms
        parsed = parse_grading_response(raw_out)
        if parsed is not None:
            return parsed, raw_out, in_tok, out_tok, None, latency_ms
        err = "parse_error"
        log(f"    parse fail attempt {attempt + 1}, retrying...")
    return None, raw_out, in_tok, out_tok, err or "parse_error", latency_ms


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)[:120]


async def run_grading_job(
    *,
    client: Anthropic,
    sem: asyncio.Semaphore,
    scenario_id: str,
    draft_id: str,
    model_key: str,
    model_id: str,
    run_idx: int,
    rubric: dict,
    draft_text: str,
    raw_dir: Path,
) -> dict[str, Any]:
    async with sem:
        loop = asyncio.get_event_loop()

        def _work():
            return grade_once(client, model=model_id, rubric=rubric, draft_text=draft_text)

        parsed, raw, in_tok, out_tok, err, latency_ms = await loop.run_in_executor(None, _work)
        raw_path = raw_dir / f"{safe_filename(scenario_id)}__{safe_filename(draft_id)}__{model_key}_run{run_idx}.json"
        payload = {
            "scenario_id": scenario_id,
            "draft_id": draft_id,
            "model": model_key,
            "run": run_idx,
            "raw_response": raw,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": latency_ms,
            "error": err,
            "grades": parsed,
        }
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  done {scenario_id} / {draft_id} / {model_key} run {run_idx}" + (f" ERR {err}" if err else ""))
        return {
            "scenario_id": scenario_id,
            "draft_id": draft_id,
            "model": model_key,
            "run": run_idx,
            "grades": parsed,
            "raw_response": raw,
            "latency_ms": latency_ms,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "error": err,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=DIR / "test_cases.json")
    ap.add_argument("-o", "--output", type=Path, default=DIR / "grading_results.json")
    ap.add_argument("--raw-dir", type=Path, default=DIR / "raw_results")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    cases = json.loads(args.input.read_text(encoding="utf-8"))
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    client = Anthropic()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = []
    for case in cases:
        sid = case["scenario_id"]
        rubric = case["rubric"]
        for d in case["drafts"]:
            did = d["draft_id"]
            text = d["text"]
            for run_idx in (1, 2, 3):
                for model_key, model_id in (("opus", MODEL_OPUS), ("sonnet", MODEL_SONNET)):
                    tasks.append(
                        run_grading_job(
                            client=client,
                            sem=sem,
                            scenario_id=sid,
                            draft_id=did,
                            model_key=model_key,
                            model_id=model_id,
                            run_idx=run_idx,
                            rubric=rubric,
                            draft_text=text,
                            raw_dir=args.raw_dir,
                        )
                    )

    log(f"Starting {len(tasks)} grading calls (concurrency={args.concurrency})...")

    async def _gather_all() -> list[Any]:
        return await asyncio.gather(*tasks)

    results = asyncio.run(_gather_all())

    # Fill latency from raw files is optional; compute from stored data — skipped in async path
    # Re-read raw_results for latency if we add timing to grade_once return - quick fix: store in payload
    out_rows = []
    for r in results:
        row = {
            "scenario_id": r["scenario_id"],
            "draft_id": r["draft_id"],
            "model": r["model"],
            "run": r["run"],
            "grades": r["grades"],
            "raw_response": r["raw_response"],
            "latency_ms": r.get("latency_ms"),
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "error": r.get("error"),
        }
        out_rows.append(row)

    args.output.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
