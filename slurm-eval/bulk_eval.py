#!/usr/bin/env python3
"""
Script: bulk_evaluation_launcher.py

Launches bulk evaluation jobs for Hugging Face checkpoints using slurm.
After all evaluation jobs have completed (or if --collect-only is specified),
this script collects all JSON outputs and merges them into a single CSV file 
in the evaluation output directory.

Usage:
    python bulk_evaluation_launcher.py --input-dir /path/to/converted_outputs \
        --eval-output-dir /path/to/eval_results [--tasks "lambada_openai,hellaswag,openbookqa"] \
        [--fewshots 0 1] [--tokenizer EleutherAI/gpt-neox-20b] [--max-parallel 2] \
        [--perplexity-jsonls /path/to/perplexity_jsonls] [--collect-only]

If --collect-only is set, evaluation jobs are not launched but the CSV is re-created 
from any evaluation outputs present in the eval output directory.
"""

import os
import subprocess
import time
from pathlib import Path
import concurrent.futures
import typer
import json
import pandas as pd

app = typer.Typer()

THIS_DIR = Path(os.path.realpath(__file__)).parent
EVAL_SCRIPT_PATH = THIS_DIR / "eval_checkpoint.py"
CONTAINER_ENVIRONMENT = f"/users/mbther/.edf/torchtitan.toml"
ACCOUNT = "a-a09"

def generate_evaluation_sbatch(
    input_subdir: Path,
    eval_output_dir: Path,
    tasks: str,
    fewshots: list,
    tokenizer: str,
    perplexity_jsonls: Path = None,
) -> Path:
    """
    Generate an sbatch script to run evaluation for a single checkpoint.
    Assumes that the input_subdir has an "hf" folder for evaluation.
    """
    hf_dir = input_subdir / "hf"
    if not hf_dir.exists():
        raise RuntimeError(f"Hugging Face folder not found in {input_subdir}")

    # Evaluation results will be written under eval_output_dir/<checkpoint_name>
    job_output_dir = eval_output_dir / input_subdir.name
    job_output_dir.mkdir(parents=True, exist_ok=True)
    logs_out = job_output_dir / "logs.out"
    logs_err = job_output_dir / "logs.err"

    # Build the command to run evaluation.
    # fewshots is passed as space-separated values.
    fewshots_args = " ".join(str(f) for f in fewshots)

    # Include perplexity_jsonls flag if provided.
    perplexity_flag = f"--perplexity-jsonls {str(perplexity_jsonls)}" if perplexity_jsonls is not None else ""

    cmd = (
        f"python {EVAL_SCRIPT_PATH} "
        f"--checkpoint-dir {hf_dir} "
        f"--output-dir {job_output_dir} "
        f'--tasks "{tasks}" '
        f"--fewshots {fewshots_args} "
        f"--tokenizer {tokenizer} {perplexity_flag}"
    )

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=eval-{input_subdir.name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output={logs_out}
#SBATCH --error={logs_err}
#SBATCH --partition=normal
#SBATCH --gpus-per-task=4
#SBATCH --environment={CONTAINER_ENVIRONMENT}
#SBATCH --account={ACCOUNT}

set -eo pipefail
echo "Running evaluation for {input_subdir.name}"
{cmd}
echo "Evaluation completed for {input_subdir.name}"
"""
    sbatch_file = job_output_dir / "run_eval.slurm"
    with open(sbatch_file, "w") as f:
        f.write(sbatch_content)
    return sbatch_file


def submit_and_wait(sbatch_path: Path):
    """
    Submit the sbatch file via the sbatch command and wait until the job finishes.
    """
    cmd = ["sbatch", str(sbatch_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"Error submitting job for {sbatch_path.parent.name}: {result.stderr}")
        return
    submission_line = result.stdout.strip()
    typer.echo(f"Submitted evaluation job for {sbatch_path.parent.name}: {submission_line}")
    try:
        job_id = submission_line.split()[-1]
    except Exception:
        typer.echo("Failed to parse job ID.")
        return

    typer.echo(f"Waiting for job {job_id} to finish...")
    while True:
        squeue_proc = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
        if job_id not in squeue_proc.stdout:
            break
        time.sleep(10)

    # Check job exit status using sacct.
    sacct_cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobIDRaw,State,ExitCode",
        "--parsable2",
        "--noheader",
    ]
    sacct_proc = subprocess.run(sacct_cmd, capture_output=True, text=True)
    if sacct_proc.returncode == 0 and sacct_proc.stdout.strip():
        job_info = sacct_proc.stdout.strip().split("|")
        if len(job_info) >= 3:
            state = job_info[1]
            exit_code = job_info[2]
            if state != "COMPLETED" or not exit_code.startswith("0:0"):
                typer.echo(f"Warning: Job {job_id} did not complete successfully: State={state}, ExitCode={exit_code}")
            else:
                typer.echo(f"Job {job_id} completed successfully.")
        else:
            typer.echo(f"Unexpected sacct output for job {job_id}: {sacct_proc.stdout}")
    else:
        typer.echo(f"Could not get job accounting info for {job_id}.")


def collect_results(eval_output_dir: Path, merge: bool = True):
    """
    Collect all JSON outputs from evaluation results stored in directories under eval_output_dir.
    The function expects each evaluation job to write a JSON file inside a folder named 'results*'.
    Merges the JSON outputs and creates a CSV (results.csv) in eval_output_dir.
    """
    eval_output_dir = Path(eval_output_dir)
    # Find all result directories recursively (looking for folders named with a leading "results")
    json_files = list(eval_output_dir.glob("**/results/*.json"))
    if not json_files:
        typer.echo(f"No JSON results found in {eval_output_dir}.")
        return

    all_results = []
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            # Expecting each JSON has a "results" key.
            if "results" not in data:
                typer.echo(f"Skipping {json_file}: no 'results' key.")
                continue
            results = data["results"]
            # Also, annotate the result with checkpoint id and fewshot state if possible.
            # We assume the parent folder of the results directory has a name pattern (e.g., step-1000 or fewshot_0).
            parent_name = json_file.parent.parent.name
            # Try to extract fewshot number from parent folder name.
            try:
                nshot = int(parent_name.split("_")[-1])
            except Exception:
                nshot = None
            parsed_results = {}
            for task, metrics in results.items():
                for key, value in metrics.items():
                    if key == "alias":
                        continue
                    new_key = f"{task}_{key}".replace(",", "_")
                    parsed_results[new_key] = value
            parsed_results["checkpoint_id"] = parent_name
            parsed_results["nshot"] = nshot
            all_results.append(parsed_results)
        except Exception as e:
            typer.echo(f"Error reading {json_file}: {e}")
            continue
    if all_results:
        df = pd.DataFrame(all_results)
        # Try to sort by checkpoint and fewshot if possible.
        if "checkpoint_id" in df.columns and "nshot" in df.columns:
            try:
                df["nshot"] = pd.to_numeric(df["nshot"])
            except Exception:
                pass
            df.sort_values(by=["checkpoint_id", "nshot"], inplace=True)
        output_csv = eval_output_dir / "results.csv"
        df.to_csv(output_csv, index=False)
        typer.echo(f"Results have been successfully written to '{output_csv}'.")
    else:
        typer.echo("No results to merge.")


@app.command()
def launch_bulk_evaluation(
    input_dir: Path = typer.Option(
        ..., help="Input directory containing subdirectories with converted checkpoints (each with an 'hf' folder)"
    ),
    eval_output_dir: Path = typer.Option(..., help="Directory to store evaluation results."),
    tasks: str = typer.Option(
        "lambada_openai,hellaswag,openbookqa", help="Comma-separated tasks string for evaluation"
    ),
    fewshots: list[int] = typer.Option([0], help="List of fewshot settings, e.g., 0 1"),
    tokenizer: str = typer.Option("EleutherAI/gpt-neox-20b", help="Tokenizer to use"),
    max_parallel: int = typer.Option(1, help="Maximum number of parallel evaluations"),
    perplexity_jsonls: Path = typer.Option(
        None, help="Optional: Directory containing JSONL files for perplexity tasks"
    ),
    collect_only: bool = typer.Option(
        False, help="If set, skip evaluation submission and only collect and export results to CSV"
    ),
):
    """
    Launch bulk evaluation jobs based on the input directory.
    If --collect-only is set, evaluation jobs are not launched but the results are collected.
    """
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    if not collect_only:
        # Find subdirectories in input_dir that contain an "hf" folder.
        candidate_dirs = [d for d in input_dir.iterdir() if d.is_dir() and (d / "hf").exists()]
        if not candidate_dirs:
            typer.echo("No valid checkpoint directories with an 'hf' subfolder found in the input directory.")
            raise typer.Exit(code=1)

        try:
            candidate_dirs.sort(key=lambda p: int(p.name.split("-")[-1]))
        except Exception:
            candidate_dirs.sort(key=lambda p: p.name)

        sbatch_files = []
        for subdir in candidate_dirs:
            sbatch_file = generate_evaluation_sbatch(
                subdir, eval_output_dir, tasks, fewshots, tokenizer, perplexity_jsonls
            )
            sbatch_files.append(sbatch_file)
            typer.echo(f"Created sbatch file for evaluation of {subdir.name} at {sbatch_file}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(submit_and_wait, sbatch_file) for sbatch_file in sbatch_files]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        typer.echo("All evaluation jobs have been processed.")

    # Now, whether evaluations were just run or not (collect-only), collect the results.
    typer.echo("Collecting and merging evaluation results...")
    collect_results(eval_output_dir, merge=True)


if __name__ == "__main__":
    app()
