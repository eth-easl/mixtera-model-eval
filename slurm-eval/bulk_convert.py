#!/usr/bin/env python3
"""
Bulk TorchTITAN-to-HuggingFace Checkpoint Converter Launcher

Usage:
    python bulk_convert_checkpoints.py --checkpoints-dir /path/to/checkpoints \
        --output-dir /path/to/output --max-parallel 4 [--tokenizer EleutherAI/gpt-neox-20b] \
        [--max-seq-len 2048] [--n-kv-heads 16]

This script scans the given checkpoints directory for "step-*" subdirectories.
For each such checkpoint, it creates an output subdirectory (e.g. output_dir/step-1000)
and writes an sbatch file which will:
    1. Install torchtitan and mixtera.
    2. Run the conversion from TorchTITAN to a temporary torch (.pt) checkpoint:
       python -m torch.distributed.checkpoint.format_utils dcp_to_torch <checkpoint_dir> checkpoint.pt
    3. Convert the temporary .pt file to a Hugging Face checkpoint by calling
       the provided conversion script (with full path):
       python /full/path/convert_to_huggingface.py --input checkpoint.pt --output hf --tokenizer <tokenizer>
           --max_seq_len <max_seq_len> [--n_kv_heads <n_kv_heads>]
Logs are saved within each output subdirectory.
The launcher submits the conversion jobs one by one, waiting for each to finish before submitting the next.
"""

import os
import subprocess
import time
from pathlib import Path
import concurrent.futures
import typer

app = typer.Typer()

CONTAINER_ENVIRONMENT = f"/users/mbther/.edf/torchtitan.toml"
TORCHTITAN_PATH = f"/users/{os.environ.get('USER')}/torchtitan-mixtera"
MIXTERA_PATH = f"/iopsstor/scratch/cscs/{os.environ.get('USER')}/mixtera"

THIS_DIR = Path(os.path.realpath(__file__)).parent
CONVERT_SCRIPT_PATH = THIS_DIR / "convert_to_huggingface.py"

ACCOUNT = "a-a09"


def get_no_conda_env():
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX", "")
    conda_bin = os.path.join(conda_prefix, "bin")
    keys_to_remove = [key for key in env if "CONDA" in key or "PYTHON" in key]
    for key in keys_to_remove:
        del env[key]
    paths = env["PATH"].split(os.pathsep)
    paths = [p for p in paths if conda_bin not in p and conda_prefix not in p]
    env["PATH"] = os.pathsep.join(paths)
    return env


def generate_sbatch_script(
    checkpoint_path: Path, output_subdir: Path, tokenizer: str, max_seq_len: int, n_kv_heads: int = None
) -> Path:
    """
    Generate an sbatch file that converts a TorchTITAN checkpoint to a Hugging Face checkpoint.
    The generated script installs torchtitan and mixtera, converts the checkpoint to torch format,
    then converts that torch checkpoint to a Hugging Face checkpoint.
    """
    # Define the temporary torch output filename and HF model folder
    torch_output = output_subdir / "torch.pt"
    hf_output_dir = output_subdir / "hf"
    logs_out = output_subdir / "logs.out"
    logs_err = output_subdir / "logs.err"

    # Build the conversion command string.
    n_kv_str = f"--n_kv_heads {n_kv_heads}" if n_kv_heads is not None else ""
    conversion_cmd = (
        f"python {CONVERT_SCRIPT_PATH} --input {torch_output} --output {hf_output_dir} "
        f"--tokenizer {tokenizer} --max_seq_len {max_seq_len} {n_kv_str}"
    )
    script_content = f"""#!/bin/bash
#SBATCH --job-name=convert-{checkpoint_path.name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output={logs_out}
#SBATCH --error={logs_err}
#SBATCH --partition=normal
#SBATCH --gpus-per-task=4
#SBATCH --environment={CONTAINER_ENVIRONMENT}
#SBATCH --account={ACCOUNT}

set -eo pipefail

echo "Installing torchtitan..."
pushd {TORCHTITAN_PATH}
pip install -e .
popd

echo "Installing mixtera..."
pushd {MIXTERA_PATH}
n=0
until [ $n -ge 5 ]
do
   if pip install -e .; then
      echo "pip install succeeded after try ($n)"
      break
   else
      n=$(($n+1))
      echo "pip install failed, retrying ($n)"
      sleep 1
   fi
done
if [ $n -ge 5 ]; then
   echo "pip install failed after 5 retries"
   exit 1
fi
popd

# Convert TorchTITAN checkpoint to torch format
echo "Converting TorchTITAN checkpoint at {checkpoint_path} to torch format..."
python -m torch.distributed.checkpoint.format_utils dcp_to_torch {checkpoint_path} {torch_output}

# Convert the temporary torch checkpoint to Hugging Face format
echo "Converting torch checkpoint to Hugging Face format..."
{conversion_cmd}

echo "Conversion completed for {checkpoint_path.name}"
"""
    sbatch_path = output_subdir / "run_conversion.slurm"
    with open(sbatch_path, "w") as f:
        f.write(script_content)
    return sbatch_path


def submit_and_wait(sbatch_path: Path):
    """
    Submit the sbatch file via sbatch command and then wait until the job has finished.
    Logs the submission and waits by polling squeue and sacct commands.
    """
    # Submit the job
    cmd = ["sbatch", str(sbatch_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, env=get_no_conda_env())
    if result.returncode != 0:
        typer.echo(f"Error submitting job for {sbatch_path.parent.name}: {result.stderr}")
        return
    submission_line = result.stdout.strip()
    typer.echo(f"Submitted job for {sbatch_path.parent.name}: {submission_line}")

    try:
        job_id = submission_line.split()[-1]
    except Exception:
        typer.echo("Failed to determine job ID from submission output.")
        return

    typer.echo(f"Waiting for job {job_id} to finish...")
    while True:
        squeue_proc = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True, env=get_no_conda_env())
        if job_id not in squeue_proc.stdout:
            break
        time.sleep(10)

    sacct_cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=JobIDRaw,State,ExitCode",
        "--parsable2",
        "--noheader",
    ]
    sacct_proc = subprocess.run(sacct_cmd, capture_output=True, text=True, env=get_no_conda_env())
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
        typer.echo(f"Could not retrieve accounting info for job {job_id}.")


@app.command()
def main(
    checkpoints_dir: Path = typer.Option(
        ...,
        help="Directory containing checkpoint subdirectories, e.g. /iopsstor/scratch/cscs/mbther/ado/torchtitan-dumps-ado/checkpoint",
    ),
    output_dir: Path = typer.Option(..., help="Directory to write conversion results (must be empty)"),
    max_parallel: int = typer.Option(
        1, help="Maximum number of parallel conversion jobs to submit (jobs will run sequentially per thread)"
    ),
    tokenizer: str = typer.Option(
        "EleutherAI/gpt-neox-20b", help="Tokenizer identifier for converting to Hugging Face format"
    ),
    max_seq_len: int = typer.Option(2048, help="Maximum sequence length for the converted model"),
    n_kv_heads: int = typer.Option(None, help="Override number of key-value heads (optional)"),
):
    # Check that the output directory exists and is empty; if not, fail early
    if output_dir.exists():
        if any(output_dir.iterdir()):
            typer.echo(f"Error: Output directory {output_dir} is not empty.")
            raise typer.Exit(code=1)
    else:
        output_dir.mkdir(parents=True)

    # Find all subdirectories that start with "step-"
    step_dirs = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("step-")]
    if not step_dirs:
        typer.echo("No checkpoint subdirectories found in the given checkpoints directory.")
        raise typer.Exit(code=1)

    try:
        step_dirs.sort(key=lambda p: int(p.name.split("-")[-1]))
    except Exception:
        typer.echo("Failed to sort checkpoint directories; ensure they follow the format step-<number>.")
        raise typer.Exit(code=1)

    sbatch_paths = []
    for step_dir in step_dirs:
        output_subdir = output_dir / step_dir.name
        try:
            output_subdir.mkdir(exist_ok=False)
        except FileExistsError:
            typer.echo(f"Error: Output subdirectory {output_subdir} already exists.")
            raise typer.Exit(code=1)

        sbatch_file = generate_sbatch_script(step_dir, output_subdir, tokenizer, max_seq_len, n_kv_heads)
        sbatch_paths.append(sbatch_file)
        typer.echo(f"Created sbatch script for {step_dir.name} at {sbatch_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(submit_and_wait, sp) for sp in sbatch_paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    typer.echo("All conversion jobs have been processed.")


if __name__ == "__main__":
    app()
