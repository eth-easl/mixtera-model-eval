from enum import Enum
import shutil
import typer
from pathlib import Path
import yaml
import subprocess
import os
import itertools
import json
from copy import deepcopy
import socket
import wandb
import time
import math
from tqdm import tqdm

SCRIPT_DIR = Path(os.path.realpath(__file__)).parent
app = typer.Typer()


class MachineType(str, Enum):
    sgsrtx = "sgsrtx"
    sgsh100 = "sgsh100"
    cscs = "cscs"


class ModelType(str, Enum):
    smollm135m = "smollm135m"
    llama1b = "llama1b"
    llama8b = "llama8b"
    llama70b = "llama70b"


# TODO: Find maximum microbatch size per GPU/model combination
MACHINE_MODEL_MICROBATCH = {
    MachineType.sgsrtx: {
        ModelType.llama1b: 1,  # tested > 1, goes OOM
        ModelType.llama8b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.sgsh100: {
        ModelType.llama1b: 4,
        ModelType.llama8b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.cscs: {
        ModelType.smollm135m: 64,  # not tested
        ModelType.llama1b: 2,  # more goes OOM
        ModelType.llama8b: 6,  # not tested
        ModelType.llama70b: 1,  # not tested
    },
}

MACHINE_MODEL_TOKENS = {
    MachineType.sgsrtx: {
        ModelType.llama1b: 500000,
        ModelType.llama8b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.sgsh100: {
        ModelType.llama1b: 2000000,
        ModelType.llama8b: 2000000,
        ModelType.llama70b: 2000000,
    },
    MachineType.cscs: {
        ModelType.smollm135m: 2000000,
        ModelType.llama1b: 2000000,
        ModelType.llama8b: 2000000,
        ModelType.llama70b: 2000000,
    },
}

OMP_NUM_THREADS = "16"

SLURM_PARTITION = "normal"
SLURM_TIME = "00:15:00"
SLURM_ACCOUNT = "a06"
SLURM_GPUS_PER_TASK = 4
SLURM_MEM = "460000"

NANOTRON_REPO_PATH = f"/iopsstor/scratch/cscs/{os.environ.get('USER')}/nanotron"
SHARED_DIR_DEFAULT = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER')}/nanotron-benchmarks")
CONTAINER_ENVIRONMENT = "/store/swissai/a06/containers/nanotron_pretrain/nanotron_pretrain.toml"


def check_nanotron_availability():
    try:
        global nanotron
        global run_script_path
        import nanotron

        run_script_path = Path(nanotron.__path__[0]).parent.parent / "run_train.py"
    except ImportError:
        typer.echo(
            "Error: 'nanotron' module is not available. Please ensure it is installed according to their instructions."
        )
        raise typer.Exit(code=1)


def ask_to_continue():
    response = input("Do you want to continue? (yes/no) [yes]: ").strip().lower()

    if response == "" or response.startswith("y"):
        return True
    elif response.startswith("n"):
        return False
    else:
        print("Invalid input, assuming 'yes'.")
        return True


def current_milli_time():
    return round(time.time() * 1000)


def load_yaml_from_file(path: str | Path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            typer.echo("Error: " + str(exc))
            raise typer.Exit(code=1) from exc


def get_data_from_wandb(project: str, run_id: str, retry: int = 0) -> dict:
    api = wandb.Api()
    # Retrieve all runs and sort them by creation date in descending order
    runs = sorted(api.runs(project), key=lambda x: x.created_at, reverse=True)
    run = next((run for run in runs if run.name.split("_", 2)[-1].startswith(run_id)), None)

    if not run:
        typer.echo(f"Error: Could not find run {run_id} in runs {[run.name for run in runs]}.")
        raise typer.Exit(code=1)

    timeout = 600  # seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        if run.state == "finished":
            break
        typer.echo("Still waiting for the run to finish on wandb.")
        time.sleep(10)
        api = wandb.Api()
        runs = sorted(api.runs(project), key=lambda x: x.created_at, reverse=True)
        run = next((run for run in runs if run.name.split("_", 2)[-1].startswith(run_id)), None)

    if run.state != "finished":
        typer.echo("Timeout reached. Run did not finish in 10 minutes.")
        raise typer.Exit(code=1)

    if (
        "global_batch_size" not in run.history().to_dict().keys()
        or "tokens_per_sec" not in run.history().to_dict().keys()
        and retry < 5
    ):
        retry += 1
        return get_data_from_wandb(project, run_id, retry)

    return run.history().to_dict()


def run_benchmark_on_sgs(config: dict, mode: MachineType) -> dict:
    # Persist yaml to disk
    bm_config_path = SCRIPT_DIR / "benchmark.yaml"
    with open(bm_config_path, "w+") as ff:
        yaml.dump(config, ff)

    # Run nanotron training on the current node
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS

    dp = int(config["parallelism"]["dp"])
    tp = int(config["parallelism"]["tp"])
    pp = int(config["parallelism"]["pp"])
    nprocs = str(dp * tp * pp)

    command = [
        "python",
        "-u",
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nprocs}",
        str(run_script_path),
        "--config-file",
        str(bm_config_path),
    ]
    typer.echo(f"Running benchmark with command {command}")
    proc = subprocess.run(command)
    if proc.returncode != 0:
        typer.echo("Process did not exit with exit code 0.")
        results = {"success": False}
    else:
        typer.echo("Benchmark done, collecting data")
        results = get_data_from_wandb(config["general"]["project"], config["general"]["run"]) | {"success": True}
        typer.echo("Data collected")

    bm_config_path.unlink()

    return results


def run_benchmark_on_cscs(config: dict, account: str, shared_dir: Path) -> dict:
    # Ensure shared directory exists
    shared_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir = shared_dir / "hf_data"
    hf_home_dir = shared_dir / "hf_home"
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home_dir.mkdir(parents=True, exist_ok=True)

    if not Path(NANOTRON_REPO_PATH).exists():
        raise RuntimeError(f"Cannot find nanotron at {NANOTRON_REPO_PATH}")

    job_name = config["general"]["run"]
    dp = int(config["parallelism"]["dp"])
    tp = int(config["parallelism"]["tp"])
    pp = int(config["parallelism"]["pp"])

    num_gpus = dp * tp * pp
    num_nodes = math.ceil(num_gpus / SLURM_GPUS_PER_TASK)

    proc_per_node = 4
    if num_gpus < 4 and num_nodes == 1:
        proc_per_node = num_gpus

    if num_nodes > 1 and num_gpus % 4 != 0:
        raise RuntimeError(f"num_nodes = {num_nodes} > 1 and num_gpus = {num_gpus} not divisible by 4, this is currently not supported.")

    # Save the benchmark configuration in the shared directory
    bm_config_path = shared_dir / f"{job_name}_benchmark.yaml"
    with open(bm_config_path, "w+") as f:
        yaml.dump(config, f)

    # Paths for logs
    log_dir = shared_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_file = log_dir / f"{job_name}_output.log"
    error_file = log_dir / f"{job_name}_output.err"

    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time={SLURM_TIME}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --no-requeue
#SBATCH --mem={SLURM_MEM}
#SBATCH --gpus-per-task={SLURM_GPUS_PER_TASK}
"""

    env_vars = f"""
export OMP_NUM_THREADS={OMP_NUM_THREADS}
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enrootlibn
export WANDB_DIR={shared_dir}/wandb
export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}
export HF_TOKEN={os.environ.get('HF_TOKEN', '')}
export MASTER_ADDR=$(hostname)
export HF_HOME={hf_home_dir}
export HF_DATASETS_CACHE={data_cache_dir}
"""

    # Commands to run
    commands = f"""
set -eo pipefail

srun -ul --container-writable --environment={CONTAINER_ENVIRONMENT} bash -c "
cd {NANOTRON_REPO_PATH}

pip install -e . --no-dependencies

numactl --membind=0-3 torchrun --nnodes=$SLURM_NNODES \\
    --nproc-per-node={proc_per_node} \\
    --node-rank=$SLURM_PROCID \\
    --master-addr=$MASTER_ADDR \\
    --master-port=$MASTER_PORT \\
    --role $(hostname -s) \\
    run_train.py --config-file {bm_config_path}
"
"""

    # Combine all parts into the SBATCH script
    sbatch_script = sbatch_header + env_vars + commands

    # Save SBATCH script to file in the shared directory
    sbatch_script_path = shared_dir / f"{job_name}_run.slurm"
    with open(sbatch_script_path, "w+") as f:
        f.write(sbatch_script)

    if num_nodes > 1:  # avoid NCCL issues
        typer.echo(f"Skipping job due to num_nodes = {num_nodes} > 1")
        return {"success": False}

    # Submit the job
    typer.echo(f"Submitting job {job_name} with sbatch script {sbatch_script_path}")
    submit_command = ["sbatch", str(sbatch_script_path)]
    proc = subprocess.run(submit_command, capture_output=True, text=True)
    if proc.returncode != 0:
        typer.echo(f"Error submitting job: {proc.stderr}")
        results = {"success": False}
    else:
        typer.echo(f"Job submitted, sbatch output: {proc.stdout}")
        # Extract job ID from sbatch output
        job_id = None
        for line in proc.stdout.strip().split("\n"):
            if "Submitted batch job" in line:
                job_id = line.strip().split()[-1]
                break
        if job_id is None:
            typer.echo("Could not determine job ID from sbatch output.")
            results = {"success": False}
        else:
            # Wait for the job to complete
            typer.echo(f"Waiting for job {job_id} to complete...")
            job_complete = False
            while not job_complete:
                squeue_cmd = ["squeue", "-j", str(job_id)]
                squeue_proc = subprocess.run(squeue_cmd, capture_output=True, text=True)
                if squeue_proc.returncode != 0:
                    typer.echo(f"Error checking job status: {squeue_proc.stderr}")
                    break
                elif job_id not in squeue_proc.stdout:
                    job_complete = True
                else:
                    typer.echo(f"Job {job_id} is still running...")
                    time.sleep(10)

            # Collect results from wandb
            typer.echo("Job completed, collecting data from WandB.")
            results = get_data_from_wandb(config["general"]["project"], config["general"]["run"]) | {"success": True}
            typer.echo("Data collected")

    return results


def run_benchmark(config: dict, mode: MachineType, account: str, shared_dir: Path) -> dict:
    if mode in [MachineType.sgsrtx, MachineType.sgsh100]:
        return run_benchmark_on_sgs(config, mode)
    elif mode == MachineType.cscs:
        return run_benchmark_on_cscs(config, account, shared_dir)

    typer.echo(f"Error: No implementation yet for mode `{mode}`")
    raise typer.Exit(code=1)


def validate_user_input(
    mode: MachineType,
    model: ModelType,
):
    if mode == MachineType.sgsrtx and model not in [ModelType.smollm135m, ModelType.llama1b]:
        typer.echo(f"RTX 3090 machines don't support model {model}")
        return False

    if mode in [MachineType.sgsrtx, MachineType.sgsh100]:
        check_nanotron_availability()

        if not run_script_path.exists():
            typer.echo(f"Cannot find run script at {run_script_path}")
            raise typer.Exit(code=1)

        # Due to SSH shenanigans, this needs to run on an sgs machine
        hostname = socket.gethostname().split(".")[0]
        if mode == MachineType.sgsh100 and hostname != "sgs-gpu07":
            typer.echo("Note that you selected H100 GPUs but this is not sgs-gpu07.")
            if not ask_to_continue():
                raise typer.Exit(code=1)

        if mode == MachineType.sgsrtx and hostname not in ["sgs-gpu01", "sgs-gpu02", "sgs-gpu03", "sgs-gpu04"]:
            typer.echo("Note that you selected RTX3090 GPUs but this is not sgs-gpu[01-04].")
            if not ask_to_continue():
                raise typer.Exit(code=1)

    return True


def load_base_config(model: ModelType) -> dict:
    if model == ModelType.llama1b:
        return load_yaml_from_file(SCRIPT_DIR / "llama1b.yaml")
    if model == ModelType.llama8b:
        return load_yaml_from_file(SCRIPT_DIR / "llama8b.yaml")
    if model == ModelType.smollm135m:
        return load_yaml_from_file(SCRIPT_DIR / "smollm135m.yaml")

    typer.echo(f"Error: No config yet for model `{model}`")
    raise typer.Exit(code=1)


def adjust_base_config(
    base_config: dict,
    dataset_path: Path,
    bm_identifier: str,
    curr_run: int,
    mode: MachineType,
    model: ModelType,
    dl_worker: int,
    dp: int,
    seq_length: int,
    batch_accumulation_per_replica: int,
    seed: int,
) -> tuple[dict, dict]:
    config = deepcopy(base_config)

    # Set wandb info
    config["general"]["project"] = bm_identifier
    config["general"]["run"] = (
        f"run{curr_run}_dp{dp}_seed{seed}_w{dl_worker}_s{seq_length}_acc{batch_accumulation_per_replica}_{mode}_{model}"
    )

    # Set number of dp nodes
    config["parallelism"]["dp"] = dp

    # Update tp/pp for node/model pairs
    if mode == MachineType.sgsrtx:
        if model == ModelType.llama1b:
            config["parallelism"]["tp"] = 2

    if mode == MachineType.cscs:
        if model in {ModelType.llama1b, ModelType.smollm135m}:
            config["parallelism"]["tp"] = 1
            config["parallelism"]["pp"] = 1

        if model == ModelType.llama8b:
            config["parallelism"]["tp"] = 2
            config["parallelism"]["pp"] = 1

    # Set microbatch size
    config["tokens"]["batch_accumulation_per_replica"] = batch_accumulation_per_replica
    config["tokens"]["micro_batch_size"] = MACHINE_MODEL_MICROBATCH[mode][model]

    # Set sequence length
    config["tokens"]["sequence_length"] = seq_length
    config["model"]["model_config"]["max_position_embeddings"] = seq_length

    # Set number of data loading workers
    assert len(config["data_stages"]) == 1, "data stages should only be 1"
    config["data_stages"][0]["data"]["num_loading_workers"] = dl_worker
    # Set seed
    config["data_stages"][0]["data"]["seed"] = seed
    config["general"]["seed"] = seed
    # Set dataset path
    config["data_stages"][0]["data"]["dataset"]["hf_dataset_or_datasets"] = str(dataset_path)

    # Number of tokens that we want to consume (will be rounded up to match batch size/seq length)
    scheduled_total_tokens = MACHINE_MODEL_TOKENS[mode][model] * dp
    batch_size = dp * config["tokens"]["batch_accumulation_per_replica"] * config["tokens"]["micro_batch_size"]
    tokens_per_step = batch_size * seq_length
    train_steps = max(
        math.ceil(scheduled_total_tokens / tokens_per_step), 10
    )  # Minimum of 10 training steps per benchmark
    config["tokens"]["train_steps"] = train_steps
    calculated_total_tokens = train_steps * tokens_per_step

    additional_info = {
        "scheduled_total_tokens": scheduled_total_tokens,
        "calculated_total_tokens": calculated_total_tokens,
        "tokens_per_step": tokens_per_step,
        "batch_size": batch_size,
        "skipped": False,
        "skip_reason": "",
    }

    # Now decide whether this run should be skipped
    total_nodes = dp * int(config["parallelism"]["tp"]) * int(config["parallelism"]["pp"])
    if mode in [MachineType.sgsrtx, MachineType.sgsh100] and total_nodes > 4:
        additional_info["skipped"] = True
        additional_info["skip_reason"] = f"total_nodes = {total_nodes} > 4 for sgs machine"

    if not config["tokens"]["batch_accumulation_per_replica"] >= config["parallelism"]["pp"] - 1:
        additional_info["skipped"] = True
        additional_info["skip_reason"] = (
            f"batch_accumulation_per_replica = {batch_accumulation_per_replica} not <= pp - 1 = {config['parallelism']['pp'] - 1:}"
        )

    # These should not happen.
    if config["model"]["model_config"]["hidden_size"] % config["model"]["model_config"]["num_attention_heads"] != 0:
        raise RuntimeError("hidden_size needs to be divisible by num_attention_heads")
    if config["model"]["model_config"]["num_attention_heads"] % config["parallelism"]["tp"] != 0:
        raise RuntimeError("num_attention_heads needs to be divisible by tp")

    return config, additional_info


def persist_results_to_json(output: Path, all_results: list[dict]):
    with open(output, "w+") as fout:
        json.dump(all_results, fout, indent=4)


@app.command()
def run_benchmarks(
    output_dir: Path,
    benchmark_name: str,
    mode: MachineType,
    model: ModelType,
    dataset_path: Path,
    dl_workers: list[int] = [0, 1, 2, 4],
    dps: list[int] = [1, 2, 4, 8, 16],
    seq_lengths: list[int] = [4096],
    batch_accumulation_per_replicas: list[int] = [1, 2, 4, 8, 16],
    seeds: list[int] = [42],
    huggingface_cache_path: Path = Path("/scratch/maximilian.boether/hfcache"),
    skip_existing: bool = False,
    account: str = SLURM_ACCOUNT,
    shared_dir: Path = SHARED_DIR_DEFAULT,
):
    if not validate_user_input(mode, model):
        raise typer.Exit(code=1)

    output_file = output_dir / f"{benchmark_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not skip_existing:
        typer.echo(f"Error: file {output_file} already exists. Specify --skip-existing to continue run.")
        raise typer.Exit(code=1)

    if output_file.exists() and skip_existing:
        bak_path = output_file.parent / f"{benchmark_name}.json.bak{current_milli_time()}"
        shutil.copyfile(output_file, bak_path)
        typer.echo(
            f"Warning: file {output_file} already exists, copied to {bak_path} since --skip-existing is enabled."
        )

    if not output_file.exists() and skip_existing:
        typer.echo(
            f"Warning: --skip-existing is enabled, but output file {output_file} does not exist. Won't skip anything"
        )

    all_results = []
    existing_runs = set()
    if skip_existing and output_file.exists():
        with open(output_file, "r") as file:
            all_results = json.load(file)

        for item in all_results:
            if item.get("skipped", False):
                continue
            if not item.get("success", True):
                continue
            _run_id = item["config"]["general"].get("run", None)
            if _run_id is not None and _run_id != "":
                existing_runs.add(_run_id)

    base_config = load_base_config(model)
    bm_identifier = "mixterabenchmark_script"
    curr_run = 0

    if huggingface_cache_path.exists():
        dataset_cache_path = huggingface_cache_path / "data"
        home_path = huggingface_cache_path / "home"
        dataset_cache_path.mkdir(exist_ok=True)
        home_path.mkdir(exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(dataset_cache_path)
        os.environ["HF_HOME"] = str(home_path)
    else:
        typer.echo(f"Note that the hf cache path {huggingface_cache_path} does not exist. Using default path by hf.")

    for seed, dl_worker, dp, seq_len, bacc in tqdm(
        list(itertools.product(seeds, dl_workers, dps, seq_lengths, batch_accumulation_per_replicas)),
        desc="Processing configurations",
    ):
        adjusted_config, additional_info = adjust_base_config(
            base_config, dataset_path, bm_identifier, curr_run, mode, model, dl_worker, dp, seq_len, bacc, seed
        )
        base_results = {
            "config": adjusted_config,
            "mode": mode,
            "model": model,
            "dataset_path": str(dataset_path),
        } | additional_info

        if additional_info["skipped"]:
            all_results.append(base_results)
            continue

        run_id = adjusted_config["general"]["run"]
        if run_id not in existing_runs:
            results = run_benchmark(adjusted_config, mode, account, shared_dir)
            all_results.append(base_results | results)
        else:
            typer.echo(f"Info: Skipping {run_id} since it already exists in the logs.")
        persist_results_to_json(output_file, all_results)

        curr_run += 1

    typer.echo("Ran all benchmarks.")


if __name__ == "__main__":
    app()
