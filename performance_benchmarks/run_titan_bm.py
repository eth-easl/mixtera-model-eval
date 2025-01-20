from enum import Enum
import shutil
import typer
from pathlib import Path
import yaml
import toml
import subprocess
import os
import itertools
import json
from copy import deepcopy
import wandb
import time
import math
from tqdm import tqdm

SCRIPT_DIR = Path(os.path.realpath(__file__)).parent
app = typer.Typer()


class ModelType(str, Enum):
    smollm135m = "smollm135m"
    llama1b = "llama1b"
    llama8b = "llama8b"
    llama70b = "llama70b"


class Dataloader(str, Enum):
    hf = "hf" # use default torchtitan repo, ensure we don't use tiktoken but  hf tokenizer
    hf_stream = "hf_stream" # use default torchtitan repo with iterable, ensure we don't use tiktoken but  hf tokenizer
    mixtera = "mixtera" # torchtitan-mixtera
    webdatasets = "webdatasets" # TBD
    mosaic = "mosaic" # TBD

# TODO: Find maximum microbatch size per GPU/model combination
MODEL_MICROBATCH = {
    ModelType.smollm135m: 256,  # not tested
    ModelType.llama1b: 2,  # not tested
    ModelType.llama8b: 6,  # not tested
    ModelType.llama70b: 1,  # not tested
}

MODEL_TOKENS = {
    ModelType.smollm135m: 2000000,
    ModelType.llama1b: 2000000,
    ModelType.llama8b: 2000000,
    ModelType.llama70b: 2000000,
}

OMP_NUM_THREADS = "64"

SLURM_PARTITION = "normal"
SLURM_TIME = "00:15:00"
SLURM_ACCOUNT = "a06"
SLURM_GPUS_PER_TASK = 4

SHARED_DIR_DEFAULT = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER')}/torchtitan-benchmarks")
CONTAINER_ENVIRONMENT = f"/users/mbther/.edf/torchtitan.toml"
TORCHTITAN_PATH = f"/users/{os.environ.get('USER')}/torchtitan-mixtera"

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

def load_base_config() -> dict:
    return load_toml_from_file(SCRIPT_DIR / "titan" / "base.toml")

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
        "global_tps" not in run.history().to_dict().keys()
        and retry < 5
    ):
        retry += 1
        return get_data_from_wandb(project, run_id, retry)

    return run.history().to_dict()

def load_yaml_from_file(path: str | Path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            typer.echo("Error: " + str(exc))
            raise typer.Exit(code=1) from exc

def load_toml_from_file(path: str | Path):
    with open(path, "r") as stream:
        try:
            return toml.load(stream)
        except toml.TomlDecodeError as exc:
            typer.echo("Error: " + str(exc))
            raise typer.Exit(code=1) from exc 

def persist_results_to_json(output: Path, all_results: list[dict]):
    with open(output, "w+") as fout:
        json.dump(all_results, fout, indent=4)

def adjust_base_config(
    base_config: dict,
    dataset_path: Path,
    bm_identifier: str,
    curr_run: int,
    model: ModelType,
    dl_worker: int,
    dp: int,
    ngpu: int,
    seq_length: int,
    seed: int,
    dataloader: Dataloader
) -> tuple[dict, dict]:
    config = deepcopy(base_config)
    if "mixtera" not in config:
        config["mixtera"] = {}

    run_name = (
        f"run{curr_run}_dprep{dp}_ngpu{ngpu}_seed{seed}_w{dl_worker}_s{seq_length}_{model}_{dataloader}"
    )
    config["metrics"]["wandb_run_name"] = run_name
    config["mixtera"]["job_id"] = run_name
    config["metrics"]["wandb_project"] = bm_identifier
    config["job"]["description"] = bm_identifier

    config["model"]["flavor"] = str(model)

    # Set number of dp nodes
    config["training"]["data_parallel_replicate_degree"] = dp
    config["training"]["data_parallel_shard_degree"] = "-1"

    # Set microbatch size
    config["training"]["batch_size"] = MODEL_MICROBATCH[model]

    # Set sequence length
    config["training"]["seq_len"] = seq_length

    # Set number of data loading workers
    config["training"]["dl_worker"] = dl_worker
    # Set seed
    config["training"]["seed"] = seed

    # Set dataset path
    config["training"]["dataset"] = str(dataset_path)
    config["training"]["dataset_path"] = str(dataset_path)

    # Handle dataloader
    if dataloader == Dataloader.hf:
        config["training"]["dataloader"] = "huggingface"
        config["training"]["disable_streaming"] = True
    elif dataloader == Dataloader.hf_stream:
        config["training"]["dataloader"] = "huggingface"
        config["training"]["disable_streaming"] = False
    elif dataloader == Dataloader.mixtera:
        raise NotImplementedError()
        config["data_stages"][0]["data"]["dataset"] = { "path": "localhost", "port": 8888, "job_id": config["general"]["run"], "chunk_size": tODO, "tunnel_via_server": todo, "chunk_reading_degree_of_parallelism": todo, "chunk_reading_mixture_type": todo, "chunk_reading_window_size": todo, "chunk_reading_prefetch_first_sample": todo}
        # chunk_reading_tokenizer, chunk_reading_sequence_len, chunk_reading_tokenization_bs
        pass # todo we have to change config["data_stages"][0]["data"]["dataset"] to a mixtera conifg
    else:
        raise NotImplementedError(f"Dataloader {dataloader} not yet supported.")

    # Number of tokens that we want to consume (will be rounded up to match batch size/seq length)
    scheduled_total_tokens = MODEL_TOKENS[model] * ngpu
    batch_size = ngpu * config["training"]["batch_size"]
    tokens_per_step = batch_size * seq_length
    train_steps = max(
        math.ceil(scheduled_total_tokens / tokens_per_step), 30
    )  # Minimum of 30 training steps per benchmark
    config["training"]["steps"] = train_steps
    calculated_total_tokens = train_steps * tokens_per_step

    additional_info = {
        "scheduled_total_tokens": scheduled_total_tokens,
        "calculated_total_tokens": calculated_total_tokens,
        "tokens_per_step": tokens_per_step,
        "batch_size": batch_size,
        "skipped": False,
        "skip_reason": "",
    }

    if dp >= ngpu and dp > 1:
        additional_info["skipped"] = True
        additional_info["skip_reason"] = f"dp = {dp} >= ngpu = {ngpu}" 
        # we cannot have dp == ngpu because that deactivates FSDP, which deactivates bfloat16
    
    if ngpu % dp != 0:
        additional_info["skipped"] = True
        additional_info["skip_reason"] = f"ngpu = {ngpu} % dp = {dp} != 0 (= {ngpu % dp})"

    return config, additional_info


def check_running_jobs(running_jobs, all_results, output_file):
    updated_running_jobs = []
    for job in running_jobs:
        job_id = job["job_id"]
        base_results = job["base_results"]
        adjusted_config = job["config"]

        # Check if the job is still running
        squeue_cmd = ["squeue", "-j", str(job_id)]
        squeue_proc = subprocess.run(squeue_cmd, capture_output=True, text=True)
        if squeue_proc.returncode != 0:
            typer.echo(f"Error checking job status: {squeue_proc.stderr}")
            # Consider the job as completed with failure
            all_results.append(base_results | {"success": False})
            persist_results_to_json(output_file, all_results)
            continue
        elif job_id in squeue_proc.stdout:
            # Job is still running
            updated_running_jobs.append(job)
        else:
            # Job has completed, check exit status and collect results
            sacct_cmd = ["sacct", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode", "--parsable2", "--noheader"]
            sacct_proc = subprocess.run(sacct_cmd, capture_output=True, text=True)

            if sacct_proc.returncode != 0:
                typer.echo(f"Error checking job exit status: {sacct_proc.stderr}")
                all_results.append(base_results | {"success": False})
                persist_results_to_json(output_file, all_results)
                continue
            else:
                sacct_output = sacct_proc.stdout.strip()
                if not sacct_output:
                    typer.echo(f"No accounting information found for job {job_id}.")
                    all_results.append(base_results | {"success": False})
                    persist_results_to_json(output_file, all_results)
                    continue

                job_info = sacct_output.split("|")
                if len(job_info) >= 3:
                    state = job_info[1]
                    exit_code = job_info[2]

                    if state != "COMPLETED" or not exit_code.startswith("0:0"):
                        typer.echo(f"Job {job_id} did not complete successfully.")
                        typer.echo(f"Job State: {state}, Exit Code: {exit_code}")
                        typer.echo(f"Please check the logs for details.")
                        all_results.append(base_results | {"success": False})
                    else:
                        typer.echo(f"Job {job_id} completed successfully.")
                        # Collect results from WandB
                        typer.echo("Collecting data from Weights & Biases.")
                        results = get_data_from_wandb(
                            adjusted_config["metrics"]["wandb_project"], adjusted_config["metrics"]["wandb_run_name"]
                        ) | {"success": True}
                        all_results.append(base_results | results)
                    persist_results_to_json(output_file, all_results)
                else:
                    typer.echo(f"Unexpected sacct output: {sacct_output}")
                    all_results.append(base_results | {"success": False})
                    persist_results_to_json(output_file, all_results)
    return updated_running_jobs

def run_benchmark(config: dict, ngpu: int, account: str, shared_dir: Path, debug_partition: bool) -> dict:
    # Ensure shared directory exists
    shared_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir = shared_dir / "hf_data"
    hf_home_dir = shared_dir / "hf_home"
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home_dir.mkdir(parents=True, exist_ok=True)

    if not Path(TORCHTITAN_PATH).exists():
        raise RuntimeError(f"Cannot find torchtitan at {TORCHTITAN_PATH}")

    job_name = config["metrics"]["wandb_run_name"]
    dp = ngpu
    tp = 1
    pp = 1

    num_gpus = dp * tp * pp
    num_nodes = math.ceil(num_gpus / SLURM_GPUS_PER_TASK)

    proc_per_node = SLURM_GPUS_PER_TASK
    if num_gpus < 4 and num_nodes == 1:
        proc_per_node = num_gpus # ensure we use this for torchtitan

    if num_nodes > 1 and num_gpus % SLURM_GPUS_PER_TASK != 0:
        raise RuntimeError(
            f"num_nodes = {num_nodes} > 1 and num_gpus = {num_gpus} not divisible by {SLURM_GPUS_PER_TASK}, this is currently not supported."
        )

    # Save the benchmark configuration in the shared directory
    bm_config_path = shared_dir / f"{job_name}_benchmark.toml"
    with open(bm_config_path, "w+b") as f:
        toml.dump(config, f)

    # Paths for logs
    log_dir = shared_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_file = log_dir / f"{job_name}_output.log"
    error_file = log_dir / f"{job_name}_output.err"

    partition = SLURM_PARTITION if not debug_partition else "debug"

    # TODO: run mixtera server on separate node if mixtera

    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time={SLURM_TIME}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={partition}
#SBATCH --no-requeue
#SBATCH --gpus-per-task={SLURM_GPUS_PER_TASK}
"""

    env_vars = f"""
export OMP_NUM_THREADS={OMP_NUM_THREADS}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_DIR={shared_dir}/wandb
export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}
export HF_TOKEN={os.environ.get('HF_TOKEN', '')}
export HF_HOME={hf_home_dir}
export HF_DATASETS_CACHE={data_cache_dir}
export LOGLEVEL=DEBUG
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
"""
    
    master_setup = """
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
"""

    # Commands to run
    commands = f"""
set -eo pipefail
{master_setup}

srun -ul --container-writable --environment={CONTAINER_ENVIRONMENT} bash -c "
pushd {TORCHTITAN_PATH}
pip install -e .
popd

# TODO: install mixtera when we support it

numactl --membind=0-3 torchrun --nnodes={num_nodes} --nproc_per_node={SLURM_GPUS_PER_TASK} --rdzv_backend c10d --rdzv_endpoint '$head_node_ip:29500' {TORCHTITAN_PATH}/train.py --job.config_file {bm_config_path} --mixtera.ip "todo" --mixtera.port 1234

"
"""

    # Combine all parts into the SBATCH script
    sbatch_script = sbatch_header + env_vars + commands

    # Save SBATCH script to file in the shared directory
    sbatch_script_path = shared_dir / f"{job_name}_run.slurm"
    with open(sbatch_script_path, "w+") as f:
        f.write(sbatch_script)

    # Submit the job
    typer.echo(f"Submitting job {job_name} with sbatch script {sbatch_script_path}")
    submit_command = ["sbatch", str(sbatch_script_path)]
    proc = subprocess.run(submit_command, capture_output=True, text=True)
    if proc.returncode != 0:
        typer.echo(f"Error submitting job: {proc.stderr}")
        return {"success": False, "job_id": None}
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
            return {"success": False, "job_id": None}
        else:
            return {"success": True, "job_id": job_id}


@app.command()
def run_benchmarks(
    output_dir: Path,
    benchmark_name: str,
    model: ModelType,
    dataset_path: Path,
    dl_workers: list[int] = [0, 1, 2, 4],
    dp_replicate_deg: list[int] = [1, 2, 4, 12], # shard degree = ngpus / dp_replicate_deg
    ngpus: list[int] = [1, 4, 8, 12, 24], # each node has 4 GPUs
    seq_lengths: list[int] = [1024, 2048],
    seeds: list[int] = [42],
    dataloaders: list[Dataloader] = [Dataloader.hf, Dataloader.hf_stream],
    mixtera_server_path: str = "", # For now, we need to set up the Mixtera server directory manually
    huggingface_cache_path: Path = Path(f"{SHARED_DIR_DEFAULT}/hfcache"),
    skip_existing: bool = False,
    account: str = SLURM_ACCOUNT,
    shared_dir: Path = SHARED_DIR_DEFAULT,
    debug_partition: bool = False,
    parallel_slurm_jobs: int = 1,
):
    if parallel_slurm_jobs < 1:
        typer.echo("Error: parallel_slurm_jobs must be at least 1.")
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

    if not all(ngpu == 1 or ngpu % 4 == 0 for ngpu in ngpus):
        typer.echo("Error: ngpu(s) need to be 1, or multiple of 4.")
        raise typer.Exit(code=1)

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
            _run_id = item["metrics"].get("wandb_run_name", None)
            if _run_id is not None and _run_id != "":
                existing_runs.add(_run_id)

    base_config = load_base_config()
    bm_identifier = "dataloader_benchmark_script"
    curr_run = 0

    if huggingface_cache_path.exists():
        dataset_cache_path = huggingface_cache_path / "data"
        home_path = huggingface_cache_path / "home"
        dataset_cache_path.mkdir(exist_ok=True)
        home_path.mkdir(exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(dataset_cache_path)
        os.environ["HF_HOME"] = str(home_path)
    else:
        typer.echo(f"Note that the hf cache path {huggingface_cache_path} does not exist. Using default path by hf. Is that ok?")
        ask_to_continue()

    running_jobs = []

    for seed, dl_worker, dp, ngpu, seq_len, dataloader in tqdm(
        list(itertools.product(seeds, dl_workers, dp_replicate_deg, ngpus, seq_lengths, dataloaders)),
        desc="Processing configurations",
    ):
        adjusted_config, additional_info = adjust_base_config(
            base_config, dataset_path, bm_identifier, curr_run, model, dl_worker, dp, ngpu, seq_len, seed, dataloader
        )
        base_results = {
            "config": adjusted_config,
            "model": model,
            "dataset_path": str(dataset_path),
        } | additional_info

        if additional_info["skipped"]:
            all_results.append(base_results)
            continue

        run_id = adjusted_config["metrics"]["wandb_run_name"]
        if run_id not in existing_runs:
            while len(running_jobs) >= parallel_slurm_jobs:
                typer.echo(f"Maximum parallel jobs reached ({parallel_slurm_jobs}). Waiting for a job to finish...")
                running_jobs = check_running_jobs(running_jobs, all_results, output_file)
                time.sleep(10)

            job_info = run_benchmark(adjusted_config, ngpu, account, shared_dir, debug_partition)
            if job_info["success"]:
                running_jobs.append(
                    {
                        "job_id": job_info["job_id"],
                        "base_results": base_results,
                        "config": adjusted_config,
                    }
                )
            else:
                all_results.append(base_results | {"success": False})
                persist_results_to_json(output_file, all_results)
        else:
            typer.echo(f"Info: Skipping {run_id} since it already exists in the logs.")

        curr_run += 1

    # After all jobs are submitted, wait for remaining jobs to finish
    if running_jobs:
        typer.echo("Waiting for all submitted jobs to finish...")
        while running_jobs:
            running_jobs = check_running_jobs(running_jobs, all_results, output_file)
            if running_jobs:
                time.sleep(10)

    persist_results_to_json(output_file, all_results)

    typer.echo("Ran all benchmarks.")


if __name__ == "__main__":
    app()
