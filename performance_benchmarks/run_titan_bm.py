from enum import Enum
import shutil
import threading

import concurrent
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
from transformers import AutoTokenizer

SCRIPT_DIR = Path(os.path.realpath(__file__)).parent
app = typer.Typer()


def check_pandas():
    try:
        import pandas  # implicitly required, otherwise wandb returns different type which breaks stuff

    except ImportError:
        typer.echo("Error: 'pandas' module is not available.")
        raise typer.Exit(code=1)


class ModelType(str, Enum):
    smollm162m = "smollm162m"
    llama1b = "llama1b"
    llama8b = "llama8b"
    llama70b = "llama70b"


class Dataloader(str, Enum):
    hf = "hf"  # use default torchtitan repo, ensure we don't use tiktoken but  hf tokenizer
    hf_stream = "hf_stream"  # use default torchtitan repo with iterable, ensure we don't use tiktoken but  hf tokenizer
    mixtera = "mixtera"  # torchtitan-mixtera
    webdatasets = "webdatasets"  # TBD
    mosaic = "mosaic"  # TBD


# TODO: Find maximum microbatch size per GPU/model combination
MODEL_MICROBATCH = {
    ModelType.smollm162m: 128,  # tested, 256 crashes (on seq leng 1024); seq len > 1024 might require even smaller
    ModelType.llama1b: 2,  # not tested
    ModelType.llama8b: 6,  # not tested
    ModelType.llama70b: 1,  # not tested
}

MODEL_TOKENS = {
    ModelType.smollm162m: 2000000,
    ModelType.llama1b: 2000000,
    ModelType.llama8b: 2000000,
    ModelType.llama70b: 2000000,
}

OMP_NUM_THREADS = "64"

SLURM_PARTITION = "normal"
SLURM_TIME = "00:15:00"
SLURM_GPUS_PER_TASK = 4

SHARED_DIR_DEFAULT = Path(f"/iopsstor/scratch/cscs/{os.environ.get('USER')}/torchtitan-benchmarks")
CONTAINER_ENVIRONMENT = f"/users/mbther/.edf/torchtitan.toml"
TORCHTITAN_PATH = f"/users/{os.environ.get('USER')}/torchtitan-mixtera"
MIXTERA_PATH = f"/users/{os.environ.get('USER')}/mixtera"


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


def get_data_from_wandb(project: str, run_id: str, num_steps: int, retry: int = 0) -> dict:
    api = wandb.Api()
    # Retrieve all runs and sort them by creation date in descending order
    runs = sorted(api.runs(project), key=lambda x: x.created_at, reverse=True)
    run = next((run for run in runs if run.name.split("_", 2)[-1].startswith(run_id)), None)

    if not run:
        typer.echo(f"Error: Could not find run {run_id} in runs {[run.name for run in runs]}.")
        raise typer.Exit(code=1)

    timeout = 600  # seconds
    start_time = time.time()
    result = {}
    while time.time() - start_time < timeout:
        if run.state == "finished":
            result = run.history().to_dict()
            if str(num_steps - 1) not in result["global_tps"].keys():
                max_key = max(int(key) for key in result["global_tps"].keys())
                typer.echo(
                    f"Run finished on wandb, but max key currently is {max_key}, waiting for key {num_steps - 1}."
                )
            else:
                break

        typer.echo("Sleeping for 10 seconds before getting data again from wandb.")
        time.sleep(10)
        api = wandb.Api()
        runs = sorted(api.runs(project), key=lambda x: x.created_at, reverse=True)
        run = next((run for run in runs if run.name.split("_", 2)[-1].startswith(run_id)), None)

    if run.state != "finished":
        typer.echo("Timeout reached. Run did not finish in 10 minutes.")
        raise typer.Exit(code=1)

    if "global_tps" not in result.keys() and retry < 5:
        retry += 1
        return get_data_from_wandb(project, run_id, retry)

    return result


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
    dump_folder: str,
    tokenizer: str,
    dataset_path: Path,
    bm_identifier: str,
    curr_run: int,
    model: ModelType,
    dl_worker: int,
    dp: int,
    ngpu: int,
    seq_length: int,
    seed: int,
    dataloader: Dataloader,
    vocab_size: int,
    mixtera_chunk_size: int,
    mixtera_chunk_reading_degree_of_parallelism: int,
    fileformat: str,
) -> tuple[dict, dict]:
    config = deepcopy(base_config)
    if "mixtera" not in config:
        config["mixtera"] = {}

    run_name = f"run{curr_run}_dprep{dp}_ngpu{ngpu}_{seed}_w{dl_worker}_s{seq_length}_{model}_{dataloader}_{mixtera_chunk_size}_{mixtera_chunk_reading_degree_of_parallelism}_{fileformat}"
    config["metrics"]["wandb_run_name"] = run_name
    config["mixtera"]["job_id"] = run_name
    config["metrics"]["wandb_project"] = bm_identifier
    config["job"]["description"] = f"{bm_identifier}/{run_name}"
    config["job"]["dump_folder"] = f"{dump_folder}/{run_name}_dumpdir"

    config["model"]["flavor"] = model.name

    # Set number of dp nodes
    config["training"]["data_parallel_replicate_degree"] = dp
    config["training"]["data_parallel_shard_degree"] = -1

    config["training"]["tokenizer"] = tokenizer

    # Set microbatch size
    config["training"]["batch_size"] = MODEL_MICROBATCH[model]

    # Set sequence length
    config["training"]["seq_len"] = seq_length

    # Set number of data loading workers
    config["training"]["dl_worker"] = dl_worker
    # Set seed
    config["training"]["seed"] = seed

    # Set dataset path
    config["training"]["dataset"] = f"benchmark_{fileformat.replace('.', '')}"
    config["training"]["dataset_path"] = str(dataset_path / fileformat)

    # Handle dataloader
    if dataloader == Dataloader.hf:
        config["training"]["dataloader"] = "huggingface"
        config["training"]["disable_streaming"] = True
    elif dataloader == Dataloader.hf_stream:
        config["training"]["dataloader"] = "huggingface"
        config["training"]["disable_streaming"] = False
    elif dataloader == Dataloader.mixtera:
        config["training"]["dataloader"] = "mixtera"
        config["mixtera"]["vocab_size"] = vocab_size
        config["mixtera"]["chunk_size"] = mixtera_chunk_size
        config["mixtera"]["tunnel_via_server"] = False
        config["mixtera"]["chunk_reading_degree_of_parallelism"] = mixtera_chunk_reading_degree_of_parallelism
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


def run_mixtera_server(
    config: dict, account: str | None, shared_dir: Path, partition: str, mixtera_server_path: str
) -> tuple[str, str, str, int]:
    job_name = config["metrics"]["wandb_run_name"]
    server_job_name = f"{job_name}_mixtera_server"
    output_file = shared_dir / f"{server_job_name}.out"
    error_file = shared_dir / f"{server_job_name}.err"
    server_ip_file = shared_dir / f"{server_job_name}_ip.txt"
    mixtera_port = 1234
    server_ip_file.unlink(missing_ok=True)

    job_server_path = f"{shared_dir}/{job_name}_mixserv"
    shutil.rmtree(job_server_path, ignore_errors=True)

    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={server_job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --time={SLURM_TIME}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={partition}
#SBATCH --environment={CONTAINER_ENVIRONMENT}
"""

    if account:
        sbatch_header += f"#SBATCH --account={account}\n"

    # Commands to run in the server job
    commands = f"""
set -eo pipefail

# Create cache directory for Mixtera server
echo 'Copying Mixtera server directory'
cp -r {mixtera_server_path} {job_server_path}

echo 'Checking cmake version'
cmake --version

# Install Mixtera
cd {MIXTERA_PATH}
n=0
until [ $n -ge 5 ]
do
   if pip install -e .; then
      echo 'pip install succeeded after try ('$n')'
      break
   else
      n=$(($n+1))
      echo 'pip install failed, retrying ('$n')'
      sleep 1
   fi
done
if [ $n -ge 5 ]; then
   echo 'pip install failed after 5 retries'
   exit 1
fi

echo "Getting server IP address"
SERVER_IP=$(hostname -I | awk '{{print $1}}')
echo "Server IP: $SERVER_IP"

# Write the server IP to a file
echo "Writing server IP to file"
echo $SERVER_IP > {server_ip_file}
sleep 1

# Start the Mixtera server
numactl --membind=0-3 python -u -m mixtera.network.server.entrypoint {job_server_path} --host $SERVER_IP --port {mixtera_port}
"""

    sbatch_script = sbatch_header + commands
    sbatch_script_path = shared_dir / f"{server_job_name}_run.slurm"
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    typer.echo(f"Submitting Mixtera server job {server_job_name} with sbatch script {sbatch_script_path}")
    submit_command = ["sbatch", str(sbatch_script_path)]
    proc = subprocess.run(submit_command, capture_output=True, text=True, env=get_no_conda_env())
    if proc.returncode != 0:
        typer.echo(f"Error submitting Mixtera server job: {proc.stderr}")
        raise RuntimeError("Failed to submit Mixtera server job.")
    else:
        typer.echo(f"Mixtera server job submitted: {proc.stdout}")
        server_job_id = None
        for line in proc.stdout.strip().split("\n"):
            if "Submitted batch job" in line:
                server_job_id = line.strip().split()[-1]
                break
        if server_job_id is None:
            typer.echo("Could not determine Mixtera server job ID from sbatch output.")
            raise RuntimeError("Failed to determine Mixtera server job ID.")

    # Wait until the server IP file appears, indicating that the server job has started
    max_wait_time = 600  # Maximum wait time in seconds (10 minutes)
    wait_interval = 5  # Wait interval in seconds
    elapsed_time = 0

    typer.echo(f"Waiting for Mixtera server to start, checking for IP file: {server_ip_file}")
    while not server_ip_file.exists() and elapsed_time < max_wait_time:
        # Check if the server job is still running
        squeue_cmd = ["squeue", "-j", str(server_job_id)]
        squeue_proc = subprocess.run(squeue_cmd, capture_output=True, text=True, env=get_no_conda_env())

        if squeue_proc.returncode != 0:
            raise RuntimeError(f"Error checking Mixtera server job status: {squeue_proc.stderr}")

        if str(server_job_id) not in squeue_proc.stdout:
            # Job is no longer running - check its exit status
            sacct_cmd = [
                "sacct",
                "-j",
                str(server_job_id),
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
                    raise RuntimeError(
                        f"Mixtera server job failed to start. Job State: {state}, Exit Code: {exit_code}. "
                        f"Please check the logs at {output_file} for details."
                    )

            raise RuntimeError("Mixtera server job ended unexpectedly before starting up")

        time.sleep(wait_interval)
        elapsed_time += wait_interval

    if not server_ip_file.exists():
        raise RuntimeError("Timed out waiting for Mixtera server to start.")

    # Read the server IP address from the file
    with open(server_ip_file, "r") as f:
        mixtera_server_ip = f.read().strip()

    typer.echo("Waiting an additional 10 seconds for Mixtera server to fully start...")
    time.sleep(10)

    typer.echo(f"Mixtera server is running at {mixtera_server_ip}:{mixtera_port}")

    return server_job_id, job_server_path, mixtera_server_ip, mixtera_port


def run_benchmark(
    config: dict,
    ngpu: int,
    account: str | None,
    shared_dir: Path,
    debug_partition: bool,
    mixtera_server_path: str,
    lock,
) -> dict:
    # Ensure shared directory exists
    with lock:
        shared_dir.mkdir(parents=True, exist_ok=True)
        data_cache_dir = shared_dir / "hf_data"
        hf_home_dir = shared_dir / "hf_home"
        data_cache_dir.mkdir(parents=True, exist_ok=True)
        hf_home_dir.mkdir(parents=True, exist_ok=True)

    job_name = config["metrics"]["wandb_run_name"]
    dp = ngpu
    tp = 1
    pp = 1

    num_gpus = dp * tp * pp
    num_nodes = math.ceil(num_gpus / SLURM_GPUS_PER_TASK)

    proc_per_node = SLURM_GPUS_PER_TASK
    if num_gpus < 4 and num_nodes == 1:
        proc_per_node = num_gpus

    if num_nodes > 1 and num_gpus % SLURM_GPUS_PER_TASK != 0:
        raise RuntimeError(
            f"num_nodes = {num_nodes} > 1 and num_gpus = {num_gpus} not divisible by {SLURM_GPUS_PER_TASK}, this is currently not supported."
        )

    # Save the benchmark configuration in the shared directory
    bm_config_path = shared_dir / f"{job_name}_benchmark.toml"
    with open(bm_config_path, "w+") as f:
        toml.dump(config, f)

    # Paths for logs
    with lock:
        log_dir = shared_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    output_file = log_dir / f"{job_name}_output.log"
    error_file = log_dir / f"{job_name}_output.err"

    partition = SLURM_PARTITION if not debug_partition else "debug"

    mixtera_ip = "no_mixtera"
    mixtera_port = 1234
    mixtera_server_job_id = None
    mixtera_server_dir = None
    if config["training"]["dataloader"] == "mixtera":
        mixtera_server_job_id, mixtera_server_dir, mixtera_ip, mixtera_port = run_mixtera_server(
            config, account, shared_dir, partition, mixtera_server_path
        )

    sbatch_header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time={SLURM_TIME}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={partition}
#SBATCH --no-requeue
#SBATCH --gpus-per-task={SLURM_GPUS_PER_TASK}
#SBATCH --environment={CONTAINER_ENVIRONMENT}
"""

    if account is not None and account != "":
        sbatch_header += f"\n#SBATCH --account={account}"

    env_vars = f"""
export OMP_NUM_THREADS={OMP_NUM_THREADS}
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

    master_setup = f"""
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
"""

    # Commands to run
    commands = f"""
set -eo pipefail
{master_setup}

srun -ul --container-writable --environment={CONTAINER_ENVIRONMENT} bash -c "
echo 'Checking cmake version'
cmake --version

# Install torchtitan
pushd {TORCHTITAN_PATH}
pip install -e .
popd

# Install mixtera
pushd {MIXTERA_PATH}
n=0

until [ \$n -ge 5 ]
do
   if pip install -e .; then
      echo 'pip install succeeded after try ('\$n')'
      break
   else
      n=\$((\$n+1))
      echo 'pip install failed, retrying ('\$n')'
      sleep 1
   fi
done
if [ \$n -ge 5 ]; then
   echo 'pip install failed after 5 retries'
   exit 1
fi

popd

numactl --membind=0-3 torchrun --nnodes={num_nodes} --nproc_per_node={proc_per_node} --rdzv_backend c10d --rdzv_endpoint '$head_node_ip:29500' {TORCHTITAN_PATH}/train.py --job.config_file {bm_config_path} --mixtera.ip "{mixtera_ip}" --mixtera.port {mixtera_port}

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
    proc = subprocess.run(submit_command, capture_output=True, text=True, env=get_no_conda_env())
    result = {}

    if mixtera_server_job_id:
        result["mixtera_server_job_id"] = mixtera_server_job_id
        result["mixtera_server_dir"] = mixtera_server_dir

    if proc.returncode != 0:
        typer.echo(f"Error submitting job: {proc.stderr}")
        result["success"] = False
        result["job_id"] = None
        return result
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
            result["success"] = False
            result["job_id"] = None
            return result
        else:
            result["success"] = True
            result["job_id"] = job_id
            return result


def cancel_mixtera_server(server_job_id):
    typer.echo(f"Cancelling Mixtera server job {server_job_id}")
    scancel_cmd = ["scancel", str(server_job_id)]
    proc = subprocess.run(scancel_cmd, capture_output=True, text=True, env=get_no_conda_env())
    if proc.returncode != 0:
        typer.echo(f"Warning: Failed to cancel Mixtera server job {server_job_id}: {proc.stderr}")
    else:
        typer.echo(f"Successfully cancelled Mixtera server job {server_job_id}")


def run_experiment(
    adjusted_config,
    base_results,
    fileformat,
    ngpu,
    shared_dir,
    debug_partition,
    account,
    mixtera_server_path,
    output_file,
    all_results,
    lock,
):
    server_path = f"{mixtera_server_path}/{fileformat}"
    run_id = adjusted_config["metrics"].get("wandb_run_name", "no_run_id?!")
    job_info = run_benchmark(adjusted_config, ngpu, account, shared_dir, debug_partition, server_path, lock)
    if job_info["success"]:
        mixtera_server_job_id = job_info.get("mixtera_server_job_id")
        mixtera_server_dir = job_info.get("mixtera_server_dir")
        job_id = job_info["job_id"]

        # Wait for the job to complete and collect results
        job_complete = False
        while not job_complete:
            time.sleep(10)
            squeue_cmd = ["squeue", "-j", str(job_id)]
            squeue_proc = subprocess.run(squeue_cmd, capture_output=True, text=True, env=get_no_conda_env())
            if job_id in squeue_proc.stdout:
                # Job is still running
                continue

            # Job has completed, check exit status and collect results
            job_complete = True
            sacct_cmd = ["sacct", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode", "--parsable2", "--noheader"]
            sacct_proc = subprocess.run(sacct_cmd, capture_output=True, text=True, env=get_no_conda_env())

            if sacct_proc.returncode != 0:
                typer.echo(f"Error checking job exit status: {sacct_proc.stderr}")
                with lock:
                    all_results.append(base_results | {"success": False})
                    persist_results_to_json(output_file, all_results)

                if mixtera_server_job_id:
                    cancel_mixtera_server(mixtera_server_job_id)
                    shutil.rmtree(mixtera_server_dir, ignore_errors=True)
            else:
                sacct_output = sacct_proc.stdout.strip()
                if not sacct_output:
                    typer.echo(f"No accounting information found for job {job_id}.")
                    with lock:
                        all_results.append(base_results | {"success": False})
                        persist_results_to_json(output_file, all_results)

                    if mixtera_server_job_id:
                        cancel_mixtera_server(mixtera_server_job_id)
                        shutil.rmtree(mixtera_server_dir, ignore_errors=True)

                job_info = sacct_output.split("|")
                if len(job_info) >= 3:
                    state = job_info[1]
                    exit_code = job_info[2]

                    if mixtera_server_job_id:
                        cancel_mixtera_server(mixtera_server_job_id)
                        shutil.rmtree(mixtera_server_dir, ignore_errors=True)

                    if state != "COMPLETED" or not exit_code.startswith("0:0"):
                        typer.echo(f"Job {job_id} did not complete successfully.")
                        typer.echo(f"Job State: {state}, Exit Code: {exit_code}")
                        typer.echo(f"Please check the logs for details.")
                        with lock:
                            all_results.append(base_results | {"success": False})
                            persist_results_to_json(output_file, all_results)
                    else:
                        typer.echo(f"Job {job_id} (run id {run_id}) completed successfully.")
                        # Collect results from WandB
                        typer.echo("Collecting data from Weights & Biases.")
                        results = get_data_from_wandb(
                            adjusted_config["metrics"]["wandb_project"], adjusted_config["metrics"]["wandb_run_name"]
                        ) | {"success": True}
                        with lock:
                            all_results.append(base_results | results)
                            persist_results_to_json(output_file, all_results)
                else:
                    typer.echo(f"Unexpected sacct output: {sacct_output}")
                    with lock:
                        all_results.append(base_results | {"success": False})
                        persist_results_to_json(output_file, all_results)
    else:
        with lock:
            all_results.append(base_results | {"success": False})
            persist_results_to_json(output_file, all_results)
        if "mixtera_server_job_id" in job_info:
            cancel_mixtera_server(job_info["mixtera_server_job_id"])
            shutil.rmtree(job_info["mixtera_server_dir"], ignore_errors=True)


@app.command()
def run_benchmarks(
    output_dir: Path,
    benchmark_name: str,
    model: ModelType,
    dataset_path: Path = Path(
        "/iopsstor/scratch/cscs/mbther/benchmark_data"
    ),  # relevant for non-mixtera. base directory for data
    mixtera_server_path: str = "/iopsstor/scratch/cscs/mbther/benchmark_mixtera_server",  # directory containing the prepared mixtera server dirs
    dl_workers: list[int] = [0, 1, 2, 4],
    dp_replicate_deg: list[int] = [1, 2, 4, 12],  # shard degree = ngpus / dp_replicate_deg
    ngpus: list[int] = [1, 4, 8, 12, 24],  # each node has 4 GPUs
    seq_lengths: list[int] = [1024, 2048],
    seeds: list[int] = [42],
    dataloaders: list[Dataloader] = [Dataloader.hf, Dataloader.hf_stream],
    fileformats: list[str] = ["jsonl"],  # supported: jsonl, jsonl.zst, parquet, webdatasets
    huggingface_cache_path: Path = Path(f"{SHARED_DIR_DEFAULT}/hfcache"),
    skip_existing: bool = False,
    account: str | None = None,
    shared_dir: Path = SHARED_DIR_DEFAULT,
    debug_partition: bool = False,
    parallel_slurm_jobs: int = 1,
    tokenizer: str = "EleutherAI/gpt-neox-20b",
    mixtera_chunk_sizes: list[int] = [512],
    mixtera_chunk_reading_degree_of_parallelisms: list[int] = [1],
):
    check_pandas()

    if not Path(TORCHTITAN_PATH).exists():
        raise RuntimeError(f"Cannot find torchtitan at {TORCHTITAN_PATH}")

    if not Path(MIXTERA_PATH).exists():
        raise RuntimeError(f"Cannot find mixtera at {MIXTERA_PATH}")

    if Dataloader.mixtera in dataloaders and (
        mixtera_server_path is None or mixtera_server_path == "" or not Path(mixtera_server_path).exists()
    ):
        raise RuntimeError(f"Mixtera server path {mixtera_server_path} does not exist, and you want to use Mixtera.")

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
            _run_id = item["config"]["metrics"].get("wandb_run_name", None)
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
        typer.echo(
            f"Note that the hf cache path {huggingface_cache_path} does not exist. Using default path by hf. Is that ok?"
        )
        ask_to_continue()

    # Calculate Mixtera vocab size as for huggingface tokenizer in torchtitan
    vocab_size = -1
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    vocab_size = max(tokenizer_obj.vocab_size, len(tokenizer_obj)) + 100
    typer.echo(f"Determined vocab size {vocab_size} for tokenizer {tokenizer}")
    del tokenizer_obj
    total_runs = 0

    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_slurm_jobs) as executor:
        futures = []
        for seed, dl_worker, dp, ngpu, seq_len, dataloader, mix_cs, mix_crdop, fileformat in tqdm(
            list(
                itertools.product(
                    seeds,
                    dl_workers,
                    dp_replicate_deg,
                    ngpus,
                    seq_lengths,
                    dataloaders,
                    mixtera_chunk_sizes,
                    mixtera_chunk_reading_degree_of_parallelisms,
                    fileformats,
                )
            ),
            desc="Scheduling experiments",
        ):
            if dataloader != Dataloader.mixtera and (
                mix_cs != mixtera_chunk_sizes[0] or mix_crdop != mixtera_chunk_reading_degree_of_parallelisms[0]
            ):
                continue  # only vary mixtera options for mixtera

            adjusted_config, additional_info = adjust_base_config(
                base_config,
                shared_dir,
                tokenizer,
                dataset_path,
                bm_identifier,
                curr_run,
                model,
                dl_worker,
                dp,
                ngpu,
                seq_len,
                seed,
                dataloader,
                vocab_size,
                mix_cs,
                mix_crdop,
                fileformat,
            )
            base_results = {
                "config": adjusted_config,
                "model": model,
                "dataset_path": str(dataset_path),
            } | additional_info

            if additional_info["skipped"]:
                with lock:
                    all_results.append(base_results)
                continue

            run_id = adjusted_config["metrics"]["wandb_run_name"]
            if run_id not in existing_runs:
                future = executor.submit(
                    run_experiment,
                    adjusted_config,
                    base_results,
                    fileformat,
                    ngpu,
                    shared_dir,
                    debug_partition,
                    account,
                    mixtera_server_path,
                    output_file,
                    all_results,
                    lock,
                )
                futures.append(future)
                total_runs += 1

        else:
            typer.echo(f"Info: Skipping {run_id} since it already exists in the logs.")

        curr_run += 1

    typer.echo("Scheduled all experiments.")
    # After all jobs are submitted, wait for remaining jobs to finish
    for future in tqdm(concurrent.futures.as_completed(futures), desc="Collecting futures", total=total_runs):
        try:
            future.result()
        except Exception as exc:
            typer.echo(f"Experiment generated an exception: {exc}")

    persist_results_to_json(output_file, all_results)

    typer.echo("Ran all benchmarks.")


if __name__ == "__main__":
    app()
