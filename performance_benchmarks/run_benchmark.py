from enum import Enum
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

# Machine: RTX 3090, H100, GH200
# Model: TinyLlama 1B, Llama7B, Llamba70B
# Model+Machine fixes pp and tp => defines base config
# TODO: Find out parameters for LLama7B and 70B.
# Llama2 vs 3? do they define different architecture params?
# what learning rate schedule to use? (probably irrelevant for tput though)
# Global batch size: Accumulation step, micro batch size
# Might depend on machine/model combination?
# Sequence Length: 1024, 2048, 4096
# Does sequence length influence batch size?
# (it should, for the same batch size if we double num tokens per sec we double the amount of data fed into the model
# Datalocation: Locally on NVMe (if applicable, probably only our machine), NFS
# Interface: Huggingface mapped, Huggingface Iterable, Nanoset, Mosaic Streaming, WebDatasets, Mixtera
# Data-Loading Workers: 0,1,4,16,32
# dp instances: 1,2,4,8,16
# for sgs-gpu machines, there is an inherent constraint since we only try single node
# for swiss ai, let's see how much we can scale up with the baselines and then with mixtera
# Mixtera-Specific Parameters: Mixture Window, Chunk Workers, stream via server, local vs server (only for dp=1, tinyllama 1b)

# This script should build up the cross product of all configurations
# Then execute each configuration
# Either: sgs-gpu machine OR SwissAI cluster
# Each training should run for 5-10 minutes (configurable?)
# Then, collect results (either wandb and/or local logging added to nanotron) after each run, build up csv with results, always persist.


class MachineType(str, Enum):
    sgsrtx = "sgsrtx"
    sgsh100 = "sgsh100"
    cscs = "cscs"


class ModelType(str, Enum):
    tinyllama = "tinyllama"
    llama7b = "llama7b"
    llama70b = "llama70b"


# TODO: Find maximum microbatch size per GPU/model combination
MACHINE_MODEL_MICROBATCH = {
    MachineType.sgsrtx: {
        ModelType.tinyllama: 1,  # tested > 1, goes OOM
        ModelType.llama7b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.sgsh100: {
        ModelType.tinyllama: 4,
        ModelType.llama7b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.cscs: {
        ModelType.tinyllama: -1,
        ModelType.llama7b: -1,
        ModelType.llama70b: -1,
    },
}

MACHINE_MODEL_TOKENS = {
    MachineType.sgsrtx: {
        ModelType.tinyllama: 500000,
        ModelType.llama7b: -1,
        ModelType.llama70b: -1,
    },
    MachineType.sgsh100: {
        ModelType.tinyllama: 2000000,
        ModelType.llama7b: 2000000,
        ModelType.llama70b: 2000000,
    },
    MachineType.cscs: {
        ModelType.tinyllama: 2000000,
        ModelType.llama7b: 2000000,
        ModelType.llama70b: 2000000,
    },
}


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

    timeout = 300  # seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        if run.state == "finished":
            break
        typer.echo("Still waiting for the run to finish on wandb.")
        time.sleep(10)
        runs = api.runs(project)
        run = next((run for run in runs if run.name.split("_", 2)[-1].startswith(run_id)), None)

    if run.state != "finished":
        typer.echo("Timeout reached. Run did not finish in 5 minutes.")
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


def run_benchmark(config: dict, mode: MachineType) -> dict:
    if mode in [MachineType.sgsh100, MachineType.sgsrtx]:
        return run_benchmark_on_sgs(config, mode)

    typer.echo(f"Error: No implementation yet for mode `{mode}`")
    raise typer.Exit(code=1)


def validate_user_input(
    mode: MachineType,
    model: ModelType,
):
    if mode == MachineType.sgsrtx and model not in [ModelType.tinyllama]:
        typer.echo(f"RTX 3090 machines don't support model {model}")
        return False

    if mode in [MachineType.sgsrtx, MachineType.sgsh100]:
        check_nanotron_availability()

        if not run_script_path.exists():
            typer.echo(f"Cannot find run script at {run_script_path}")
            raise typer.Exit(code=1)

        # due to SSH shenanagans, this needs to run on an sgs machine
        hostname = socket.gethostname().split(".")[0]
        if mode == MachineType.sgsh100 and hostname != "sgs-gpu07":
            typer.echo("Note that you selected H100 gpus but this is not sgs-gpu07.")
            if not ask_to_continue():
                raise typer.Exit(code=1)

        if mode == MachineType.sgsrtx and hostname not in ["sgs-gpu01", "sgs-gpu02", "sgs-gpu03", "sgs-gpu04"]:
            typer.echo("Note that you selected RTX3090 gpus but this is not sgs-gpu[01-04].")
            if not ask_to_continue():
                raise typer.Exit(code=1)

    return True


def load_base_config(model: ModelType) -> dict:
    if model == ModelType.tinyllama:
        return load_yaml_from_file(SCRIPT_DIR / "tinyllama.yaml")

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

    # set wandb info
    config["general"]["project"] = bm_identifier
    config["general"]["run"] = f"run{curr_run}_dp{dp}_seed{seed}_w{dl_worker}_s{seq_length}_acc{batch_accumulation_per_replica}_{mode}_{model}"

    # set number of dp nodes
    config["parallelism"]["dp"] = dp

    # update tp nodes if necessary
    if mode == MachineType.sgsrtx and model == ModelType.tinyllama:
        config["parallelism"]["tp"] = 2

    # set microbatch size
    # TODO: We need to figure out whether batch_accumulation_per_replica matters. Can we always leave that at 1 for throughput benchmarks? For training, we don't want too small batches becaues that hurts convergence. However, for tput, in my understanding, increasing it to 2 will double the factual batch size, but keep the number of forward passes. So it should not affect throughput. Need to benchmark that though.
    config["tokens"]["batch_accumulation_per_replica"] = batch_accumulation_per_replica
    config["tokens"]["micro_batch_size"] = MACHINE_MODEL_MICROBATCH[mode][model]

    # set sequence length
    config["tokens"]["sequence_length"] = seq_length
    config["model"]["model_config"]["max_position_embeddings"] = seq_length

    # set number of data loading workers
    assert len(config["data_stages"]) == 1, "data stages should only be 1"
    config["data_stages"][0]["data"]["num_loading_workers"] = dl_worker
    # set seed
    config["data_stages"][0]["data"]["seed"] = seed
    config["general"]["seed"] = seed
    # set dataset path
    config["data_stages"][0]["data"]["dataset"]["hf_dataset_or_datasets"] = str(dataset_path)

    # number of tokens that we want to consume (will be rounded up to match batch size/seq length)
    scheduled_total_tokens = MACHINE_MODEL_TOKENS[mode][model] * dp
    batch_size = dp * config["tokens"]["batch_accumulation_per_replica"] * config["tokens"]["micro_batch_size"]
    tokens_per_step = batch_size * seq_length
    train_steps = max(math.ceil(scheduled_total_tokens / tokens_per_step), 10) # minimum of 10 training steps per benchmark
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

    # now decide whether this run should be skipped
    total_nodes = dp * int(config["parallelism"]["tp"]) * int(config["parallelism"]["pp"])
    if mode in [MachineType.sgsrtx, MachineType.sgsh100] and total_nodes > 4:
        additional_info["skipped"] = True
        additional_info["skip_reason"] = f"total_nodes = {total_nodes} > 4 for sgs machine"

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
    dl_workers: list[int] = [0, 1, 4, 16, 32],
    dps: list[int] = [1, 2, 4, 8, 16],
    seq_lengths: list[int] = [1024, 2048, 4096],
    batch_accumulation_per_replicas: list[int] = [1],
    seeds: list[int] = [42],
    huggingface_cache_path: Path = Path("/scratch/maximilian.boether/hfcache"),
):
    if not validate_user_input(mode, model):
        raise typer.Exit(code=1)

    output_file = output_dir / f"{benchmark_name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        typer.echo(f"Error: file {output_file} already exists")
        raise typer.Exit(code=1)

    base_config = load_base_config(model)
    all_results = []
    bm_identifier = "mixterabenchmark_script"
    curr_run = 0

    if huggingface_cache_path.exists():
        os.environ["HF_DATASETS_CACHE"] = str(huggingface_cache_path)
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

        all_results.append(base_results | run_benchmark(adjusted_config, mode))

        persist_results_to_json(output_file, all_results)
        curr_run += 1

    typer.echo("Ran all benchmarks.")


if __name__ == "__main__":
    app()
