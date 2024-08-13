from enum import Enum
import typer
from pathlib import Path
import yaml
from tqdm import tqdm
import subprocess

app = typer.Typer()

# Machine: RTX 3090, H100, GH200
# Model: TinyLlama 1B, Llama7B, Llamba70B
    # Model+Machine fixes pp and tp => defines base config
    # TODO: Find out parameters for LLama7B and 70B. 
    # Llama2 vs 3? do they define different architecture params?
    # what learning rate schedule to use?
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
    sgs = "sgs"
    cscs = "cscs"

def check_nanotron_availability():
    try:
        global nanotron
        import nanotron
    except ImportError:
        typer.echo(
            "Error: 'nanotron' module is not available. Please ensure it is installed according to their instructions."
        )
        raise typer.Exit(code=1)


@app.command()
def run_benchmarks(
    mode: MachineType,
    dl_workers: list[int] = [0,1,4,16,32],
    dp: list[int] = [1,2,4,8,16],
):
    pass


if __name__ == "__main__":
    app()
