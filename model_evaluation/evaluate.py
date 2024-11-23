import os
import typer
from pathlib import Path
import yaml
from tqdm import tqdm
import subprocess
import os
import yaml
import subprocess
import shutil

app = typer.Typer()


def check_nanotron_availability():
    try:
        global nanotron
        import nanotron
    except ImportError:
        typer.echo(
            "Error: 'nanotron' module is not available. Please ensure it is installed according to their instructions."
        )
        raise typer.Exit(code=1)


def generate_yaml_tasks(jsonl_dir, yaml_output_dir):
    task_names = []

    jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]
    for jsonl_file in jsonl_files:
        task_name = os.path.splitext(jsonl_file)[0]
        task_names.append(task_name)

        task_yaml = {
            "task": task_name,
            "dataset_path": "json",
            "output_type": "loglikelihood_rolling",
            "test_split": "train",
            "doc_to_text": "{{text}}",
            "metric_list": [
                {"metric": "word_perplexity", "aggregation": "weighted_perplexity", "higher_is_better": False},
                {"metric": "byte_perplexity", "aggregation": "weighted_perplexity", "higher_is_better": False},
                {"metric": "bits_per_byte", "aggregation": "bits_per_byte", "higher_is_better": False},
            ],
            "metadata": {"version": 1.0, "description": f"Perplexity evaluation on {jsonl_file}"},
            "dataset_kwargs": {"data_files": {"train": os.path.join(jsonl_dir, jsonl_file)}},
            "max_num_fewshot": 0,
        }

        yaml_file_path = os.path.join(yaml_output_dir, f"{task_name}.yaml")
        with open(yaml_file_path, "w+") as f:
            yaml.dump(task_yaml, f)

    return task_names


def check_lm_eval_availability():
    result = subprocess.run(["lm_eval", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        typer.echo("Error: 'lm_eval' CLI tool is not available. Please ensure it is installed and in your PATH.")
        raise typer.Exit(code=1)


def list_checkpoints(directory: Path):
    checkpoints = [d for d in directory.iterdir() if d.is_dir()]
    return checkpoints


def get_tokenizer_name(checkpoint_dir: Path):
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["tokenizer"]["tokenizer_name_or_path"]


def get_parallelism_config(checkpoint_dir: Path):
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config.get("parallelism", {})


def check_output_dir(output_dir: Path):
    if output_dir.exists():
        typer.echo(f"Warning: Output directory {output_dir} already exists.")
        if not typer.confirm("Do you want to continue?"):
            raise typer.Abort()


def convert_checkpoints(hf_output_dir: Path, selected_checkpoints: dict[int, Path]):
    hf_output_dir.mkdir(parents=True, exist_ok=True)

    conversion_script_path = str(
        Path(nanotron.__path__[0]).parent.parent / "examples" / "llama" / "convert_nanotron_to_hf.py"
    )
    for _, checkpoint in tqdm(selected_checkpoints.items(), desc="Converting Checkpoints"):
        tokenizer_name = get_tokenizer_name(checkpoint)
        save_path = hf_output_dir / f"{checkpoint.name}-hf"
        parallelism_config = get_parallelism_config(checkpoint)
        tp = parallelism_config.get("tp", 1)
        pp = parallelism_config.get("pp", 1)

        total_procs_for_conversion = tp * pp

        if save_path.exists():
            typer.echo(f"Error: Directory {save_path} already exists - this is unexpected.")
            raise typer.Exit(code=1)

        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={total_procs_for_conversion}",
                conversion_script_path,
                "--checkpoint_path",
                str(checkpoint),
                "--save_path",
                str(save_path),
                "--tokenizer_name",
                tokenizer_name,
            ],
            check=True,
        )

        typer.echo(f"Converted {checkpoint.name} to Hugging Face format at {save_path}")


def evaluate_checkpoints(
    hf_output_dir: Path,
    output_dir: Path,
    selected_checkpoints: dict[int, Path],
    tasks: str,
    data_parallel: int,
    seed: int,
    num_fewshots: list[int],
    include_yaml_dir: Path,
    additional_task_names: list[str],
):
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    for _, checkpoint in tqdm(selected_checkpoints.items(), desc="Evaluating Models"):
        save_path = hf_output_dir / f"{checkpoint.name}-hf"
        if not save_path.exists():
            typer.echo(f"Expected model directory {save_path} does not exist. Skipping.")
            continue

        tokenizer_name = get_tokenizer_name(checkpoint)
        parallelism_config = get_parallelism_config(checkpoint)
        tp = parallelism_config.get("tp", 1)
        pp = parallelism_config.get("pp", 1)

        if pp > 1:
            typer.echo(f"Error: lm_eval currently only supports pp = 1 with VLLM, but pp = {pp}!")
            raise typer.Exit(code=1)

        if len(additional_task_names) > 0:
            tasks += "," + ",".join(additional_task_names)

        for num_fewshot in num_fewshots:
            typer.echo(f"Running with {num_fewshot} fewshot examples.")
            fewshot_output_dir = output_dir / str(num_fewshot)
            fewshot_output_dir.mkdir(exist_ok=True)
            cmd = [
                "lm_eval",
                "--model",
                "vllm",
                "--model_args",
                f"pretrained={save_path},trust_remote_code=True,tensor_parallel_size={tp},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size={data_parallel},tokenizer={tokenizer_name},seed={seed}",
                "--tasks",
                tasks,
                "--num_fewshot",
                str(num_fewshot),
                "--batch_size",
                "auto",
                "--include_path",
                str(include_yaml_dir),
                "--output_path",
                str(fewshot_output_dir / f"results_{checkpoint.name}"),
            ]
            subprocess.run(cmd, check=True)

    typer.echo("Successfully ran conversion and evaluation. Use `parse_results.py` to convert the data into a csv.")


@app.command()
def convert_and_evaluate(
    checkpoint_dir: Path,
    output_dir: Path,
    tasks: str = "lambada_openai,hellaswag,openbookqa,winogrande,glue,arc_easy,arc_challenge,mmlu,squadv2",
    data_parallel: int = 1,
    skip_conversion: bool = False,
    seed: int = 1337,
    use_all_chkpnts: bool = False,
    num_fewshots: list[int] = [0, 1],
    perp_jsonls: Path = Path(__file__).parent / "perplexity_data",
):
    check_nanotron_availability()
    check_lm_eval_availability()
    check_output_dir(output_dir)

    ### Setup custom task yamls
    yaml_path = Path(__file__).parent / "yaml_temp"
    if yaml_path.exists():
        shutil.rmtree(yaml_path)
    yaml_path.mkdir(parents=True)
    if perp_jsonls.exists() and perp_jsonls.is_dir():
        task_names = generate_yaml_tasks(perp_jsonls, yaml_path)

    ### Get all available checkpoints
    checkpoints = list_checkpoints(checkpoint_dir)
    checkpoint_dict = {i: cp for i, cp in enumerate(checkpoints)}

    ### Decide which checkpoints to use
    if not use_all_chkpnts:
        # Ask user which checkpoints to exclude
        typer.echo("Available checkpoints:")
        for idx, cp in checkpoint_dict.items():
            typer.echo(f"ID {idx}: {cp.name}")

        exclusions = []
        while True:
            exclude = typer.prompt("Enter checkpoint ID to exclude (or 'done' to finish)", default="done")
            if exclude.lower() == "done":
                break
            if int(exclude) in checkpoint_dict:
                exclusions.append(int(exclude))
            else:
                typer.echo(f"Unknown ID: {exclude}")

        selected_checkpoints = {i: cp for i, cp in checkpoint_dict.items() if i not in exclusions}
    else:
        # Use all checkpoints if flag is supplied instead of having user interaction
        selected_checkpoints = {i: cp for i, cp in checkpoint_dict.items()}

    hf_output_dir = output_dir / "hf_ckpts"

    if not skip_conversion:
        convert_checkpoints(hf_output_dir, selected_checkpoints)

    ### Evaluate the checkpoints
    evaluate_checkpoints(
        hf_output_dir, output_dir, selected_checkpoints, tasks, data_parallel, seed, num_fewshots, yaml_path, task_names
    )


if __name__ == "__main__":
    app()
