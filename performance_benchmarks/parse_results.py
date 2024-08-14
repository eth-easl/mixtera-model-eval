import typer
from pathlib import Path
import pandas as pd
import json
import numpy as np

app = typer.Typer()


def compute_stats(values):
    if values:
        return {
            "avg": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
            "min": np.min(values),
            "max": np.max(values),
        }
    else:
        return {"avg": None, "stderr": None, "median": None, "min": None, "max": None}


@app.command()
def parse_results(json_path: Path, output_path: Path):
    with open(json_path, "r") as file:
        data = json.load(file)

    processed_data = []

    for item in data:
        if item.get("skipped", False):
            continue

        mode = item.get("mode", "")
        model = item.get("model", "")
        seed = int(item["config"]["general"].get("seed", -1))
        num_loading_workers = int(item["config"]["data_stages"][0]["data"].get("num_loading_workers", -1))
        run_id = item["config"]["general"].get("run", "")
        project = item["config"]["general"].get("project", "")
        dp = int(item["config"]["parallelism"].get("dp", -1))
        pp = int(item["config"]["parallelism"].get("pp", -1))
        tp = int(item["config"]["parallelism"].get("tp", -1))
        sequence_length = int(item["config"]["tokens"].get("sequence_length", -1))
        train_steps = int(item["config"]["tokens"].get("train_steps", -1))
        batch_accumulation_per_replica = int(item["config"]["tokens"].get("batch_accumulation_per_replica", -1))
        micro_batch_size = int(item["config"]["tokens"].get("micro_batch_size", -1))

        batch_size = int(item.get("batch_size", -1))
        assert batch_size == item["global_batch_size"]["0"]

        tokens_per_second = [val for key, val in item["tokens_per_sec"].items() if key not in ["0", 0]]
        tokens_per_second_per_gpu = [val for key, val in item["tokens_per_sec_per_gpu"].items() if key not in ["0", 0]]
        elapsed_time_per_iteration_ms = [
            val for key, val in item["elapsed_time_per_iteration_ms"].items() if key not in ["0", 0]
        ]
        model_tflops_per_gpu = [val for key, val in item["model_tflops_per_gpu"].items() if key not in ["0", 0]]

        _max_key = str(max([int(key) for key in item["model_tflops_per_gpu"].keys()]))
        final_consumed_tokens = item["consumed_tokens"][_max_key]
        final_runtime = item["_runtime"][_max_key]
        total_elapsed_time_s = np.sum(elapsed_time_per_iteration_ms) / 1000.0
        global_tput_elapsed_time = final_consumed_tokens / total_elapsed_time_s
        global_tput_runtime = final_consumed_tokens / (final_runtime / 1000.0)

        processed_data.append(
            {
                "mode": mode,
                "model": model,
                "seed": seed,
                "num_loading_workers": num_loading_workers,
                "run_id": run_id,
                "project": project,
                "dp": dp,
                "pp": pp,
                "tp": tp,
                "sequence_length": sequence_length,
                "train_steps": train_steps,
                "batch_accumulation_per_replica": batch_accumulation_per_replica,
                "micro_batch_size": micro_batch_size,
                "batch_size": batch_size,
                "final_consumed_tokens": final_consumed_tokens,
                "final_runtime": final_runtime,
                "total_elapsed_time_s": total_elapsed_time_s,
                "global_tput_elapsed_time": global_tput_elapsed_time,
                "global_tput_runtime": global_tput_runtime,
                **{k + "_tokens_per_second": v for k, v in compute_stats(tokens_per_second).items()},
                **{k + "_tokens_per_second_per_gpu": v for k, v in compute_stats(tokens_per_second_per_gpu).items()},
                **{
                    k + "_elapsed_time_per_iteration_ms": v
                    for k, v in compute_stats(elapsed_time_per_iteration_ms).items()
                },
                **{k + "_model_tflops_per_gpu": v for k, v in compute_stats(model_tflops_per_gpu).items()},
            }
        )

    df = pd.DataFrame(processed_data)
    df = df.sort_values(
        by=[
            "mode",
            "model",
            "seed",
            "num_loading_workers",
            "dp",
            "pp",
            "tp",
            "sequence_length",
            "micro_batch_size",
            "train_steps",
        ]
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
