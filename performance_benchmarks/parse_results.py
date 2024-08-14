import typer
from pathlib import Path
import pandas as pd
import json

app = typer.Typer()


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
        seed = int(item["config"]["general"].get("seed", 0))
        num_loading_workers = int(item["config"]["data_stages"][0]["data"].get("num_loading_workers", 0))
        run_id = item["config"]["general"].get("run", "")
        dp = int(item["config"]["parallelism"].get("dp", 0))
        pp = int(item["config"]["parallelism"].get("pp", 0))
        tp = int(item["config"]["parallelism"].get("tp", 0))
        sequence_length = int(item["config"]["tokens"].get("sequence_length", 0))
        train_steps = int(item["config"]["tokens"].get("train_steps", 0))
        batch_accumulation_per_replica = int(item["config"]["tokens"].get("batch_accumulation_per_replica", 0))
        micro_batch_size = int(item["config"]["tokens"].get("micro_batch_size", 0))

        processed_data.append(
            {
                "mode": mode,
                "model": model,
                "seed": seed,
                "num_loading_workers": num_loading_workers,
                "run_id": run_id,
                "dp": dp,
                "pp": pp,
                "tp": tp,
                "sequence_length": sequence_length,
                "train_steps": train_steps,
                "batch_accumulation_per_replica": batch_accumulation_per_replica,
                "micro_batch_size": micro_batch_size,
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
