import typer
from pathlib import Path
import pandas as pd
import json

app = typer.Typer()


def read_json_results(results_dir):
    """Read the JSON file in the specified results directory and validate its existence."""
    json_files = list(results_dir.glob("**/*.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected exactly one JSON file in {results_dir}, found {len(json_files)}.")
    with open(json_files[0], "r") as file:
        data = json.load(file)
    if "results" not in data:
        raise ValueError(f"'results' key not found in JSON file {json_files[0]}.")
    return data["results"]


def parse_results(data):
    """Parse the results data to flatten the structure for DataFrame conversion."""
    parsed_data = {}
    for task, metrics in data.items():
        for key, value in metrics.items():
            if key != "alias":  # Exclude the 'alias' key
                new_key = f"{task}_{key}".replace(",", "_")
                parsed_data[new_key] = value
    return parsed_data


@app.command()
def collect_and_export_results(output_dir: Path):
    output_dir = Path(output_dir)
    result_dirs = [d for d in output_dir.glob("results_*") if d.is_dir()]

    all_results = []
    schema_set = set()  # To track consistency of schema across all JSON files

    for dir in result_dirs:
        results = read_json_results(dir)
        parsed_results = parse_results(results)
        parsed_results["checkpoint_id"] = dir.name.split("_")[1]  # Extract checkpoint ID from directory name
        all_results.append(parsed_results)

        # Update schema_set to ensure all parsed results have the same schema
        if not schema_set:
            schema_set.update(parsed_results.keys())
        elif schema_set != set(parsed_results.keys()):
            raise ValueError("Inconsistent data schema across result JSON files.")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "results.csv", index=False)
        typer.echo(f"Results have been successfully written to '{output_dir / 'results.csv'}'.")
    else:
        typer.echo("No results found.")


if __name__ == "__main__":
    app()
