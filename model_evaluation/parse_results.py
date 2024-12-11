import typer
from pathlib import Path
import pandas as pd
import json

app = typer.Typer()


def read_json_results(results_dir, merge=False):
    """Read JSON files in the specified results directory and return merged results if merge is True."""
    json_files = list(results_dir.glob("**/*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {results_dir}.")

    if not merge:
        if len(json_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {results_dir}, found {len(json_files)}.")
        json_file = json_files[0]
        with open(json_file, "r") as file:
            data = json.load(file)
        if "results" not in data:
            raise ValueError(f"'results' key not found in JSON file {json_file}.")
        return data["results"]
    else:
        # Merge mode: Read all JSON files and merge results, giving precedence to newer files
        # Sort JSON files by modification time (newest first)
        json_files = sorted(json_files, key=lambda f: f.stat().st_mtime, reverse=True)
        merged_results = {}
        for json_file in json_files:
            with open(json_file, "r") as file:
                data = json.load(file)
            if "results" not in data:
                raise ValueError(f"'results' key not found in JSON file {json_file}.")
            results = data["results"]
            for task, metrics in results.items():
                if task not in merged_results:
                    merged_results[task] = metrics  # Keep the metrics from the newer file
        return merged_results


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
def collect_and_export_results(
    output_dir: Path,
    merge: bool = typer.Option(
        False,
        help="If set, merge results from multiple JSON files, giving precedence to newer files.",
    ),
):
    output_dir = Path(output_dir)
    nshot_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.isnumeric()]
    all_results = []

    for nshot_dir in nshot_dirs:
        nshot = int(nshot_dir.name)
        result_dirs = [d for d in nshot_dir.glob("results_*") if d.is_dir()]

        schema_set = set()  # To track consistency of schema across all JSON files

        for dir in result_dirs:
            results = read_json_results(dir, merge=merge)
            parsed_results = parse_results(results)
            parsed_results["checkpoint_id"] = dir.name.split("_")[1]  # Extract checkpoint ID from directory name
            parsed_results["nshot"] = nshot
            all_results.append(parsed_results)

            # Update schema_set to ensure all parsed results have the same schema
            if not schema_set:
                schema_set.update(parsed_results.keys())
            elif schema_set != set(parsed_results.keys()):
                raise ValueError("Inconsistent data schema across result JSON files.")

    if all_results:
        df = pd.DataFrame(all_results)
        df[["checkpoint_id", "nshot"]] = df[["checkpoint_id", "nshot"]].apply(pd.to_numeric)
        column_order = ["checkpoint_id", "nshot"] + [col for col in df.columns if col not in ["checkpoint_id", "nshot"]]
        df = df[column_order]
        df.sort_values(by=["checkpoint_id", "nshot"], inplace=True)
        df.to_csv(output_dir / "results.csv", index=False)
        typer.echo(f"Results have been successfully written to '{output_dir / 'results.csv'}'.")
    else:
        typer.echo("No results found.")


if __name__ == "__main__":
    app()
