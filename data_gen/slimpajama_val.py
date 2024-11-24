import argparse
import json
import os
import sys
from xopen import xopen
from tqdm import tqdm

# Mapping of redpajama_set_name to output filename
REDPAJAMA_SET_NAME_TO_FILENAME = {
    "RedPajamaCommonCrawl": "sp_commoncrawl",
    "RedPajamaC4": "sp_c4",
    "RedPajamaWikipedia": "sp_wikipedia",
    "RedPajamaStackExchange": "sp_stackexchange",
    "RedPajamaGithub": "sp_github",
    "RedPajamaArXiv": "sp_arxiv",
    "RedPajamaBook": "sp_book",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split SlimPajama dataset into component JSONL files.")
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Input directory containing .jsonl.zst files (required)"
    )
    parser.add_argument(
        "-o", "--output-dir", default=".", help="Output directory for the component files (default: current directory)"
    )
    return parser.parse_args()


def find_input_files(input_dir):
    """Recursively find all .jsonl.zst files in the input directory."""
    input_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".jsonl.zst"):
                input_files.append(os.path.join(root, filename))
    return input_files


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List of input files to process
    input_files = find_input_files(input_dir)
    if not input_files:
        print(f"No .jsonl.zst files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Dictionary to keep file handles for each component
    file_handles = {}
    # Dictionary to keep count of lines for each category
    category_counts = {key: 0 for key in REDPAJAMA_SET_NAME_TO_FILENAME.keys()}

    # Processing files
    try:
        for input_file in tqdm(input_files, desc="Processing files", unit="file"):
            try:
                with xopen(input_file, "rt", encoding="utf-8") as f_in:
                    for line_number, line in enumerate(f_in, 1):
                        line = line.strip()
                        if not line:
                            continue  # Skip empty lines
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(
                                f"Error decoding JSON in file '{input_file}', line {line_number}: {e}", file=sys.stderr
                            )
                            sys.exit(1)

                        # Extract redpajama_set_name
                        try:
                            redpajama_set_name = data["meta"]["redpajama_set_name"]
                        except KeyError:
                            print(
                                f"Missing 'redpajama_set_name' in metadata in file '{input_file}', line {line_number}",
                                file=sys.stderr,
                            )
                            sys.exit(1)

                        if redpajama_set_name not in REDPAJAMA_SET_NAME_TO_FILENAME:
                            print(
                                f"Unknown redpajama_set_name '{redpajama_set_name}' in file '{input_file}', line {line_number}",
                                file=sys.stderr,
                            )
                            sys.exit(1)

                        output_filename = REDPAJAMA_SET_NAME_TO_FILENAME[redpajama_set_name]
                        output_path = os.path.join(output_dir, output_filename + ".jsonl")

                        if output_filename not in file_handles:
                            try:
                                # Open the file in append mode to write entries
                                file_handles[output_filename] = open(output_path, "a", encoding="utf-8")
                            except IOError as e:
                                print(f"Error opening file '{output_path}': {e}", file=sys.stderr)
                                sys.exit(1)

                        # Write the original line to the appropriate output file
                        file_handles[output_filename].write(line + "\n")

                        # Update category count
                        category_counts[redpajama_set_name] += 1
            except IOError as e:
                print(f"Error reading file '{input_file}': {e}", file=sys.stderr)
                sys.exit(1)
    finally:
        # Close all file handles
        for fh in file_handles.values():
            fh.close()

    # Print counts for each category
    print("\nSamples written for each category:")
    statistics_by_file = {}
    for redpajama_set_name, count in category_counts.items():
        print(f"{redpajama_set_name}: {count} samples")
        statistics_by_file[REDPAJAMA_SET_NAME_TO_FILENAME[redpajama_set_name]] = count

    # Write statistics to JSON files
    with open(os.path.join(output_dir, "statistics_by_group.json"), "w", encoding="utf-8") as fp:
        json.dump(category_counts, fp, indent=4)
    with open(os.path.join(output_dir, "statistics_by_file.json"), "w", encoding="utf-8") as fp:
        json.dump(statistics_by_file, fp, indent=4)


if __name__ == "__main__":
    main()
