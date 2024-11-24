# Note fineweb does not have an explicit validation set!

import argparse
import os
import sys
import json
from tqdm import tqdm
import pyarrow.parquet as pq


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert FineWeb Parquet validation data to JSONL.")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory containing Parquet files (required)")
    parser.add_argument(
        "-o", "--output-file", default="fineweb_val.jsonl", help="Output JSONL file (default: fineweb_val.jsonl)"
    )
    return parser.parse_args()


def find_parquet_files(input_dir):
    """Recursively find all .parquet files in the input directory."""
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(root, filename))
    return parquet_files


def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_file = args.output_file

    # List of input Parquet files to process
    parquet_files = find_parquet_files(input_dir)
    if not parquet_files:
        print(f"No .parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    total_records = 0  # To keep count of total records processed

    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for parquet_file in tqdm(parquet_files, desc="Processing files", unit="file"):
                try:
                    with open(parquet_file, "rb") as f_in:
                        parquet_reader = pq.ParquetFile(f_in)
                        for batch in parquet_reader.iter_batches():
                            records = batch.to_pylist()
                            for record in records:
                                if "text" in record:
                                    text = record["text"]
                                    # Create JSON object
                                    json_obj = {"text": text}
                                    json_line = json.dumps(json_obj, ensure_ascii=False)
                                    f_out.write(json_line + "\n")
                                    total_records += 1
                                else:
                                    print(f"Record in file '{parquet_file}' is missing 'text' field.", file=sys.stderr)
                                    sys.exit(1)
                except Exception as e:
                    print(f"Error processing file '{parquet_file}': {e}", file=sys.stderr)
                    sys.exit(1)
    except IOError as e:
        print(f"Error opening output file '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal records processed: {total_records}")
    print(f"Output written to '{output_file}'")


if __name__ == "__main__":
    main()
