import argparse
import json
import os
import sys
from xopen import xopen
from tqdm import tqdm

# Mapping of pile_set_name to output filename
PILE_SET_NAME_TO_FILENAME = {
    "Pile-CC": "pile_val_cc",
    "PubMed Central": "pile_val_pubmed_central",
    "Books3": "pile_val_books3",
    "OpenWebText2": "pile_val_openwebtext2",
    "ArXiv": "pile_val_arxiv",
    "Github": "pile_val_github",
    "FreeLaw": "pile_val_freelaw",
    "StackExchange": "pile_val_stackexchange",
    "USPTO Backgrounds": "pile_val_uspto_backgrounds",
    "PubMed Abstracts": "pile_val_pubmed_abstracts",
    "Gutenberg (PG-19)": "pile_val_gutenberg",
    "OpenSubtitles": "pile_val_opensubtitles",
    "Wikipedia (en)": "pile_val_wikipedia",
    "DM Mathematics": "pile_val_dm_mathematics",
    "Ubuntu IRC": "pile_val_ubuntu_irc",
    "BookCorpus2": "pile_val_bookcorpus2",
    "EuroParl": "pile_val_europarl",
    "HackerNews": "pile_val_hackernews",
    "YoutubeSubtitles": "pile_val_youtubesubtitles",
    "PhilPapers": "pile_val_philpapers",
    "NIH ExPorter": "pile_val_nih_exporter",
    "Enron Emails": "pile_val_enron",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split val.jsonl.zst into component JSONL files.")
    parser.add_argument(
        "-i", "--input", default="val.jsonl.zst", help="Input compressed JSONL file (default: val.jsonl.zst)"
    )
    parser.add_argument(
        "-o", "--output-dir", default=".", help="Output directory for the component files (default: current directory)"
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        default=False,
        help="If set, merge all samples per category into a single sample (default: False)",
    )
    return parser.parse_args()


def count_total_lines(file_path):
    """Count total number of lines in the compressed file."""
    line_count = 0
    print("Counting total number of lines in the input file...")
    with xopen(file_path, "rt", encoding="utf-8") as f:
        for _ in tqdm(f, desc="Counting lines", unit=" lines", mininterval=1.0):
            line_count += 1
    return line_count


def main():
    args = parse_arguments()
    input_path = args.input
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to keep count of lines for each category
    category_counts = {key: 0 for key in PILE_SET_NAME_TO_FILENAME.keys()}

    if args.merged:
        print("Merging all samples for each category into a single sample per file.")
        print(
            "Note: This requires enough memory to hold all texts for each category.\n"
            "Ensure your system has sufficient memory."
        )
        # Dictionary to accumulate texts per category
        merged_samples = {key: "" for key in PILE_SET_NAME_TO_FILENAME.keys()}
    else:
        # Dictionary to keep file handles for each component
        file_handles = {}

    # First, count total number of lines for progress bar
    total_lines = count_total_lines(input_path)

    try:
        with xopen(input_path, "rt", encoding="utf-8") as f_in:
            with tqdm(total=total_lines, desc="Processing", unit=" lines", mininterval=1.0) as pbar:
                for line_number, line in enumerate(f_in, 1):
                    line = line.strip()
                    pbar.update(1)
                    if not line:
                        continue  # Skip empty lines
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {line_number}: {e}", file=sys.stderr)
                        sys.exit(1)

                    # Extract pile_set_name
                    try:
                        pile_set_name = data["meta"]["pile_set_name"]
                    except KeyError:
                        print(f"Missing 'pile_set_name' in metadata on line {line_number}", file=sys.stderr)
                        sys.exit(1)

                    if pile_set_name not in PILE_SET_NAME_TO_FILENAME:
                        print(f"Unknown pile_set_name '{pile_set_name}' on line {line_number}", file=sys.stderr)
                        sys.exit(1)

                    text = data["text"].strip()
                    if not text:
                        continue  # Skip empty texts

                    # Update category count
                    category_counts[pile_set_name] += 1

                    if args.merged:
                        # Accumulate text in memory
                        merged_samples[pile_set_name] += text + "\n"
                    else:
                        output_filename = PILE_SET_NAME_TO_FILENAME[pile_set_name]
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

    finally:
        if not args.merged:
            # Close all file handles
            for fh in file_handles.values():
                fh.close()

    if args.merged:
        # Write the merged samples to respective output files
        print("\nWriting merged samples to output files...")
        for pile_set_name, accumulated_text in merged_samples.items():
            output_filename = PILE_SET_NAME_TO_FILENAME[pile_set_name]
            output_path = os.path.join(output_dir, output_filename + ".jsonl")
            if accumulated_text.strip():
                try:
                    with open(output_path, "w", encoding="utf-8") as f_out:
                        merged_sample = {
                            "text": accumulated_text.strip(),
                            "meta": {
                                "pile_set_name": pile_set_name,
                                "num_samples": category_counts[pile_set_name],
                            },
                        }
                        f_out.write(json.dumps(merged_sample) + "\n")
                except IOError as e:
                    print(f"Error writing to file '{output_path}': {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"No text accumulated for pile_set_name '{pile_set_name}', skipping file.")

    # Print counts for each category
    print("\nSamples written for each category:")
    statistics_by_file = {}
    for pile_set_name, count in category_counts.items():
        print(f"{pile_set_name}: {count} samples")
        statistics_by_file[PILE_SET_NAME_TO_FILENAME[pile_set_name]] = count

    with open(os.path.join(output_dir, "statistics_by_group.json"), "w+", encoding="utf-8") as fp:
        json.dump(category_counts, fp)
    with open(os.path.join(output_dir, "statistics_by_file.json"), "w+", encoding="utf-8") as fp:
        json.dump(statistics_by_file, fp)


if __name__ == "__main__":
    main()
