import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_file(url, target_path):
    """Attempt to download a file from 'url' to 'target_path' up to 3 tries."""
    tries = 3
    for attempt in range(tries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 404:
                return None  # Signal that file does not exist
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True  # File downloaded successfully
        except requests.RequestException as e:
            if attempt < tries - 1:  # Retry if not the last attempt
                continue
            else:
                raise Exception(f"Failed to download {url} after {tries} attempts") from e


def download_chunk_files(chunk_id, base_url, target_dir):
    """Download all files for a given chunk in batches until a 404 is encountered."""
    os.makedirs(
        target_dir,
        exist_ok=True,
    )
    batch_size = 500
    file_index = 0

    while True:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            for _batch_id in range(batch_size):
                file_url = f"{base_url}/chunk{chunk_id}/example_train_{file_index}.jsonl.zst?download=true"
                target_path = os.path.join(target_dir, f"example_train_{file_index}.jsonl.zst")
                futures[executor.submit(download_file, file_url, target_path)] = file_index
                file_index += 1

            break_after_loop = False
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    break_after_loop = True

            if break_after_loop:
                return


def main():
    base_url = "https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train"
    target_dir_base = "/scratch/maximilian.boether/datasets/slimpajama"
    chunks_to_download = [0, 1, 2, 3]
    for chunk_id in chunks_to_download:
        target_dir = os.path.join(target_dir_base, f"chunk{chunk_id}")
        download_chunk_files(chunk_id, base_url, target_dir)


if __name__ == "__main__":
    main()
