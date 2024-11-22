import datasets
from lm_eval.base import Task, rf
from lm_eval.metrics import weighted_perplexity, bits_per_byte
import os
import glob
from pathlib import Path


class BaseCPerplexityTask(Task):
    VERSION = 1

    def __init__(self, dataset_name, data_file):
        self.TASK_NAME = dataset_name
        self.DATA_FILE = data_file
        super().__init__()

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def test_docs(self):
        return datasets.load_dataset("json", data_files=self.DATA_FILE)

    def doc_to_text(self, doc):
        return doc["text"]

    def construct_requests(self, doc, ctx):
        return rf.loglikelihood_rolling(ctx)

    def process_results(self, doc, results):
        loglikelihood, token_count = results
        text = doc["text"]
        word_count = len(text.split())
        byte_count = len(text.encode("utf-8"))

        return {
            "token_perplexity": (loglikelihood, token_count),
            "word_perplexity": (loglikelihood, word_count),
            "byte_perplexity": (loglikelihood, byte_count),
            "bits_per_byte": (loglikelihood, byte_count),
        }

    def aggregation(self):
        return {
            "token_perplexity": weighted_perplexity,
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    def higher_is_better(self):
        return {"token_perplexity": False, "word_perplexity": False, "byte_perplexity": False, "bits_per_byte": False}


DATASET_DIR = Path(__file__).parent / "perplexity_data"
jsonl_files = glob.glob(os.path.join(DATASET_DIR, "*.jsonl"))

for jsonl_file in jsonl_files:
    dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    class_name = "CPlex_" + dataset_name.replace("-", "_").capitalize()

    def init(self, dataset_name=dataset_name, data_file=jsonl_file):
        BaseCPerplexityTask.__init__(self, dataset_name, data_file)

    new_class = type(class_name, (BaseCPerplexityTask,), {"__init__": init})
    globals()[class_name] = new_class
