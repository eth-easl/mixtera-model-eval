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
        return True

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
task_classes = {}

for jsonl_file in jsonl_files:
    dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    class_name = "CPlex_" + dataset_name.replace("-", "_").capitalize()

    def init(self, dataset_name=dataset_name, data_file=jsonl_file):
        BaseCPerplexityTask.__init__(self, dataset_name, data_file)

    new_class = type(class_name, (BaseCPerplexityTask,), {"__init__": init})
    globals()[class_name] = new_class
    task_classes[dataset_name] = new_class


class CPerpAll(Task):
    VERSION = 1
    TASK_NAME = "cperp_all"

    def __init__(self):
        super().__init__()
        self.subtasks = {}
        for dataset_name, task_class in task_classes.items():
            self.subtasks[dataset_name] = task_class()

    def has_validation_docs(self):
        return any(task.has_validation_docs() for task in self.subtasks.values())

    def has_test_docs(self):
        return any(task.has_test_docs() for task in self.subtasks.values())

    def test_docs(self):
        # Yield documents with a reference to their subtask
        for name, task in self.subtasks.items():
            if task.has_test_docs():
                for doc in task.test_docs():
                    doc["_subtask"] = name  # Mark the document with its subtask name
                    yield doc

    def doc_to_text(self, doc):
        subtask = self.subtasks[doc["_subtask"]]
        return subtask.doc_to_text(doc)

    def construct_requests(self, doc, ctx):
        subtask = self.subtasks[doc["_subtask"]]
        return subtask.construct_requests(doc, ctx)

    def process_results(self, doc, results):
        subtask = self.subtasks[doc["_subtask"]]
        return {(doc["_subtask"], k): v for k, v in subtask.process_results(doc, results).items()}

    def aggregation(self):
        agg = {}
        for name, task in self.subtasks.items():
            for k, fn in task.aggregation().items():
                agg[(name, k)] = fn
        return agg

    def higher_is_better(self):
        hib = {}
        for name, task in self.subtasks.items():
            for k, flag in task.higher_is_better().items():
                hib[(name, k)] = flag
        return hib
