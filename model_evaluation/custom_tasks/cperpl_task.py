import datasets
from lm_eval.api.task import PerplexityTask, Task
import os
import glob
from pathlib import Path


class BaseCPerplexityTask(PerplexityTask):
    VERSION = 1

    def __init__(self, dataset_name, data_file):
        super().__init__()
        self.DATASET_PATH = "json"
        self.DATASET_NAME = None
        self.dataset_name = dataset_name
        self.data_file = data_file

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["train"]

    def download(self, **kwargs):
        # Load your dataset from the specified JSONL file
        data_files = {"train": self.data_file}
        self.dataset = self.dataset = self._load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=data_files,
        )

    def doc_to_text(self, doc):
        return doc["text"]


DATASET_DIR = Path(__file__).parent / "perplexity_data"
jsonl_files = glob.glob(os.path.join(DATASET_DIR, "*.jsonl"))
task_classes = {}

for jsonl_file in jsonl_files:
    dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
    class_name = "CPlex_" + dataset_name.replace("-", "_").capitalize()

    new_class = type(
        class_name,
        (BaseCPerplexityTask,),
        {
            "__init__": lambda self, dataset_name=dataset_name, data_file=jsonl_file: BaseCPerplexityTask.__init__(
                self, dataset_name, data_file
            ),
            "TASK_NAME": dataset_name,
        },
    )
    globals()[class_name] = new_class
    task_classes[dataset_name] = new_class


class CPerpAll(Task):
    VERSION = 1
    TASK_NAME = "cperp_all"

    def __init__(self):
        super().__init__()
        self.subtasks = {}
        for dataset_name, task_class in task_classes.items():
            # Initialize each subtask
            self.subtasks[dataset_name] = task_class()

    def download(self, **kwargs):
        # Each subtask needs to download its own data
        for task in self.subtasks.values():
            task.download(**kwargs)

    def has_training_docs(self):
        return any(task.has_training_docs() for task in self.subtasks.values())

    def has_validation_docs(self):
        return any(task.has_validation_docs() for task in self.subtasks.values())

    def has_test_docs(self):
        return any(task.has_test_docs() for task in self.subtasks.values())

    def training_docs(self):
        for name, task in self.subtasks.items():
            if task.has_training_docs():
                for doc in task.training_docs():
                    doc["_subtask"] = name
                    yield doc

    def validation_docs(self):
        for name, task in self.subtasks.items():
            if task.has_validation_docs():
                for doc in task.validation_docs():
                    doc["_subtask"] = name
                    yield doc

    def test_docs(self):
        for name, task in self.subtasks.items():
            if task.has_test_docs():
                for doc in task.test_docs():
                    doc["_subtask"] = name
                    yield doc

    def doc_to_text(self, doc):
        # Redirect to the appropriate subtask's doc_to_text
        subtask = self.subtasks[doc["_subtask"]]
        return subtask.doc_to_text(doc)

    def construct_requests(self, doc, ctx=None):
        # Redirect to the appropriate subtask's construct_requests
        subtask = self.subtasks[doc["_subtask"]]
        return subtask.construct_requests(doc)

    def process_results(self, doc, results):
        # Redirect to the appropriate subtask's process_results
        subtask = self.subtasks[doc["_subtask"]]
        res = subtask.process_results(doc, results)
        # Prefix the metric names with the subtask name to prevent collisions
        return {f"{doc['_subtask']}_{k}": v for k, v in res.items()}

    def aggregation(self):
        agg = {}
        for name, task in self.subtasks.items():
            task_agg = task.aggregation()
            for k, fn in task_agg.items():
                agg[f"{name}_{k}"] = fn
        return agg

    def higher_is_better(self):
        hib = {}
        for name, task in self.subtasks.items():
            task_hib = task.higher_is_better()
            for k, flag in task_hib.items():
                hib[f"{name}_{k}"] = flag
        return hib
