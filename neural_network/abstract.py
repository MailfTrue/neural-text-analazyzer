import random

from abc import ABC
from tqdm import tqdm

import spacy
from spacy.language import Example
from spacy.util import minibatch, compounding


class AbstractNeuralNetworkTrainer(ABC):
    labels: list
    spacy_nlp_model_name: str = "ru_core_news_md"
    model_save_path: str = "model_artifacts"

    def __init__(self, spacy_nlp_model_name: str | None = None, model_save_path: str | None = None):
        self.loaded_model = None
        if spacy_nlp_model_name is not None:
            self.spacy_nlp_model_name = spacy_nlp_model_name
        if model_save_path is not None:
            self.model_save_path = model_save_path

        if not self.labels:
            raise ValueError("labels must be not empty")

    def message_is_valid(self, message):
        raise NotImplementedError()

    def get_message_text(self, message):
        raise NotImplementedError()

    def get_spacy_label(self, message):
        raise NotImplementedError()

    def load_training_data(self, data_directory: str = "./train", split: float = 0.8, limit: int = 0) -> tuple:
        raise NotImplementedError()

    def train_model(
            self,
            training_data: list,
            test_data: list,
            iterations: int = 20,
    ) -> None:
        nlp = spacy.load(self.spacy_nlp_model_name)
        if "textcat" not in nlp.pipe_names:
            textcat = nlp.add_pipe("textcat", last=True)
        else:
            textcat = nlp.get_pipe("textcat")

        for label in self.labels:
            textcat.add_label(label)

        training_excluded_pipes = [
            pipe for pipe in nlp.pipe_names if pipe != "textcat"
        ]
        with nlp.disable_pipes(training_excluded_pipes):
            optimizer = nlp.begin_training()
            batch_sizes = compounding(
                8, 128.0, 1.001
            )
            for __ in tqdm(range(iterations), desc="Training"):
                loss = {}
                random.shuffle(training_data)
                batches = minibatch(training_data, size=batch_sizes)
                for batch in batches:
                    examples = [Example.from_dict(nlp.make_doc(text), labels) for (text, labels) in batch]
                    nlp.update(
                        examples,
                        drop=0.2,
                        sgd=optimizer,
                        losses=loss
                    )
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(self.model_save_path)
