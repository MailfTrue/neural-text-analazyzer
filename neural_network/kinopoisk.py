import json
import random

from tqdm import tqdm
from .abstract import AbstractNeuralNetworkTrainer
from .mixins import EmotionNeuralNetworkTrainerMixin


class KinopoiskNeuralNetworkTrainer(AbstractNeuralNetworkTrainer):
    def message_is_valid(self, message: dict):
        return self.get_message_text(message).strip()

    def get_message_text(self, message: dict):
        return message['content']

    def get_spacy_label(self, message: dict):
        raise NotImplementedError()

    def load_training_data(self, data_directory: str = "./train", split: float = 0.8, limit: int = 0) -> tuple:
        messages = []
        with open("./kinopoisk.jsonl", encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading training data"):
                _message = json.loads(line)
                if self.message_is_valid(_message):
                    text = self.get_message_text(_message)
                    spacy_label = self.get_spacy_label(_message)
                    messages.append((text, spacy_label))

                    if limit and len(messages) >= limit:
                        break

        random.shuffle(messages)

        if limit:
            messages = messages[:limit]
        split = int(len(messages) * split)
        return messages[:split], messages[split:]


class EmotionKinopoiskNeuralNetworkTrainer(KinopoiskNeuralNetworkTrainer, EmotionNeuralNetworkTrainerMixin):
    def get_spacy_label(self, message: dict):
        return {"cats": {"pos": "Good" == message['grade3'],
                         "neutral": "Neutral" == message['grade3'],
                         "neg": "Bad" == message['grade3']}}

    def message_is_valid(self, message: dict):
        return self.get_message_text(message).strip() and str(message['grade3']) in ["Good", "Bad", "Neutral"]
