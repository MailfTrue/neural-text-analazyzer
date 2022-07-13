import spacy
import fire
from pprint import pprint
from neural_network import (EmotionExcelNeuralNetworkTrainer,
                            EmotionKinopoiskNeuralNetworkTrainer,
                            RelevanceExcelNeuralNetworkTrainer)


def train():
    # Учим эмоциональную модель из наших данных
    emotion = EmotionExcelNeuralNetworkTrainer(spacy_nlp_model_name="ru_core_news_lg",
                                               model_save_path="models/emotion_model")
    emotion.train_model(*emotion.load_training_data())

    # Обогащаем через отзывы
    emotion_kinopoisk = EmotionKinopoiskNeuralNetworkTrainer(spacy_nlp_model_name="models/emotion_model",
                                                             model_save_path="models/emotion_model")
    emotion_kinopoisk.train_model(*emotion_kinopoisk.load_training_data())

    # Учим модель релевантных сообщений из наших данных
    relevance = RelevanceExcelNeuralNetworkTrainer(spacy_nlp_model_name="ru_core_news_md",
                                                   model_save_path="models/relevance_model")
    relevance.train_model(*relevance.load_training_data())


def test():
    emotion_spacy = spacy.load("models/emotion_model")

    with open('test.txt', encoding='utf-8') as f:
        text = f.read()
        return {k: round(v, 3) for k, v in emotion_spacy(text).cats.items()}


if __name__ == '__main__':
    fire.Fire()
