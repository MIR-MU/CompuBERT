from typing import List

import torch
from sentence_transformers import SentencesDataset
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.readers import InputExample
from sentence_transformers.util import paired_embeddings_for_dataloader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_manhattan_distances, paired_euclidean_distances
from torch.utils.data import DataLoader
import numpy as np


class QuestionResponder:

    def __init__(self, main_similarity: SimilarityFunction=None, device=None):
        self.main_similarity = main_similarity
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    @staticmethod
    def examples_for_q_answers(q_text: str, a_texts: List[str], a_dists: List[float] = None):
        if a_dists is None:
            a_dists = [0]*len(a_texts)
        for a_text, a_dist in zip(a_texts, a_dists):
            yield InputExample("Infer_example", [q_text, a_text], a_dist)

    @staticmethod
    def _dataloader_from_examples(examples, model, batch_size=8, shuffle=False):
        train_data = SentencesDataset(examples, model, show_progress_bar=True)
        return DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)

    def ranked_answers_for_question(self, model, question: str, answers_text: List[str]):
        qas_examples = self.examples_for_q_answers(question, answers_text)
        qas_loader = self._dataloader_from_examples(list(qas_examples), model)
        qas_loader.collate_fn = model.smart_batching_collate

        embeddings1, embeddings2 = paired_embeddings_for_dataloader(self.device, model, qas_loader, get_labels=False)

        try:
            cosine_distances = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
            euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
            dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
            all_dists = [cosine_distances, manhattan_distances, euclidean_distances, dot_products]
        except Exception as e:
            print(embeddings1)
            print(embeddings2)
            raise e

        return np.argsort(all_dists[int(self.main_similarity.value)])
