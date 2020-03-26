from typing import Tuple, Iterable, Dict

from sentence_transformers import SentenceTransformer

from ARQMathCode.Entities.Post import Question
from . import SimilarityFunction, EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

import torch
import logging
from tqdm import tqdm
from ..util import paired_embeddings_for_dataloader
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from annoy import AnnoyIndex
from pytrec_eval import RelevanceEvaluator


class IREvaluator(EmbeddingSimilarityEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    index = None
    finalized_index = False

    def __init__(self, model: SentenceTransformer, dataloader: DataLoader, rel_questions_map: Dict[int, int],
                 rel_judgements_file: str, main_similarity: SimilarityFunction = None,
                 name: str = '', show_progress_bar: bool = None, device=None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        super().__init__(dataloader, main_similarity, name, show_progress_bar, device)
        self.model = model
        # shape = (post_id, is_question, <embeddings>)
        # index init
        self.index = np.empty(shape=(self.model.get_sentence_embedding_dimension(), 0))
        self.annoy_index = AnnoyIndex(2e6)

        self.judgements = dict()
        # evaluator init
        with open(rel_judgements_file, "r") as f:
            for ln in f.readlines():
                l_content = [item.strip() for item in ln.split("\t")]
                try:
                    q = rel_questions_map[int(l_content[0])]
                    try:
                        self.judgements[q][l_content[2]] = int(l_content[-1])
                    except KeyError:
                        self.judgements[q] = {l_content[2]: int(l_content[-1])}
                except KeyError:
                    raise KeyError("Key %s not found in rel_questions_map" % l_content[0])
        self.rel_questions_map = rel_questions_map
        self.evaluator = RelevanceEvaluator(self.judgements)

    def add_to_index(self, questions: Tuple[int, Iterable[Question]]):
        # index only certain topics, or everything?
        # Indexing everything
        batch_index = []
        batch_type = []
        batch_sents = []
        for q_i, q in questions:
            batch_index.append(q_i)
            batch_type.append(1)
            batch_sents.append(q.body)
            for a in q.answers:
                batch_index.append(a.post_id)
                batch_type.append(0)
                batch_sents.append(a.body)
        batch_embs = self.model.encode(batch_sents)
        batch_out = np.concatenate((batch_index, batch_type), axis=1)
        self.index = np.concatenate((self.index, batch_out), axis=0)
        for i, emb in enumerate(batch_embs):
            self.annoy_index = self.annoy_index.add(i, emb)

        return batch_index

    def finalize_index(self, annoy_trees=10):
        self.finalized_index = True
        self.annoy_index.build(annoy_trees)
        self.annoy_index.save("annoy_t10.pkl")

    def _get_ranked_list(self, question_body: str, no_ranked_results=10e5):
        # return the most similar answers for a body of given piece of text
        question_emb = self.model.encode([question_body])[0]
        out = self.annoy_index.get_nns_by_vector(question_emb, n=no_ranked_results, include_distances=True)
        # TODO: distance are just to check the ranking, then remove it
        print(len(out))
        return dict(zip(self.index[[o[0] for o in out]], [o[1] for o in out]))

    def __call__(self, questions: dict, trec_metric="ntcg", **kwargs) -> float:
        if not self.finalized_index:
            self.finalize_index()

        questions_bodies = {k: questions[v].body for k, v in self.rel_questions_map}
        questions_predicted_nns = {k: self._get_ranked_list(v) for k, v in questions_bodies}

        def trec_metric():
            return self.evaluator.evaluate(questions_predicted_nns)

        return super.__call__(**kwargs, additional_evaluator=trec_metric)
