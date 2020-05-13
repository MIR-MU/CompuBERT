from typing import List

from sentence_transformers import SentenceTransformer

from ARQMathCode.post_reader_record import DataReaderRecord
from arqmath_eval import get_topics, get_judged_documents, get_ndcg
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

import datetime
import random
from typing import Tuple, Iterable, Dict

import numpy as np
from annoy import AnnoyIndex
from arqmath_eval import get_judged_documents
from arqmath_eval.common import get_ndcg
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


from ARQMathCode.Entities.Post import Question
from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord
from ARQMathCode.topic_file_reader import TopicReader
from . import SimilarityFunction, EmbeddingSimilarityEvaluator


class ArqmathEvaluator(EmbeddingSimilarityEvaluator):
    """
    Lightweight implementation of IREvaluator - without content indexing,
    Utilizing the get_ndcg for small validation set of Task1-votes

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    index = None
    finalized_index = False
    trec_metric = "euclidean_ndcg"
    task = 'task1-votes'
    subset = 'small-validation'

    def __init__(self, model: SentenceTransformer, dataloader: DataLoader, post_parser_postproc: PostParserRecord,
                 main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None, device=None):

        super().__init__(dataloader, main_similarity, name, show_progress_bar, device)
        self.model = model
        self.post_parser = post_parser_postproc

    def eval_transformer(self, subsample: int = False):
        results = {}
        all_questions_ids = get_topics(task=self.task, subset=self.subset)
        # all_questions = dict([(int(qid), self.post_parser.map_questions[int(qid)]) for qid in all_questions_ids])
        all_questions = dict([(int(qid), "Question %s content" % qid) for qid in all_questions_ids])
        if subsample:
            all_questions = all_questions[:subsample]

        for i, (qid, question) in enumerate(all_questions.items()):
            results[qid] = {}
            judged_answer_ids = get_judged_documents(task=self.task, subset=self.subset, topic=str(qid))
            question_e = self.model.encode([question], batch_size=8)
            try:
                # answers_bodies = [self.post_parser.map_just_answers[int(aid)].body for aid in judged_answer_ids]
                answers_bodies = ["Answer %s body" % aid for aid in judged_answer_ids]
            except KeyError:
                print("Key error at qid %s" % qid)
                answers_bodies = []
                answers_bodies = ["Answer %s body" % aid for aid in judged_answer_ids]
            if not answers_bodies:
                print("No evaluated answers for question %s, dtype %s" % (qid, str(type(qid))))
                continue
            answers_e = self.model.encode(answers_bodies, batch_size=8)
            answers_dists = cosine_similarity(np.array(question_e), np.array(answers_e))[0]

            for aid, answer_sim in sorted(zip(judged_answer_ids, answers_dists), key=lambda qid_dist: qid_dist[1],
                                          reverse=True):
                print(aid, answer_sim)
                results[qid][aid] = float(answer_sim)

        ndcg_val = get_ndcg(results, task=self.task, subset=self.subset)
        print("Computed ncdg on %s questions: %s" % (len(all_questions), ndcg_val))
        return ndcg_val

    def __call__(self, *args, eval_all_metrics=True, **kwargs) -> float:
        def trec_metric_f():
            return self.eval_transformer()

        if eval_all_metrics:
            return super(ArqmathEvaluator, self).__call__(*args, **kwargs, additional_evaluator=trec_metric_f)
        else:
            return self.eval_transformer()
