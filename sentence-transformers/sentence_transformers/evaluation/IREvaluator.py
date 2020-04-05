from typing import Tuple, Iterable

import numpy as np
from annoy import AnnoyIndex
from pytrec_eval import RelevanceEvaluator
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from ARQMathCode.Entities.Post import Question
from ARQMathCode.topic_file_reader import TopicReader
from . import SimilarityFunction, EmbeddingSimilarityEvaluator


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

    eval_topics_path = "../question_answer/eval_dir/Task1_Samples_V2.0.xml"
    rel_questions_map = {1: 3082747,
                         2: 3163489,
                         3: 3237032}

    def __init__(self, model: SentenceTransformer, dataloader: DataLoader, rel_judgements_file: str,
                 eval_topics_path, trec_metric="ndcg", main_similarity: SimilarityFunction = None,
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
        self.index = np.empty(shape=(2, 0))
        self.annoy_index = AnnoyIndex(self.model.get_sentence_embedding_dimension())

        self.judgements = dict()
        # evaluator init
        with open(rel_judgements_file, "r") as f:
            for ln in f.readlines():
                l_content = [item.strip() for item in ln.split("\t")]
                try:
                    q = self.rel_questions_map[int(l_content[0])]
                    try:
                        self.judgements[str(q)][l_content[2]] = int(l_content[-1])
                    except KeyError:
                        self.judgements[str(q)] = {l_content[2]: int(l_content[-1])}
                except KeyError:
                    raise KeyError("Key %s not found in rel_questions_map" % l_content[0])
        self.evaluator = RelevanceEvaluator(self.judgements, measures={trec_metric})
        self.trec_metric = trec_metric
        self.eval_texts = self._eval_texts_from_xml(eval_topics_path)

    @staticmethod
    def _eval_texts_from_xml(eval_xml_path):
        reader = TopicReader(eval_xml_path)
        return {reader.topic_map[t_key]: str(t_vals.question) for t_key, t_vals in reader.map_topics.items()}

    def add_to_index(self, questions: Tuple[int, Iterable[Question]], infer_batch=32, reload_embs_dir=False):
        # index only certain topics, or everything?
        # -> Index everything
        batch_index = []
        batch_type = []
        batch_sents = []
        for q_i, q in questions:
            batch_index.append(q_i)
            batch_type.append(1)
            batch_sents.append(q.body)
            if q.answers is None:
                continue
            for a in q.answers:
                batch_index.append(a.post_id)
                batch_type.append(0)
                batch_sents.append(a.body)
        batch_out = np.concatenate(([batch_index], [batch_type]), axis=0)
        self.index = np.concatenate((self.index, batch_out), axis=1)
        if not reload_embs_dir:
            batch_embs = self.model.encode(batch_sents, show_progress_bar=True, batch_size=infer_batch)
            for i, emb in enumerate(batch_embs):
                self.annoy_index.add_item(i, emb)
        else:
            self.annoy_index.load(reload_embs_dir)
            self.finalized_index = True
        return batch_index

    def finalize_index(self, annoy_trees=100):
        self.annoy_index.build(annoy_trees)
        self.finalized_index = True
        self.annoy_index.save("annoy_t10.pkl")

    def _get_ranked_list(self, question_body: str, no_ranked_results=int(10e5)):
        # return the most similar answers for a body of given piece of text
        question_emb = self.model.encode([question_body])[0]
        ranked_results, dists = self.annoy_index.get_nns_by_vector(question_emb, n=no_ranked_results, include_distances=True)
        return dict(zip(map(str, ranked_results), dists))

    def __call__(self, *args, eval_all_metrics=False, **kwargs) -> float:
        if not self.finalized_index:
            self.finalize_index()

        # question_bodies = {k: question_bodies[v] for k, v in self.rel_questions_map}
        questions_predicted_nns = {str(self.rel_questions_map[k]): self._get_ranked_list(v) for k, v in self.eval_texts.items()}

        def trec_metric_f():
            results_each = self.evaluator.evaluate(questions_predicted_nns)
            return float(np.mean([v[self.trec_metric] for v in results_each.values()]))

        if eval_all_metrics:
            return super(IREvaluator, self).__call__(*args, **kwargs, additional_evaluator=trec_metric_f)
        else:
            return trec_metric_f()
