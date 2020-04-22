import datetime
from typing import Tuple, Iterable

import numpy as np
from annoy import AnnoyIndex
from arqmath_eval import get_judged_documents
from arqmath_eval import ndcg
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from ARQMathCode.Entities.Post import Question
from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord
from ARQMathCode.topic_file_reader import TopicReader
from . import SimilarityFunction, EmbeddingSimilarityEvaluator


class IREvaluator(EmbeddingSimilarityEvaluator):
    """
    TODO: read https://drive.google.com/file/d/1y2EHcTLuA63VS6IL5wc29TlG9wIDDLg0/view
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    index = None
    finalized_index = False

    eval_topics_path = "../question_answer/eval_dir/Task1_Samples_V2.0.xml"

    def __init__(self, model: SentenceTransformer, dataloader: DataLoader, post_parser: PostParserRecord,
                 eval_topics_path, main_similarity: SimilarityFunction = None,
                 name: str = '', show_progress_bar: bool = None, device=None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        super().__init__(dataloader, main_similarity, name, show_progress_bar, device)
        self.model = model
        self.post_parser = post_parser
        # shape = (post_id, is_question, <embeddings>)
        # index init
        self.index = np.empty(shape=(2, 0))
        self.annoy_index = AnnoyIndex(self.model.get_sentence_embedding_dimension())

        self.eval_texts = self._eval_texts_from_xml(eval_topics_path)

    @staticmethod
    def _eval_texts_from_xml(eval_xml_path):
        reader = TopicReader(eval_xml_path)
        return {t_key: str(t_vals.question) for t_key, t_vals in reader.map_topics.items()}

    def add_to_index(self, questions: Tuple[int, Iterable[Question]], infer_batch=32, reload_embs_dir=False):
        # index only certain topics, or everything?
        # -> Index everything
        batch_index = []
        batch_type = []
        batch_sents = []
        for q_i, q in questions:
            # 'questions' are never in relevant judgements as answers
            # batch_index.append(q_i)
            # batch_type.append(1)
            # batch_sents.append(q.body)
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

    def index_judged_questions(self, reload_embs_dir=False):
        relevant_qs = dict()
        for relevant_qi in get_judged_documents("task1"):
            try:
                parent_id = self.post_parser.map_just_answers[int(relevant_qi)].parent_id
            except KeyError as e:
                print("IREvaluator error: judged answer %s was not loaded and can not be evaluated" % relevant_qi)
                raise e
            relevant_qs[parent_id] = self.post_parser.map_questions[parent_id]
        self.add_to_index(relevant_qs.items(), reload_embs_dir=reload_embs_dir)

    def clear_index(self):
        self.annoy_index.unload()
        self.annoy_index = AnnoyIndex(self.model.get_sentence_embedding_dimension(), 'euclidean')
        self.trec_metric = "euclidean_ndcg"
        self.finalized_index = False

    def finalize_index(self, annoy_trees=100, out_file=None):
        self.annoy_index.build(annoy_trees)
        self.finalized_index = True
        if out_file is None:
            out_file = "annoy_index_%s_%s_items.pkl" % (str(datetime.date.today()), len(self.index[0]))
        self.annoy_index.save(out_file)

    def _postproc_scores(self, answers_distances: dict, scale_to=(0, 4)):
        dists = np.array(list(answers_distances.values()))
        norm_dists = (dists-min(dists))/(max(dists)-min(dists))
        norm_similarities = 1-norm_dists
        scaled_similarities = (norm_similarities-scale_to[0])*(scale_to[1]-scale_to[0])
        return dict(zip(answers_distances.keys(), scaled_similarities))

    def _get_ranked_list(self, question_body: str, no_ranked_results=1000):
        # return the most similar answers for a body of given piece of text
        question_emb = self.model.encode([question_body])[0]
        ranked_results_i, dists = self.annoy_index.get_nns_by_vector(question_emb, n=no_ranked_results,
                                                                     include_distances=True)
        ranked_results = self.index[0, ranked_results_i]
        dists_dict = dict(zip(map(lambda a_key: str(int(a_key)), ranked_results), dists))
        similarity_dict = self._postproc_scores(dists_dict)
        return similarity_dict

    def __call__(self, *args, eval_all_metrics=True, reindex=True, **kwargs) -> float:
        if reindex:
            self.clear_index()
            self.index_judged_questions()
        if not self.finalized_index:
            self.finalize_index()

        self.questions_predicted_nns = {str(k): self._get_ranked_list(v) for k, v in self.eval_texts.items()}

        def trec_metric_f():
            return ndcg(self.questions_predicted_nns)

        if eval_all_metrics:
            return super(IREvaluator, self).__call__(*args, **kwargs, additional_evaluator=trec_metric_f)
        else:
            return ndcg(self.questions_predicted_nns)
