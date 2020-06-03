from enum import Enum
from typing import Iterable, List, Dict
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ARQMathCode.Entities.Post import Question
from ARQMathCode.post_reader_record import DataReaderRecord
from ARQMathCode.topic_file_reader import TopicReader
from preproc.question_answer.blank_substituer import BlankSubstituer
from preproc.question_answer.infix_substituer import InfixSubstituer
from preproc.question_answer.polish_substituer import PolishSubstituer


class Preprocessing(Enum):
    LATEX = 0
    PREFIX = 1
    INFIX = 2


class EvalQuestion(object):
    def __init__(self, qid: str, body: str):
        self.qid = qid
        self.body = body


class SBertIRSystem:
    question_index = []
    answers_index = []
    questions_embeddings = None
    answers_embeddings = None

    def __init__(self, model_path: str, questions_path: str, preprocessing: Preprocessing, identifier: str,
                 *preprocessor_args, use_cuda=True, limit_posts=None):
        if preprocessing == Preprocessing.LATEX:
            self.preprocessor = BlankSubstituer()
        elif preprocessing == Preprocessing.PREFIX:
            self.preprocessor = PolishSubstituer(*preprocessor_args)
        elif preprocessing == Preprocessing.INFIX:
            self.preprocessor = InfixSubstituer(*preprocessor_args)
        else:
            raise NotImplementedError("No default preprocessor. Pick one from %s" % Preprocessing.enum_members)

        reader = DataReaderRecord(questions_path, limit_posts=limit_posts)

        self.parser = self.preprocessor.process_parser(reader.post_parser)
        self.device = "cuda" if use_cuda else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        self.identifier = identifier

    def index_questions(self, questions_ids: Iterable[int] = None, infer_batch_size=16) -> None:
        if questions_ids is not None:
            self.question_index = np.array(list(questions_ids))
        else:
            self.question_index = np.array(list(self.parser.map_questions.keys()))
        questions_bodies = [self.parser.map_questions[int(qid)].body for qid in self.question_index]
        self.questions_embeddings = np.array(self.model.encode(questions_bodies, batch_size=infer_batch_size,
                                                               show_progress_bar=True))

    def index_eval_questions(self, eval_topics_xml: str = "question_answer/eval_dir/Topics_V2.0.xml",
                             infer_batch_size=16):
        topic_reader = TopicReader(eval_topics_xml)
        topics = topic_reader.map_topics.values()
        self.question_index = np.array(
            [topic.topic_id for topic in topics if topic.topic_id not in topic_reader.eval_topics.keys()])
        questions_bodies = [topic.question for topic in topics if topic.topic_id not in topic_reader.eval_topics.keys()]
        question_bodies_postproc = [self.preprocessor.subst_body(body) for body in questions_bodies]
        self.questions_embeddings = np.array(self.model.encode(question_bodies_postproc, batch_size=infer_batch_size,
                                                               show_progress_bar=True))
        for qid, qbody in zip(self.question_index, question_bodies_postproc):
            self.parser.map_questions[qid] = EvalQuestion(qid, qbody)
        return self.question_index

    def index_answers(self, answers_ids: Iterable[int] = None, infer_batch_size=16) -> None:
        if answers_ids is not None:
            self.answers_index = list(answers_ids)
        else:
            self.answers_index = list(self.parser.map_just_answers.keys())
        answers_bodies = [self.parser.map_just_answers[int(qid)].body for qid in self.answers_index]
        self.answers_embeddings = np.array(self.model.encode(answers_bodies, batch_size=infer_batch_size,
                                                             show_progress_bar=True))

    def _ranked_results_for_questions(self, questions_ids: np.ndarray = None, answers_ids: np.ndarray = None) \
            -> Dict[int, Dict[int, str]]:
        from sklearn.metrics.pairwise import cosine_similarity
        if questions_ids is not None:
            selected_index = np.argwhere([self.question_index == qid for qid in questions_ids])[:, 0]
            selected_q_embeddings = self.questions_embeddings[selected_index]
        else:
            questions_ids = self.question_index
            selected_q_embeddings = self.questions_embeddings
        if answers_ids is not None:
            selected_index = np.arqwhere([self.answers_index == aid for aid in answers_ids])[:, 0]
            selected_a_embeddings = self.answers_embeddings[selected_index]
        else:
            answers_ids = np.array(self.answers_index)
            selected_a_embeddings = self.answers_embeddings

        similarities = cosine_similarity(selected_q_embeddings, selected_a_embeddings)

        results = {}
        for q_id, q_similarities in tqdm(zip(questions_ids, similarities), desc="Collecting ranked lists for Qs"):
            answers_sorted_idx = (-q_similarities).argsort(axis=0)
            answers_sorted = answers_ids[answers_sorted_idx].flatten()
            answers_sims = q_similarities[answers_sorted_idx]
            results[q_id] = dict(zip(answers_sorted, answers_sims))
        return results

    def dump_arqmath_response_ratings(self, submit_dir: str):
        # https://gitlab.fi.muni.cz/sojka/mir-general/-/issues/28#note_99180
        filename = f'{submit_dir}/Task-1-QA/MIRMU-task1-{self.identifier}-auto-both-P.tsv'
        self.dump_response_ratings(filename, top_n=1000)

    def dump_response_ratings(self, file_path: str, questions_ids: Iterable[int] = None,
                              identifier=None, top_n: int = None) -> None:
        results = self._ranked_results_for_questions(questions_ids)
        with open(file_path, 'wt') as f:
            for q_id, answers in results.items():
                a_pairs = list(answers.items())[:top_n] if top_n is not None else list(answers.items())
                for rank, (a_id, a_score) in enumerate(a_pairs):
                    line = f'{q_id}\t{a_id}\t{rank}\t{a_score}\t{self.identifier if identifier is None else identifier}'
                    print(line, file=f)

    def dump_response_bodies(self, json_path: str, questions_ids: Iterable[int], topn: int = 10) -> None:
        import json
        results = self._ranked_results_for_questions(np.array(list(questions_ids)))
        results_str = dict()
        for q_id, answers in results.items():
            if type(q_id) == np.int:
                q_id = int(q_id)
            q_body = self.parser.map_questions[q_id].body
            results_str[q_id] = {"body": q_body,
                                 "answers": {}}
            for rank, (a_id, a_score) in enumerate(list(answers.items())[:topn]):
                a_body = self.parser.map_just_answers[int(a_id)].body
                results_str[q_id]["answers"][int(a_id)] = {"rank": rank,
                                                           "cosine_sim_score": float(a_score),
                                                           "body": a_body}
        with open(json_path, 'wt') as f:
            f.write(json.dumps(results_str))

    def save_index(self, save_dir: str):
        np.save(os.path.join(save_dir, "questions_embeddings_%s" % self.identifier), self.questions_embeddings)
        np.save(os.path.join(save_dir, "answers_embeddings_%s" % self.identifier), self.answers_embeddings)

        np.save(os.path.join(save_dir, "question_index_%s" % self.identifier), self.question_index)
        np.save(os.path.join(save_dir, "answer_index_%s" % self.identifier), self.answers_index)

    def load_index(self, load_dir: str):
        self.questions_embeddings = np.load(os.path.join(load_dir, "questions_embeddings_%s.npy" % self.identifier))
        self.answers_embeddings = np.load(os.path.join(load_dir, "answers_embeddings_%s.npy" % self.identifier))

        self.question_index = np.load(os.path.join(load_dir, "question_index_%s.npy" % self.identifier))
        self.answers_index = np.load(os.path.join(load_dir, "answer_index_%s.npy" % self.identifier))
