from typing import List, Tuple, Iterable

from scipy.stats import zscore
import numpy as np
from sentence_transformers.readers import InputExample
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize


from ARQMathCode.Entities.Post import Question


def upvotes_to_similarities(all_q_votes):
    if len(all_q_votes) < 2:
        return [1.0] if all_q_votes[0] > 0 else [0.0]
    z_vals = zscore(all_q_votes)
#     z_vals_inv = (-1)*z_vals
    z_vals_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
    return z_vals_norm


def examples_from_questions_tup(questions: Iterable[Tuple[int, Question]]):
    for q_i, q in questions:
        if q_i % 10000 == 0:
            print("Loading %s" % q_i)
        if q.answers is None:
            continue
        all_q_upvotes = [a.score for a in q.answers]
        all_q_sims = upvotes_to_similarities(all_q_upvotes)
        if np.isnan(all_q_sims).any():
            # skip the votes with equally-rated answers: these are mostly 0-votes: we do not know anything about
            continue
        for q_sent in sent_tokenize(q.body):
            for a_i, a in enumerate(q.answers):
                for a_sent in a.body:
                    yield InputExample("%s_%s" % (q_i, a_i), [q_sent, a_sent], all_q_sims[a_i])


def examples_for_q_answers(q_text: str, a_texts: List[str], a_dists=List[float]):
    for a_text, a_dist in zip(a_texts, a_dists):
        yield InputExample("Infer_example", [q_text, a_text], a_dist)


def dataloader_from_examples(examples, model, batch_size=8, shuffle=False):
    train_data = SentencesDataset(examples, model, show_progress_bar=True)
    return DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
