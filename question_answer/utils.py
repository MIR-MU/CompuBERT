from scipy.stats import zscore
from sentence_transformers.readers import InputExample
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader


def upvotes_to_distance(all_q_votes):
    if len(all_q_votes) < 2:
        return [1.0] if all_q_votes[0] > 0 else [0.0]
    z_vals = zscore(all_q_votes)
#     z_vals_inv = (-1)*z_vals
    z_vals_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
    return z_vals_norm


def examples_from_questions(questions):
    for q_i, q in questions.items():
        if q_i % 10000 == 0:
            print("Loading %s" % q_i)
        if q.answers is None:
            continue
        all_q_upvotes = [a.score for a in q.answers]
        all_q_dists = upvotes_to_distance(all_q_upvotes)

        for a_i, a in enumerate(q.answers):
            yield InputExample("%s_%s" % (q_i, a_i), [q.body, a.body], all_q_dists[a_i])


def dataloader_from_examples(examples, model, batch_size, shuffle):
    train_data = SentencesDataset(examples, model, show_progress_bar=True)
    return DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
