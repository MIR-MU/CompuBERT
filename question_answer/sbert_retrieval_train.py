from ARQMathCode.post_reader_record import DataReaderRecord
from scipy.stats import zscore
from sentence_transformers import SentencesDataset, SentenceTransformer, losses
from torch.utils.data import DataLoader

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path)


def upvotes_to_distance(all_q_votes):
    z_vals = zscore(all_q_votes)
    z_vals_inv = (-1)*z_vals
    return z_vals_inv


def examples_from_questions(questions):
    for q_i, q in questions.items():
        for a_i, a in q.answers:
            yield InputExample("%s_%s" % (q_i, a_i), [q.body, a.body], upvotes_to_distance(a.score))


examples_len = len(dr.post_parser.map_questions)

train_dev_test_split = (int(0.8*examples_len), int(0.9*examples_len))

all_examples = list(examples_from_questions(dr.post_parser.map_questions))

train_examples = all_examples[:train_dev_test_split[0]]
train_data = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)

dev_examples = all_examples[train_dev_test_split[0]:train_dev_test_split[1]]
dev_data = SentencesDataset(dev_examples, model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=32)

train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
