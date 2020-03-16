# TODO: not the main train script - just for local tests. Go to ../sbert_retrieval_train.py

from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.polish_substituer import PolishSubstituer
from question_answer.utils import examples_from_questions_tup
from sentence_transformers import SentenceTransformer, losses, SentencesDataset
import pickle

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cpu"

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device)

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=100)

# postprocessor = PolishSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv')
# postproc_questions = list(postprocessor.process_questions(dr.post_parser.map_questions))

# all_examples = list(examples_from_questions_tup(postproc_questions))
all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions.items()))
examples_len = len(all_examples)

# train_dev_test_split = (int(0.1*examples_len), int(0.2*examples_len))
train_dev_test_split = (6, 8)

train_data = SentencesDataset(all_examples[:train_dev_test_split[0]], model, show_progress_bar=True)
# pickle.dump(train_data, open("train_data.pkl", "wb"))
# train_data = pickle.load(open("train_data.pkl", "rb"))

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

dev_data = SentencesDataset(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model, show_progress_bar=True)
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=15)
dev_loader = DataLoader(train_data, batch_size=2, sampler=dev_sampler)

train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)

model.fit(train_objectives=[(train_loader, train_loss)],
          evaluator=evaluator,
          epochs=2,
          evaluation_steps=5,
          warmup_steps=2,
          output_path="../question_answer/out",
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False})

print("done")
