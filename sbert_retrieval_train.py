from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.polish_substituer import PolishSubstituer
from question_answer.utils import examples_from_questions_tup
from sentence_transformers import SentenceTransformer, losses, SentencesDataset
import pickle

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cuda"

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens',
                            logfile="train_sampled_eval5.log", tboard_logdir="train_sampled_eval5.tboard")
# model = SentenceTransformer('/home/xstefan3/arqmath/compubert/out_whole', logfile="train_whole_sampled_eval.log")

clef_home_directory_file_path = '/home/xstefan3/arqmath/data/Collection'
dr = DataReaderRecord(clef_home_directory_file_path)
# dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=1000)

postprocessor = PolishSubstituer('/home/xstefan3/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv')
postproc_questions = list(postprocessor.process_questions(dr.post_parser.map_questions))

all_examples = list(examples_from_questions_tup(postproc_questions))
# all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions))
examples_len = len(all_examples)

train_dev_test_split = (int(0.8*examples_len), int(0.9*examples_len))

# single-time preprocessing support
train_data = SentencesDataset(all_examples[:train_dev_test_split[0]], model, show_progress_bar=True)
pickle.dump(train_data, open("train_data.pkl", "wb"))
# train_data = pickle.load(open("train_data.pkl", "rb"))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

dev_data = SentencesDataset(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model, show_progress_bar=True)
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=2000)
dev_loader = DataLoader(dev_data, batch_size=16, sampler=dev_sampler)

train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)

model.fit(train_objectives=[(train_loader, train_loss)],
          evaluator=evaluator,
          epochs=10,
          evaluation_steps=2000,
          warmup_steps=int(217206/5),
          output_path="train_sampled_eval4",
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False})

