from torch.utils.data import RandomSampler

from ARQMathCode.post_reader_record import DataReaderRecord
from question_answer.utils import examples_from_questions_tup, dataloader_from_examples
from sentence_transformers import SentenceTransformer, losses, SentencesDataset

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from scipy.stats import zscore
from sentence_transformers.readers import InputExample
from sentence_transformers import SentencesDataset
from torch.utils.data import DataLoader
import numpy as np

device = "cuda"

model = SentenceTransformer('/home/xstefan3/arqmath/compubert/out_whole_sampled_eval')

clef_home_directory_file_path = '/home/xstefan3/arqmath/data/Collection'
dr = DataReaderRecord(clef_home_directory_file_path)

all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions))
examples_len = len(all_examples)

train_dev_test_split = (int(0.8*examples_len), int(0.9*examples_len))

# model = SentenceTransformer('/home/xstefan3/arqmath/compubert/out_whole', logfile="train_whole_sampled_eval.log")

test_data = SentencesDataset(all_examples[train_dev_test_split[1]:], model, show_progress_bar=True)
# test_sampler = RandomSampler(dev_data, replacement=True, num_samples=250)

test_loader = DataLoader(test_data, batch_size=16)

evaluator = EmbeddingSimilarityEvaluator(test_loader, show_progress_bar=True, device=device)

test_val = model.evaluate(evaluator)
print(test_val)
