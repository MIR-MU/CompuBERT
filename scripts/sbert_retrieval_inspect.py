# TODO: not the main train script - just for local tests. Go to ../sbert_retrieval_train.py

from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.unique_prefix_substituer import UniquePrefixSubstituer
from question_answer.utils import examples_from_questions_tup
from sentence_transformers import SentenceTransformer, losses, SentencesDataset
from scripts.loader_to_tsv import dump_to_tsv

from sentence_transformers.evaluation import IREvaluator
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cpu"

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device)

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=1000)

# postprocessor = UniquePrefixSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv',
#                                        "/home/michal/Documents/projects/arqmath/compubert/question_answer/out/0_BERT/vocab.txt")
# postproc_questions = list(postprocessor.process_questions(dr.post_parser.map_questions))
# postprocessor.extend_sbert_vocab(model)

all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions.items()))
# all_examples = list(examples_from_questions_tup(postproc_questions))
examples_len = len(all_examples)

train_dev_test_split = (int(0.8*examples_len), int(0.9*examples_len))
# train_dev_test_split = (6, 8)

train_data = SentencesDataset(all_examples[:train_dev_test_split[0]], model, show_progress_bar=True)
# pickle.dump(train_data, open("train_data.pkl", "wb"))
# train_data = pickle.load(open("train_data.pkl", "rb"))

train_loader = DataLoader(train_data, batch_size=5, shuffle=False)

dev_data = SentencesDataset(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model, show_progress_bar=True)
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=15)
dev_loader = DataLoader(train_data, batch_size=5, sampler=dev_sampler)

dump_to_tsv(train_loader, ids_token_map=model[0].tokenizer.ids_to_tokens, out_file='nopreproc_dump.tsv', first_n=10)
