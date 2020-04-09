# TODO: not the main train script - just for local tests. Go to ../sbert_retrieval_train.py

from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.unique_prefix_substituer import UniquePrefixSubstituer
from preproc.question_answer.polish_substituer import PolishSubstituer
from question_answer.utils import examples_from_questions_tup
from sentence_transformers import SentenceTransformer, losses, SentencesDataset

from sentence_transformers.evaluation import IREvaluator
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cuda"

# model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device)
model = SentenceTransformer('/data/arqmath/models/train_sampled_eval9', device=device)

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=10000)

# postprocessor = UniquePrefixSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv',
#                                        "/home/michal/Documents/projects/arqmath/compubert/question_answer/out/0_BERT/vocab.txt")
# postprocessor = PolishSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv')

# postproc_questions = list(postprocessor.process_questions(dr.post_parser.map_questions))
postproc_questions = dr.post_parser.map_questions
# postprocessor.extend_sbert_vocab(model)

all_examples = list(examples_from_questions_tup(postproc_questions.items()))
# all_examples = list(examples_from_questions_tup(postproc_questions))
examples_len = len(all_examples)

# train_dev_test_split = (int(0.1*examples_len), int(0.2*examples_len))
# # train_dev_test_split = (6, 8)

# train_data = SentencesDataset(all_examples, model, show_progress_bar=True)
# pickle.dump(train_data, open("train_data.pkl", "wb"))
# train_data = pickle.load(open("train_data.pkl", "rb"))

# train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

dev_data = SentencesDataset(all_examples[:10000], model, show_progress_bar=True)
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=15)
dev_loader = DataLoader(dev_data, batch_size=2, sampler=dev_sampler)

train_loss = losses.CosineSimilarityLoss(model=model)

# evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)
evaluator = IREvaluator(model, dev_loader, post_parser=dr.post_parser, show_progress_bar=True, device=device,
                        eval_topics_path="../question_answer/eval_dir/Task1_Samples_V2.0.xml")
# evaluator.add_to_index(dr.post_parser.map_questions.items())
# evaluator.index_judged_questions(dr.post_parser, reload_embs_dir="annoy_t10.pkl")
evaluator.index_judged_questions(dr.post_parser)
# evaluator.add_to_index(dr.post_parser.map_questions.items(), reload_embs_dir="annoy_t10.pkl")

print(evaluator(model, "../question_answer/out"))

print("done")
