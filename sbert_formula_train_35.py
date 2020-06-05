# Created from run 29: attempt for experiment 9 reproduction, +
# correction of get_ndcg tracking,
# persistence of best model based on ndcg

from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.polish_substituer import PolishSubstituer
from preproc.question_answer.blank_substituer import BlankSubstituer
from formula_retrieval.utils import examples_from_questions_tup
from sentence_transformers import models, SentenceTransformer, losses, SentencesDataset
import pickle

# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from sentence_transformers.evaluation import IREvaluator
from sentence_transformers.evaluation import ArqmathEvaluator


device = "cuda"
experiment_id = 35

model = SentenceTransformer('bert-base-nli-stsb-mean-tokens',
                            logfile="train_sampled_eval%s.log" % experiment_id,
                            tboard_logdir="train_sampled_eval%s.tboard" % experiment_id)

clef_home_directory_file_path = '/home/xstefan3/arqmath/data/Collection_v1.0'

dr = DataReaderRecord(clef_home_directory_file_path)

postprocessor = BlankSubstituer()
postproc_parser = postprocessor.process_parser(dr.post_parser)

# postproc_questions = list(postproc_parser.map_questions)

all_examples = list(examples_from_questions_tup(postproc_parser.map_questions.items()))
# all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions))
examples_len = len(all_examples)

train_dev_test_split = (int(0.8*examples_len), int(0.9*examples_len))

# single-time preprocessing support
train_data = SentencesDataset(all_examples[:train_dev_test_split[0]], model, show_progress_bar=True)
pickle.dump(train_data, open("train_data_run%s_nopreproc_fix1_v1.0.pkl" % experiment_id, "wb"))
# train_data = pickle.load(open("train_data_run%s_nopreproc_html_v2_v1.0.pkl" % 29, "rb"))

train_loader = DataLoader(train_data, batch_size=14, shuffle=True)

dev_data = SentencesDataset(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model, show_progress_bar=True)
pickle.dump(dev_data, open("dev_data_run%s_nopreproc_fix1_v1.0.pkl" % experiment_id, "wb"))
# dev_data = pickle.load(open("dev_data_run%s_nopreproc_fix1_v1.0.pkl" % 29, "rb"))
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=2000)
dev_loader = DataLoader(dev_data, batch_size=14, sampler=dev_sampler, shuffle=False)

train_loss = losses.CosineSimilarityLoss(model=model)
# evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)
# evaluator = IREvaluator(model, dev_loader, post_parser=dr.post_parser, eval_topics_path="question_answer/eval_dir/Task1_Samples_V2.0.xml",
#                         show_progress_bar=True, device=device)
# postproc_parser = pickle.load(open("parser_run28_nopreproc_html_v2_v1.0.pkl", "rb"))
evaluator = ArqmathEvaluator(model, dev_loader, post_parser_postproc=postproc_parser, show_progress_bar=True, device=device)

model.fit(train_objectives=[(train_loader, train_loss)],
          evaluator=evaluator,
          epochs=7,
          evaluation_steps=1280,
          warmup_steps=30000,
          output_path="train_sampled_eval%s" % experiment_id,
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False})

