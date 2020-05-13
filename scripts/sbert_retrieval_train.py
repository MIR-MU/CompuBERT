# TODO: not the main train script - just for local tests. Go to ../sbert_retrieval_train.py

from torch.utils.data import RandomSampler, DataLoader

from ARQMathCode.post_reader_record import DataReaderRecord
from preproc.question_answer.unique_prefix_substituer import UniquePrefixSubstituer
# from preproc.question_answer.polish_substituer import PolishSubstituer
from preproc.question_answer.blank_substituer import BlankSubstituer
# from preproc.question_answer.external_substituer import ExternalSubstituer

from question_answer.utils import examples_from_questions_tup
from sentence_transformers import models, SentenceTransformer, losses, SentencesDataset

from sentence_transformers.evaluation import IREvaluator
from sentence_transformers.evaluation import ArqmathEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cpu"

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model = SentenceTransformer('/data/arqmath/models/train_sampled_eval9', device=device)

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection_v1.0'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=1000)

# postprocessor = UniquePrefixSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection/formula_prefix.V0.2.tsv',
#                                        "/home/michal/Documents/projects/arqmath/compubert/question_answer/out/0_BERT/vocab.txt")
# postprocessor = PolishSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection_v1.0/formula_prefix.V1.0.tsv')
# postprocessor = ExternalSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection_Vit/Posts_V1_0_prefix.json.gz')
postprocessor = BlankSubstituer()

postproc_parser = postprocessor.process_parser(dr.post_parser)
# postproc_questions = list(postprocessor.process_questions(dr.post_parser.map_questions))
# postproc_questions = list(dr.post_parser.map_questions.items())
# postprocessor.extend_sbert_vocab(model)

all_examples = list(examples_from_questions_tup(postproc_parser.map_questions.items()))
# all_examples = list(examples_from_questions_tup(dr.post_parser.map_questions.items()))
examples_len = len(all_examples)

# train_dev_test_split = (int(0.1*examples_len), int(0.2*examples_len))
train_dev_test_split = (18, 24)

train_data = SentencesDataset(all_examples[:train_dev_test_split[0]], model, show_progress_bar=True)
# pickle.dump(train_data, open("train_data.pkl", "wb"))
# train_data = pickle.load(open("train_data.pkl", "rb"))

train_loader = DataLoader(train_data, batch_size=6, shuffle=True)

dev_data = SentencesDataset(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model, show_progress_bar=True)
dev_sampler = RandomSampler(dev_data, replacement=True, num_samples=15)
dev_loader = DataLoader(train_data, batch_size=6, sampler=dev_sampler)

train_loss = losses.CosineSimilarityLoss(model=model)

# evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)
evaluator = ArqmathEvaluator(model, dev_loader, post_parser_postproc=postproc_parser, show_progress_bar=True, device=device)
# index all- not necessary for the current eval
# evaluator.add_to_index(dr.post_parser.map_questions.items())

print(evaluator(model, dev_loader))

model.fit(train_objectives=[(train_loader, train_loss)],
          evaluator=evaluator,
          epochs=2,
          evaluation_steps=1,
          warmup_steps=2,
          output_path="../question_answer/out",
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False})

print("done")
