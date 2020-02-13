from ARQMathCode.post_reader_record import DataReaderRecord
from question_answer.utils import examples_from_questions, dataloader_from_examples
from sentence_transformers import SentenceTransformer, losses

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

device = "cpu"

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device, logfile="train.log")

clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=10000)

all_examples = list(examples_from_questions(dr.post_parser.map_questions))
examples_len = len(all_examples)

# train_dev_test_split = (int(0.1*examples_len), int(0.2*examples_len))
train_dev_test_split = (160, 320)

train_loader = dataloader_from_examples(all_examples[:train_dev_test_split[0]], model,
                                        batch_size=16, shuffle=True)

dev_loader = dataloader_from_examples(all_examples[train_dev_test_split[0]:train_dev_test_split[1]], model,
                                      batch_size=16, shuffle=False)

train_loss = losses.CosineSimilarityLoss(model=model)
evaluator = EmbeddingSimilarityEvaluator(dev_loader, show_progress_bar=True, device=device)

model.fit(train_objectives=[(train_loader, train_loss)],
          evaluator=evaluator,
          epochs=10,
          evaluation_steps=10,
          warmup_steps=100,
          output_path="out",
          optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False})

print("done")
