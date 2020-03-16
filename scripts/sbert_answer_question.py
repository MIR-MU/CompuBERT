# model answering demo
from sentence_transformers.question_responder import QuestionResponder
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction

device = "cpu"

model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens', device=device)

responder = QuestionResponder(main_similarity=SimilarityFunction.COSINE, device=device)

ranked_list = responder.ranked_answers_for_question(model, question="What the hell?",
                                                    answers_text=["Whatever", "Nevermind", "I do not know."])

print("done")
