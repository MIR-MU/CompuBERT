# # TODO: not the main train script - just for local tests. Go to ../sbert_retrieval_train.py
# from typing import List
#
# from sentence_transformers import SentenceTransformer
#
# from ARQMathCode.post_reader_record import DataReaderRecord
# from arqmath_eval import get_topics, get_judged_documents, get_ndcg
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import os
#
# from preproc.question_answer.polish_substituer import PolishSubstituer
#
# device = "cpu"
#
# clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection_v1.0'
# dr = DataReaderRecord(clef_home_directory_file_path)
# postprocessor = PolishSubstituer('/data/arqmath/ARQMath_CLEF2020/Collection_v1.0/formula_prefix.V1.0.tsv')
#
#
# def report_ndcg_results(result_tsv_name: str, results: dict):
#     with open(result_tsv_name, 'wt') as f:
#         for topic, documents in results.items():
#             top_documents = sorted(documents.items(), key=lambda x: x[1], reverse=True)[:1000]
#             for rank, (document, similarity_score) in enumerate(top_documents):
#                 line = '{}\txxx\t{}\t{}\t{}\txxx'.format(topic, document, rank + 1, similarity_score)
#                 print(line, file=f)
#
#
# def eval_prefix_transformer(model_dir: str = '/data/arqmath/models/train_sampled_eval9', subsample: int = False):
#     model = SentenceTransformer(model_dir, device=device)
#
#     task = 'task1-votes'
#     subset = 'validation'
#     results = {}
#     all_questions_ids = get_topics(task=task, subset=subset)
#     all_questions_raw = [(qid, dr.post_parser.map_questions[int(qid)]) for qid in all_questions_ids]
#     # all_questions = all_questions_raw
#     all_questions = list(postprocessor.process_questions(all_questions_raw))
#     if subsample:
#         all_questions = all_questions[:subsample]
#
#     for qid, question in all_questions:
#         results[qid] = {}
#         judged_answer_ids = get_judged_documents(task=task, subset=subset, topic=qid)
#         question_e = model.encode([question.body])
#         answers_bodies = [dr.post_parser.map_just_answers[int(aid)].body for aid in judged_answer_ids]
#         answers_e = model.encode(answers_bodies)
#         answers_dists = cosine_similarity(np.array(question_e), np.array(answers_e))[0]
#         for aid, answer_sim in sorted(zip(judged_answer_ids, answers_dists), key=lambda qid_dist: qid_dist[1], reverse=True):
#             print(aid, answer_sim)
#             results[qid][aid] = float(answer_sim)
#
#     ndcg_val = get_ndcg(results, task=task, subset=subset)
#     return ndcg_val, results
#
#
# def eval_all_dirs(dirs: List[str], results_tsv_dir: str, summary_path: str):
#     summary_f = open(summary_path, "w")
#     for model_dir in dirs:
#         ndcg, results = eval_prefix_transformer(model_dir, subsample=100)
#         print("%s:\t%s" % (model_dir, ndcg), file=summary_f)
#         report_ndcg_results(os.path.join(results_tsv_dir, model_dir.split("/")[-1]+".tsv"), results)
#
#
# eval_all_dirs(["/data/arqmath/models/train_sampled_eval%s" % i for i in [9]],
#               "/home/michal/Documents/projects/arqmath/compubert/arqmath_eval_out",
#               "arqmath_eval_summary.tsv")
# print("done")
