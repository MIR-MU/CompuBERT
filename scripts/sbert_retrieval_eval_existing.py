from arqmath_eval import get_topics, get_judged_documents
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

task = 'task1-votes'
subset = 'validation'
results = {}
all_questions_ids = get_topics(task=task, subset=subset)
all_questions = dict([(int(qid), postproc_parser.map_questions[int(qid)]) for qid in all_questions_ids])

for i, (qid, question) in tqdm(list(enumerate(all_questions.items())), desc="Collecting answers for %s questions" % len(all_questions)):
    results[qid] = {}
    judged_answer_ids = get_judged_documents(task=task, subset=subset, topic=str(qid))
    question_e = model_saved.encode([question.body], batch_size=8)
    answers_bodies = [postproc_parser.map_just_answers[int(aid)].body for aid in judged_answer_ids]
    if not answers_bodies:
        print("No evaluated answers for question %s, dtype %s" % (qid, str(type(qid))))
        continue
    answers_e = model_saved.encode(answers_bodies, batch_size=8)
    answers_dists = cosine_similarity(np.array(question_e), np.array(answers_e))[0]
    for aid, answer_sim in sorted(zip(judged_answer_ids, answers_dists), key=lambda qid_dist: qid_dist[1], reverse=True):
        # print("Q %s, A %s: sim: %s" % (qid, aid, answer_sim))
        results[qid][aid] = float(answer_sim)


def report_ndcg_results(result_tsv_name: str, results: dict):
    with open(result_tsv_name, 'wt') as f:
        for topic, documents in results.items():
            top_documents = sorted(documents.items(), key=lambda x: x[1], reverse=True)[:1000]
            for rank, (document, similarity_score) in enumerate(top_documents):
                line = '{}\txxx\t{}\t{}\t{}\txxx'.format(topic, document, rank + 1, similarity_score)
                print(line, file=f)


report_ndcg_results("validation_ranking_logs_eval%s" % experiment_id, results)

