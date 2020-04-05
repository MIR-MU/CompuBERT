from pytrec_eval import RelevanceEvaluator
import random


# to precisely capture the one-digit decimal format from example
def random_score(out_intvl=(0, 40)):
    return random.choice(range(*out_intvl))/10


def random_IRSystem_eval(rel_judgements_file='task1_qrel.V0.1.tsv', trec_metric="ndcg"):
    judgements = dict()

    with open(rel_judgements_file, "r") as f:
        for ln in f.readlines():
            l_content = [item.strip() for item in ln.split("\t")]
            try:
                q = int(l_content[0])
                try:
                    judgements[str(q)][l_content[2]] = int(l_content[-1])
                except KeyError:
                    judgements[str(q)] = {l_content[2]: int(l_content[-1])}
            except KeyError:
                raise KeyError("Key %s not found in rel_questions_map" % l_content[0])
    evaluator = RelevanceEvaluator(judgements, measures={trec_metric})
    # answers retrieved in random order get random_score
    random_judgements = {kq: dict(sorted(((ka, random_score()) for ka in kv.keys()), key=lambda _: random.random()))
                         for kq, kv in judgements.items()}

    out = evaluator.evaluate(random_judgements)
    print(out)
    return out


random_IRSystem_eval()
