from arqmath_eval import ndcg
from arqmath_eval.common import PARSED_RELEVANCE_JUDGEMENTS
import random

def random_score(out_intvl=(0, 1000)):
    return random.choice(range(*out_intvl))/1000

# answers retrieved in random order get random_score
random_judgements = {kq: dict(sorted(((ka, random_score()) for ka in kv.keys()), key=lambda _: random.random()))
                     for kq, kv in PARSED_RELEVANCE_JUDGEMENTS['train']['task1'].items()}
ndcg(random_judgements)
