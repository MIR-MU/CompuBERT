import unicodedata

import re
from typing import Dict, Iterable, Tuple
from tqdm import tqdm

# we want the polish interpretation to match the natural language as well as possible,
# to utilize pre-trained words representations
from ARQMathCode.Entities.Post import Question

alias_map = {"eq": "equals",
             "sup": "powered",
             "log": "logarithm",
             "ln": "natural logarithm",
             "sqrt": "squared root",
             "SUB": "lower index",
             "N!": " ",
             "V!": "variable ",
             "T!": " ",
             "F!": "fraction ",
             "R!": "radical ",
             "W!": " ",
             "M!": " ",
             "O!": "function ",
             "U!": "function "}


class PolishSubstituer:
    matching_template = '<span class="math-container" id="%s">'
    formulas_map = dict()

    def __init__(self, preproc_formulas_tsv: str):
        with open(preproc_formulas_tsv, "r") as tsv_f:
            for line in tqdm(tsv_f.readlines()[1:], desc="Loading Polish notation map"):
                line_parts = line.split("\t")
                self.formulas_map[int(line_parts[0])] = line_parts[-1]

    @staticmethod
    def _drop_original_math(body: str, formula_sep='$'):
        body_parts = body.split(formula_sep)
        noformula_parts = [part for i, part in enumerate(body_parts) if i % 2 == 0]
        return ' '.join(noformula_parts)

    def _process_body(self, body: str):
        body_out = unicodedata.normalize('NFKC', body)
        body_out = body_out.replace("</span>", "")
        for alias_k in alias_map.keys():
            body_out = body_out.replace(alias_k, alias_map[alias_k])
        body_out = self._drop_original_math(body_out)
        return body_out

    def subst_body(self, qa_body: str) -> str:
        body_out = qa_body
        for match in re.finditer(self.matching_template % r'(\d+)', qa_body):
            match_id = int(match.groups()[0])
            # Throw out non-processed formulae
            body_out = body_out.replace(self.matching_template % match_id, self.formulas_map.get(match_id, ""))
        body_out = body_out.replace("</span>", "")
        return self._process_body(body_out)

    def process_questions(self, questions: Dict[int, Question]) -> Tuple[int, Iterable[Question]]:
        for q_i, q in tqdm(questions.items(), desc="Parsing Polish notation"):
            q.body = self.subst_body(q.body)
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = self.subst_body(a.body)
            yield q_i, q
