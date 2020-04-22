import unicodedata

import re
from typing import Dict, Iterable, Tuple
from tqdm import tqdm

# we want the polish interpretation to match the natural language as well as possible,
# to utilize pre-trained words representations
from ARQMathCode.Entities.Post import Question

alias_map = {"-": " ",
             "eq": "=",
             "geq": ">=",
             "leq": "<=",
             "gt": ">",
             "lt": "<",
             "SUP": "^",
             "log": "log",
             "ln": "log",
             "sqrt": "squared root",
             "plus": "+",
             "minus": "-",
             "divide": "/",
             "times": "*",
             "much": " ",
             "percent": "%",
             "limit": "lim",
             "SUB": "",
             "N!": " ",
             "V!": "var ",
             "T!": " ",
             "F!": " ",
             "R!": " ",
             "W!": " ",
             "M!": " ",
             "O!": " ",
             "U!": " "}


class InfixSubstituer:
    matching_template = '<span class="math-container" id="%s">'
    formulas_map = dict()
    # all_ops = dict()

    def __init__(self, preproc_formulas_tsv: str):
        with open(preproc_formulas_tsv, "r") as tsv_f:
            for line in tqdm(tsv_f.readlines()[1:], desc="Loading Infix notation map"):
                line_parts = line.split("\t")
                self.formulas_map[int(line_parts[0])] = line_parts[-1]

    @staticmethod
    def _drop_xml_tags(body: str):
        return re.sub('<[^<]+>', "", body)

    @staticmethod
    def _drop_original_math(body: str, formula_sep='$'):
        body_parts = body.split(formula_sep)
        noformula_parts = [part for i, part in enumerate(body_parts) if i % 2 == 0]
        return ' '.join(noformula_parts)

    def _process_formula(self, formula: str):
        formula_out = formula
        for alias_k in alias_map.keys():
            # for part in [p for p in formula_out.split() if 'O!' in p]:
            #     try:
            #         self.all_ops[part] += 1
            #     except KeyError:
            #         self.all_ops[part] = 1
            formula_out = formula_out.replace(alias_k, alias_map[alias_k])
        return formula_out

    def _process_body(self, body: str):
        body_out = body
        body_out = unicodedata.normalize('NFKC', body_out)
        body_out = self._drop_original_math(body_out)
        body_out = self._drop_xml_tags(body_out)
        return body_out

    def subst_body(self, qa_body: str) -> str:
        body_out = qa_body
        for match in re.finditer(self.matching_template % r'(\d+)', qa_body):
            match_id = int(match.groups()[0])
            # Replace the initial formula tag (matching_template) with preprocessed formula
            # formula_postproc = self._process_formula(self.formulas_map.get(match_id, ""))
            # try to do nothing, no replacements
            formula_postproc = self.formulas_map.get(match_id, "")
            body_out = body_out.replace(self.matching_template % match_id, formula_postproc)
        return self._process_body(body_out)

    def process_questions(self, questions: Dict[int, Question]) -> Tuple[int, Iterable[Question]]:
        for q_i, q in tqdm(questions.items(), desc="Parsing Polish notation"):
            q.body = self.subst_body(q.body)
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = self.subst_body(a.body)
            yield q_i, q
