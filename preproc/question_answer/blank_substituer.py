import unicodedata

import re
from typing import Dict, Iterable, Tuple
from tqdm import tqdm

from ARQMathCode.Entities.Post import Question


class BlankSubstituer:
    matching_template = '<span class="math-container" id="%s">'
    formulas_map = dict()

    @staticmethod
    def _drop_xml_tags(body: str):
        return re.sub('<[^<]+>', "", body)

    def _process_formula(self, formula: str):
        return "$"+self._drop_xml_tags(formula)+"$"

    def replace_math(self, body_in: str, math_id: int, new_math_content: str):
        math_match = re.findall(self.matching_template % math_id + r".+?</span>", body_in)[0]
        body_out = body_in.replace(math_match, new_math_content)
        return body_out

    def subst_body(self, qa_body: str) -> str:
        body_out = qa_body
        for match in re.finditer(self.matching_template % r'(\d+)', qa_body):
            match_id = int(match.groups()[0])
            # Replace the initial formula tag (matching_template) with preprocessed formula
            formula = self._process_formula(self.formulas_map.get(match_id, ""))
            body_out = self.replace_math(body_out, match_id, formula)
        return self._drop_xml_tags(body_out)

    def process_questions(self, questions: Dict[int, Question]) -> Tuple[int, Iterable[Question]]:
        for q_i, q in tqdm(questions.items(), desc="Parsing Original math notation"):
            q.body = self.subst_body(q.body)
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = self.subst_body(a.body)
            yield q_i, q
