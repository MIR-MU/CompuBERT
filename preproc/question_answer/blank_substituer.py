import unicodedata
from lxml import etree

import re
from typing import Dict, Iterable, Tuple
from tqdm import tqdm

from ARQMathCode.Entities.Post import Question, Answer
from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord


class BlankSubstituer:
    matching_template = '<span class="math-container" id="%s">'
    formulas_map = dict()

    @staticmethod
    def _drop_xml_tags(body: str):
        if not body:
            return ''
        html5_parser = etree.HTMLParser(huge_tree=True)
        html5_document = etree.XML(body, html5_parser)
        return ' '.join(html5_document.itertext())

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
        for q_i, q in tqdm(questions.items(), desc="Parsing questions: Original math notation"):
            q.body = self.subst_body(q.body)
            if q.answers is None:
                continue
            # for a in q.answers:
            #     a.body = self.subst_body(a.body)
            # yield q_i, q

    def process_answers(self, answers: Dict[int, Answer]) -> Tuple[int, Iterable[Question]]:
        for a_i, a in tqdm(answers.items(), desc="Parsing answers: Original math notation"):
            a.body = self.subst_body(a.body)
            # yield a_i, a

    def process_parser(self, parser: PostParserRecord) -> PostParserRecord:
        """Process questions and answers bodies in the parser"""
        self.process_questions(parser.map_questions)
        self.process_answers(parser.map_questions)
        for q_i, q in tqdm(parser.map_questions.items(), desc="Replacing questions answers: Original math notation"):
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = parser.map_just_answers[a.post_id].body
        return parser
