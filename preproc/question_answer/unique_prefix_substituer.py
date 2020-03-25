import unicodedata

import re
from typing import Dict, Iterable, Tuple

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# we want the polish interpretation to match the natural language as well as possible,
# to utilize pre-trained words representations
from ARQMathCode.Entities.Post import Question


class UniquePrefixSubstituer:
    matching_template = '<span class="math-container" id="%s">'
    formulas_map = dict()
    alias_map = dict()
    unicode_iter_start = 800

    def __init__(self, preproc_formulas_tsv: str, exclude_vocab_path: str):
        with open(preproc_formulas_tsv, "r") as tsv_f:
            for line in tqdm(tsv_f.readlines()[1:], desc="Loading Polish notation map"):
                line_parts = line.split("\t")
                self.formulas_map[int(line_parts[0])] = line_parts[-1]

        self.exclude_vocab_list = [v for v in map(str.strip, open(exclude_vocab_path, "r").readlines()) if len(v) == 1]

    @staticmethod
    def _drop_xml_tags(body: str):
        return re.sub('<[^<]+>', "", body)

    @staticmethod
    def _drop_original_math(body: str, formula_sep='$'):
        body_parts = body.split(formula_sep)
        noformula_parts = [part for i, part in enumerate(body_parts) if i % 2 == 0]
        return ' '.join(noformula_parts)

    def _get_first_unused_utf8(self) -> str:
        for utf_i in range(self.unicode_iter_start, int(10e6)):
            symbol = chr(utf_i)
            if symbol not in self.exclude_vocab_list and symbol not in self.alias_map.values():
                self.unicode_iter_start  = utf_i
                return symbol
        raise ValueError("Run out of Unicode symbols :(")

    def _get_item(self, item: str):
        # add the new aliases to unknown items
        if item not in self.alias_map.keys():
            self.alias_map[item] = self._get_first_unused_utf8()
        return self.alias_map[item]

    def _postproc(self, formula: str):
        formula_out = formula
        return re.sub(r".!", "", formula_out)

    def _process_formula(self, formula: str):
        formula_out = formula
        formula_out = unicodedata.normalize('NFKC', formula_out)
        # find all X!Y items
        for match in re.finditer(r"([A-Z]![a-zA-Z0-9]+) ", formula_out):
            item = match.groups()[0]
            replaced_item = self._get_item(item)
            formula_out = formula_out.replace(item, replaced_item)

        return self._postproc(formula_out)

    def _process_body(self, body: str):
        body_out = body
        # body_out = unicodedata.normalize('NFKC', body_out)
        body_out = self._drop_original_math(body_out)
        body_out = self._drop_xml_tags(body_out)
        return body_out

    def subst_body(self, qa_body: str) -> str:
        body_out = qa_body
        for match in re.finditer(self.matching_template % r'(\d+)', qa_body):
            match_id = int(match.groups()[0])
            # Replace the initial formula tag (matching_template) with preprocessed formula
            formula_postproc = self._process_formula(self.formulas_map.get(match_id, ""))
            body_out = body_out.replace(self.matching_template % match_id, formula_postproc)
        return self._process_body(body_out)

    def process_questions(self, questions: Dict[int, Question]) -> Tuple[int, Iterable[Question]]:
        for q_i, q in tqdm(questions.items(), desc="Parsing Unique Polish notation"):
            q.body = self.subst_body(q.body)
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = self.subst_body(a.body)
            yield q_i, q

    def extend_sbert_vocab(self, model: SentenceTransformer):
        tokenizer = model._modules['0'].tokenizer
        bert = model._modules['0'].bert

        tokenizer.add_tokens(list(self.alias_map.keys()))
        bert.resize_token_embeddings()
