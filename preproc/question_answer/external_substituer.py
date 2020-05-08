from typing import Dict, Iterable, Tuple
from tqdm import tqdm

import gzip
import json
from multiprocessing import Pool
import re


from ARQMathCode.Entities.Post import Question

POOL_NUM_WORKERS = None
POOL_CHUNKSIZE = 1
JSON_LINE_REGEX = re.compile(r'"(?P<document_name>[^"]*)": (?P<json_document>.*),')


class ExternalSubstituer:
    matching_template = '<span class="math-container" id="%s">'

    @staticmethod
    def read_json_file(filename, total_number_of_documents=2477487):
        number_of_documents = 0
        with gzip.open(filename, 'rt') as f:
            with Pool(POOL_NUM_WORKERS) as pool:
                for result in pool.imap(
                        ExternalSubstituer.read_json_file_worker, tqdm(f,
                            desc='Reading documents from {}'.format(filename),
                            total=total_number_of_documents,),
                        POOL_CHUNKSIZE,
                ):
                    if result is not None:
                        number_of_documents += 1
                        assert number_of_documents <= total_number_of_documents, \
                            'Expected {} documents, but just read document number {}'.format(
                                total_number_of_documents,
                                number_of_documents,
                            )
                        document_name, document = result
                        yield (document_name, document)
        assert number_of_documents == total_number_of_documents, \
            'Expected {} documents, but read only {}'.format(
                total_number_of_documents,
                number_of_documents,
            )

    @staticmethod
    def read_json_file_worker(line):
        line = line.strip()
        if line in ('{', '}'):
            return None
        match = re.fullmatch(JSON_LINE_REGEX, line)
        document_name = match.group('document_name')
        document = json.loads(match.group('json_document'))
        return (document_name, document)

    def __init__(self, gz_file_path: str):
        self.formulas_map = dict()
        for str_id, content_list in self.read_json_file(gz_file_path):
            self.formulas_map[int(str_id)] = ' '.join([token[5:] for token in content_list])

    def subst_body(self, q_i: int) -> str:
        return self.formulas_map[q_i]

    def process_questions(self, questions: Dict[int, Question]) -> Tuple[int, Iterable[Question]]:
        for q_i, q in tqdm(questions.items(), desc="Replacing with external bodies"):
            q.body = self.subst_body(q_i)
            if q.answers is None:
                continue
            for a in q.answers:
                a.body = self.subst_body(a.post_id)
            yield q_i, q
