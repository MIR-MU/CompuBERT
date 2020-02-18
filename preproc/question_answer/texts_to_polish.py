from ARQMathCode.post_reader_record import DataReaderRecord
from functools import reduce


def tex_to_polish(text_formula):
    # TODO
    return text_formula


def fulltext_to_polish(entity_body_text, formula_sep="$"):
    body_parted = entity_body_text.split(formula_sep)
    # formulas are always separated by a PAIR of "$"s, thus they reside on every second position (if correctly formed)
    formulas = [part for i, part in enumerate(body_parted) if i % 2 == 1]
    if len(formulas) == 0:
        return entity_body_text
    parts = [part for i, part in enumerate(body_parted) if i % 2 == 0]
    formulas_polish = list(map(tex_to_polish, formulas)) + [""]
    body_replaced_l = list(reduce(lambda x, part_tups: x + list(part_tups), zip(parts, formulas_polish), []))
    return formula_sep.join(body_replaced_l)


clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=10000)

for q_i, q in dr.post_parser.map_questions.items():
    # question text replacement
    q.body = fulltext_to_polish(q.body)
    for a in q.answers:
        a.body = fulltext_to_polish(a.body)
