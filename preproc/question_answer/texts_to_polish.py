from ARQMathCode.post_reader_record import DataReaderRecord
from functools import reduce

from TangentS.math_tan.math_extractor import MathExtractor

alias_map = {"eq": "equals",
             "sup": "powered",
             "log": "logarithm",
             "ln": "natural logarithm",
             "sqrt": "squared root"}


def alias(val, is_var=False):
    if is_var:
        return "variable %s" % val
    return alias_map.get(val.lower(), val)


def latex_to_polish_list(expression):
    seen_ops = set()
    for transition in MathExtractor.parse_from_tex_opt(expression).get_pairs(window=1, eob=True):
        trans_tuple = transition.split("\t")
        # is var
        if trans_tuple[0].startswith("V"):
            yield alias(trans_tuple[0].split("!")[1], is_var=True)
            continue
        # trans_tuple[0]+trans_tuple[-1] is presumed to be op's unique ID in OPTree
        if trans_tuple[0] + trans_tuple[-1] not in seen_ops:
            yield alias(trans_tuple[0].split("!")[1])
            seen_ops.add(trans_tuple[0] + trans_tuple[-1])
        else:
            seen_ops.remove(trans_tuple[0] + trans_tuple[-1])


def latex_to_polish(expression):
    return ' '.join(latex_to_polish_list(expression))


def fulltext_to_polish(entity_body_text, formula_sep="$"):
    body_parted = entity_body_text.split(formula_sep)
    # formulas are always separated by a PAIR of "$"s, thus they reside on every second position (if correctly formed)
    formulas = [part for i, part in enumerate(body_parted) if i % 2 == 1]
    if len(formulas) == 0:
        return entity_body_text
    parts = [part for i, part in enumerate(body_parted) if i % 2 == 0]
    formulas_polish = list(map(latex_to_polish, formulas)) + [""]
    body_replaced_l = list(reduce(lambda x, part_tups: x + list(part_tups), zip(parts, formulas_polish), []))
    return formula_sep.join(body_replaced_l)


clef_home_directory_file_path = '/data/arqmath/ARQMath_CLEF2020/Collection'
dr = DataReaderRecord(clef_home_directory_file_path, limit_posts=10000)

for q_i, q in dr.post_parser.map_questions.items():
    # question text replacement
    q.body = fulltext_to_polish(q.body)
    for a in q.answers:
        a.body = fulltext_to_polish(a.body)
