from question_answer.sbert_ir_system import SBertIRSystem, Preprocessing

irsystem = SBertIRSystem('/data/arqmath/models/train_sampled_eval9',
                         questions_path='/data/arqmath/ARQMath_CLEF2020/Collection',
                         preprocessing=Preprocessing.LATEX,
                         identifier='model_9',
                         use_cuda=False)

irsystem.index_questions(irsystem.parser.map_questions.keys())
irsystem.index_answers(irsystem.parser.map_just_answers.keys())
irsystem.save_index("/home/michal/Documents/projects/arqmath/compubert/tmp")

irsystem.dump_arqmath_response_ratings()
# irsystem.dump_response_bodies("/home/michal/Documents/projects/arqmath/compubert/tmp/model_9_serps.json",
#                               list(irsystem.parser.map_questions.keys())[:10])

# irsystem.load_index("/home/michal/Documents/projects/arqmath/compubert/tmp")
