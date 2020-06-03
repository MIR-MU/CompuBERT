from question_answer.sbert_ir_system import SBertIRSystem, Preprocessing

data_root = "/run/media/stefanik/Evin exterak/ideapad_bkp"

irsystem = SBertIRSystem(data_root+'/data/arqmath/models/train_sampled_eval9',
                         questions_path=data_root+'/data/arqmath/ARQMath_CLEF2020/Collection',
                         preprocessing=Preprocessing.LATEX,
                         identifier="CompuBERT_model9",
                         use_cuda=True)

# irsystem.index_questions(irsystem.parser.map_questions.keys())
eval_ids = irsystem.index_eval_questions(infer_batch_size=2)
irsystem.index_answers(irsystem.parser.map_just_answers.keys(), infer_batch_size=2)
irsystem.save_index(data_root+"/data/projects/arqmath/compubert/tmp")

irsystem.dump_arqmath_response_ratings(data_root+'/projects/arqmath/compubert/tmp/submit_dir')
irsystem.dump_response_bodies(data_root+"/projects/arqmath/compubert/tmp/model_9_serps.json",
                              eval_ids[:3])

# irsystem.load_index("/home/michal/Documents/projects/arqmath/compubert/tmp")
print("done")