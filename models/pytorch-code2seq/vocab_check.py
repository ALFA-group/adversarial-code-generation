import pickle

vocab_path = "../../final-models/code2seq/sri/py150/normal/model/vocabulary.pkl"

vocab = pickle.load(open(vocab_path, "rb"))

token_to_id = vocab["token_to_id"]

print(token_to_id["@R_1@"])
