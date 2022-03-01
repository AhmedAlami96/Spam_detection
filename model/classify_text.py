from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from model.model import LSTMModel, MNBModel, ETCModel
import pickle

with open('./model/dict_params.pickle', 'rb') as f:
    dict_params = pickle.load(f)

le = dict_params['label_encoder']
tok = dict_params['tokenizer']
max_len = dict_params['max_len']
m = LSTMModel()

with open('./model/mnb_params.pickle', 'rb') as f:
    dict_params_mnb = pickle.load(f)
cv = dict_params_mnb['counteVectorizer']
mnb = MNBModel()

with open('./model/etc_params.pickle', 'rb') as f:
    dict_params_etc = pickle.load(f)

tf_idf = dict_params_etc['tfidfVectorizer']
etc = ETCModel()

def classify_text_LSTM(raw_text):
    model = m.get_model()
    global tok
    global max_len

    sequences = tok.texts_to_sequences([raw_text])
    sequences_padded = sequence.pad_sequences(sequences,maxlen=max_len)
    proba_spam = model.predict(sequences_padded[0:1])[0][0]

    return proba_spam

def classify_text_MNB(raw_text):
    classifier = mnb.get_model()
    sequence = cv.transform([raw_text]).toarray()
    proba_spam = classifier.predict_proba(sequence)[0][1]
    return proba_spam

def classify_text_ETC(raw_text):
    classifier = etc.get_model()
    sequence = tf_idf.transform([raw_text]).toarray()
    proba_spam = classifier.predict_proba(sequence)[0][1]
    return proba_spam
