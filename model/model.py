from keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from keras.models import Model
from sklearn.externals import joblib
import os

class LSTMModel:
    def __init__(self):
        self.model = None

    def get_model(self):
        if self.model:
            return self.model
        max_words = 1000
        max_len = 150
        inputs = Input(name='inputs',shape=[max_len])
        layer = Embedding(max_words,50,input_length=max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1,name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        self.model = Model(inputs=inputs,outputs=layer)
        self.model.load_weights(os.getcwd()+"/model/trained_models/lstm_model.h5")
        return self.model

class MNBModel:
    def __init__(self):
        self.model = None
    
    def get_model(self):
        if self.model:
            return self.model
        else:
            self.model = joblib.load(os.getcwd()+"/model/trained_models/mnv.joblib")
            return self.model

class ETCModel:
    def __init__(self):
        self.model = None
    
    def get_model(self):
        if self.model:
            return self.model
        else:
            self.model = joblib.load(os.getcwd()+"/model/trained_models/etc.joblib")
            return self.model
print('kolchi mziane')