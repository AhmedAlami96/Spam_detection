from flask import Flask, render_template, request
from model.classify_text import classify_text_MNB, classify_text_LSTM, classify_text_ETC
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        sms = request.form['sms']
        classifier = request.form['classifier']
        if classifier == 'LSTM':
            probas_spam = None
            proba_spam = classify_text_LSTM(sms)
        elif classifier == 'MNB':
            probas_spam = None
            proba_spam = classify_text_MNB(sms)
        elif classifier == 'ETC':
            probas_spam = None
            proba_spam = classify_text_ETC(sms)
            if proba_spam == 0:
                proba_spam = 0.0000001
            else:
                proba_spam = 0.9999999
        else:
            proba_spam = None
            probas_spam = {}
            probas_spam['LSTM'] = classify_text_LSTM(sms)
            probas_spam['MNB'] = classify_text_MNB(sms)
            probas_spam['ETC'] = classify_text_ETC(sms)
            classifier = 'ALL'

        # return render_template('index.html', proba_spam=proba_spam[0][0])
        return render_template('index.html', proba_spam=proba_spam, probas_spam=probas_spam, sms=sms, classifier=classifier)
    return render_template('index.html', classifier='ALL')

if __name__ == '__main__':
    app.run()