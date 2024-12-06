import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import pickle
from gensim.models import KeyedVectors

app = Flask(__name__)

model = pickle.load(open('models/vector_model-ver1.pkl','rb'))
wv = KeyedVectors.load("vectors/wordvectors.kv", mmap='r')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    answers = [str(x) for x in request.form.values()]

    neuroticism = 0
    agreeable = 0
    conscientiousness = 0
    openness_to_experience = 0
    extraverted = 0

    df = pd.DataFrame({})
    df['responses'] = pd.Series(answers)
    df['tokens'] = df['responses'].apply(spacy_tokenizer)
    df['vec'] = df['tokens'].apply(sent_vec)
    X = np.array(df['vec'].tolist())
    predicted = model.predict(X)
    df['pred_persona'] = pd.Series(predicted)

    for persona in df['pred_persona']:
        if persona == "neuroticism":
            neuroticism += 1
        elif persona == "agreeable":
            agreeable += 1
        elif persona == "conscientiousness":
            conscientiousness += 1
        elif persona == "openness to experience":
            openness_to_experience += 1
        elif persona == "extraverted":
            extraverted += 1

        persona_array = {
        "neuroticism": neuroticism,
        "agreeable": agreeable,
        "conscientiousness": conscientiousness,
        "openness_to_experience": openness_to_experience,
        "extraverted": extraverted
    }
        
    return jsonify(persona_array)

def sent_vec(sent):


    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
 
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res/ctr
    return wv_res


def spacy_tokenizer(sentence):
    nlp = spacy.load('en_core_web_sm')

    doc = nlp(sentence)

    stop_words = STOP_WORDS
    punctuations = string.punctuation

    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens


if __name__ == '__main__':
    app.run(debug=True)