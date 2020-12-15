import flask
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import codecs
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model 

# Loading the model
model = load_model("model/network.h5") 
# loading
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():

    SEQUENCE_LENGTH = 29 

    def preprocess_text(sen):
        sentence = remove_tags(sen)
        sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
        sentence = re.sub(r'\s+', ' ', str(sentence))

        return sentence
    
    TAG_RE = re.compile(r'<[^>]+>')
    def remove_tags(text):
        return TAG_RE.sub('', str(text))

    def get_predictions(text):
        text = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([text])
        # pad the sequence
        sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
        # get the prediction
        prediction = model.predict(sequence)[0]
        pred = list(prediction)
        # one-hot encoded vector, revert using np.argmax
        return pred.index(np.max(list(prediction)))


    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        email_text = flask.request.form['email_text']

        prediction = get_predictions(email_text)
        seriousness = ""
        if(prediction == 0):
            seriousness = "Our Model can't decide if this is relevant or not " 
        
        elif(prediction == 1):
            seriousness = "This tweet is not relevant to a disaster"

        else:
            seriousness = "This tweet is relevant to a disaster, you should consider investigating"

        return flask.render_template('main.html',
        original_input={
            "email_text":email_text,
                                                        
        },
                                        result=seriousness
                                        )

if __name__ == '__main__':
    app.run()
