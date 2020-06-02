import re
import string
import nltk
from nltk.corpus import stopwords
import yaml
import os

nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
from nltk import word_tokenize

# load configuration file
with open(os.path.join(".", 'config', 'config.yml'), 'r', encoding='utf-8') as conf_file:
    config = yaml.load(conf_file, Loader=yaml.SafeLoader)

def clean_text_similarity(text, del_num=False):
    # print("Text:",text)
    # clear string starting from ECE
    # clear text starting from www.
    # clear punctuation
    # clear numbers
    # clear textnumber mix
    # clear page num
    parathesis = r"\(.+\)"
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    text = re.sub("Annex .+", "", text)
    text = re.sub(parathesis, " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"ECE[\S]+\s*(?:\d+)*", " ", text)
    text = re.sub(r"Rev[\S]+\s*(?:\d+)*", " ", text)
    text = re.sub(r"www[\S]+", " ", text)
    text = re.sub(r"}\s.*", " ", text)
    # text = re.sub(r"\/\d\s.*", " ", text)
    text = re.sub(r"page\s\d+", "", text)
    text = re.sub(r"_{2,}", " ", text)
    text = re.sub(r"\d/ .*", " ", text)
    text = re.sub(r"(\d+)(mm|cm)", "\1 \2", text)
    text = re.sub(r"\d\sAs defined in.+html", " ", text)
    if del_num is True:
        text = re.sub(r'\w*\d\w*', '', text).strip()
    text = re.sub(r"…", " ", text)
    text = re.sub(r"-", "", text)
    # text = re.sub(r"(?: a | an | the )"," ",text)
    text = re.sub(r"\s\d+\s", " ", text)
    text = re.sub(r"\s+", ' ', text)
    return text.strip()


def tokenizer_stemming(text, stop=True):
    wn = config["Preprocessing"]["wordnet"]
    porter = config["Preprocessing"]["porter"]
    tokenized_text = word_tokenize(text.lower())
    if stop:
        stop = set(stopwords.words('english'))
        stop.add("shall")
        stop.add("aa")
        stop.add("\x01")
        stop.add("paragraph")
        stop.add("km")
        stop.add("mm")
        stop.add("h")
        tokenized_text = [i.lower() for i in tokenized_text if i.lower() not in stop]
    if wn:
        wordnet = WordNetLemmatizer()
        tokenized_text = [wordnet.lemmatize(word).lower() for word in tokenized_text]
    if porter:
        p = PorterStemmer()
        tokenized_text = [p.stem(word).lower() for word in tokenized_text]

    return tokenized_text
