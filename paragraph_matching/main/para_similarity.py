# function: load yaml config file
# run selected construction
# evaluation
# print report

import pandas as pd
import os
import itertools
import json
import numpy as np
from CorpusGenerator import CorpusGenerator
from Text_Processing import tokenizer_stemming
from Text_Processing import clean_text_similarity
import yaml
from TW_EU_tf_idf import output_tf_idf
from gensim.models.phrases import Phrases, Phraser
import TW_EU_LSI
import TW_EU_w2v
import TW_EU_d2v
from gensim import corpora
from gensim import models
# load configuration file
with open(os.path.join(".", 'config', 'config.yml'), 'r', encoding='utf-8') as conf_file:
    config = yaml.load(conf_file, Loader=yaml.SafeLoader)
# generate whole corpus for general representation construction
all_corpus = CorpusGenerator(country_list=["TW", "EU"],
                             task="SIM", data_location="default",
                             generate_only_for=None, filtered=None)
print("start to prepare whole dataset")
corpus_dataframe = all_corpus.read_from_location()
print(corpus_dataframe)
tokenized_cleaned_corpus = list(
    corpus_dataframe["Text"].apply(clean_text_similarity).apply(lambda x: x.lower()).apply(tokenizer_stemming))
print(tokenized_cleaned_corpus)
# bigram, trigram for gensim model
bigram = Phrases(tokenized_cleaned_corpus,
                 min_count=config["Modelling"]["PHRASER"]["config"]["MIN_COUNT"],
                 threshold=100)
trigram = Phrases(bigram[tokenized_cleaned_corpus], threshold=100)
bigram = Phraser(bigram)
trigram = Phraser(trigram)

if config["Preprocessing"]["trigram"] is True:
    sentences = [trigram[bigram[sent]] for sent in tokenized_cleaned_corpus]
else:
    trigram = None
    if config["Preprocessing"]["bigram"] is True:
        sentences = [bigram[sent] for sent in tokenized_cleaned_corpus]
    else:
        bigram=None
        sentences = tokenized_cleaned_corpus


print(sentences)
# for phrase, score in trigram.export_phrases(bi_sentences):
# print(phrase, score)


DATA_FOLDER = config["FileLocation"]["DATA_FOLDER"]
TW_FOLDER = config["FileLocation"]["TW_FOLDER"]
EU_FOLDER = config["FileLocation"]["EU_FOLDER"]
OUTPUT_FOLDER = config["FileLocation"]["OUTPUT_FOLDER"]
# pairing file
pairing = pd.read_csv(os.path.join(DATA_FOLDER, config["FileLocation"]["PAIRING_FILE"]))
filtered = pairing[(pairing["Status"] == "Y") & (pairing["similarity_score"] > 0.23)]
filtered = filtered.reset_index(drop=True)
pairs = [(filtered.loc[i, "TW"].split("-")[0], filtered.loc[i, "EU"]) for i in range(len(filtered))]
pairs = list(set(pairs))

dictionary = corpora.Dictionary(sentences)
# tfidf
bow_corpus = [dictionary.doc2bow(text) for text in sentences]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# TFIDF


if "TFIDF" in config["Generate"]:
    print("Starting TFIDF Presentation")
    if config["Modelling"]["TFIDF"]["metrics"]["cosine"]["output"] is True:

        output_tf_idf(pairs, filtered, dictionary=dictionary, tfidf_model=tfidf,
                      bigram=bigram, trigram=trigram,
                      outputloc=OUTPUT_FOLDER,
                      foldername=config["Modelling"]["TFIDF"]["metrics"]["cosine"]["foldername"],
                      output=True)

# LSI
if "LSI" in config["Generate"]:
    print("starting LSI Presentation")
    if config["Modelling"]["LSI"]["metrics"]["cosine"]["output"] is True:
        lsi_model = TW_EU_LSI.build_lsi(corpus_tfidf, dictionary, config["Modelling"]["LSI"]["config"]["NUM_TOPIC"])
        TW_EU_LSI.output_sli(pairs, filtered, tfidf, lsi_model, dictionary, outputloc=OUTPUT_FOLDER,
                             foldername=config["Modelling"]["LSI"]["metrics"]["cosine"]["foldername"],
                             bigram=bigram, trigram=trigram,
                             output=True)

# word2vec
if "W2V" in config["Generate"]:
    print("starting Word2Vec Presentation")
    if config["Modelling"]["WORD2VEC"]["construct"] is True:
        print("start retraining word2vec model")
        TW_EU_w2v.construct_w2v(corpus=sentences,
                                min_count=config["Modelling"]["WORD2VEC"]["config"]["MIN_COUNT"],
                                window=config["Modelling"]["WORD2VEC"]["config"]["WINDOW"],
                                size=config["Modelling"]["WORD2VEC"]["config"]["SIZE"],
                                alpha=config["Modelling"]["WORD2VEC"]["config"]["ALPHA"],
                                min_alpha=config["Modelling"]["WORD2VEC"]["config"]["MIN_ALPHA"],
                                negative=config["Modelling"]["WORD2VEC"]["config"]["NEGATIVE"],
                                workers=config["Modelling"]["WORD2VEC"]["config"]["WORKERS"],
                                progress_per=config["Modelling"]["WORD2VEC"]["config"]["PROGRESS_PER"],
                                epochs=config["Modelling"]["WORD2VEC"]["config"]["EPOCHS"],
                                model_loc=os.path.join(config["Modelling"]["WORD2VEC"]["config"]["MODEL_DIR"],
                                                       config["Modelling"]["WORD2VEC"]["config"]["MODEL_NAME"]))

    for metric in config["Modelling"]["WORD2VEC"]["metrics"]:
        if config["Modelling"]["WORD2VEC"]["metrics"][metric]["output"] is True:
            TW_EU_w2v.output_w2v(pairs, filtered,
                                 trained_model=os.path.join(config["Modelling"]["WORD2VEC"]["config"]["MODEL_DIR"],
                                                            config["Modelling"]["WORD2VEC"]["config"]["MODEL_NAME"]),
                                 size=config["Modelling"]["WORD2VEC"]["config"]["SIZE"],
                                 bigram=bigram,
                                 trigram=trigram,
                                 outputloc=OUTPUT_FOLDER,
                                 foldername=config["Modelling"]["WORD2VEC"]["metrics"][metric]["foldername"],
                                 metrics=metric,
                                 n_best=config["Modelling"]["WORD2VEC"]["config"]["NUM_BEST"],
                                 output=True)

# Doc2Vec
if "D2V" in config["Generate"]:
    print("starting Doc2Vec Presentation")
    if config["Modelling"]["DOC2VEC"]["construct"] is True:
        print("start retraining word2vec model")
        TW_EU_d2v.construct_d2v(corpus=sentences,
                                size=config["Modelling"]["DOC2VEC"]["config"]["SIZE"],
                                min_count=config["Modelling"]["DOC2VEC"]["config"]["MIN_COUNT"],
                                epochs=config["Modelling"]["DOC2VEC"]["config"]["EPOCHS"],
                                progress_per=config["Modelling"]["DOC2VEC"]["config"]["PROGRESS_PER"],
                                model_loc=os.path.join(config["Modelling"]["DOC2VEC"]["config"]["MODEL_DIR"],
                                                       config["Modelling"]["DOC2VEC"]["config"]["MODEL_NAME"]))

    if config["Modelling"]["DOC2VEC"]["metrics"]["cosine"]["output"] is True:
        TW_EU_d2v.output_d2v(pairs, filtered,
                             trained_model=os.path.join(config["Modelling"]["DOC2VEC"]["config"]["MODEL_DIR"],
                                                       config["Modelling"]["DOC2VEC"]["config"]["MODEL_NAME"]),
                             size=config["Modelling"]["DOC2VEC"]["config"]["SIZE"],
                             n_best=config["Modelling"]["DOC2VEC"]["config"]["NUM_BEST"],
                             bigram=bigram,
                             trigram=trigram,
                             outputloc=OUTPUT_FOLDER,
                             foldername=config["Modelling"]["DOC2VEC"]["metrics"]["cosine"]["foldername"],
                             output=True)

