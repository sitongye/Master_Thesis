import pandas as pd
import os
import itertools
import json
import numpy as np
from CorpusGenerator import CorpusGenerator
from Text_Processing import tokenizer_stemming
from Text_Processing import clean_text_similarity
from gensim import models
from collections import defaultdict
from gensim import corpora
from gensim import similarities


def build_lsi(corpus_tfidf, dictionary, num_topic):
    # dictionary for vocab
    # dictionary = corpora.Dictionary(corpus)
    # bow_corpus = [dictionary.doc2bow(text) for text in corpus]

    # tf-idf transformation
    # tfidf = models.TfidfModel(bow_corpus)  # initialize tfidf model
    #corpus_tfidf = tfidf[bow_corpus]
    ##lsi
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary,
                                num_topics=num_topic)  # initialize an LSI transformation
    return lsi_model


def transform_instance_lsi(dictionary, raw_text_list, tf_idf_model, lsi_model, bigram=None, trigram=None):
    tokenized = [tokenizer_stemming(clean_text_similarity(text).lower()) for text in raw_text_list]
    if bigram:
        sents = [bigram[sent] for sent in tokenized]
    if trigram:
        sents = [trigram[sent] for sent in sents]
    if (bigram is None) and (trigram is None):
        sents = tokenized
    bow_corpus = [dictionary.doc2bow(text) for text in sents]
    corpus_tfidf = tf_idf_model[bow_corpus]
    corpus_lsi = lsi_model[corpus_tfidf]
    return corpus_lsi


def output_sli(pairs, filtered, tfidf, lsi_model, dictionary, outputloc, foldername, num_best="all", bigram=None,trigram=None,output=False):
    for pair in pairs:
        print("prepare for: ", pair)
        result_dict = {}
        # select data
        TW_LIST = list(filter(lambda x: x.split("-")[0] == pair[0], filtered["TW"]))
        # EU_LIST = [pair[1].split("_")[-1]]
        TW_TITLES = dict()
        for doc in TW_LIST:
            title = list(filtered.loc[filtered.TW == doc]["TW_TITLE"])[0]
            TW_TITLES[doc] = "{} {}".format(doc.split("_")[-1], title)
        result_dict["Taiwan_Regulation_Title"] = TW_TITLES
        print("prepare for TW:", pair[0])
        # generate cleaned dataframe for both EU and TW
        TW_data = CorpusGenerator(country_list=["TW"],
                                  task="SIM", data_location="default",
                                  generate_only_for={"TW": [x.split("_")[-1] for x in TW_LIST]}, filtered=None)
        TW_dataframe = TW_data.read_from_location()
        eu_filename = "UN_Regulation_No._{}.csv".format(pair[1].split("_")[-1])
        result_dict["UN_Regulation_Title"] = eu_filename.split(".csv")[0]
        print("prepare for EU", pair[1])
        EU_data = CorpusGenerator(country_list=["EU"],
                                  task="SIM", data_location="default",
                                  generate_only_for={"EU": [pair[1].split("_")[-1]]}, filtered=None)
        EU_dataframe = EU_data.read_from_location()

        print("length of TW dataframe {}:".format(pair[0]), len(TW_dataframe))
        print("length of EU dataframe {}:".format(pair[1]), len(EU_dataframe))
        if num_best == "all":
            instance = similarities.MatrixSimilarity(
                transform_instance_lsi(dictionary, list(EU_dataframe["Text"]), tfidf, lsi_model, bigram=bigram,
                                       trigram=trigram), num_best=len(EU_dataframe))
        result_dict_lsi = {}
        result_dict_lsi["UN_Regulation_Title"] = eu_filename.split(".csv")[0]
        result_dict_lsi["Taiwan_Regulation_Title"] = TW_TITLES
        result_dict_lsi["Paragraphs"] = []
        for index in range(len(TW_dataframe)):
            query = transform_instance_lsi(dictionary, [TW_dataframe.loc[index, "Text"]], tfidf, lsi_model,
                                           bigram=bigram, trigram=trigram)
            sims = instance[query][0]
            if len(sims) != 0:
                # print('Query:')
                # print("Index: ", TW_dataframe.loc[index,"Index"])
                # print(TW_dataframe.loc[index,"Text"])
                tw_dict = {}
                tw_dict["Index"] = TW_dataframe.loc[index, "Index"]
                tw_dict["Text"] = TW_dataframe.loc[index, "Text"]
                tw_dict["Similar Paragraphs"] = []
                for i in range(len(sims)):
                    # print("Index: ", EU_dataframe.loc[sims[i][0], "Index"])
                    #print(EU_dataframe.loc[sims[i][0], "Text"])
                    para_dict = {}
                    para_dict["Similarity Rank"] = i+1
                    para_dict["Similarity Score"] = sims[i][1]
                    # print("Here: ", sims[i][0])
                    para_dict["Index"] = EU_dataframe.loc[sims[i][0], "Index"]
                    if i in range(3):
                        para_dict["Text"] = EU_dataframe.loc[sims[i][0], "Text"]
                    tw_dict["Similar Paragraphs"].append(para_dict)
                result_dict_lsi["Paragraphs"].append(tw_dict)

        if output is True:
            if os.path.exists(os.path.join(outputloc, foldername)) is False:
                os.mkdir(os.path.join(outputloc, foldername))
            with open(os.path.join(outputloc, foldername, '{}_{}.json'.format(pair[0], pair[1])),
                      'w') as file:
                json.dump(result_dict_lsi, file)
