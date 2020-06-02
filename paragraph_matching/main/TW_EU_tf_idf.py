import pandas as pd
import os
import itertools
import json
import numpy as np
from CorpusGenerator import CorpusGenerator
from Text_Processing import tokenizer_stemming
from Text_Processing import clean_text_similarity
from gensim.similarities import MatrixSimilarity
# train_tf_idf on whole corpus


def transform_text(raw_text_list, tfidf_model, dictionary, bigram=None, trigram=None, mode="tfidf"):
    """

    :param raw_text_list:
    :param mode: "tfidf", "bow"
    :param dictionary:
    :return:
    """
    tokenized = [tokenizer_stemming(clean_text_similarity(text)) for text in raw_text_list]
    if bigram:
        sents = [bigram[sent] for sent in tokenized]
    if trigram:
        sents = [trigram[sent] for sent in sents]
    if (bigram is None) and (trigram is None):
        sents = tokenized
    bow_corpus = [dictionary.doc2bow(text) for text in sents]
    corpus_tfidf = tfidf_model[bow_corpus]
    if mode == "tfidf":
        return corpus_tfidf
    elif mode == "bow":
        return bow_corpus


def output2dict(sims, TW_query_index, TW_dataframe, EU_dataframe):
    # for one query paragraph in TW
    # after setting num_best to the whole length of EU,
    # output all parapgrah that is none_zero
    tw_dict = {}
    tw_dict["Index"] = TW_dataframe["Index"][TW_query_index]
    tw_dict["Text"] = TW_dataframe["Text"][TW_query_index]
    tw_dict["Similar Paragraphs"] = []

    for i in range(len(sims)):
        para_index = sims[i][0]
        para_dict = {}
        para_dict["Similarity Rank"] = i + 1
        para_dict["Similarity Score"] = sims[i][1]
        para_dict["Index"] = EU_dataframe["Index"][para_index]
        # para_dict["Text"] = EU_dataframe["Text"][para_index]
        tw_dict["Similar Paragraphs"].append(para_dict)
    return tw_dict

# construct dataframe, append dataframe when more than one exists
# comparison starts from "Market side"


def output_tf_idf(pairs, filtered, dictionary, tfidf_model, bigram, trigram, outputloc, foldername, output=False):
    for pair in pairs:
        print("prepare for: ", pair)
        result_dict = {}
        # select data
        TW_LIST = list(filter(lambda x: x.split("-")[0] == pair[0], filtered["TW"]))
        EU_LIST = list(pair[1].split("_")[-1])
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
        EU_tf_idf = transform_text(EU_dataframe["Text"],
                                   bigram=bigram, trigram=trigram,
                                   tfidf_model=tfidf_model,
                                   dictionary=dictionary)
        TW_tf_idf = transform_text(TW_dataframe["Text"],
                                   tfidf_model=tfidf_model,
                                   dictionary=dictionary,
                                   bigram=bigram, trigram=trigram)
        print("length of TW dataframe {}:".format(pair[0]), len(TW_dataframe))
        print("length of EU dataframe {}:".format(pair[1]), len(EU_dataframe))
        # construct tf-idf
        index = MatrixSimilarity(EU_tf_idf, num_features=len(dictionary), num_best=len(EU_tf_idf))
        result_dict["Paragraphs"] = []
        for query_index in range(len(TW_tf_idf)):
            # TW_dataframe.loc[TW_dataframe.loc[:,"Index"]==key, "EU_Top1_idx"] = value[0][0]
            # if len(EU_dataframe.loc[EU_dataframe.loc[:,"Index"]==value[0][0],"Text"])!=0:
            #    TW_dataframe.loc[TW_dataframe.loc[:,"Index"]==key, "EU_Top1_txt"] = EU_dataframe.loc[EU_dataframe.loc[:,"Index"]==value[0][0],"Text"].values[0]
            sims = index[TW_tf_idf[query_index]]
            tw_dict = output2dict(sims, query_index, TW_dataframe, EU_dataframe)
            result_dict["Paragraphs"].append(tw_dict)
            # output result_dict to json
        if output:
            if os.path.exists(os.path.join(outputloc, foldername)) is False:
                os.mkdir(os.path.join(outputloc, foldername))
            with open(os.path.join(outputloc, foldername, '{}_{}.json'.format(pair[0], pair[1])),
                      'w') as file:
                json.dump(result_dict, file)
