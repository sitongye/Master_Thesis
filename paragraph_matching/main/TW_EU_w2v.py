import pandas as pd
import os
import itertools
import json
import numpy as np
from CorpusGenerator import CorpusGenerator
from sklearn.metrics.pairwise import cosine_similarity
from Text_Processing import tokenizer_stemming
from Text_Processing import clean_text_similarity
from pyemd import emd
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import gensim


def construct_w2v(corpus, min_count, window, size, alpha,
                  min_alpha, negative, workers, progress_per,
                  epochs, model_loc):
    w2v_model = Word2Vec(min_count=min_count,
                         window=window,
                         size=size,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         negative=negative,
                         workers=workers)
    w2v_model.build_vocab(corpus, progress_per=progress_per)
    w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)
    print(list(w2v_model.wv.vocab))
    print("complete w2v training")
    w2v_model.save(model_loc)


def model2features(model_input, corpus, size, bigram, trigram):
    # load model
    # model.bin
    filename = model_input
    model = Word2Vec.load(filename)
    features = None
    index_to_be_deleted = []
    tokenized = [tokenizer_stemming(sent) for sent in corpus]
    if bigram:
        sents = [bigram[sent] for sent in tokenized]
    if trigram:
        sents = [trigram[sent] for sent in sents]
    else:
        sents = tokenized
    for i in range(len(sents)):
        input_sent = [word for word in sents[i] if word in model.wv.vocab]
        if len(input_sent) != 0:
            doc_vector = np.mean(model[input_sent], axis=0)
            doc_vector = doc_vector.reshape(1, size)
            if features is None:
                features = doc_vector
            else:
                features = np.vstack((features, doc_vector))
        else:
            index_to_be_deleted.append(i)
    return model, features, index_to_be_deleted


def output_w2v(pairs, filtered, trained_model, size, bigram, trigram, outputloc, foldername, metrics, n_best, output):
    """
    :param pairs: 
    :param filtered: 
    :param trained_model: 
    :param metrics: ["cosine", "wmd"]
    :return: 
    """
    print("metric", metrics, "if output: ", output)
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
        # construct w2v
        if metrics == "cosine":
            model, TW_features, to_be_deleted_tw = model2features(trained_model,
                                                                  TW_dataframe["Text"].apply(
                                                                      clean_text_similarity).apply(
                                                                      lambda x: x.lower()), size, bigram=bigram,
                                                                  trigram=trigram)
            print("TW features shape:", TW_features.shape)
            _, EU_features, to_be_deleted_eu = model2features(trained_model,
                                                              EU_dataframe["Text"].apply(clean_text_similarity).apply(
                                                                  lambda x: x.lower()), size, bigram=bigram,
                                                              trigram=trigram)
            print("EU features shape:", EU_features.shape)
            if len(to_be_deleted_tw) != 0:
                TW_dataframe = TW_dataframe.drop(index=to_be_deleted_tw)
            if len(to_be_deleted_eu) != 0:
                EU_dataframe = EU_dataframe.drop(index=to_be_deleted_eu)

            if output is True:
                cosine_sim = pd.DataFrame(cosine_similarity(TW_features, EU_features),
                                          index=TW_dataframe["Index"].values, columns=EU_dataframe['Index'].values)
                sim = cosine_sim.apply(lambda s: list(
                    zip(s.nlargest(len([k for k in s if k > 0])).index, s.nlargest(len([k for k in s if k > 0])))),
                                       axis=1).to_dict()  # dict
                result_dict["Paragraphs"] = []
                for key, value in sim.items():
                    # TW_dataframe.loc[TW_dataframe.loc[:,"Index"]==key, "EU_Top1_idx"] = value[0][0]
                    # if len(EU_dataframe.loc[EU_dataframe.loc[:,"Index"]==value[0][0],"Text"])!=0:
                    # TW_dataframe.loc[TW_dataframe.loc[:,"Index"]==key, "EU_Top1_txt"] = EU_dataframe.loc[EU_dataframe.loc[:,"Index"]==value[0][0],"Text"].values[0]
                    tw_dict = {}
                    tw_dict["Index"] = key
                    tw_dict["Text"] = TW_dataframe.loc[TW_dataframe.loc[:, "Index"] == key, "Text"].values[0]
                    tw_dict["Similar Paragraphs"] = []
                    for para in value:
                        para_dict = {}
                        rank = value.index(para) + 1
                        para_dict["Similarity Rank"] = rank
                        para_dict["Similarity Score"] = para[1]
                        para_dict["Index"] = para[0]
                        if len(EU_dataframe.loc[EU_dataframe.loc[:, "Index"] == para[0], "Text"]) != 0:
                            if rank < 4:
                                para_dict["Text"] = \
                                    EU_dataframe.loc[EU_dataframe.loc[:, "Index"] == para[0], "Text"].values[0]
                        tw_dict["Similar Paragraphs"].append(para_dict)
                    result_dict["Paragraphs"].append(tw_dict)
                if os.path.exists(os.path.join(outputloc, foldername)) is False:
                    os.mkdir(os.path.join(outputloc, foldername))
                with open(os.path.join(outputloc, foldername, '{}_{}.json'.format(pair[0], pair[1])),
                          'w') as file:
                    json.dump(result_dict, file)
                del model

        # w2v + wordmover_distance, reuse the word2vec model
        elif metrics == "wmd":
            TW_dataframe = TW_dataframe.reset_index(drop=True)
            EU_dataframe = EU_dataframe.reset_index(drop=True)
            print("TW ", len(TW_dataframe))
            print("EU ", len(EU_dataframe))
            EU_dataframe["cleaned"] = EU_dataframe["Text"].apply(clean_text_similarity).apply(
                lambda x: tokenizer_stemming(x))
            model = Word2Vec.load(trained_model)
            if trigram:
                cleaned = trigram[bigram[list(EU_dataframe["cleaned"])]]
            else:
                if bigram:
                    cleaned = bigram[list(EU_dataframe["cleaned"])]
                else:
                    cleaned = list(EU_dataframe["cleaned"])
            wmd_corpus = []
            for doc in cleaned:
                doc = [word for word in doc if word in model.wv.vocab]
                wmd_corpus.append(doc)
            instance = WmdSimilarity(wmd_corpus, model, num_best=len(EU_dataframe))
            result_dict_w2v = {}
            result_dict_w2v["UN_Regulation_Title"] = eu_filename.split(".csv")[0]
            result_dict_w2v["Taiwan_Regulation_Title"] = TW_TITLES
            result_dict_w2v["Paragraphs"] = []
            TW_dataframe["cleaned"] = TW_dataframe["Text"].apply(clean_text_similarity).apply(
                lambda x: tokenizer_stemming(x))
            if trigram:
                tw_cleaned = trigram[bigram[list(TW_dataframe["cleaned"])]]
            else:
                if bigram:
                    tw_cleaned = bigram[list(TW_dataframe["cleaned"])]
                else:
                    tw_cleaned = list(TW_dataframe["cleaned"])
            for index in range(len(TW_dataframe)):
                query = tw_cleaned[index]
                sims = instance[query]
                if len(sims) != 0:
                    # print('Query:')
                    # print("Index: ", TW_dataframe.loc[index,"Index"])
                    # print(TW_dataframe.loc[index,"Text"])
                    tw_dict = {}
                    tw_dict["Index"] = TW_dataframe.loc[index, "Index"]
                    tw_dict["Text"] = TW_dataframe.loc[index, "Text"]
                    tw_dict["Similar Paragraphs"] = []
                    for i in range(len(sims)):
                        # print("Index: ", EU_dataframe.loc[sims[i][0],"Index"])
                        # print(EU_dataframe.loc[sims[i][0],"Text"])
                        para_dict = {}
                        para_dict["Similarity Rank"] = i + 1
                        para_dict["Similarity Score"] = sims[i][1]
                        # print("Here: ", sims[i][0])
                        para_dict["Index"] = EU_dataframe.loc[sims[i][0], "Index"]
                        if i in range(3):
                            para_dict["Text"] = EU_dataframe.loc[sims[i][0], "Text"]
                        tw_dict["Similar Paragraphs"].append(para_dict)
                    result_dict_w2v["Paragraphs"].append(tw_dict)
            if output is True:
                if os.path.exists(os.path.join(outputloc, foldername)) is False:
                    os.mkdir(os.path.join(outputloc, foldername))
                with open(os.path.join(outputloc, foldername, '{}_{}.json'.format(pair[0], pair[1])),
                          'w') as file:
                    json.dump(result_dict_w2v, file)
