import pandas as pd
import os
import json
import numpy as np
from CorpusGenerator import CorpusGenerator
from sklearn.metrics.pairwise import cosine_similarity
from Text_Processing import tokenizer_stemming
from Text_Processing import clean_text_similarity
import gensim
from gensim.models.doc2vec import Doc2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def construct_d2v(corpus, size, min_count, epochs, progress_per, model_loc):
    print("Train Doc2Vec")
    tagged = [gensim.models.doc2vec.TaggedDocument(words=lst_t, tags=str(i)) for i, lst_t in enumerate(corpus)]
    d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=size, min_count=min_count, epochs=epochs)
    d2v_model.build_vocab(tagged, progress_per=progress_per)
    d2v_model.train(tagged, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    d2v_model.save(model_loc)


def model2features_d2v(model_input, corpus, vector_size, bigram, trigram):
    # load model
    # model.bin
    filename = model_input
    model = Doc2Vec.load(filename)
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
        if len(sents) != 0:
            doc_vector = model.infer_vector(input_sent)
            doc_vector = doc_vector.reshape(1, vector_size)
            if features is None:
                features = doc_vector
            else:
                features = np.vstack((features, doc_vector))
        else:
            index_to_be_deleted.append(i)
    return model, features, index_to_be_deleted


def output_d2v(pairs, filtered, trained_model, size, n_best, bigram, trigram, outputloc, foldername,output):
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
        # construct d2v
        model, TW_features, to_be_deleted_tw = model2features_d2v(trained_model,
                                                                  TW_dataframe["Text"].apply(
                                                                      clean_text_similarity).apply(
                                                                      lambda x: x.lower()), size, bigram, trigram)
        print("TW features shape:", TW_features.shape)
        _, EU_features, to_be_deleted_eu = model2features_d2v(trained_model,
                                                              EU_dataframe["Text"].apply(clean_text_similarity).apply(
                                                                  lambda x: x.lower()), size, bigram, trigram)
        print("EU features shape:", EU_features.shape)
        if len(to_be_deleted_tw) != 0:
            TW_dataframe = TW_dataframe.drop(index=to_be_deleted_tw)
        if len(to_be_deleted_eu) != 0:
            EU_dataframe = EU_dataframe.drop(index=to_be_deleted_eu)

        cosine_sim = pd.DataFrame(cosine_similarity(TW_features, EU_features),
                                  index=TW_dataframe["Index"].values, columns=EU_dataframe['Index'].values)
        # cosine_sim.to_excel("cosine_sim.xlsx",index=False)
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
                        para_dict["Text"] = EU_dataframe.loc[EU_dataframe.loc[:, "Index"] == para[0], "Text"].values[0]
                tw_dict["Similar Paragraphs"].append(para_dict)
            result_dict["Paragraphs"].append(tw_dict)

        if output:
            if os.path.exists(os.path.join(outputloc, foldername)) is False:
                os.mkdir(os.path.join(outputloc, foldername))
            with open(os.path.join(outputloc, foldername, '{}_{}.json'.format(pair[0], pair[1])),
                      'w') as file:
                json.dump(result_dict, file)
