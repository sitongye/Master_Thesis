#!/usr/bin/env python
# coding: utf-8


# 1. import libraries
import pandas as pd
import os
import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from spacy.displacy import render, serve
import networkx as nx
import itertools
import json
import pylab
from spacy.util import filter_spans
import re
import tensorflow as tf
from transformers import *
import torch
from torch.nn.utils.rnn import pad_sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import spacy
from keras.models import Model
from keras.layers import Input, LSTM, Lambda, Dense, Concatenate
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
nltk.download('wordnet')
from BertMerger import BadTERMMerger


import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# functions for baseline models
def bow_col2feat(train_data, test_data, column, paradict, bow=True):
    features_train, features_test, vectorizer = bag_of_word(train_data[column].apply(lambda x: " ".join(x)),
                                                            test_data[column].apply(lambda x: " ".join(x)),
                                                            paradict, bow=bow)
    return features_train, features_test, vectorizer

# stack features, for example "pos" tag, "dep" tag together as import
def bow_train_feature_stack(train_data, test_data, paradict, column_list, bow=True):
    features_stack_train = None
    features_stack_test = None
    for column in column_list:
        features_train, features_test, _ = bow_col2feat(train_data, test_data, column, paradict, bow=bow)
        features_stack_train = hstack((features_stack_train, features_train))
        features_stack_test = hstack((features_stack_test, features_test))
    return features_stack_train, features_stack_test

# vectorizer is selectable between bagofwords or tfidf vectorizer
def bag_of_word(train, test, paradict, bow=True):
    vectorizer = CountVectorizer(ngram_range=paradict["NGRAM_RANGE"],
                                 max_features=paradict["MAX_FEATURES"],
                                 binary=paradict["BINARY"],
                                 max_df=paradict["MAX_DF"])
    if bow is False:
        vectorizer = TfidfVectorizer(ngram_range=paradict["NGRAM_RANGE"],
                                     max_features=paradict["MAX_FEATURES"],
                                     binary=paradict["BINARY"],
                                     max_df=paradict["MAX_DF"])
    train_data_features = vectorizer.fit_transform(train)
    test_data_features = vectorizer.transform(test)
    return train_data_features, test_data_features, vectorizer

# load trained NER model for "QuantitativeValue"
# define nlp pipeline
nlp = spacy.load(os.path.join(".", "ner", "parameter"))
nlp.add_pipe(nlp.create_pipe('merge_entities'))
term_merger = BadTERMMerger(nlp)
nlp.add_pipe(term_merger, before="ner")


def return_quant_token(row):
    q_t = None
    i = None
    text_doc = nlp(row["Text"])
    for t in text_doc:
        if row["Quant"].strip().lower() in t.text.lower():
            q_t = t
            i = t.i
    return q_t, i

def return_target_token(row):
    t_t = None
    i = None
    text_doc = nlp(row["Text"])
    for t in text_doc:
        if str(row["target"]).strip().lower() in t.text.lower():
            t_t = t
            i = t.i
    return t_t, i


def sent_snippet(row):
    span = None
    if (row["Quant_Token"][1] is not None) and (row["target_Token"][1] is not None):
        start = min(row["Quant_Token"][1], row["target_Token"][1])
        end = max(row["Quant_Token"][1], row["target_Token"][1])
        span = row["Tokens"][start:end + 1]
    return span

def read_labeled_data(data_path):
    gesamt = pd.DataFrame()
    for path, _, files in os.walk(data_path):
        for FILE_NAME in tqdm(files):
            try:
                test_input = pd.read_excel(os.path.join(path, FILE_NAME), usecols=["Text", "Quant", "Target", "Label"],
                                           dtype={"Text": str, "Quant": str, "Target": str})
                test_input.dropna(subset=["Label"], inplace=True, axis=0)
                test_input = test_input.reset_index(drop=True)
                test_input.rename(columns={"Target": "target", "Label": "label"}, inplace=True)
                test_input["label"] = test_input["label"].apply(lambda x: int(x))
                test_input["Tokens"] = test_input.Text.apply(lambda x: [t for t in nlp(x)])
                test_input["Quant_Token"] = test_input.apply(return_quant_token, axis=1)
                test_input["target_Token"] = test_input.apply(return_target_token, axis=1)
                test_input["Sent_Span"] = test_input.apply(sent_snippet, axis=1)
                valided_index = []
                for row in range(len(test_input)):
                    tokens = test_input.loc[row, "Tokens"]
                    quant_index = test_input.loc[row, "Quant_Token"][1]
                    target_index = test_input.loc[row, "target_Token"][1]
                    if quant_index is not None:
                        if tokens[quant_index].head == tokens[target_index].head:
                            valided_index.append(row)
                cleaned = test_input.loc[valided_index, :]
                cleaned = test_input.reset_index(drop=True)
                print(FILE_NAME)
                gesamt = gesamt.append(cleaned, ignore_index=True)
            except:
                print("ERROR!:", FILE_NAME)
                continue
    return gesamt

# labeled data path for relation extraction
LABELED_DATA = r"C:\Users\Salome YE\OneDrive\Masterthese\data\processed_data\labeled"
tokenized_df = read_labeled_data(data_path=LABELED_DATA)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# functions for bert embedding
# tokenisation
def bert_token_pre(dataframe, index, output="all"):
    """
    output: "all"- output whole sentence
    output: "sub" - output separately 
    """
    end = max(dataframe.loc[index, "Quant_Token"][1], dataframe.loc[index, "target_Token"][1])
    start = min(dataframe.loc[index, "Quant_Token"][1], dataframe.loc[index, "target_Token"][1])
    span_before = " ".join([t.text for t in dataframe.loc[index, "Tokens"][:start - 1]])
    span_current = " ".join([t.text for t in dataframe.loc[index, "Tokens"][start:end + 1]])
    span_after = " ".join([t.text for t in dataframe.loc[index, "Tokens"][end + 1:]])
    ber_tok_before = tokenizer.tokenize(span_before)
    ber_tok_span = tokenizer.tokenize(span_current)
    ber_tok_after = tokenizer.tokenize(span_after)
    ber_tok_all = ber_tok_before + ber_tok_span + ber_tok_after
    if output == "all":
        return ber_tok_all
    elif output == "sub":
        return ber_tok_before, ber_tok_span, ber_tok_after

def padding(tokenized_list, max_length):
    if len(tokenized_list) < max_length:
        tokens = ["[CLS]"] + tokenized_list + ["[SEP]"] + (max_length - len(tokenized_list)) * ["[PAD]"]
    else:
        tokens = ["[CLS]"] + tokenized_list + ["[SEP]"]
    return tokens

# 1. add [CLS] [SEP] tokens
def output_maxlen(dataframe):
    maxlen = 0
    maxlen = max([len(tokenizer.tokenize(text)) for text in dataframe["Text"]])
    return maxlen

# prepare for input_id
def get_hidden_state(dataframe, model):
    input_ids = []
    attention_masks = []
    for index in range(len(dataframe)):
        tokenized_list = bert_token_pre(dataframe, index, output="all")
        padded = padding(tokenized_list, output_maxlen(dataframe))
        attention_mask = [1 if t != "[PAD]" else 0 for t in padded]
        encoded = tokenizer.encode_plus(padded, add_special_tokens=False, pad_to_max_length=True,
                                        is_pretokenized=True, return_tensors="pt")
        input_id = encoded["input_ids"]
        input_ids.append(input_id)
        attention_masks.append(torch.tensor(attention_mask))
    print(len(input_ids))
    print(len(attention_masks))
    input_ids = torch.cat(input_ids, dim=0)
    ats = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        last_hidden_states = model(input_ids=input_ids)[0].numpy()
    return last_hidden_states


def generate_train_data(train_df, last_hidden_states_np, maxtextspan):
    length_bert_tokens = []
    lstm_inputs = []
    for i in range(len(train_df)):
        bf, span, after = bert_token_pre(train_df, i, output="sub")
        all_bert_tokens = bert_token_pre(train_df, i, output="all")
        bert_index = (len(bf), len(after))
        # print(bert_index)
        length_bert_tokens.append(len(all_bert_tokens[len(bf):len(bf) + len(span)]))
        bert_span_start = len(bf) + 1
        bert_span_end = len(bf) + len(span)
        hidden_state = last_hidden_states_np[i, len(bf):len(bf) + len(span), :]
        lstm_inputs.append(hidden_state)
    padded = pad_sequences(lstm_inputs, maxlen=maxtextspan, padding='post')
    return padded




num2label = {0: "NoRelation",
             1: "MinValue",
             2: "MaxValue",
             3: "IsValue"}
