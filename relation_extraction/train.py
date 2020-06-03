# 0. import libraries
import numpy as np
from Relation_Extraction import *
import tensorflow as tf
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import *
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Lambda, Dense, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
nltk.download('wordnet')
from BertMerger import BadTERMMerger
import matplotlib.pyplot as plt


# 1. Data Preprocessing
# 1.1 load labelled data
LABELED_DATA = r"C:\Users\Salome YE\OneDrive\Masterthese\data\processed_data\labeled"
# read from labelled data location and generate dataframe with text spans and indexing of entites
tokenized_df = read_labeled_data(data_path=LABELED_DATA)

num2label = {0: "NoRelation",
             1: "MinValue",
             2: "MaxValue",
             3: "IsValue"}

# 1.2 basic cleaning
tokenized_df_bert = tokenized_df[tokenized_df.Sent_Span != None]
tokenized_df_bert = tokenized_df_bert[tokenized_df_bert.Quant_Token != (None, None)]
tokenized_df_bert = tokenized_df_bert[tokenized_df_bert.target_Token != (None, None)]
tokenized_df_bert = tokenized_df_bert.reset_index(drop=True)

# 2. training data preprocessing
# 2.1 training and testing set split
X_train, X_test, y_train, y_test = train_test_split(
    tokenized_df_bert.loc[:, ~tokenized_df_bert.columns.isin(["label"])], tokenized_df_bert["label"], test_size=0.2,
    random_state=42)
# 2.2 add linguistic features to dataframe, including pos tags, dependency parse
X_train["span"] = X_train["Sent_Span"].apply(lambda x: [k.text for k in x])
X_test["span"] = X_test["Sent_Span"].apply(lambda x: [k.text for k in x])
X_train["pos"] = X_train["Sent_Span"].apply(lambda x: [k.pos_ for k in x])
X_test["pos"] = X_test["Sent_Span"].apply(lambda x: [k.pos_ for k in x])
X_train["dep"] = X_train["Sent_Span"].apply(lambda x: [k.dep_ for k in x])
X_test["dep"] = X_test["Sent_Span"].apply(lambda x: [k.dep_ for k in x])
for df in [X_train, X_test, y_train, y_test]:
    df.reset_index(drop=True, inplace=True)

# 3. Baseline Models
# 3.1 parameter setting for bagofwords of tfidf vectorizer
BOW_PREDICT = {
    "NGRAM_RANGE": (1, 3),
    "MAX_FEATURES": 30,
    "BINARY": False,
    "MAX_DF": 0.95}

# 3.2 feature stacking
features_stack_train, features_stack_test = bow_train_feature_stack(X_train, X_test, BOW_PREDICT,
                                                                    ["span", "pos", "dep"], bow=False)
# 3.3 Linear Regression Model
lr = LogisticRegression(random_state=45, solver="lbfgs").fit(features_stack_train, y_train)
y_pred_lr = lr.predict(features_stack_test)

# 3.3.1 Linear Regression Result Evaluation
print("Linear Regression Model results: ")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print(accuracy_score(y_test, y_pred_lr))

# 3.4 Support Vector Machine Model
svclassifier = SVC(kernel='linear')
svclassifier.fit(features_stack_train, y_train)
y_pred_svc = svclassifier.predict(features_stack_test)

#  3.4.1 Support Vector Machine Model Evaluation
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
print(accuracy_score(y_test, y_pred_svc))

# 4. Neural Network Architecture
num2label = {0: "NoRelation",
             1: "MinValue",
             2: "MaxValue",
             3: "IsValue"}

# 4.1. parameter setting
MAXTEXTSPAN = max(tokenized_df_bert["Sent_Span"].apply(lambda x: len(x)))
SEN_LEN = MAXTEXTSPAN
MAX_LEN = output_maxlen(tokenized_df)
MODEL = BertModel.from_pretrained("bert-base-cased")
NUM_CLASSES = len(set(tokenized_df_bert["label"]))
DIM = 768
EPOCHS = 100
BATCH_SIZE = 32

# 4.2 Bidirectional LSTM architecture
Bert_input = Input(shape=(SEN_LEN, DIM))
X, X_lstm1, _ = LSTM(768, dropout=0.3, return_state=True, return_sequences=True)(Bert_input)
X, X_lstm2, _ = LSTM(768, dropout=0.3, go_backwards=True, return_state=True)(X)
X = Concatenate()([X_lstm1, X_lstm2])
X = Dense(4, activation="softmax")(X)
general_model = Model(Bert_input, X, name="bertforRX")
general_model.summary()
general_model.compile(Adam(5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

#4.3 prepare train and test data
# 4.3.1 Bert Embedding as LSTM input
train_last_hidden_states_np = get_hidden_state(X_train, MODEL)
X_train_input = generate_train_data(X_train, train_last_hidden_states_np,MAXTEXTSPAN)
test_last_hidden_states_np = get_hidden_state(X_test, MODEL)
X_test_input = generate_train_data(X_test, test_last_hidden_states_np, MAXTEXTSPAN)

# 4.4 training
history = general_model.fit(x=X_train_input,
                            y=to_categorical(y_train), batch_size=BATCH_SIZE,
                            validation_data=(X_test_input, to_categorical(y_test)),
                            epochs=EPOCHS)

# 4.5 result and evaluation
#print(history.history.keys())
# 4.5.1 summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# 4.5.2 summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 4.6 save model
general_model.save('bert_relation_extraction')

y_pred = general_model.predict(X_test_input, batch_size=32, verbose=10)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool, target_names=['NoRelation', 'MinValue', 'MaxValue', 'IsValue']))
