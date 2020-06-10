# Optimizing Domain-specific Regulation Comparison with Semantic Text Matching & Relation Extraction 

## Introduction
This repository is for the source codes of master thesis: "Optimizing Domain-specific Regulation Comparison with Semantic Text Matching & Relation Extraction", the goals of which is to examine the potential for Natural Language Processing technologies to optimize current manual regulation maintenance process in an automotive manufacturer enterprise and proposes concepts of architecture and implementation of a text-matching as well as a relation extraction mechanism that integrates entity recognition to extract key information that is interested by experts from the regulations that could accelerate the manual information processes. 

## The source codes consists of following two parts:
### Part 1: Paragraph Matching
<img src="archiv/code structure.png">

The directory includes following files:

* main
  * CorpusGenerator.py
  * config
    * config.yml
  * trained models
  * para_similarity.py
  * TW_EU_tf_idf.py
  * TW_EU_LSI.py
  * TW_EU_w2v.py
  * TW_EU_d2v.py
  * Evaluate.py
  * evaluation_plot.ipynb
  
 #### user guide:
 Paragraph matching mechanism supports four alternive text representation approaches: 
 1. TF-IDF
 2. LSI
 3. word2vec
 4. doc2vec
 
correspondent configurations van be done under config YAML file in config directory. Supported Configuations include:
1. File location
  1. data folder
  2. document pairing file folder
  3. result output folder
  4. groud truth folder
2. list of approaches that need to be generated in one execution
3. Preprocessing approaches (boolean values)
  1. bigram / trigram
  2. porter stemming / wordnet lemmatization
4. Modeling 
  For each approach:
  1. parameters
  2. similarity metrics
    1. cosine similarity 
    2. word mover distance
  3. if the embedding needs to be pretrained
  
an example can be seen under [example config](paragraph_matching/main/config)
  
 
- - - -
### Part 2: Relation Extraction
modelling of relation extration consists of following steps: 
1. Recognition of defined "Quantitative Value" Entity: pretrained customized SpaCy model under "ner" folder
2. extraction of tokens with "noun" as Part of Speech under same tree in dependency parsing with named entity in the last step and pairing with named entity
3. classification of above entity pairs into four defined classes: MinValue, MaxValue, IsValue, NoRelation

The architecture is summarized as follow:
<img src="archiv/relation extraction.png">


