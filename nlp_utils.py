import string
import nltk
import re
import numpy as np
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from utils import flatten

def process_text(text: str, remove_punctuation: bool = False, 
                 remove_stopword: bool = False, remove_text_brackets: bool = False,
                 split_sentences: bool = True) -> (list,list[tuple]):
    """
    This is a utility function for cleaning up text quickly for NLP.
    
    text (str):                     takes in some text that is processed
    remove_punctuation (bool):      remove punctuation from the text
    remove_stopword (bool):         remove stopwords from the text
    remove_text_brackets (bool):    remove the text inbetween brackets
    split_sentences (bool):         split the text into sentences if active, otherwise just splits by words
    
    returns two variables, the first one corresponds to the words in the sentence, the second contains words as 
    well as postags.
       
    [[word1sent1, word2sent1, word3sent1], [word1sent2, word2sent2, word3sent2]], 
    [[(word1sent1, tag1sent1, word2sent1, tag1sent1)], [word1sent2, tag1sent2, word2sent2, tag2sent2]]
    
    """
    if remove_text_brackets:
        text = re.sub("[\(\[].*?[\)\]]", "", text)
    sentences: list = nltk.sent_tokenize(text) # split the text into sentences
    if remove_punctuation: # remove punctuation
        sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences] # remove punctuation
    l: list = [nltk.word_tokenize(s) for s in sentences] # split sentences into lists of words
    ln: list = []
    tags: list = []
    for s in l:
        temp_tags = nltk.pos_tag(s)
        ln.append([])
        tags.append([])
        for w,t in zip(s,temp_tags):
            if remove_stopword:
                if w not in stopwords:
                    ln[-1].append(w)
                    tags[-1].append(t)
            else:
                ln[-1].append(w)
                tags[-1].append(t)

    l = ln
    # if output == "str":
    #     out = [" ".join([w for s in l for w in s])][0]
    sentences = [sent for sent in l if len(sent) >0]
    tagged_sentences = [tag for tag in tags if len(tag) >0]
    
    if not split_sentences:
        sentences = [word for sent in sentences for word in sent]
        tagged_sentences = [tag_word for tag in tags for tag_word in tag]
    return sentences, tagged_sentences
    
def process_text_to_str(text: str, remove_punctuation: bool = False, 
                        remove_stopword: bool = False, remove_text_brackets: bool = False) -> str:
    """
    
    This is a utility function for cleaning up text quickly for NLP. It returns a cleaned string.
    
    text (str):                     takes in some text that is processed
    remove_punctuation (bool):      remove punctuation from the text
    remove_stopword (bool):         remove stopwords from the text
    remove_text_brackets (bool):    remove the text inbetween brackets
    
    returns a string with cleaned text
    """
    text, tags = process_text(text, remove_punctuation=remove_punctuation, remove_stopword=remove_stopword)
    return " ".join([w for s in text for w in s])


# ----------------------------------------------------------------
# BAG OF WORD UTILITIES

import copy
from sortedcontainers import SortedDict


def create_corpus(texts: str, overlap_only: bool = False):
    """ Creates the corpus to create Bag of Word embeddings

    Args:
        texts (str): list of texts for which to create the embeddings
        overlap_only (bool, optional): Whether to create an overlap only. Defaults to False.

    Returns:
        _type_: a list of words representing the corpus
    """
    if overlap_only:
        all_words = [word.lower() for text in texts for word in text]
        words = dict.fromkeys(all_words,0)
        for word in all_words:
            words[word] += 1
        corpus = []
        for word in all_words:
            if words[word] > 1:
                corpus.append(word)
        return corpus
    
    else:
        corpus = [word.lower() for text in texts for word in text]
        return np.unique(corpus)
    

def create_bow_encoding(texts: list, words: list) -> list[dict]:
    """
    texts (list[list[str]]): a list of texts for which we want to create the embeddings. Each text is a list of words
    words (list[str]): a list of all words for which we want to create the embeddings
    
    returns a list of dictionaries, where each dictionary corresponds to one text
    """
    if not isinstance(texts[0], list):
        texts = [texts]
    words = SortedDict.fromkeys(words,0)
    all_counts = []
    for text in texts:
        temp_words = copy.deepcopy(words)
        for word in text:
            try:
                temp_words[word.lower()] += 1 # if the word is not in the original dictionary ignore it
            except:
                pass
        all_counts.append(temp_words)
    return all_counts

def make_bow(texts: str, overlap_only: bool = False) -> tuple[list[dict], list[str]]:
    """ Takes in a list of texts and returns the bag of word encoding of the texts. 

    Args:
        texts list[list[str]]: A list containing multiple texts which are each tokenized into their words

    Returns:
        tuple(list[dict], list[str]): returns a list of dictionaries (the bow encodings for each text) and a list of words that make up the corpus
    """
    corpus = create_corpus(texts, overlap_only=overlap_only)
    return create_bow_encoding(texts, corpus), corpus