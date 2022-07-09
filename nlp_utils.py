import string
import nltk
import re
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


def process_text(text: str, remove_punctuation: bool = False, 
                 remove_stopword: bool = False, remove_text_brackets: bool = False) -> (list,list[tuple]):
    """
    This is a utility function for cleaning up text quickly for NLP.
    
    text (str):                     takes in some text that is processed
    remove_punctuation (bool):      remove punctuation from the text
    remove_stopword (bool):         remove stopwords from the text
    remove_text_brackets (bool):    remove the text inbetween brackets
    
    
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
    return [sent for sent in l if len(sent) >0], [tag for tag in tags if len(tag) >0]
    
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