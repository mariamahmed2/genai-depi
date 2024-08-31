import re 
from nltk.corpus import stopwords

from nltk import PorterStemmer, WordNetLemmatizer
import joblib
from typing import List, Literal

# vec 
bow = joblib.load('bow.pkl')
tfidf = joblib.load('tfidf.pkl')

# model
svc_bow = joblib.load('svc_bow.pkl')
svc_tfidf = joblib.load('svc_tfidf.pkl')



def cleaning(text: str):
    # tags
    text = remove_pattern(input_txt=text, pattern= r'@[\w]*')

    # url
    text = remove_pattern(input_txt=text, pattern=r'https?://\S+|www\.\S+')
    
    # repeated 
    text = remove_excessive_repeated_characters(input_string=text)

    # emoji
    text = convert_emoticons(text=text)

    # punc
    text = text.replace('[^a-zA-Z#]', ' ')

    # short 
    text= ' '.join([w for w in text.split() if len(w)>3])

    # num
    text = remove_pattern(input_txt=text, pattern='(?<=\w)\d+|\d+(?=\w)')

    # speacial
    text = remove_pattern(input_txt=text, pattern=r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\|\/]')
    
    # redun 
    clean_text = remove_redundant_words_extra_spaces(text=text)


    return clean_text


# token amd lemma 
def text_lemma(text: str) -> str:
    tokenized_text = text.split()
    ## Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemma_word = [lemmatizer.lemmatize(word) for word in tokenized_text]

    lemma_text = ' '.join(lemma_word)

    return lemma_text








## Remove unwanted text patterns from the tweets
def remove_pattern(input_txt: str, pattern: str):
    ''' This Function takes the input and pattern you want to remove

    Args:
    *****
        (input_text: str) --> The text you want to apply the function to it.
        (pattern: str) --> The pattern you want to remove from the text.
    '''
  
    input_txt = re.sub(pattern, '', input_txt)
    return input_txt

## A Function to remove excessive repeated chars while preserving correct words 
def remove_excessive_repeated_characters(input_string, max_repeats=2):
    ## Define a regular expression pattern to match consecutive repeated characters
    pattern = f"(\\w)\\1{{{max_repeats},}}"
    ## Replace the matched pattern with a single occurrence of the character
    cleaned_string = re.sub(pattern, r"\1", input_string)
    
    return cleaned_string


emoticon_meanings = {
    ":)": "Happy",
    ":(": "Sad",
    ":D": "Very Happy",
    ":|": "Neutral",
    ":O": "Surprised",
    "<3": "Love",
    ";)": "Wink",
    ":P": "Playful",
    ":/": "Confused",
    ":*": "Kiss",
    ":')": "Touched",
    "XD": "Laughing",
    ":3": "Cute",
    ">:(": "Angry",
    ":-O": "Shocked",
    ":|]": "Robot",
    ":>": "Sly",
    "^_^": "Happy",
    "O_o": "Confused",
    ":-|": "Straight Face",
    ":X": "Silent",
    "B-)": "Cool",
    "<(‘.'<)": "Dance",
    "(-_-)": "Bored",
    "(>_<)": "Upset",
    "(¬‿¬)": "Sarcastic",
    "(o_o)": "Surprised",
    "(o.O)": "Shocked",
    ":0": "Shocked",
    ":*(": "Crying",
    ":v": "Pac-Man",
    "(^_^)v": "Double Victory",
    ":-D": "Big Grin",
    ":-*": "Blowing a Kiss",
    ":^)": "Nosey",
    ":-((": "Very Sad",
    ":-(": "Frowning",
}


## Function to replace emoticons with their meanings
def convert_emoticons(text: str):
    ''' This Function is to replace the emoticons with thier meaning instead 
    '''
    for emoticon, meaning in emoticon_meanings.items():
        text = text.replace(emoticon, meaning)
    return text

## A Function to remove redundant words like (I've, You'll)
## Again, It is challenging

def remove_redundant_words_extra_spaces(text: str):
    ## Remove contractions using regular expressions
    contraction_pattern = re.compile(r"'\w+|\w+'\w+|\w+'")
    text = contraction_pattern.sub('', text)

    ## Define a set of stopwords
    stop_words = set(stopwords.words("english"))

    ## Remove stopwords and extra spaces
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    clean_text = ' '.join(filtered_words)

    ## Remove extra spaces
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def text_vec (text: str, method: Literal['BOW', 'TF-IDF']):
    # apply vec
    if method == 'BOW':
        x_process = bow.transfoem([text]).toarray()

    else:
        x_process = tfidf.transfoem([text]).toarray()
    return x_process

def predect_class(x_process, method: Literal['BOW', 'TF-IDF'] ):
    svc_bow.predict([x_process])[0]
    if method == 'BOW':
        y_predict = svc_bow.predict([x_process])[0]

    else:
        y_predict = svc_tfidf.predict([x_process])[0]
    return y_predict
