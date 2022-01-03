# Deep Learning
from torch.optim.lr_scheduler import OneCycleLR
from utils import contractions_dict
from tqdm import tqdm
import pandas as pd
import re
from bs4 import BeautifulSoup

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# ## Text preprocessing
#
# Text preprocessing steps include a few essential tasks to further clean the available text data. It includes tasks like:-
#
# 1. Stop-Word Removal : In English words like a, an, the, as, in, on, etc. are considered as stop-words so according to our requirements we can remove them to reduce vocabulary size as these words don't have some specific meaning
#
# 2. Lower Casing : Convert all words into the lower case because the upper or lower case may not make a difference for the problem. And we are reducing vocabulary size by doing so.
#
# 3. Stemming : Stemming refers to the process of removing suffixes and reducing a word to some base form such that all different variants of that word can be represented by the same form (e.g., “walk” and “walking” are both reduced to “walk”).
#
# 4. Tokenization : NLP software typically analyzes text by breaking it up into words (tokens) and sentences.

stopwords = stopwords.words('english')

# Why "not" a stopword
# https://datascience.stackexchange.com/questions/15765/nlp-why-is-not-a-stop-word
# Remove 'not' from stopwords
keep_words = ['not', 'no']
for w in keep_words:
    stopwords.remove(w)


def remove_stopwords(text: str, stopwords: list):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
    text = pattern.sub('', text)
    return text


def remove_website_links(text):
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)
    return text


def remove_html_tags(text: str):
    soup = BeautifulSoup(text, 'html.parser')  # Removes HTML tags
    return soup.get_text()


def remove_emoji(text: str):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def expand_contractions(s, contractions_dict: dict = contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    return contractions_re.sub(replace, s)


def text_cleaning(paragraph):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    0. Lower case
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''

    paragraph = paragraph.lower()
    paragraph = re.sub(r"’", "'", paragraph)
    paragraph = remove_website_links(paragraph)
    paragraph = remove_html_tags(paragraph)
    paragraph = remove_emoji(paragraph)

    # paragraph = paragraph.replace('.', ' .')
    # paragraph = re.sub(r"[^...]", " ", paragraph)
    # paragraph = re.sub(r"[^a-zA-Z\d]", " ", paragraph)  # Remove special Charecters
    sents = nltk.sent_tokenize(paragraph)
    # paragraph = paragraph.replace(' .', '.')
    # return sents

    for i in range(len(sents)):
        sent = sents[i]
        sent = expand_contractions(sent, contractions_dict)
        # sent = remove_stopwords(sent, stopwords)
        # sent = re.sub(r"[^a-zA-Z\d]", " ", sent)  # Remove special Charecters
        sent = re.sub(' +', ' ', sent)  # Remove Extra Spaces
        sent = sent.strip()
        sents[i] = sent
    sents = [sent for sent in sents if sent != '  ']
    paragraph = ' . '.join(sents)
    # paragraph = '<s> ' + paragraph + ' </s>'
    return paragraph


label_mapping = {'moderate': 0, 'not depression': 1, 'severe': 2}

tqdm.pandas()

# Load dataset
train_df = pd.read_csv('dataset/train_80.tsv', sep='\t')
dev_df = pd.read_csv('dataset/dev_20.tsv', sep='\t')
dev_df = dev_df.rename(columns={'Text data': 'Text_data'})

train_df.Text_data = train_df.Text_data.progress_apply(text_cleaning)
dev_df.Text_data = dev_df.Text_data.progress_apply(text_cleaning)

train_df.Label = train_df.Label.progress_apply(lambda x: label_mapping[x])
dev_df.Label = dev_df.Label.progress_apply(lambda x: label_mapping[x])

train_df.to_csv('dataset/train_80_prepr.tsv', index=False, sep='\t')
dev_df.to_csv('dataset/dev_20_prepr.tsv', index=False, sep='\t')

# TEST
test_df = pd.read_csv('dataset/dev_with_labels.tsv', sep='\t')
test_df = test_df.rename(columns={'Text data': 'Text_data'})

test_df.Text_data = test_df.Text_data.progress_apply(text_cleaning)
test_df.Label = test_df.Label.progress_apply(lambda x: label_mapping[x])
test_df.to_csv('dataset/dev_with_labels_prepr.tsv', index=False, sep='\t')