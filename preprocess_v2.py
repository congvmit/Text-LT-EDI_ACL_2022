# Deep Learning

from tqdm import tqdm
import pandas as pd
import mipkit

text_cleaner = mipkit.nlp.TextCleaner(fix_unicode=True,
                                      to_ascii=False,
                                      lower=True,
                                      normalize_whitespace=True,
                                      no_line_breaks=True,
                                      strip_lines=False,
                                      keep_two_line_breaks=False,
                                      no_urls=True,
                                      no_emails=True,
                                      no_phone_numbers=True,
                                      no_numbers=False,
                                      no_digits=False,
                                      no_currency_symbols=True,
                                      no_punct=False,
                                      no_emoji=True,
                                      no_html_tags=True,
                                      no_contractions=True,
                                      no_website_links=True,
                                      no_stopwords=False)

label_mapping = {'moderate': 0, 'not depression': 1, 'severe': 2}

tqdm.pandas()

# Load dataset
train_df = pd.read_csv('dataset/train_80.tsv', sep='\t')
dev_df = pd.read_csv('dataset/dev_20.tsv', sep='\t')
dev_df = dev_df.rename(columns={'Text data': 'Text_data'})

train_df.Text_data = train_df.Text_data.progress_apply(text_cleaner)
dev_df.Text_data = dev_df.Text_data.progress_apply(text_cleaner)

train_df.Label = train_df.Label.progress_apply(lambda x: label_mapping[x])
dev_df.Label = dev_df.Label.progress_apply(lambda x: label_mapping[x])

train_df.to_csv('dataset/train_80_prepr.tsv', index=False, sep='\t')
dev_df.to_csv('dataset/dev_20_prepr.tsv', index=False, sep='\t')

# TEST
test_df = pd.read_csv('dataset/dev_with_labels.tsv', sep='\t')
test_df = test_df.rename(columns={'Text data': 'Text_data'})

test_df.Text_data = test_df.Text_data.progress_apply(text_cleaner)
test_df.Label = test_df.Label.progress_apply(lambda x: label_mapping[x])
test_df.to_csv('dataset/dev_with_labels_prepr.tsv', index=False, sep='\t')