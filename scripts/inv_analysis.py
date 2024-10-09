'''
Exploratory data analysis on inv turns
'''
import re
import warnings
from glob import glob
from tqdm import tqdm
import string
import os
import configparser
from itertools import combinations
from collections import Counter
import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')


def read_files(files_list):
    """
    read and save investigators' utterances into dataframe

    Args:
        files_list (list): a list of files containing investigators' utterances

    Returns:
        pd.DataFrame: all investigators utterances in a unified dataframe
    """
    total_df = pd.DataFrame()
    for tran in tqdm(
        files_list,
        desc=f"Reading INV utterances"):
        pid = os.path.basename(tran).split(".")[0]
        sub_df = pd.read_json(tran, lines=True)
        try:
            sub_df = sub_df[['text']]
            sub_df['pid'] = pid
            total_df = pd.concat([total_df, sub_df])
        except KeyError:
            pass
    return total_df


def load_inv_wls(config):
    matched_df = pd.read_csv("../data/matched_wls.csv")
    inv_path = os.path.join(config['DATA']['wls_output'], "INV/txt")
    inv_files = glob(f"{inv_path}/*.jsonl")
    inv_df = read_files(inv_files)
    inv_df['pid']  = '20000' + inv_df['pid'].astype(str)
    inv_df['pid'] = inv_df['pid'].astype(int) 
    # inv_df = inv_df.loc[inv_df['pid'].isin(matched_ids)]
    total_df = matched_df.merge(inv_df, on = "pid")
    return total_df


def load_inv_pitt(config):
    """
    load *matched* utterances from investigators in the Pitt corpus

    Args:
        config (dict): config reader

    Returns:
        pd.DataFrame: the utterances from investigators in the matched Pitt corpus
    """
    matched_ids = pd.read_csv("../data/matched_pitt.csv")
    matched_ids = matched_ids['pid'].values.tolist()
    inv_con_path = os.path.join(config['DATA']['pitt_output'], "control/INV/txt")
    inv_con_files = glob(f"{inv_con_path}/*.jsonl")
    inv_dem_path = os.path.join(config['DATA']['pitt_output'], "dementia/INV/txt")
    inv_dem_files = glob(f"{inv_dem_path}/*.jsonl")
    inv_con_df = read_files(inv_con_files)
    inv_con_df['dx'] = int(0)
    inv_dem_df = read_files(inv_dem_files)
    inv_dem_df['dx'] = int(1)
    inv_df = pd.concat([inv_con_df, inv_dem_df])
    inv_df = inv_df.loc[inv_df['pid'].isin(matched_ids)]
    return inv_df

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and token.is_alpha]
    return tokens


def get_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def analyze_ngrams(df, text_column, n):
    all_ngrams = []
    for utterance in df[text_column]:
        tokens = preprocess_text(utterance)
        all_ngrams.extend(get_ngrams(tokens, n))
    
    ngram_freq = Counter(all_ngrams)
    df_ngrams = pd.DataFrame(ngram_freq.most_common(), columns=[f'{n}-gram', 'Frequency'])
    return df_ngrams

def ngram_driver(df, utterance_column, n):
    print(f"============ Top 10 {n}-grams:=========")
    bigrams_df = analyze_ngrams(df, utterance_column, n)
    print(bigrams_df.head(10))

def segment_sentences(text):
    doc = nlp(text)
    return [strip_end_punctuation(sent.text.strip()) for sent in doc.sents]

def strip_end_punctuation(sentence):
    return sentence.rstrip(string.punctuation)

def find_top_10_sentences(sentences):
    # Count the occurrences of each sentence
    sentence_counts = Counter(sentences)
    
    # Get the 10 most common sentences
    top_10 = sentence_counts.most_common(10)
    
    return top_10


def utterance_analysis(df, utterance_column, model):
    all_utterances = [sentence for utterance in df[utterance_column] for sentence in segment_sentences(utterance)]
    all_utterances = [re.sub(r'\s?\x15', '', item) for item in all_utterances]
    all_utterances = [item for item in all_utterances if item]
    print(f"Number of utterances: {len(all_utterances)}")
    # top 10 common sentence
    top_10 = find_top_10_sentences(all_utterances)
    for sentence, count in top_10:
        print(f"'{sentence}' - {count} occurrences")
    all_utterances = list(set(all_utterances))
    print(f"Number of utterances after de-duplicate: {len(all_utterances)}")
    utterance_pairs = list(combinations(all_utterances, 2))
    df = pd.DataFrame(utterance_pairs, columns=['utterance1', 'utterance2'])
    df['score'] = calculate_similarities(df, model, 1000)
    print(df['score'].describe())
    return df


def calculate_similarities(df, model, batch_size=1000):
    total_rows = len(df)
    similarities = np.zeros(total_rows)
    
    for i in tqdm(range(0, total_rows, batch_size)):
        batch = df.iloc[i:i+batch_size]
        sentences1 = batch['utterance1'].tolist()
        sentences2 = batch['utterance2'].tolist()
        
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        
        batch_similarities = model.similarity(embeddings1, embeddings2)
        
        # Extract the diagonal of the similarity matrix
        batch_similarities_diagonal = np.diagonal(batch_similarities.cpu().numpy())
        
        similarities[i:i+len(batch_similarities_diagonal)] = batch_similarities_diagonal
    
    return similarities

def sim_driver(df, utterance_column, model):
    con_df = df.loc[df['dx'] == 0]
    dem_df = df.loc[df['dx'] == 1]
    print("============ Control ================")
    con_res = utterance_analysis(con_df, utterance_column, model)
    print("============ Dementia ===============")
    dem_res = utterance_analysis(dem_df, utterance_column, model)
    con_res['dx'] = 0
    dem_res['dx'] = 1
    res_df = pd.concat([con_res, dem_res])
    return res_df


if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    if not os.path.exists("../data/pitt_inv.tsv") or not os.path.exists("../data/wls_inv.tsv"):
        inv_pitt = load_inv_pitt(config_parser)
        inv_pitt = inv_pitt[~((inv_pitt['text'] == ".") | (inv_pitt['pid'] == "") | (inv_pitt['text'].isna()))]
        inv_wls = load_inv_wls(config_parser)
        inv_wls = inv_wls[~((inv_pitt['text'] == ".") | (inv_wls['pid'] == "") | (inv_wls['text'].isna()))]
        inv_pitt = inv_pitt.groupby(['pid', 'dx'])['text'].agg(result=lambda x: "".join(x)).reset_index()
        inv_wls = inv_wls.groupby(['pid', 'dx'])['text'].agg(result=lambda x: "".join(x)).reset_index()
        inv_pitt.to_csv("../data/pitt_inv.tsv", sep="\t", index=False)
        inv_wls.to_csv("../data/wls_inv.tsv", sep="\t", index=False)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        utterance_column = "result"
        nlp = spacy.load('en_core_web_trf')
        inv_pitt = pd.read_csv("../data/pitt_inv.tsv", sep="\t")
        inv_wls = pd.read_csv("../data/wls_inv.tsv", sep="\t")
        print("============ Pitt corups ============")
        # ngram_driver(inv_pitt, utterance_column, 3)
        pitt_res = sim_driver(inv_pitt, utterance_column, model)
        pitt_res.to_csv("../data/pitt_sim.csv", index=False)
        print("============ WLS corups =============")
        wls_res = sim_driver(inv_wls, utterance_column, model)
        wls_res.to_csv("../data/wls_sim.csv", index=False)
