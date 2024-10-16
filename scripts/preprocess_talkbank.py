'''
Pre-processing for the Pitt corpus and WLS corpus,
'''
import warnings
import re
from glob import glob
from tqdm import tqdm
import shutil
import json
import string
import pickle
import argparse
import os
import configparser
from TRESTLE import TextWrapperProcessor, AudioProcessor
import spacy
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import lru_cache

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command-line arguments with nested structure."""
    parser = argparse.ArgumentParser(description="Process TalkBank corpus data.")
    
    subparsers = parser.add_subparsers(dest="component", required=True, help="The component of TalkBank")
    
    # Common parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--preprocess", action="store_true", help="Perform pre-processing")
    parent_parser.add_argument("--indicator", required=True, help="Speaker indicator (follow CLAN annotation)")
    
    # Pitt component parser
    pitt_parser = subparsers.add_parser("pitt", help="Process Pitt corpus", parents=[parent_parser])
    pitt_parser.add_argument("--subset", required=True, help="The subset of Pitt corpus")
    
    # WLS component parser
    wls_parser = subparsers.add_parser("wls", help="Process WLS corpus", parents=[parent_parser])
    
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_word_dist():
    """
    Load word distribution from a pickle file.

    Returns:
        dict: Word distribution dictionary.
    """
    with open("word_dist.pkl", "rb") as f:
        return pickle.load(f)


def get_par_trans(input_sample, preprocess_rules, require_audio=True):
    """Perform text and audio pre-processing."""
    #if not os.path.exists(input_sample['text_output_path']):
    subset_info = f", {input_sample['subset']}" if input_sample['subset'] else ''
    print(f"Text preprocessing for {input_sample['component']}{subset_info} on {input_sample['indicator'][1:]} transcripts")
    wrapper_processor = TextWrapperProcessor(
        data_loc=input_sample, txt_patterns=preprocess_rules)
    wrapper_processor.process()
    
    if require_audio and not os.path.exists(input_sample['audio_output_path']):
        processor = AudioProcessor(data_loc=input_sample, sample_rate=16000)
        processor.process_audio()

def process_transcript(tran):
    """Process a single transcript file."""
    with open(tran, "r") as json_file:
        turns = sum(1 for _ in json_file)
    return turns


def get_turns(input_sample):
    """Get the number of PAR and INV turns from the component."""
    par_path = input_sample["text_output_path"]
    inv_path = par_path.replace("PAR", "INV")
    all_par_trans = glob(f"{par_path}/*.jsonl")
    all_inv_trans = glob(f"{inv_path}/*.jsonl")

    # Create a dictionary to store both PAR and INV turns for each PID
    turn_dict = {}
    
    # Process PAR turns
    for tran in tqdm(
        all_par_trans, desc=f"Processing {input_sample['component']} PAR transcripts"):
        pid = os.path.basename(tran).split(".")[0]
        if input_sample['component'] == "wls":
            pid = f"20000{pid}"
        par_turns = process_transcript(tran)
        turn_dict[pid] = {"pid": pid, "par_turns": par_turns, "inv_turns": 0}
    
    # Process INV turns
    for tran in tqdm(
        all_inv_trans, desc=f"Processing {input_sample['component']} INV transcripts"):
        pid = os.path.basename(tran).split(".")[0]
        if input_sample['component'] == "wls":
            pid = f"20000{pid}"
        inv_turns = process_transcript(tran)
        if pid in turn_dict:
            turn_dict[pid]["inv_turns"] = inv_turns
        else:
            turn_dict[pid] = {"pid": pid, "par_turns": 0, "inv_turns": inv_turns}
    
    # Convert the dictionary to a list of dictionaries
    turn_list = list(turn_dict.values())
    return pd.DataFrame(turn_list)


def get_info(input_sample, spacy_processor):
    """Get participants' average POS tags, log lexical, and TTR for talkbank component"""
    stop_words = set(stopwords.words("english"))
    word_dist = load_word_dist()
    all_trans = glob(f"{input_sample['text_output_path']}/*.jsonl")

    pos_list = []
    for tran in tqdm(
        all_trans,
        desc=f"Processing {input_sample['component']} transcripts for POS tags"):
        pid = os.path.basename(tran).split(".")[0]
        if input_sample['component'] == "wls":
            pid = f"20000{pid}"
        pos_counts = {"pid": pid}
        with open(tran, "r") as json_file:
            text = " ".join(json.loads(jline)['text'] for jline in json_file)
            text = re.sub(r"\s+", " ", text)
        if text:
            # POS tags
            doc = spacy_processor(text)
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            # clause
            clause_count = 0
            for sent in doc.sents:
                # Count root verbs (main clauses)
                clause_count += len([token for token in sent if token.dep_ == "ROOT"])
                
                # Count subordinate clauses
                clause_count += len([token for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl"]])
            total_words = word_tokenize(text)
            total_words = [word for word in total_words 
                        if word not in string.punctuation]
            words = set(word for word in word_tokenize(text) 
                        if word not in string.punctuation and word not in stop_words)
            # type-to-token ratio
            ttr = round(len(words)/len(total_words), 2)
            # log lexical frequency
            log_lf = [np.log(word_dist.get(word, 1)) for word in words]
            log_lf = [lf for lf in log_lf if lf != -np.inf and lf != 0.0]
            avg_lf = round(sum(log_lf) / len(log_lf), 2) if log_lf else 0
            pos_counts['LF'] = avg_lf
            pos_counts['TTR'] = ttr
            pos_counts['CLAUSE'] = clause_count
            pos_list.append(pos_counts)
        else:
            pass

    pos_df = pd.DataFrame(pos_list)
    pos_df.drop(["X", "NUM"], axis=1, errors='ignore', inplace=True)
    pos_df.fillna(0, inplace=True)
    return pos_df


def load_or_process_data(input_sample):
    """
    Load previously processed data or process new data for POS and turns.

    Args:
        input_sample (dic): The input meta data.
        subset (str): The subset of the Pitt corpus.

    Returns:
        pandas.DataFrame: The processed subset data.
    """
    nlp = spacy.load('en_core_web_trf')
    pos_df = get_info(input_sample, nlp)
    turn_df = get_turns(input_sample)
    df = pd.merge(pos_df, turn_df, on="pid")
    df['pid'] = df['pid'].astype(int)
    
    return df


def get_meta_from_header(subset):
    """
    Extract metadata from CHAT file headers.

    Args:
        subset (str): The subset of the corpus to process.

    Returns:
        pandas.DataFrame: Dataframe containing extracted metadata.
    """
    all_trans = glob(os.path.join(f"{config['DATA']['pitt_input']}{subset}/txt/", "*.cha"))
    meta_keys = ["age", "gender", "dx", "mmse"]
    
    meta_list = []
    for tran in tqdm(all_trans, desc=f"Processing {subset} transcripts for metadata"):
        pid = os.path.basename(tran).split(".")[0]
        with open(tran, "r") as cha_file:
            meta_header = next(line.strip() for line in cha_file if line.startswith('@ID:\teng|Pitt|PAR|'))
        
        meta_header = meta_header.split("|")[3:]
        meta_header.remove("Participant")
        meta_header[0] = int(meta_header[0].split(";")[0])
        
        meta_dict = dict(zip(meta_keys, meta_header))
        meta_dict['pid'] = pid
        meta_list.append(meta_dict)
    
    return pd.DataFrame(meta_list)

def merge_with_meta_pitt(meta_data_path):
    # get mmse score for each transcript
    meta_df = pd.read_csv(meta_data_path)
    meta_df = meta_df[["id", "educ", "mms", "mmse2", "mmse3", "mmse4", "mmse5", "mmse6", "mmse7", "lastmms"]]
    meta_df.rename(columns={"id": "pid", "mms": "mmse1"}, inplace=True)
    meta_df['pid'] = meta_df['pid'].astype(str).str.zfill(3)
    melted_df = pd.melt(meta_df, id_vars=['pid', 'educ'],
                    value_vars=[f'mmse{i}' for i in range(1, 8)],
                    var_name='mmse_num', value_name='mmse')
    melted_df = melted_df.dropna(subset=['mmse'])
    melted_df['mmse_num'] = melted_df['mmse_num'].str.extract('(\d+)').astype(int) - 1
    melted_df['pid'] = melted_df['pid'] + '-' + melted_df['mmse_num'].astype(str)
    melted_df.drop(columns=['mmse_num'], axis=1, inplace=True)
    mmse_df = melted_df[['pid', "educ", "mmse"]].reset_index(drop=True)
    # get subset metadata from header
    dem_df = get_meta_from_header("dementia")
    con_df = get_meta_from_header("control")
    full_df = pd.concat([con_df, dem_df])
    full_df.drop(columns=['mmse'], axis=1, inplace=True)
    full_df = full_df.merge(mmse_df, on=['pid'])
    return full_df

def merge_with_meta_wls(meta_data_path):
    meta_df = pd.read_csv(meta_data_path)
    # TICSm score for dementia cutoff
    meta_df = meta_df.dropna(subset=['TICSm score'])
    # get diagnosis via positive predictive value summary
    # meta_df = meta_df.loc[meta_df['Positive predictive value summary outcome'].isin([1,2,3,6,8])]
    # remove participants who did not complete long interview
    # meta_df = meta_df.loc[meta_df['Level of cognitive impairment via Consensus']!= -2]
    meta_df.rename(columns={'idtlkbnk': 'pid',
                            'sex': 'gender',
                            'age 2020': 'age',
                            'TICSm score': 'score'},
                    inplace=True)
    meta_df = meta_df.loc[(meta_df['score'] <= 27 )| (meta_df['score'] > 31)]
    meta_df['dx'] = np.where(meta_df['score'] <= 27, 1, 0)
    meta_df['gender'] = np.where(
        meta_df['gender'] == 1, 'male', 'female'
    )
    meta_df = meta_df[['pid', 'age', 'gender','dx', 'score']]
    meta_df = meta_df.loc[meta_df['age']!= -2]
    # add education
    educ_file = meta_data_path.replace("diagnosis", 'educ')
    educ_df = pd.read_csv(educ_file)
    educ_df.rename(
        columns={'idtlkbnk': 'pid',
                    'education': 'educ',},
        inplace=True)
    meta_df['pid'] = meta_df['pid'].astype(int)
    meta_df = meta_df.merge(educ_df, on=['pid'])
    return meta_df


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    pargs = parse_args()
    cha_txt_patterns = {
        # remove the parentheses and keep the content only if it's English letters
        r'\([^a-zA-Z]*\)': '',
        r'\(([^a-zA-Z]*)\)': "",
        r'[()]': "",
        # remove special form markers
        r'@\w': '',
        # remove unitelligent words
        r'(\w)\1\1': '',
        # replace single underscore as a whitespace
        r'(?<!_)_(?!_)': ' ',
        r'\[.*?\]': "",
        # keep repetitions
        r'&-(\w+)': r'\1',
        # keep invited interruptions
        r'&\+(\w+)': r'\1',
        # remove gestures
        r'&=(\w+)': "",
        # keep phrase revision
        r'\<([^<>]*)\>': r'\1',
        # removing trailling off utterances
        r'\+..': "",
        # remove non-ascii characters
        r'[^\x00-\x7F]+': '',
        # remove addtional whitespace between the last word and the last punctuation
        r'\s+([.,!?;:])|([.,!?;:])\s+': r'\1\2',
        # remove additional whitespace
        r'\s+': ' ',
    }
    subset = getattr(pargs, 'subset', '')
    sample = {
        "format": ".cha",
        "component": pargs.component,
        "subset": getattr(pargs, 'subset', None),
        "indicator": pargs.indicator,
        "text_input_path": os.path.join(
            config['DATA'][f'{pargs.component}_input'], getattr(pargs, 'subset', '') or '', "txt"),
        "audio_input_path": os.path.join(
            config['DATA'][f'{pargs.component}_input'], getattr(pargs, 'subset', '') or '', "audio"),
        "text_output_path": os.path.join(
            config['DATA'][f'{pargs.component}_output'], getattr(pargs, 'subset', '') or '', pargs.indicator[1:], "txt"),
        "audio_output_path": os.path.join(
            config['DATA'][f'{pargs.component}_output'], getattr(pargs, 'subset', '') or '', pargs.indicator[1:], "audio"),
        "audio_type": ".mp3",
        "speaker": pargs.indicator,
        "content": r'@G:	Cookie\n(.*?)@End' if pargs.component == "pitt" else r'@Bg:	Activity\n.*?@Eg:	Activity',
    }
    if pargs.preprocess:
        # if os.path.exists(config['DATA'][f"{sample['component']}_output"]):
        #     shutil.rmtree(config['DATA'][f"{sample['component']}_output"])
        get_par_trans(
            sample, cha_txt_patterns, require_audio=False)
    else:
        subset_suffix = f"_{subset}" if subset else ""
        meta_prefix = config['META']['prefix']
        # get meta data
        meta_file = os.path.join(meta_prefix, f"{sample['component']}.csv")
        if sample['component'] == "pitt":
            meta_df = merge_with_meta_pitt(config['META'][f"{sample['component']}_meta"])
        else:
            meta_df = merge_with_meta_wls(config['META'][f"{sample['component']}_meta"])
        # get turns data
        turn_file = os.path.join(meta_prefix, f"{sample['component']}{subset_suffix}_turns.csv")
        turn_df = load_or_process_data(sample)
        # merge for further analysis
        full_file = os.path.join(meta_prefix, f"{sample['component']}_total.csv")
        if sample['component'] == 'pitt':
            con_file = os.path.join(meta_prefix, f"{sample['component']}_control_turns.csv")
            dem_file = os.path.join(meta_prefix, f"{sample['component']}_dementia_turns.csv")
            
            con_df = pd.read_csv(con_file)
            dem_df = pd.read_csv(dem_file)
            
            full_df = pd.concat([con_df, dem_df]).merge(meta_df, on=['pid'])
        else:
            full_df = turn_df.merge(meta_df, on=['pid'])
        full_df = full_df.sample(frac=1)
        full_df.to_csv(full_file, index=False)
