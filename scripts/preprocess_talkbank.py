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
    parser.add_argument("--preprocess", action="store_true", help="Perform pre-processing")
    
    subparsers = parser.add_subparsers(dest="component", required=True, help="The component of TalkBank")
    
    # Pitt component parser
    pitt_parser = subparsers.add_parser("pitt", help="Process Pitt corpus")
    pitt_parser.add_argument("--subset", required=True, help="The subset of Pitt corpus")
    pitt_parser.add_argument("--indicator", required=True, help="Speaker indicator (follow CLAN annotation)")
    
    # WLS component parser
    wls_parser = subparsers.add_parser("wls", help="Process WLS corpus")
    wls_parser.add_argument("--indicator", required=True, help="Speaker indicator (follow CLAN annotation)")
    
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


def get_par_trans(sample, preprocess_rules, require_audio=True):
    """Perform text and audio pre-processing."""
    if not os.path.exists(sample['text_output_path']):
        subset_info = f", {sample['subset']}" if sample['subset'] else ''
        print(f"Text preprocessing for {sample['component']}{subset_info} on {sample['indicator'][1:]} transcripts")
        TextWrapperProcessor(data_loc=sample, txt_patterns=preprocess_rules).process()
    
    if require_audio and not os.path.exists(sample['audio_output_path']):
        AudioProcessor(data_loc=sample, sample_rate=16000).process_audio()

def process_transcript(tran):
    """Process a single transcript file."""
    pid = os.path.basename(tran).split(".")[0]
    with open(tran, "r") as json_file:
        turns = sum(1 for _ in json_file)
    return {"pid": pid, "inv_turns": turns}

def get_turns(subset, config):
    """Get the number of INV turns for the Pitt corpus subset."""
    all_trans = glob(os.path.join(f"{config['DATA']['pitt_output']}{subset}/INV/txt/*.jsonl"))
    
    turn_list = []
    for tran in tqdm(all_trans, desc=f"Processing {subset} transcripts for INV turns"):
        turn_list.append(process_transcript(tran))
    
    return pd.DataFrame(turn_list)


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
        "text_input_path": os.path.join(config['DATA'][f'{pargs.component}_input'], getattr(pargs, 'subset', '') or '', "txt"),
        "audio_input_path": os.path.join(config['DATA'][f'{pargs.component}_input'], getattr(pargs, 'subset', '') or '', "audio"),
        "text_output_path": os.path.join(config['DATA'][f'{pargs.component}_output'], getattr(pargs, 'subset', '') or '', pargs.indicator[1:], "txt"),
        "audio_output_path": os.path.join(config['DATA'][f'{pargs.component}_output'], getattr(pargs, 'subset', '') or '', pargs.indicator[1:], "audio"),
        "audio_type": ".mp3",
        "speaker": pargs.indicator,
        "content": r'@G:	Cookie\n(.*?)@End' if pargs.component == "pitt" else r'@Bg:	Activity\n.*?@Eg:	Activity'
    }
    print(sample)