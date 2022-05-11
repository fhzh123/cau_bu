import os
import time
import h5py
import pickle
import logging
import numpy as np
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizeing
from tokenizer.spacy_tokenize import spacy_tokenizing
from utils import TqdmLoggingHandler, write_log

def preprocessing(args):

    start_time = time.time()