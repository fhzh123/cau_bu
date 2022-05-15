import os
import time
import h5py
import pickle
import logging
import numpy as np
import pandas as pd
# Import custom modules
from tokenizer.spm_tokenize import spm_tokenizing
from tokenizer.plm_tokenize import plm_tokenizeing
from tokenizer.spacy_tokenize import spacy_tokenizing
from utils import TqdmLoggingHandler, write_log

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    src_sequences = dict()
    trg_label = dict()

    args.data_path = os.path.join(args.data_path,'GVFC')
    
    gvfc_dat = pd.read_csv(os.path.join(args.data_path, 'GVFC_headlines_and_annotations.csv'))
    gvfc_dat = gvfc_dat.replace(99, 0)
    src_text = gvfc_dat['news_title'].tolist()
    trg_class = gvfc_dat['Q3 Theme1'].tolist()

    # Train / Test split
    data_len = len(src_text)
    valid_index = np.random.choice(data_len, int(data_len * 0.15), replace=False)
    train_index = list(set(range(data_len)) - set(valid_index))
    test_index = np.random.choice(train_index, int(data_len * 0.1), replace=False)
    train_index = list(set(train_index) - set(test_index))

    src_sequences['train'] = list(src_text[i] for i in train_index)
    src_sequences['valid'] = list(src_text[i] for i in valid_index)
    src_sequences['test'] = list(src_text[i] for i in test_index)

    trg_label['train'] = list(trg_class[i] for i in train_index)
    trg_label['valid'] = list(trg_class[i] for i in valid_index)
    trg_label['test'] = list(trg_class[i] for i in test_index)

    #===================================#
    #==========Pre-processing===========#
    #===================================#

    write_log(logger, 'Tokenizer setting...')
    start_time = time.time()

    if args.tokenizer == 'spm':
        processed_src, word2id = spm_tokenizing(src_sequences, args)
    elif args.tokenizer == 'spacy':
        processed_src, word2id = spacy_tokenizing(src_sequences, args)
    else:
        processed_src, word2id = plm_tokenizeing(src_sequences, args)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')

    #===================================#
    #==============Saving===============#
    #===================================#

    write_log(logger, 'Parsed sentence saving...')
    start_time = time.time()

    # Path checking
    save_path = os.path.join(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.tokenizer == 'spm':
        save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.pkl'
    else:
        save_name = f'processed_{args.tokenizer}.pkl'

    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump({
            'train_src_input_ids': processed_src['train'],
            'train_trg_labels': trg_label['train'],
            'valid_src_input_ids': processed_src['valid'],
            'valid_trg_labels': trg_label['valid']
        }, f)

    with open(os.path.join(save_path, 'test_' + save_name), 'wb') as f:
        pickle.dump({
            'test_src_input_ids': processed_src['test'],
            'test_trg_labels': trg_label['test']
        }, f)

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'wb') as f:
        pickle.dump({
            'src_word2id': word2id['src']
        }, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')