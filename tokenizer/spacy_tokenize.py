import spacy

def spacy_tokenizing(src_sequences, trg_sequences, args):

    # 1) Source lanugage
    processed_src, processed_trg, word2id = dict(), dict(), dict()

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

    parsed_src_train = list()

    for src in src_sequences['train']:
        tokenizer_train_src = [token.text for token in spacy_en.tokenizer(src)]
    for src in src_sequences['valid']:
        tokenizer_valid_src = [token.text for token in spacy_en.tokenizer(src)]
    for src in src_sequences['test']:
        tokenizer_test_src = [token.text for token in spacy_en.tokenizer(src)]

    