from transformers import  BertTokenizer, BartTokenizer, T5Tokenizer

def plm_tokenizeing(text_sequences, args):
    processed_text_dict, word2id = dict(), dict()

    if args.tokenizer == 'bert':
        tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # 1) Source lanugage
    for phase in ['train', 'valid', 'test']:
        processed_text_dict[phase] = \
        tokenizer(
            text_sequences[phase],
            max_length=args.max_len,
            padding='max_length',
            truncation=True
        )
    
    word2id['src'] = tokenizer.get_vocab()

    return processed_text_dict, word2id