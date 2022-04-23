from utils import TqdmLoggingHandler, write_log

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    # write_log(logger, "Load data...")
    # gc.disable()

    # save_path = os.path.join(args.preprocess_path, args.tokenizer)
    # if args.tokenizer == 'spm':
    #     save_name = f'processed_{args.data_name}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    # else:
    #     save_name = f'processed_{args.data_name}_{args.tokenizer}.hdf5'

    # with h5py.File(os.path.join(save_path, save_name), 'r') as f:
    #     train_src_input_ids = f.get('train_src_input_ids')[:]
    #     train_trg_input_ids = f.get('train_trg_input_ids')[:]
    #     valid_src_input_ids = f.get('valid_src_input_ids')[:]
    #     valid_trg_input_ids = f.get('valid_trg_input_ids')[:]

    # with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
    #     data_ = pickle.load(f)
    #     src_word2id = data_['src_word2id']
    #     trg_word2id = data_['trg_word2id']
    #     src_vocab_num = len(src_word2id)
    #     trg_vocab_num = len(trg_word2id)
    #     del data_

    # gc.enable()
    # write_log(logger, "Finished loading data!")

    # dataset = get_dataset(
    #     dataset_name=dataset_name, train_docs=train_docs, wiki_sup=wiki_sup)

    # print(f'Data Examples:')
    # print(dataset['train'][0], '\n', '=' * 100)
    # print(dataset['train'][1], '\n', '=' * 100)

    # dataloaders = get_dataloaders(
    #     dataset=dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     dynamic_shape=True,
    #     max_src_length=MAX_SRC_LENGTH,
    #     max_tgt_length=MAX_TGT_LENGTH,
    #     shuffle=True)

    # if pretrained_ckpt is not None:
    #     log_dir = f'logs/{dataset_name}_plus/docs{train_docs}/'
    # else:
    #     log_dir = f'logs/{dataset_name}/docs{train_docs}/'

    # if os.path.exists(log_dir):
    #     print(f'log_dir \"{log_dir}\" exists. training skipped.')
    #     return
    # os.makedirs(log_dir)