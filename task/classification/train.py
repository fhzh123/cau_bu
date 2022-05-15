# Import modules
import os
import gc
import h5py
import pickle
import logging
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.dataset import CustomDataset
# from model.custom_transformer.transformer import Transformer
# from model.plm.bart import Bart
from model.loss import label_smoothing_loss
from optimizer.utils import shceduler_select, optimizer_select
from transformers import BertForSequenceClassification
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
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.task, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_name = f'processed_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.pkl'
    else:
        save_name = f'processed_{args.tokenizer}.pkl'

    with open(os.path.join(save_path, save_name), 'rb') as f:
        data_ = pickle.load(f)
        train_src_input_ids = data_['train_src_input_ids']
        train_trg_labels = data_['train_trg_labels']
        valid_src_input_ids = data_['valid_src_input_ids']
        valid_trg_labels = data_['valid_trg_labels']
        del data_

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    # 2) Dataloader setting
    dataset_dict = {
        'train': CustomDataset(tokenizer=args.tokenizer, input_ids_list=train_src_input_ids['input_ids'], 
                               label_list=train_trg_labels, attention_mask_list=train_src_input_ids['attention_mask'], 
                               token_type_ids_list=train_src_input_ids['token_type_ids'], 
                               min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(tokenizer=args.tokenizer, input_ids_list=valid_src_input_ids['input_ids'], 
                               label_list=valid_trg_labels, attention_mask_list=valid_src_input_ids['attention_mask'], 
                               token_type_ids_list=valid_src_input_ids['token_type_ids'], 
                               min_len=args.min_len, max_len=args.max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')
    if args.model_type == 'custom_transformer':
        model = Transformer(src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                            pad_idx=args.pad_id, bos_idx=args.bos_id, eos_idx=args.eos_id,
                            d_model=args.d_model, d_embedding=args.d_embedding, n_head=args.n_head,
                            dim_feedforward=args.dim_feedforward,
                            num_common_layer=args.num_common_layer, num_encoder_layer=args.num_encoder_layer,
                            num_decoder_layer=args.num_decoder_layer,
                            src_max_len=args.src_max_len, trg_max_len=args.trg_max_len,
                            dropout=args.dropout, embedding_dropout=args.embedding_dropout,
                            trg_emb_prj_weight_sharing=args.trg_emb_prj_weight_sharing,
                            emb_src_trg_weight_sharing=args.emb_src_trg_weight_sharing, 
                            variational=args.variational, parallel=args.parallel)
        tgt_subsqeunt_mask = model.generate_square_subsequent_mask(args.trg_max_len - 1, device)
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=10)
    model = model.to(device)
    
    # 2) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        checkpoint = torch.load(os.path.join(args.model_save_path, 'checkpoint.pth.tar'))
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Traing start!')

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        start_time_e = time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                write_log(logger, 'Validation start...')
                val_loss = 0
                val_acc = 0
                model.eval()
            for i, (src_sequence, src_segment, src_att, trg_label) in enumerate(tqdm(dataloader_dict[phase], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

                # Optimizer setting
                optimizer.zero_grad(set_to_none=True)

                # Input, output setting
                src_sequence = src_sequence.to(device, non_blocking=True)
                src_segment = src_segment.to(device, non_blocking=True)
                src_att = src_att.to(device, non_blocking=True)
                trg_label = trg_label.to(device, non_blocking=True)

                # Train
                if phase == 'train':

                    with autocast():
                        predicted = model(src_sequence)
                        predicted = predicted.view(-1, predicted.size(-1))
                        nmt_loss = label_smoothing_loss(predicted, trg_sequence_gold, trg_pad_idx=args.pad_id)
                        total_loss = nmt_loss + kl

                    scaler.scale(total_loss).backward()
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(nmt_loss)

                    # Print loss value only training
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        acc = (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)
                        iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.3f | train_acc:%03.2f%% | learning_rate:%1.6f | spend_time:%02.2fmin" % \
                            (epoch, i, len(dataloader_dict['train']), 
                            total_loss.item(), acc*100, optimizer.param_groups[0]['lr'], 
                            (time() - start_time_e) / 60)
                        write_log(logger, iter_log)
                        freq = 0
                    freq += 1

                # Validation
                if phase == 'valid':
                    with torch.no_grad():
                        predicted, kl = model(src_sequence, trg_sequence, 
                                              non_pad_position=non_pad, tgt_subsqeunt_mask=tgt_subsqeunt_mask)
                        nmt_loss = F.cross_entropy(predicted, trg_sequence_gold)
                        total_loss = nmt_loss + kl
                    val_loss += total_loss.item()
                    val_acc += (predicted.max(dim=1)[1] == trg_sequence_gold).sum() / len(trg_sequence_gold)

            if phase == 'valid':

                if args.scheduler == 'reduce_valid':
                    scheduler.step(val_loss)
                if args.scheduler == 'lambda':
                    scheduler.step()

                val_loss /= len(dataloader_dict[phase])
                val_acc /= len(dataloader_dict[phase])
                write_log(logger, 'Validation Loss: %3.3f' % val_loss)
                write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))
                save_path = os.path.join(args.model_save_path, args.task, args.data_name, args.tokenizer)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_file_name = os.path.join(save_path, 
                                              f'checkpoint_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}_v_{args.variational}_p_{args.parallel}.pth.tar')
                if val_acc > best_val_acc:
                    write_log(logger, 'Checkpoint saving...')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict()
                    }, save_file_name)
                    best_val_acc = val_acc
                    best_epoch = epoch
                else:
                    else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc.item()*100, 2)})% is better...'
                    write_log(logger, else_log)

    # 3) Print results
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc.item(), 2)}')

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