# Import modules
import os
import sys
import tqdm
import logging
import argparse
# Import PyTorch
import torch.nn.functional as F

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_check(args):
    # Preprocessing Path Checking
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    if not os.path.exists(os.path.join(args.preprocess_path, args.task)):
        os.mkdir(os.path.join(args.preprocess_path, args.task))

    if not os.path.exists(os.path.join(args.preprocess_path, args.task, args.data_name)):
        os.mkdir(os.path.join(args.preprocess_path, args.task, args.data_name))

    # Model Checkpoint Path Checking
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    if not os.path.exists(os.path.join(args.model_save_path, args.task)):
        os.mkdir(os.path.join(args.model_save_path, args.task))

    if not os.path.exists(os.path.join(args.model_save_path, args.task, args.data_name)):
        os.mkdir(os.path.join(args.model_save_path, args.task, args.data_name))

    # Testing Results Path Checking
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)


def write_log(logger, message):
    if logger:
        logger.info(message)