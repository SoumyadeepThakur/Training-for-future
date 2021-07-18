import sys

sys.path.append("../GI")

import argparse
from trainer_new import *
from preprocess import *
import random
import os
import numpy as np

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):

    seed_torch(args.seed)
    
    args.device = "cpu"
    args.data = "moons"
    args.bs = 100
    args.train_algo = "grad-int"
    args.early_stopping = True
    args.plot = True
    if args.preprocess:
    
        load_moons(11)

    if args.model=='cida':
        import main_moons
    else:
        trainer = GradRegTrainer(args)
        
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--model', default='GI', help="Model")
    parser.add_argument('--preprocess', action='store_true', help="Do we pre-process the data?")
    parser.add_argument('--epoch_finetune',default=25, help="Needs to be int, number of epochs for transformer/ordinal classifier",type=int)
    parser.add_argument('--epoch_classifier',default=35, help="Needs to be int, number of epochs for classifier",type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--delta', default=0.0, type=float)
    parser.add_argument('--max_k', type=int)
    parser.add_argument('--multistep',action='store_true')
    parser.add_argument('--ensemble',action='store_true')
    parser.add_argument('--encoder',action='store_true',help="Do we use encodings?")
    parser.add_argument('--pretrained',action='store_true')
    args = parser.parse_args()
    main(args)
