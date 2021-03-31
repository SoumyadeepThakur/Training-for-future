'''Main script to run stuff

[description]
'''

import argparse
from trainer_GI import *
from preprocess import *
import random
import os

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
    if args.use_cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    if args.preprocess:
        print("Preprocessing")
        if args.data == "mnist":
            load_Rot_MNIST(args.encoder)
        if args.data == "moons":
            load_moons()
        if args.data == "sleep":
            load_sleep()
        if args.data == "cars":
            load_comp_cars()
        if args.data == "house":
            load_house_price()
        if args.data == "house_classifier":
            load_house_price_classification()             
    if args.train_algo == "transformer":
        trainer = TransformerTrainer(args)
    elif args.train_algo == "grad":
        trainer = CrossGradTrainer(args)
    elif args.train_algo == "meta":
        trainer = MetaTrainer(args)
    elif args.train_algo == "grad_reg":
        trainer = GradRegTrainer(args)
    elif args.train_algo == "hybrid":
        trainer = HybridTrainer(args)
    
    trainer.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--train_algo',help="String, needs to be one of grad or transformer")
    parser.add_argument('--data',help="String, needs to be one of mnist, sleep, moons, cars")
    parser.add_argument('--epoch_finetune',default=20,help="Needs to be int, number of epochs for transformer/ordinal classifier",type=int)
    parser.add_argument('--epoch_classifier',default=20,help="Needs to be int, number of epochs for classifier",type=int)
    parser.add_argument('--bs',default=100,help="Batch size",type=int)
    parser.add_argument('--early_stopping',action='store_true',help="Early Stopping for finetuning")
    parser.add_argument('--use_cuda',action='store_true',help="Should we use a GPU")
    parser.add_argument('--preprocess',action='store_true',help="Do we pre-process the data?")
    parser.add_argument('--encoder',action='store_true',help="Do we use encodings?")
    parser.add_argument('--delta',default=0.0,type=float)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args)
