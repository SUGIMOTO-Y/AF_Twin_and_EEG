import argparse
from utilities.Load_Dataloader import *
from utilities.Evaluation import *
from utilities.set_model import *
from utilities.set_args import *
from utilities.Finetuning import *
from Model.Adapter import *
from Model.Model import *

def main_process(args_): 
    args = set_args(args_)
    args, test_eeg, test_img, train_eeg = load_dataloader(args)
    model = set_model(args)
    if args.finetuning_flag:
        model = load_pretrain_weight(args, model)
        model = Finetuning(args, train_eeg, model)
    Acc = CalculateAcc(args, test_eeg, model, test_img)
    print(f'subject: {args.TargetSub} | Top-1 Accuracy: {Acc[0]} |Top-5 Accuracy:{Acc[1]}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, help='The target subject (from 1 to 10)')
    parser.add_argument('-a', '--adapter_mode', type=int, help='Types of AF: (0.AF-Twin, 1.AF-EEG)')
    parser.add_argument('-f', '--finetuning', action='store_true', help='Finetuning on the target data or not')
    args = parser.parse_args()
    main_process(args)