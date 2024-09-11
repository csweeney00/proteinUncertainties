# 1D Regression Task

from data_generation import unimodal_1d_regression
from data_generation import np_to_torch_dataset
from models import FFNetMeanVar

import numpy as np
import torch
import argparse
import os

def ensemble(args, train_loader):
    ensemble = []
    ensemble_seeds = np.random.randint(low = 1, high = 1000, size = args.num_models)
    for i in range(args.num_models):
        # Changing the random seeds for each model so that each ensemble has a different randomisation.
        np.random.seed(ensemble_seeds[i])
        torch.manual_seed(ensemble_seeds[i])

        model = FFNetMeanVar(1,1)
        model.train_model(args, train_loader)
        ensemble.append(model)
    return ensemble

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate Loops Distributions')
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.1)
    argparser.add_argument('--wd', type=float, default=0.01)
    argparser.add_argument('--num_epochs', type=int, default=100)
    argparser.add_argument('--epoch_print_rate', type=int, default='1')
    argparser.add_argument('--early_stopping', type=bool, default=False)
    argparser.add_argument('--hp_search', type=bool, default=False)
    argparser.add_argument('--dry_run', type=bool, default=False)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--data_seed', type=int, default=42)
    argparser.add_argument('--num_models', type=int, default=1)
    argparser.add_argument('--ensemble_idx', type=int, default=0)
    argparser.add_argument('--device')
    argparser.add_argument('--train_size', type = int, default=100)
    args = argparser.parse_args()
    args.device = torch.device("cpu")

    print("============== Making training data ==============")
    np.random.seed(args.data_seed)      # We keep the seeds that generate the data fixed across all runs
    torch.manual_seed(args.data_seed)

    x_train, y_train = unimodal_1d_regression(n=int((args.train_size*0.7)//1), epi='HIGH', ale='LOW', hetero=True)
    x_val, y_val = unimodal_1d_regression(n=int((args.train_size*0.3)//1), epi='HIGH', ale='LOW', hetero=True)
    x_test, y_test = unimodal_1d_regression(n=150, epi='LOW', ale='LOW', hetero=True)
    
    train_loader = np_to_torch_dataset(np.expand_dims(x_train, -1), y_train, args.batch_size, None)
    val_loader = np_to_torch_dataset(np.expand_dims(x_val, -1), y_val, 50, None)
    train_val_loader = np_to_torch_dataset(np.expand_dims(np.hstack([x_train, x_val]), -1), np.hstack([y_train, y_val]), args.batch_size, None)
    test_loader = np_to_torch_dataset(np.expand_dims(x_test, -1), y_test, 50, None)

    if args.hp_search:
        print("============== Hyperparameter Search ==============")

        og_num_models = args.num_models     # Just train one model for the hyperparameter tuning stage
        args.num_models = 1
        lr_options = [0.001, 0.01, 0.05, 0.1, 0.5]
        num_epochs_options = [400, 500, 600, 700, 800, 900]

        if args.early_stopping:
            hyper_params = {'lr':lr_options}
        else:
            hyper_params = {'lr':lr_options, 'num_epochs':num_epochs_options}
        num_hyper_param_options = np.prod([len(options) for options in hyper_params.values()])

        hp_losses = []

        for i in range(num_hyper_param_options):
            print("Hyper param option", i)
            args.lr = lr_options[i // len(num_epochs_options)]
            if not args.early_stopping:
                args.num_epochs = num_epochs_options[i % len(num_epochs_options)]
            model = ensemble(args, train_loader)[0]

            hp_losses.append(model.eval_model(args, val_loader, torch.nn.GaussianNLLLoss(), need_var = True))

        print(hp_losses)
        min_idx = np.argmin(hp_losses)
        args.lr = lr_options[min_idx // len(lr_options)]
        print('lr', args.lr)
        if not args.early_stopping:
            args.num_epochs = num_epochs_options[min_idx % len(lr_options)]
            print('epochs', args.num_epochs)

        args.num_models = og_num_models
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


    print("============== Training Model ==============")
    

    ensemble = ensemble(args, train_val_loader)

    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists(os.path.join('models', 'gap_' + str(args.train_size) + '_models')):
        os.mkdir('models/gap_' + str(args.train_size) + '_models')

    for i in range(len(ensemble)):
        model = ensemble[i]
        torch.save(ensemble[0].state_dict(), 'models/gap_' + str(args.train_size) + '_models/ensemble_model_' + str(args.ensemble_idx) + '_' + str(i) + '.pt')
    
    
