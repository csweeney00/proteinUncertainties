# 3D Regression Task

from data_generation import unimodal_3d_regression
from data_generation import np_to_torch_dataset
from models import FFNetMeanVar3D

import numpy as np
import torch
import argparse
import os
import wandb
from sklearn.preprocessing import StandardScaler

def ensemble(args, train_loader, val_loader):
    ensemble = []
    ensemble_seeds = np.random.randint(low = 1, high = 1000, size = args.num_models)
    for i in range(args.num_models):
        # Changing the random seeds for each model so that each ensemble has a different randomisation.
        np.random.seed(ensemble_seeds[i])
        torch.manual_seed(ensemble_seeds[i])

        model = FFNetMeanVar3D(args.data_size, 3*args.data_size, args.data_size, args.batch_size, args.band_size, args.inner_width)
        model.train_model(args, train_loader, val_loader)
        ensemble.append(model)
    return ensemble

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate Loops Distributions')
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.1)
    argparser.add_argument('--num_epochs', type=int, default=100)
    argparser.add_argument('--epoch_print_rate', type=int, default='1')
    argparser.add_argument('--early_stopping', type=bool, default=False)
    argparser.add_argument('--hp_search', type=bool, default=False)
    argparser.add_argument('--dry_run', type=bool, default=False)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--num_models', type=int, default=1)
    argparser.add_argument('--data_seed', type=int, default=42)
    argparser.add_argument('--ensemble_idx', type=int, default=0)
    argparser.add_argument('--device')
    argparser.add_argument('--train_size', type = int, default=100)
    argparser.add_argument('--data_size', type = int, default=200)
    argparser.add_argument('--band_size', type = int, default=2)
    argparser.add_argument('--inner_width', type=int, default=150)
    argparser.add_argument('--sigma', type=float, default=1)            # We don't alter sigma as this doesn't alter the covariance strength
    argparser.add_argument('--lengthscale_idx', type=int, default=2)
    args = argparser.parse_args()
    args.device = torch.device("cpu")

    lengthscales = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 2]
    lengthscale = lengthscales[args.lengthscale_idx]

    print("============== Making training data ==============")

    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)

    t_train, z_train = unimodal_3d_regression(n=args.train_size, data_size=args.data_size, sigma = args.sigma, lengthscale=lengthscale, epi='LOW', ale='LOW', hetero=True)
    t_val, z_val = unimodal_3d_regression(n=int((args.train_size*0.3)//1), data_size=args.data_size, sigma = args.sigma, lengthscale=lengthscale, epi='LOW', ale='LOW', hetero=True)
    t_test, z_test = unimodal_3d_regression(n=args.train_size, data_size=args.data_size, sigma = args.sigma, lengthscale=lengthscale, epi='LOW', ale='LOW', hetero=True)

    # Scaling didn't affect predictions so ignore
    scaler = StandardScaler()
    scaler.fit(t_train)
    
    train_loader = np_to_torch_dataset(t_train, z_train, args.batch_size, None)
    val_loader = np_to_torch_dataset(t_val, z_val, args.batch_size, None)
    test_loader = np_to_torch_dataset(t_test, z_test, args.batch_size, None)    
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hp_search:
        print("============== Hyperparameter Search ==============")

        og_num_models = args.num_models     # Just train one model for the hyperparameter tuning stage
        args.num_models = 1
        lr_options = [0.001, 0.01, 0.05, 0.1, 0.5]
        num_epochs_options = [200, 300, 400, 500, 600, 800, 900]

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

            hp_losses.append(model.eval_model(args, val_loader, torch.nn.GaussianNLLLoss()))

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

    wandb.init(
    # set the wandb project where this run will be logged
    project="Toy-Parametric-Regression",
    entity = 'csweeney00',
    # track hyperparameters and run metadata
    config={
    "dataset": "3d-regression",
    "epochs": args.num_epochs,
    "num_models": args.num_models
    }
    )
    
    ensemble = ensemble(args, train_loader, val_loader)

    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists(os.path.join('models', 'models_3d_ls_'+str(lengthscale))):
        os.mkdir('models/models_3d_ls_'+str(lengthscale))

    for i in range(len(ensemble)):
        model = ensemble[i]
        torch.save(ensemble[0].state_dict(), 'models/models_3d_ls_'+str(lengthscale)+'/ensemble_model_' + str(args.ensemble_idx) + '_' + str(i) + '.pt')

    wandb.finish()

    
