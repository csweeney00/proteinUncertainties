# Plot Results

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn
import torch
import os
import argparse
from scipy.stats import pearsonr
from scipy.stats import multivariate_normal 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_generation import unimodal_1d_regression, unimodal_3d_regression
from data_generation import np_to_torch_dataset
from models import FFNetMeanVar, FFNetMeanVar3D
from data_generation import f, f_var, g, g_noise

# Plots predictions (mean predictions and uncertainty) of certain ensembles
def plot_1d_multi_ensemble_results(args, ensembles, train_loaders, test_loader):

    n = len(ensembles)
    n_samples = [150, 110, 70, 30]
    fig, axes = plt.subplots(4,len(n_samples), figsize = (14,8))
    for i in range(0, len(ensembles), 8):
        ensemble = ensembles[i]
        train_loader = train_loaders[i]
        means_train_i, vars_train_i, means_test_i, vars_test_i = [], [], [], []
        for model in ensemble:
            x_values_train, y_values_train, means, vars = model.get_all_predictions(args, train_loader)

            sorted_train_inds = x_values_train.argsort()
            x_values_train, y_values_train = x_values_train[sorted_train_inds[::-1]], y_values_train[sorted_train_inds[::-1]]
            means, vars = means[sorted_train_inds[::-1]], vars[sorted_train_inds[::-1]]

            means_train_i.append(means)
            vars_train_i.append(vars)

            x_values_test, y_values_test, means, vars = model.get_all_predictions(args, test_loader)

            sorted_test_inds = x_values_test.argsort()
            x_values_test, y_values_test = x_values_test[sorted_test_inds[::-1]], y_values_test[sorted_test_inds[::-1]]
            means, vars = means[sorted_test_inds[::-1]], vars[sorted_test_inds[::-1]]

            means_test_i.append(means)
            vars_test_i.append(vars) 

        x = np.linspace(min(x_values_test), max(x_values_test), 100)

        train_epistemic = np.var(means_train_i, axis = 0)
        test_epistemic = np.var(means_test_i, axis = 0)
        train_aleatoric = np.mean(vars_train_i, axis = 0)
        test_aleatoric = np.mean(vars_test_i, axis = 0)
        total_train_unc = np.mean(np.array(means_train_i)**2, axis=0) - np.mean(np.array(means_train_i), axis=0)**2 + np.mean(vars_train_i, axis=0)
        total_test_unc = np.mean(np.array(means_test_i)**2, axis=0) - np.mean(np.array(means_test_i), axis=0)**2 + np.mean(vars_test_i, axis=0)

        #total_train_unc = train_epistemic + train_aleatoric
        #total_test_unc = test_epistemic + test_aleatoric

        mean_train_pred = np.mean(means_train_i, axis = 0)
        mean_test_pred = np.mean(means_test_i, axis = 0)

        idx = i // 8
        axes[0,idx].plot(x, f(x), color = 'k', label = 'True Mean')
        axes[0,idx].plot(x_values_train, mean_train_pred, color = 'cadetblue', label = 'Pred Mean')
        axes[0,idx].scatter(x_values_train, y_values_train, color = 'steelblue', s = 1 )
        axes[0,idx].fill_between(x = x_values_train, 
                                y1 = mean_train_pred - 1.96*np.sqrt(total_train_unc), 
                                y2 = mean_train_pred + 1.96*np.sqrt(total_train_unc), 
                                alpha = .5, 
                                color = 'cadetblue',
                                label = 'Total Uncertainty (95% CI)')

        axes[1,idx].plot(x, f(x), color = 'k', label = 'True Mean')
        axes[1,idx].plot(x_values_test, mean_test_pred, color = 'cadetblue', label = 'Pred Mean')
        axes[1,idx].scatter(x_values_test, y_values_test, color = 'steelblue', s = 1)
        axes[1,idx].fill_between(x = x_values_test, 
                                y1 = mean_test_pred - 1.96*np.sqrt(total_test_unc), 
                                y2 = mean_test_pred + 1.96*np.sqrt(total_test_unc), 
                                alpha = .5, 
                                color = 'cadetblue',
                                label = 'Total Uncertainty (95% CI)')

        axes[2,idx].plot(x, f(x), color = 'k', label = 'True Mean')
        axes[2,idx].plot(x_values_test, mean_test_pred, color = 'cadetblue', label = 'Pred Mean')
        axes[2,idx].scatter(x_values_test, y_values_test, color = 'steelblue', s = 1)
        axes[2,idx].fill_between(x = x_values_test, 
                                y1 = mean_test_pred - 1.96*np.sqrt(test_epistemic), 
                                y2 = mean_test_pred + 1.96*np.sqrt(test_epistemic), 
                                alpha = .5, 
                                color = 'cadetblue',
                                label = 'Epistemic Uncertainty (95% CI)')

        axes[3,idx].plot(x, f(x), color = 'k', label = 'True Mean')
        axes[3,idx].plot(x_values_test, mean_test_pred, color = 'cadetblue', label = 'Pred Mean')
        axes[3,idx].scatter(x_values_test, y_values_test, color = 'steelblue', s = 1)
        axes[3,idx].fill_between(x = x_values_test, 
                                y1 = mean_test_pred - 1.96*np.sqrt(test_aleatoric), 
                                y2 = mean_test_pred + 1.96*np.sqrt(test_aleatoric), 
                                alpha = .5, 
                                color = 'cadetblue',
                                label = 'Aleatoric Uncertainty (95% CI)')

    axes[0][0].set_ylabel('Train Predictions \n (Total Uncertainty)')
    axes[1][0].set_ylabel('Test Predictions \n (Total Uncertainty)')
    axes[2][0].set_ylabel('Test Predictions \n (Epistemic Uncertainty)')
    axes[3][0].set_ylabel('Test Predictions \n (Aleatoric Uncertainty)')

    for i in range(len(n_samples)):
        lab = 'x (' + str(n_samples[i]) + ' Train Points)'
        axes[3][i].set_xlabel(lab)

    plt.suptitle("Ensemble Performance Against Number of Train Points")

    plt.savefig('ensemble_train_test_performance.png',bbox_inches='tight')

def plot_across_ensemble_error_correlation(args, ensembles, train_loaders, test_loader):
    test_aleatorics = []
    for i in range(len(ensembles)):
        ensemble = ensembles[i]
        vars_test_i = []
        for model in ensemble:
            x_values_test, _, means, vars = model.get_all_predictions(args, test_loader)

            sorted_test_inds = x_values_test.argsort()
            _, vars = means[sorted_test_inds[::-1]], vars[sorted_test_inds[::-1]]

            vars_test_i.append(vars) 

        test_aleatorics.append(np.mean(vars_test_i, axis = 0))

    _, axes = plt.subplots(1,1, figsize = (5,4))
    print([i+1 for i in range(len(ensembles))])
    print(np.mean(test_aleatorics, axis=1))
    axes.plot([i+1 for i in range(len(ensembles))], np.mean(test_aleatorics, axis=1), '-o', color='cadetblue', )

    axes.set_xticks([i+1 for i in range(len(ensembles))])
    axes.set_xticklabels(["120", "110", "100", "90", "80", "70", "60", "50", "40", "30", "20", "10"])
    #axes.set_xticklabels(["50", "40", "30", "20", "10"])
    axes.set_xlabel(r'Num Train Points')
    axes.set_ylabel(r'Aleatoric-Variance Mean Squared Error')
    axes.set_title(r'MSE of Aleatoric Uncertainty and Variance Against Num Train Points on Test Set')

    plt.savefig('across_ensemble_error_correlation.png', bbox_inches='tight')

# Plots the predictions made by the ensembles on train and test set
# Also plots the aleatoric uncertainty across the curve when compared
# to known variance (g_noise)
def plot_3d_multi_ensemble_results(args, ensembles, train_loaders, test_loaders, lengthscales):

    ensemble_train_means, ensemble_test_means, ensemble_train_vars, ensemble_test_vars = [], [], [], []
    ensemble_t_train_values, ensemble_t_test_values, ensemble_z_train_values, ensemble_z_test_values = [], [], [], []

    for i in range(len(ensembles)):
        ensemble = ensembles[i]

        means_train, vars_train, means_test, vars_test = [], [], [], []
        for model in ensemble:
            # The means are actually triples of x,y,z values
            t_values_train, z_values_train, means, vars = model.get_all_predictions(args, train_loaders[i])
            means, vars = model.reshape_preds(means, vars, args.train_size)

            sorted_train_inds = t_values_train.argsort(axis=1)
            ens_means_train = []
            ens_vars_train = []
            sorted_t_values_train = []
            sorted_z_values_train = []
            for j in range(len(sorted_train_inds)):
                ts = sorted_train_inds[i]
                sorted_t_values_train.append(t_values_train[j][ts[::-1]])
                sorted_z_values_train.append(z_values_train[j][ts[::-1]])
                sorted_means, sorted_vars = means[j][ts[::-1]], vars[j][ts[::-1]]

                ens_means_train.append(sorted_means)
                ens_vars_train.append(sorted_vars)
            means_train.append(ens_means_train)
            vars_train.append(ens_vars_train)

            t_values_test, z_values_test, means, vars = model.get_all_predictions(args, test_loaders[i])
            means, vars = model.reshape_preds(means, vars, args.test_size)

            sorted_test_inds = t_values_test.argsort(axis=1)
            ens_means_test = []
            ens_vars_test = []
            sorted_t_values_test = []
            sorted_z_values_test = []
            for j in range(len(sorted_test_inds)):
                ts = sorted_test_inds[j]
                sorted_t_values_test.append(t_values_test[j][ts[::-1]])
                sorted_z_values_test.append(z_values_test[j][ts[::-1]])
                sorted_means, sorted_vars = means[j][ts[::-1]], vars[j][ts[::-1]]

                ens_means_test.append(sorted_means)
                ens_vars_test.append(sorted_vars)
            means_test.append(ens_means_test)
            vars_test.append(ens_vars_test)

        mean_train_pred = np.mean(np.mean(means_train, axis = 0),axis=0)
        mean_test_pred = np.mean(np.mean(means_test, axis = 0), axis=0)
        mean_train_var = np.mean(np.mean(vars_train, axis=0), axis=0)        # Aleatoric Uncertainty
        mean_test_var = np.mean(np.mean(vars_test, axis=0), axis=0)          # Aleatoric Uncertainty
        t_values_train = sorted_t_values_train[0]
        t_values_test = sorted_t_values_test[0]
        z_values_train = sorted_z_values_train[0]
        z_values_test = sorted_z_values_test[0]

        ensemble_train_means.append(mean_train_pred)
        ensemble_test_means.append(mean_test_pred)
        ensemble_train_vars.append(mean_train_var)
        ensemble_test_vars.append(mean_test_var)
        ensemble_t_train_values.append(t_values_train)
        ensemble_t_test_values.append(t_values_test)
        ensemble_z_train_values.append(z_values_train)
        ensemble_z_test_values.append(z_values_test)

    t = np.expand_dims(np.linspace(-1*np.pi, np.pi, args.data_size), 0)
    points = g(t)[0]
    points = np.array(points)

    fig, axes = plt.subplots(1,2, figsize = (10,5), subplot_kw=dict(projection='3d'))
    axes[0].plot(points[:,0], points[:,1], points[:,2], color='k')
    axes[1].plot(points[:,0], points[:,1], points[:,2], color='k')
    for i in range(len(ensembles)):
        mean_train_pred = ensemble_train_means[i]
        z_values_train = ensemble_z_train_values[i]

        axes[0].scatter(mean_train_pred[:,0], mean_train_pred[:,1], mean_train_pred[:,2], color=colour_list[i], label = str(lengthscales[i]))
        axes[1].scatter(mean_train_pred[:,0], mean_train_pred[:,1], mean_train_pred[:,2], color=colour_list[i], label = str(lengthscales[i]))

    axes[1].view_init(elev=90, azim=-90)
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    axes[1].set_zticks([])
    axes[1].set_zticks([], minor=True)

    plt.suptitle("Train Set")
    plt.savefig('train_pred_3d_ensembles.png', bbox_inches='tight')

  
    fig, axes = plt.subplots(1,2, figsize = (10,5), subplot_kw=dict(projection='3d'))
    axes[0].plot(points[:,0], points[:,1], points[:,2], color='k')
    axes[1].plot(points[:,0], points[:,1], points[:,2], color='k')
    for i in range(len(ensembles)):
        mean_test_pred = ensemble_test_means[i]
        z_values_test = ensemble_z_test_values[i]

        axes[0].scatter(mean_test_pred[:,0], mean_test_pred[:,1], mean_test_pred[:,2], color=colour_list[i], label = str(lengthscales[i]))
        axes[1].scatter(mean_test_pred[:,0], mean_test_pred[:,1], mean_test_pred[:,2], color=colour_list[i], label = str(lengthscales[i]))

    axes[1].view_init(elev=90, azim=-90)
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    axes[1].set_zticks([])
    axes[1].set_zticks([], minor=True)

    plt.suptitle("Test Set")
    plt.savefig('test_pred_3d_ensembles.png', bbox_inches='tight')
  
    # project onto first two principal components to visualise variance predictions
    pca = PCA(n_components=2).fit(points)
    points = pca.transform(points)
    x, y = points[:,0], points[:,1]
    t = t[0]
    cov = lambda x,y : 1**2 * np.exp(-(x-y)**2 / (2*(0.1)**2)) + np.sin(x/3)**2 * (x == y)
    #cov = lambda x,y : 1**2 * np.exp(-(x-y)**2 / (2*(0.1)**2)) + 0.2*np.abs(x-y)*(np.abs(x-y) > np.pi) + np.sin(x/3)**2 * (x == y)
    noise = [cov(t1,t1) for t1 in t]

    fig, axes = plt.subplots(1,2, figsize = (8,4))
    axes[0].plot(t, 1.96*np.sqrt(noise), color='k', label="True Variance")
    axes[0].plot(t, -1.96*np.sqrt(noise), color='k')
    axes[1].plot(t, 1.96*np.sqrt(noise), color='k', label="True Variance")
    axes[1].plot(t, -1.96*np.sqrt(noise), color='k')
    for i in range(len(ensembles)):
        mean_train_pred = ensemble_train_means[i]
        mean_test_pred = ensemble_test_means[i]
        mean_train_var = ensemble_train_vars[i]
        mean_test_var = ensemble_test_vars[i]
        t_values_train = ensemble_t_train_values[i]
        t_values_test = ensemble_t_test_values[i]
        z_values_train = ensemble_z_train_values[i]
        z_values_test = ensemble_z_test_values[i]

        #mean_train_var = mean_train_var - (np.max(mean_train_var) - np.max(noise))
        #mean_test_var = mean_test_var - (np.max(mean_test_var) - np.max(noise))

        z_values_train = pca.transform(z_values_train)
        mean_train_pred = pca.transform(mean_train_pred)
        z_values_test = pca.transform(z_values_test)
        mean_test_pred = pca.transform(mean_test_pred)

        axes[0].plot(t_values_train, 1.96*mean_train_var[:,-1].flatten(), color=colour_list[i], label = str(lengthscales[i]))
        axes[1].plot(t_values_test, 1.96*mean_test_var[:,-1].flatten(), color=colour_list[i], label = str(lengthscales[i])) 
        axes[0].plot(t_values_train, -1.96*mean_train_var[:,-1].flatten(), color=colour_list[i])
        axes[1].plot(t_values_test, -1.96*mean_test_var[:,-1].flatten(), color=colour_list[i])        

    axes[0].set_title('Train Set Aleatoric Estimates')
    axes[0].set_xlabel('t', loc="right", fontsize=12)
    axes[0].set_ylabel('')
    axes[0].set_ylabel(r'$\pm$1.96 $\times$ Aleatoric Uncertainty', fontsize=11)
    axes[1].set_title('Test Set Aleatoric Estimates')
    axes[1].set_xlabel('t', loc="right", fontsize=12)
    #axes[1].set_ylabel('Aleatoric Uncertainty')

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.spines['bottom'].set_position('zero')
        a.xaxis.set_label_coords(1, 0.45)

    axes[1].legend(loc = 'upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig('train_test_epi_ale_pca_3d_ensembles.png', bbox_inches='tight')

    
    true_var_mean = np.mean(noise)
    fig, axes = plt.subplots(1,2, figsize=(8, 4))
    maes_train = []
    maes_test = []
    for i in range(len(ensembles)):
        mean_train_var = ensemble_train_vars[i][:,-1]
        mean_test_var = ensemble_test_vars[i][:,-1]

        aligned_train_var = mean_train_var - (np.mean(mean_train_var) - true_var_mean)
        aligned_test_var = mean_test_var - (np.mean(mean_test_var) - true_var_mean)

        absolute_train_errors = np.abs(noise - aligned_train_var)
        absolute_test_errors = np.abs(noise - aligned_test_var)
        maes_train.append(np.mean(absolute_train_errors))
        maes_test.append(np.mean(absolute_test_errors))
        

    axes[0].plot(lengthscales, maes_train, color=colour_list[4])
    axes[1].plot(lengthscales, maes_test, color=colour_list[4])

    
    axes[0].set_xlabel("Lengthscale")
    axes[0].set_ylabel("Mean Absolute Error")
    axes[1].set_xlabel("Lengthscale")
    axes[0].set_title("Train Set")
    axes[1].set_title("Test Set")

    axes[1].legend(loc = 'upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig('aumac_3d_ensembles.png', bbox_inches='tight')

    print(lengthscales, maes_train, maes_test)

def plot_covariance_matrices(args, lengthscales):
    t_values = np.linspace(-1*np.pi, np.pi, args.data_size)
    fig, axes = plt.subplots(2,4, figsize=(25,10))
    for i in range(0,len(lengthscales),2):
        lengthscale = lengthscales[i]
        cov = lambda x,y : args.sigma**2 * np.exp(-(x-y)**2 / (2*lengthscale**2)) + np.sin(x/3)**2 * (x == y)
        cov_matrix = np.array([[cov(t1, t2) for t2 in t_values] for t1 in t_values])

        im = axes[0,i//2].imshow(cov_matrix, cmap='coolwarm', interpolation='nearest', vmin = -(2 + args.sigma**2), vmax = 2 + args.sigma**2, aspect='auto')
        axes[0,i//2].spines[:].set_visible(False)
        axes[0,i//2].set_title("Lengthscale:"+str(lengthscale), fontsize=18)
        axes[0,i//2].set_xticklabels([])
        axes[0,i//2].set_yticklabels([])
    for i in range(0,len(lengthscales),2):
        lengthscale = lengthscales[i]
        cov = lambda x,y : args.sigma**2 * np.exp(-(x-y)**2 / (2*lengthscale**2)) - 0.3*np.abs(x-y)*(np.abs(x-y) > np.pi) + np.sin(x/3)**2 * (x == y)
        cov_matrix = np.array([[cov(t1, t2) for t2 in t_values] for t1 in t_values])

        im = axes[1,i//2].imshow(cov_matrix, cmap='coolwarm', interpolation='nearest', vmin = -(2 + args.sigma**2), vmax = 2 + args.sigma**2, aspect='auto')
        axes[1,i//2].spines[:].set_visible(False)
        #axes[1,i//2].set_title("Lengthscale:"+str(lengthscale))
        axes[1,i//2].set_xticklabels([])
        axes[1,i//2].set_yticklabels([])
    
    axes[0,0].set_ylabel("RBF Kernel", fontsize=16)
    axes[1,0].set_ylabel("Altered RBF Kernel", fontsize=16)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=16) 

    plt.suptitle("Sample of Covariance Matrices Used in Training", fontsize=30)
    plt.savefig('covariance_matrix_plots.png', bbox_inches='tight')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate Loops Distributions')
    argparser.add_argument('--batch_size', type=int, default=1)
    argparser.add_argument('--dry_run', type=bool, default=False)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--data_seed', type=int, default=42)
    argparser.add_argument('--train_size', type=int, default=100)
    argparser.add_argument('--device')
    argparser.add_argument('--three_d', type=bool, default=False)
    argparser.add_argument('--data_size', type = int, default=200)
    argparser.add_argument('--test_size', type = int, default=1)
    argparser.add_argument('--inner_width', type = int, default=150)
    argparser.add_argument('--sigma', type=float, default=1)
    argparser.add_argument('--band_size', type = int, default=1)
    args = argparser.parse_args()
    args.device = torch.device("cpu")

    colour_list = ['#ffffd9', '#edf8b1', '#ddeda0', '#c7e9b4', '#9cd9a5', '#7fcdbb', '#51c0b3', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']
    colour_list = colour_list[::-1]

    print("============== Making training data ==============")

    np.random.seed(args.data_seed)
    torch.manual_seed(args.data_seed)    

    lengthscales = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 2]

    if args.three_d:
        train_loaders = []
        test_loaders = []
        for lengthscale in lengthscales:
            t_train, z_train = unimodal_3d_regression(n=args.train_size, data_size=args.data_size, sigma = args.sigma, lengthscale=lengthscale, epi='LOW', ale='LOW', hetero=True)

            scaler = StandardScaler()
            scaler.fit(t_train)

            train_loader = np_to_torch_dataset(t_train, z_train, args.batch_size, None)
            train_loaders.append(train_loader)

            t_test, z_test = unimodal_3d_regression(n=args.test_size, data_size=args.data_size, sigma = args.sigma, lengthscale=lengthscale, epi='LOW', ale='LOW', hetero=True, test=True)
            test_loader = np_to_torch_dataset(t_test, z_test, args.batch_size, None)
            test_loaders.append(test_loader)
    else:
        train_loaders = []
        train_samples = [150,145,140,135,130,125,120, 115,110, 105,100, 95,90, 85,80, 75,70, 65,60, 55,50, 45,40, 35,30, 25,20, 15,10]
        for n in train_samples:
            x_train, y_train = unimodal_1d_regression(n=n, epi='HIGH', ale='LOW', hetero=True)
            train_val_loader = np_to_torch_dataset(np.expand_dims(x_train, -1), y_train, args.batch_size, None)
            train_loaders.append(train_val_loader)

        x_test, y_test = unimodal_1d_regression(n=150, epi='LOW', ale='LOW', hetero=True)
        test_loader = np_to_torch_dataset(np.expand_dims(x_test, -1), y_test, 50, None)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("============== Plotting results ==============")

    if args.three_d:
        dirs = ['./models/models_3d_ls_' + str(lengthscale) for lengthscale in lengthscales]
        
        ensembles = []

        for i in range(len(dirs)):
            dir = dirs[i]
            ensemble = []
            for file in os.listdir(dir):
                model = FFNetMeanVar3D(args.data_size, 3*args.data_size, args.data_size, args.batch_size, args.band_size, args.inner_width)
                model.load_state_dict(torch.load(dir + '/' + file))
                ensemble.append(model)
            ensembles.append(ensemble)

        plot_covariance_matrices(args, lengthscales)
        plot_3d_multi_ensemble_results(args, ensembles, train_loaders, test_loaders, lengthscales)

    else:

        dirs = ['./models/gap_150_models',
                './models/gap_145_models',
                './models/gap_140_models',
                './models/gap_135_models',
                './models/gap_130_models',
                './models/gap_125_models',
                './models/gap_120_models',
                './models/gap_115_models',
                './models/gap_110_models',
                './models/gap_105_models',
                './models/gap_100_models',
                './models/gap_95_models', 
                './models/gap_90_models',
                './models/gap_85_models',
                './models/gap_80_models',
                './models/gap_75_models',
                './models/gap_70_models',
                './models/gap_65_models',
                './models/gap_60_models', 
                './models/gap_55_models',
                './models/gap_50_models',
                './models/gap_45_models',
                './models/gap_40_models', 
                './models/gap_35_models',
                './models/gap_30_models',
                './models/gap_25_models',
                './models/gap_20_models', 
                './models/gap_15_models',
                './models/gap_10_models']
        
        ensembles = []

        for dir in dirs:
            ensemble = []
            for file in os.listdir(dir):
                model = FFNetMeanVar(1,1)
                model.load_state_dict(torch.load(dir + '/' + file))
                ensemble.append(model)
            ensembles.append(ensemble)

        #plot_ensemble_epi_ale_error_correlation(args, ensembles, train_loaders, test_loader)
        plot_1d_multi_ensemble_results(args, ensembles, train_loaders, test_loader)
        #plot_across_ensemble_error_correlation(args, ensembles, train_loaders, test_loader)
