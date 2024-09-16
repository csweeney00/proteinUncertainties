# Loops Distributions vs Experimental Flexibility.

import pandas as pd
import numpy as np
import os
import time
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ranksums
from scipy.stats import norm
from scipy.stats import uniform
import statsmodels.api as sm
from scipy.interpolate import CubicSpline
import scipy.integrate as integrate
from collections import Counter
import editdistance

from unipressed import IdMappingClient # maps pdb codes to uniprot identifiers
from Bio.PDB import PDBParser
from Bio.PDB.Dice import extract

from build_loop_dists import get_clusters, get_cluster_means_and_covar, get_conformation_means_and_var, get_all_confs

def link_to_pdb(link):
    return link[-8:-4]

def get_pdbs_from_loop_conformation(loop_df):
    loop_codes = loop_df.aa_seq
    loop_conformations = loop_df.conformation_indxs
    loop_to_conf_dict = {loop_codes.iloc[i] : loop_conformations.iloc[i] for i in range(len(loop_codes))}

    # Key is loop sequence, value is list of pdbs codes for each conformation
    loop_to_conf_pdb_dict = {}
    missed=[]

    dataset_filepath = "./loopsDatasets"
    for file in os.listdir(dataset_filepath):
        dataset = pd.read_pickle(dataset_filepath + "/" + file)
        dataset = dataset[dataset.index.isin(loop_codes)]
        for index, entry in dataset.iterrows():
            idxs = list(set([int(x) for x in loop_to_conf_dict[index]]))
            try:
                loop_to_conf_pdb_dict[index] = np.array(list(map(link_to_pdb, entry[2])))[idxs]
            except:
                missed.append(index)

    return loop_to_conf_pdb_dict, missed

def get_conformation_chain_res(loop_seq, conformation_idx):
    dataset_filepath = "./loopsDatasets"
    found = False
    for file in os.listdir(dataset_filepath):
        dataset = pd.read_pickle(dataset_filepath + "/" + file)
        if loop_seq in dataset.index:
            found = True
            break
    
    if found == False:
        return -1
    
    entry = dataset.loc[loop_seq]
    chain = entry.chains[conformation_idx]
    res = entry.resis[conformation_idx]

    return chain, res

def get_loop_n_conformation_clusters(loop_seq):
    dataset_filepath = "./loopsDatasets"
    found = False
    for file in os.listdir(dataset_filepath):
        dataset = pd.read_pickle(dataset_filepath + "/" + file)
        if loop_seq in dataset.index:
            found = True
            break
    
    if found == False:
        return -1
    
    entry = dataset.loc[loop_seq]
    n_conformations = entry.n_structures
    n_cluster_conformations = entry.loc['n_conformations_1.25']

    return n_conformations, n_cluster_conformations

def get_protein(uniprot_accession):
    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
    url = f"{api_endpoint}{uniprot_accession}"  # Construct the URL for API

    try:
        # Use a timeout to handle potential connection issues
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            # Raise an exception for better error handling
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

def get_plddt(uniprot_accession, loop_seq, conformation_idx):
    protein_info = get_protein(uniprot_accession)

    if protein_info:
        # Extract PDB and PNG URLs
        pdb_url = protein_info[0].get('pdbUrl')

        if pdb_url:
            # Retrieve the PDB data from the URL
            pdb_data = requests.get(pdb_url).text
            # Save the file so that PDBParser can read it
            pdb_file_name = uniprot_accession + ".pdb"
            pdb_file = open(pdb_file_name, "w")
            pdb_file.write(pdb_data)
            pdb_file.close()

            parser = PDBParser()
            structure = parser.get_structure(uniprot_accession, pdb_file_name)

            try:
                chain, res = get_conformation_chain_res(loop_seq, conformation_idx)
            except:
                print("couldnt get plddt for", loop_seq)
                raise Exception("Failed to find loop sequence")

            start_res, stop_res = res[0], res[-1]
            extract(structure, chain, start_res, stop_res, pdb_file_name)

            constricted_structure = parser.get_structure(uniprot_accession, pdb_file_name)

            plddts = []
            for model in constricted_structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.name == 'CA':
                                atom_plddt = atom.get_bfactor()
                                plddts.append(atom_plddt)
            return np.array(plddts)

        else:
            print("Failed to retrieve PDB URL.")
    else:
        print("Failed to retrieve protein information.")

def get_b_factors(loop_seq, conformation_idx):
    # loop pdbs are not stored on cluster node
    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/Loops'
    pdb_file_name = vols_home + '/loopsPDBs/' + loop_seq + '_' + str(conformation_idx) + '.pdb'
    parser = PDBParser()
    structure = parser.get_structure(loop_seq, pdb_file_name)

    b_factors = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == 'CA':
                        atom_plddt = atom.get_bfactor()
                        b_factors.append(atom_plddt)
    return np.array(b_factors)

def get_empirical_variance(aa_seq):
    try:
        n_conformations, n_cluster_conformations = get_loop_n_conformation_clusters(aa_seq)
    except:
        print(aa_seq)
        raise Exception("Failed to find loop sequence")
    
    loops = get_all_confs(aa_seq, n_conformations)
    try:
        _, cov = get_conformation_means_and_var(loops, aa_seq)
    except:     # Exception will occur if there is a missing residue. I don't know wy it's still here bc I though I filtered them out previously
        cov = None

    if cov is not None:
        cov = cov[2:-2]
    else:
        print(aa_seq)
    return cov

# We don't obtain the actual aleatoric uncertainty which is the average of the
# variance predictions of each ensemble. Instead we use the individual variance
# predictions to have a larger sample size.
def get_predictive_variance(aa_seq):
    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/ensemble_results'
    # should be 5 x 15
    vars = [np.load(os.path.join(vols_home, 'ale_uncertainty_'+str(i+1)+'_test.npy'), allow_pickle=True).item()[aa_seq].flatten() for i in range(5)]
    return vars

def plot_flexibility_measures(sequence, plddts, b_factors, emp_vars, pred_vars):

    colour_list = ['#f1b6da', '#fdae61', '#abdda4', '#3288bd']

    fig, axes = plt.subplots(1,2, figsize = (10,5))

    plddts = [smoothing(plddt)[1] for plddt in plddts]
    b_factors = [smoothing(b_factor)[1] for b_factor in b_factors]
    emp_vars = [smoothing(emp_var)[1] for emp_var in emp_vars]
    pred_vars = [smoothing(pred_var)[1] for pred_var in pred_vars] 

    xs = [i+1 for i in range(len(pred_vars[0]))]

    print(np.array(plddts).shape, np.array(b_factors).shape, np.array(emp_vars).shape, np.array(pred_vars).shape)

    axes[0].plot(xs, np.mean(emp_vars,axis=0), color=colour_list[2], label = 'Empirical Variance')
    axes[0].plot(xs, np.mean(pred_vars,axis=0), color=colour_list[3], label = 'Aleatoric Uncertainty')
    axes[1].plot(xs, np.mean(emp_vars,axis=0), color=colour_list[2], label = 'Empirical Variance')
    axes[1].plot(xs, np.mean(pred_vars,axis=0), color=colour_list[3], label = 'Aleatoric Uncertainty')
    
    
    axes[0].plot(xs, np.mean(plddts, axis=0), color=colour_list[0], label = 'pLDDT')
    axes[0].fill_between(x=xs, y1 = np.mean(plddts, axis=0) - np.std(plddts, axis=0), y2 = np.mean(plddts, axis=0) + 1.96*np.std(plddts, axis=0), color=colour_list[0], alpha=.3)

    axes[1].plot(xs, np.mean(b_factors, axis=0), color=colour_list[1], label = 'B-Factor')
    axes[1].fill_between(x=xs, y1 = np.mean(b_factors, axis=0) - np.std(b_factors, axis=0), y2 = np.mean(b_factors, axis=0) + 1.96*np.std(b_factors, axis=0), color=colour_list[1], alpha=.3)
    
    handles=[]
    handles.append(mpatches.Patch(color=colour_list[0], label='pLDDT'))
    handles.append(mpatches.Patch(color=colour_list[1], label='B-Factor'))
    handles.append(mpatches.Patch(color=colour_list[2], label='Empirical Variance'))
    handles.append(mpatches.Patch(color=colour_list[3], label='Aleatoric Uncertainty'))

    plt.suptitle(sequence)
    axes[0].set_xlabel('Backbone Atom')
    axes[1].set_xlabel('Backbone Atom')
    axes[0].set_ylabel('Flexibility')
    axes[0].legend(handles=handles, loc = 'lower left', bbox_to_anchor=(0.1, -0.2), ncol=4)

    plt.savefig('flexibility_comparisons_'+sequence+'.png',bbox_inches='tight')

def calculate_rmsd_pred_mean(aa_seq, ensemble_idxs):
    try:
        n_conformations, n_cluster_conformations = get_loop_n_conformation_clusters(aa_seq)
    except:
        print(aa_seq)
        raise Exception("Failed to find loop sequence")

    loops = get_all_confs(aa_seq, n_conformations)
    mean, _ = get_conformation_means_and_var(loops, aa_seq)

    ensemble_mean_pred

def gen_loop_to_conf_pdb_dict(test_dataset):
    loop_to_conf_pdb_dict, missed = get_pdbs_from_loop_conformation(test_dataset)
    np.save('loop_to_conf_pdb_dict', loop_to_conf_pdb_dict, allow_pickle=True)

def gen_loop_to_conf_uniprot_dict(loop_to_conf_pdb_dict):
    req_pdbs = []
    for code_list in loop_to_conf_pdb_dict.values():
        req_pdbs += [code for code in code_list]
    req_pdbs = list(set(req_pdbs))
    
    request = IdMappingClient.submit(
        source="PDB", dest="UniProtKB", ids=req_pdbs
    )
    time.sleep(10)
    results = list(request.each_result())

    loop_to_conf_uniprot_dict = {result["from"] : result["to"] for result in results}
    np.save('loop_to_conf_uniprot_dict', loop_to_conf_uniprot_dict, allow_pickle=True)

def gen_use_these_sequences(test_dataset, loop_to_conf_pdb_dict):
    train_data = pd.read_csv("test_data_len_11.csv")
    loop_seqs = list(set([seq.split("_")[0] for seq in train_data.ID]))
    successful_seqs = []

    for loop_seq in loop_seqs:
        pdb_codes = loop_to_conf_pdb_dict[loop_seq]
        successful_codes = []
        for i in range(len(pdb_codes)):
            code = pdb_codes[i]
            try:
                uniprot_accession = loop_to_conf_uniprot_dict[code]
                conformation_idx = int(list(test_dataset[test_dataset.aa_seq == loop_seq].conformation_indxs)[0][i])
                plddts = get_plddt(uniprot_accession, loop_seq, conformation_idx)
                if len(plddts) != 0:
                    successful_codes.append(code)
            except:
                continue
        if len(successful_codes) >= 0.8*len(pdb_codes):
            successful_seqs.append(loop_seq)
    
    np.save("use_these_sequences_test.npy", successful_seqs, allow_pickle=True)

def brownian_motion(x0, n, dt, delta):
    ### Source: https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html ###
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    return out

def wilcoxon_random_projections(obs1, obs2):
    n1 = len(obs1)
    n2 = len(obs2)

    obs1 = [smoothing(obs)[0] for obs in obs1]
    obs2 = [smoothing(obs)[0] for obs in obs2]

    z = np.empty((1,101))
    z[:, 0] = 0.0

    bm = [brownian_motion(z, 101, 0.5, 1)[0][0] for i in range(n1)]
    x = np.linspace(start=1,stop=11,num=101)
    f = lambda t,i : obs1[i](t) * bm[i][int(10*(t-1))]
    ys = [list(map(lambda s : f(s, i), x)) for i in range(n1)]
    obs1_random_projections = [integrate.simpson(ys[i], x=x) for i in range(n1)]

    bm = [brownian_motion(z, 101, 0.5, 1)[0][0] for i in range(n2)]
    x = np.linspace(start=1,stop=11,num=101)
    f = lambda t,i : obs2[i](t) * bm[i][int(10*(t-1))]
    ys = [list(map(lambda s : f(s, i), x)) for i in range(n2)]
    obs2_random_projections = [integrate.simpson(ys[i], x=x) for i in range(n2)]

    statistic, p_value = ranksums(obs1_random_projections, obs2_random_projections, alternative='two-sided', nan_policy="raise")
    return statistic, p_value

def smoothing(curve):
    x = [i+1 for i in range(len(curve))]
    smoothed = CubicSpline(x, curve)
    new_x = np.linspace(start=1, stop=max(x), num=30)
    return smoothed, smoothed(new_x)

def plot_p_values(p_values):
    p_values = {loop : p_values[loop] for loop in p_values.keys() if p_values[loop][-1] >= 3}       # To be a valid test, we need at least 8 samples (5 in pred, 3 in others)
    print(p_values)
    fig, axes = plt.subplots(1,1,figsize=(5,5))
    colour_list = ['#f1b6da', '#fdae61', '#abdda4', '#3288bd']
    plddts = [p_values[loop][0] for loop in p_values.keys()]
    b_factors = [p_values[loop][1] for loop in p_values.keys()]

    axes.hist(b_factors, bins=50, color=colour_list[2], alpha=.8, label="B-Factor", edgecolor = 'k', linewidth=.5, align='right')
    axes.hist(plddts, bins=50, color=colour_list[0], alpha=.8, label="pLDDT", edgecolor = 'k', linewidth=.5,align='mid')
    plt.vlines([0.05], ymin = 0, ymax = plt.ylim()[1], colors=['k'], label = r'$\alpha = 0.05$')
    axes.set_title("Histogram of P-Values Obtained from Wilcoxon \n Rank Sum Tests with Random Projections")
    axes.set_ylabel("Frequency")
    axes.legend(loc="upper right")

    plt.savefig("p_value_hist.png")


    n_conformations = [p_values[loop][-1] + 5 for loop in p_values.keys()]
    fig,axes = plt.subplots(1,1,figsize=(5,5))
    axes.scatter(n_conformations, plddts, color=colour_list[2], label='pLDDT')
    axes.scatter(n_conformations, b_factors, color=colour_list[0], label='B-Factor')
    axes.set_xlabel('Number of Samples')
    axes.set_ylabel('p-value')
    axes.set_title("Scatter Plot of P-Values Against Number of Samples")
    axes.legend(loc="lower right")
    plt.savefig("p_value_scatter.png")

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    sm.qqplot(np.array(plddts), uniform, line="45", ax=axes[0])
    sm.qqplot(np.array(b_factors), uniform, line="45", ax=axes[1])
    axes[0].set_title("QQ-Plot for pLDDT P-Values")
    axes[1].set_title("QQ-Plot for B-Factor P-Values")
    plt.savefig("p_value_qq.png")

def calculate_hamming_dist(code):
    train_set = pd.read_csv("train_data_len_11.csv")
    val_set = pd.read_csv("val_data_len_11.csv")
    min_dist = np.inf
    for train_code in train_set.ID:
        dist = editdistance.eval(code, train_code)
        if dist < min_dist:
            min_dist = dist
    for val_code in val_set.ID:
        dist = editdistance.eval(code, val_code)
        if dist < min_dist:
            min_dist = dist
    return min_dist

if __name__ == "__main__":
    test_dataset = pd.read_pickle('loops_dataset_len_11.pkl') # will contain the loop code and conformation indices
    #original_dataset = pd.read_pickle('20231211_bsheet_loops_w2ndstruc_conformations.pkl')

    loop_to_conf_pdb_dict = np.load('loop_to_conf_pdb_dict.npy', allow_pickle=True).item()  
    loop_to_conf_uniprot_dict = np.load('loop_to_conf_uniprot_dict.npy', allow_pickle=True).item()
    #successful_seqs = gen_use_these_sequences(test_dataset, loop_to_conf_pdb_dict)
    successful_seqs = np.load("use_these_sequences_test.npy", allow_pickle=True) # sequences with at least 80% of confs in AF2 DB
  
    for loop_seq in successful_seqs:
        #loop_seq = 'YVTYGKSPSRP'
        pdb_codes = loop_to_conf_pdb_dict[loop_seq]
        num_conformations_used = 0

        across_conf_plddts = []
        across_conf_b_factors = []
        across_conf_emp_vars = []
        across_conf_pred_vars = []
        try:
            pred_vars = get_predictive_variance(loop_seq)       # some predictions weren't successful, I don't know why yet
        except:
            continue
        pred_vars = [pred_var[2:-2] for pred_var in pred_vars]
        pred_vars = [(pred_var - np.min(pred_var)) / np.std(pred_var) for pred_var in pred_vars]
        across_conf_pred_vars += pred_vars
        for i in range(len(pdb_codes)):
            pdb_code = pdb_codes[i]
            try:        # not all conformations will have a code - we restricted the inference data so that at least 80% do.
                uniprot_accession = loop_to_conf_uniprot_dict[pdb_code]
                conformation_idx = int(list(test_dataset[test_dataset.aa_seq == loop_seq].conformation_indxs)[0][i])
                plddts = get_plddt(uniprot_accession, loop_seq, conformation_idx)
            except:
                continue
            if len(plddts) == 15:
                last_loop_seq = pdb_code
                last_conformation_idx = conformation_idx
                b_factors = get_b_factors(loop_seq, conformation_idx)
                empirical_vars = get_empirical_variance(loop_seq)
                if empirical_vars is not None:
                    plddts = plddts[2:-2]
                    b_factors = b_factors[2:-2]
                    empirical_vars = empirical_vars
                                        
                    plddts = (100 - plddts)
                    plddts = (plddts - np.min(plddts)) / np.std(plddts)
                    b_factors = (b_factors - np.min(b_factors)) / (np.std(b_factors) + 0.001)
                    empirical_vars = (empirical_vars - np.min(empirical_vars)) / np.std(empirical_vars)

                    across_conf_plddts.append(plddts)
                    across_conf_b_factors.append(b_factors)
                    if np.isnan(b_factors).any() != True:
                        across_conf_emp_vars.append(empirical_vars)
                    num_conformations_used+=1
        #print("plddts", np.mean(np.mean(across_conf_plddts,axis=0)))
        #print("b_factors", np.mean(np.mean(across_conf_b_factors, axis=0)))
    
        if len(across_conf_plddts) != 0:
            plot_flexibility_measures(loop_seq, across_conf_plddts, across_conf_b_factors, across_conf_emp_vars, across_conf_pred_vars)
    #    if len(across_conf_plddts) != 0:
    #        p_plddt = wilcoxon_random_projections(across_conf_pred_vars, across_conf_plddts)[1]
    #        p_b_factor = wilcoxon_random_projections(across_conf_pred_vars, across_conf_b_factors)[1]
    #        p_emp = wilcoxon_random_projections(across_conf_pred_vars, across_conf_emp_vars)[1]
    #        p_values[loop_seq] = [p_plddt, p_b_factor, p_emp, num_conformations_used]
    #print(len(p_values))
    #np.save("p_values_test.npy", p_values, allow_pickle=True)
    
    


