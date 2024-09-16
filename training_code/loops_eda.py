# Loops Exploratory Data Analysis

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def get_loops_dataset():
    loops_dataset = pd.read_pickle('/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/loops_dataset_len_11.pkl')
    return loops_dataset

def get_all_data():
    csv_file = pd.read_csv('/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code/data/all_data.csv')
    return csv_file

def group_all_data(all_data_csv):
    sequences = Counter(["".join([char for char in idx if not(char.isnumeric())]) for idx in all_data_csv.ID])
    model_data_df = pd.DataFrame.from_dict(sequences, orient='index', columns=['num_conformations'])
    return model_data_df

def plot_sequence_length(sequences):
    lengths = Counter([len(seq) for seq in sequences])

    print("Min Length", np.min(list(lengths.keys())), 
          "Max Length", np.max(list(lengths.keys())), 
          "Total Num Seq", len(sequences),
          "Mean Length position", np.mean(list(lengths.values())))
    
    print("Len PDF", {sorted(lengths.keys())[i] : lengths[sorted(lengths.keys())[i]] for i in range(len(lengths)) })
    print("Len CDF", {sorted(lengths.keys())[i] : sum([lengths[sorted(lengths.keys())[j]] for j in range(i+1)]) for i in range(len(lengths)) })

    fig = plt.figure(figsize=(5,6))
    plt.hist([len(seq) for seq in sequences], bins=50, log=True, color='cadetblue', edgecolor = 'k')
    plt.vlines([min(lengths.keys()), max(lengths.keys())], ymin = 0, ymax = plt.ylim()[1], colors=['red'])
    plt.xlabel("Sequence Length")
    plt.ylabel("Log Frequency")
    plt.title("Histogram of Sequence Lengths in Loops Dataset")

    plt.savefig("sequence_length_count.png", bbox_inches="tight")

def plot_num_conformations(conf_indxs):
    num_conformations = Counter([conf_indx for conf_indx in conf_indxs])

    print("Min Num Confs", np.min(list(num_conformations.keys())), 
          "Max Num Confs", np.max(list(num_conformations.keys())),
          "Total Num Confs", np.sum(conf_indxs),
          "Mean Conf position", np.mean(list(num_conformations.values())))
    
    print("Conf PDF", {sorted(num_conformations.keys())[i] : num_conformations[sorted(num_conformations.keys())[i]] for i in range(len(num_conformations)) })
    print("Conf CDF", {sorted(num_conformations.keys())[i] : sum([num_conformations[sorted(num_conformations.keys())[j]] for j in range(i+1)]) for i in range(len(num_conformations)) })

    fig = plt.figure(figsize=(8,6))
    plt.hist([conf_indx for conf_indx in conf_indxs], bins=50, log=True, color='cadetblue',alpha=.7, edgecolor = 'k')
    plt.vlines([min(num_conformations.keys()), max(num_conformations.keys())], ymin = 0, ymax = plt.ylim()[1], colors=['red'])
    plt.xlabel("Number of Conformations")
    plt.ylabel("Log Frequency")
    plt.title("Histogram of Associated Conformations to a Sequence in Length 11 Loops Dataset")

    plt.savefig("num_conformations_count.png", bbox_inches="tight")

def plot_num_cluster_centers(dataset_dir):
    return

if __name__ == "__main__":
    loops_dataset = get_loops_dataset()
    print(loops_dataset)
    model_dataset = group_all_data(get_all_data())
    model_dataset = model_dataset.filter(items = loops_dataset.aa_seq, axis=0)
    print(model_dataset)

    #print("num loops conformations", sum([len(idxs) for idxs in loops_dataset.conformation_indxs]))
    #print("num model loops conformations", sum([idxs for idxs in model_dataset.num_conformations]))

    sequences = model_dataset.index
    conf_indxs = model_dataset.num_conformations
    
    ########## Sequence Length ##########

    #plot_sequence_length(sequences)

    ######## Total Conformations ########
    
    plot_num_conformations(conf_indxs)
    
    ########## Cluster Centres ##########

    #plot_num_cluster_centers("")
