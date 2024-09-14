import pandas as pd
import numpy as np
import os
import argparse
from os import listdir
from os.path import isfile, join
from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer

def extract_code(id):
    numbs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return ''.join([id[i] for i in range(len(id)) if id[i] not in numbs])

def new_all_data_file(all_data, seq_length):
    # Ensure that the included loops have a main sequence length of seq_length and 4 anchoring residues in total
    row_filter = [i for i in range(len(all_data)) if (len(extract_code(all_data.iloc[i].ID)) == seq_length) and (len(all_data.iloc[i].Seq) == seq_length+4)]
    return all_data.iloc[row_filter]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Prepare data for training')
    argparser.add_argument('--directory', type=str, default='/data/localhost/not-backed-up/chsweeney/training_code')
    args = argparser.parse_args()

    all_data = pd.read_csv(args.directory + '/data/all_data.csv')
    all_data = new_all_data_file(all_data, 11)

    code_dict_end_idx = {extract_code(all_data.iloc[i].ID):i for i in range(len(all_data))}
    keys = list(code_dict_end_idx.keys())
    code_dict_beg_idx = {keys[i] : code_dict_end_idx[keys[i-1]] + 1 for i in range(1, len(keys))}
    code_dict_beg_idx[keys[0]] = code_dict_end_idx[keys[0]]

    # The most isolated points (3 edit distance/hamming distance away) are 
    # TRWSIYSRWPM
    # We're going to include half of these in the test set.
    codes = list(code_dict_beg_idx.keys())
    
    test_codes = pd.read_pickle('test_codes.pkl').aa_seq.to_numpy()
    isolated_test_indxs = [i for i in range(len(codes)) if codes[i] in test_codes]

    train_indxs = np.sort(np.random.choice([i for i in range(len(codes)-1) if i not in isolated_test_indxs],replace = False, size = int((0.9 * len(codes)) // 1)))
    test_indxs = [i for i in range(len(codes)) if i not in train_indxs]

    val_indxs = np.sort(np.random.choice(train_indxs, replace = False, size = int((0.2 * len(train_indxs)) // 1)))
    train_indxs = [i for i in train_indxs if i not in val_indxs]

    codes = np.array(codes)
    train_codes = codes[train_indxs]
    val_codes = codes[val_indxs]
    test_codes = codes[test_indxs]

    with open(os.path.join(args.directory, "train_data_len_11.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    for code in train_codes:
        for i in range(code_dict_beg_idx[code], code_dict_end_idx[code]+1):
            with open(os.path.join(args.directory, "train_data_len_11.csv"), "a+") as file:
                # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
                idx = all_data.iloc[i][0][len(code):]
                file.write("{}_{},{},{}\n".format(code, idx, all_data.iloc[i][1], 0.0))  

    with open(os.path.join(args.directory, "val_data_len_11.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    for code in val_codes:
        for i in range(code_dict_beg_idx[code], code_dict_end_idx[code]+1):
            with open(os.path.join(args.directory, "val_data_len_11.csv"), "a+") as file:
                # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
                idx = all_data.iloc[i][0][len(code):]
                file.write("{}_{},{},{}\n".format(code, idx, all_data.iloc[i][1], 0.0))  

    with open(os.path.join(args.directory, "test_data_len_11.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    # test set does not ask for predictions of multiple conformations, just of a single code.
    for code in test_codes:
        with open(os.path.join(args.directory, "test_data_len_11.csv"), "a+") as file:
            # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
            i = code_dict_end_idx[code]
            file.write("{},{},{}\n".format(code, all_data.iloc[i][1], 0.0))


    # now create the quick run through datasets
    quick_train_codes = train_codes[:50]
    quick_val_codes = val_codes[:10]

    with open(os.path.join(args.directory, "quick_train_data_len_11.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    for code in quick_train_codes:
        for i in range(code_dict_beg_idx[code], code_dict_end_idx[code]+1):
            with open(os.path.join(args.directory, "quick_train_data_len_11.csv"), "a+") as file:
                # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
                idx = all_data.iloc[i][0][len(code):]
                file.write("{}_{},{},{}\n".format(code, idx, all_data.iloc[i][1], 0.0))  
    
    with open(os.path.join(args.directory, "quick_val_data_len_11.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    for code in quick_val_codes:
        for i in range(code_dict_beg_idx[code], code_dict_end_idx[code]+1):
            with open(os.path.join(args.directory, "quick_val_data_len_11.csv"), "a+") as file:
                # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
                idx = all_data.iloc[i][0][len(code):]
                file.write("{}_{},{},{}\n".format(code, idx, all_data.iloc[i][1], 0.0))
