import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def build_loops_dataset():
    onlyfiles = [f for f in listdir('./pdbs') if isfile(join('./pdbs', f))]
    codes = set([f.split("_")[0] for f in onlyfiles])

    # Limiting codes to sequences of length 11
    codes = [code for code in codes if len(code) == 11]

    data = []
    print("Producing data")
    for code in codes:
        conformations = list(filter(lambda x: x.startswith(code), onlyfiles))
        conformation_indxs = [conf.split("_")[1][:-4] for conf in conformations]
        data.append([code, conformation_indxs])

    df = pd.DataFrame(data=data, columns = ['aa_seq', 'conformation_indxs'])
    print("Saving the dataframe")
    df.to_pickle('./loops_dataset_len_11.pkl')
    return

if __name__ == "__main__":
    build_loops_dataset()
