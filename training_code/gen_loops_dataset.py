# Python Script to generate Loops dataset

import pickle as pkl
import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBParser
from Bio.PDB import PDBList
from Bio.PDB import MMCIFParser
import argparse

def delete_files(files):
    for file in files:
        os.remove(file)

def get_loop_structures(id, files, parser):
    structures = []
    for file in files:
        file_id = id + str(len(structures))
        structure = parser.get_structure(file_id, file)
        structures.append(structure)
    return structures

def download_loop_proteins(df, parser, pdb_list):
    structure_df = pd.DataFrame(columns=["id", "structures"])
    for i in range(len(df)):
        id = str(df.index[i])
        file_names = df.iloc[i].loc['paths']
        codes = [file[-8:-4] for file in file_names][:3]
        pdb_list.download_pdb_files(codes, pdir = './pdbs')
        files = ['./pdbs/' + code + '.cif' for code in codes]
        structures = get_loop_structures(id, files, parser)
        delete_files(list(set(files)))

        structure_df.loc[i] = [id, structures]
    return structure_df

def get_loop_residues(df, structure_df):
    residue_df = pd.DataFrame(columns=['id', 'residues'])
    for j in range(len(structure_df)):
        row = structure_df.iloc[j]
        loop_residues = df.iloc[j].loc['resis'] - 2 # I think the first residue is indexed by 2 rather than 0 so trying to correct offset
        structure_residues_list = []
        for i in range(len(row[1])):
            structure = row[1][i]
            structure_residues = list(structure.get_residues())
            structure_residues = [structure_residues[k] for k in loop_residues[i]]
            structure_residues_list.append(structure_residues)
        residue_df.loc[j] = [row[0], structure_residues_list]
    return residue_df

def get_loop_coordinates(residue_df):
    coord_df = pd.DataFrame(columns=['id', 'coords'])
    for i in range(len(residue_df)):
        row = residue_df.iloc[i]
        id = row[0]
        residues_atoms = [list(res.get_atoms()) for res in row[1][0]]
        coords = [[list(atom.get_coord()) for atom in residue_atoms] for residue_atoms in residues_atoms]
        coord_df.loc[i] = [id, coords]
    return coord_df

def generate_coordinate_dataset(df, parser, pdb_list):
    coord_df = pd.DataFrame(columns=['id', 'coords'])
    for i in range(len(df)):
        row = df.iloc[i].to_frame().T
        structure_df = download_loop_proteins(row, parser, pdb_list)
        residue_df = get_loop_residues(row, structure_df)
        coords_row = get_loop_coordinates(residue_df)
        coord_df.loc[i] = coords_row.iloc[0]
    return coord_df

# splits up the dataset into smaller more manageable chunks so that the processing can be easily parallelised
def split_dataset(num_splits, bs):
    df_size = len(bs) // num_splits
    for i in range(num_splits-1):
        df = bs.iloc[i*df_size: (i+1)*df_size]
        df.to_pickle('./loopsDatasets/bs_loops_' + str(i) + '.pkl')
    df = bs.iloc[(num_splits-1)*df_size:]
    df.to_pickle('./loopsDatasets/bs_loops_' + str(num_splits-1) + '.pkl')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate Loops Dataset')
    argparser.add_argument('--dataframe_file_head', type=str,
                         help='the filepath from current directory to the stored loop dataframe to convert')
    argparser.add_argument('--file_num', type=int, default=0,
                         help='the index of the loop dataframe')
    argparser.add_argument('--dry_run', type=bool, default=False,
                         help='do a single pass through the program')
    argparser.add_argument('--n_splits', type=int)
    args = argparser.parse_args()

    beta_sheet_loops = pkl.load(open('./20231211_bsheet_loops_w2ndstruc_conformations.pkl', 'rb'))
    split_dataset(args.n_splits, beta_sheet_loops)
    
    df = pd.read_pickle(args.dataframe_file_head + str(args.file_num) + '.pkl')
    if args.dry_run == True:
        df = df.iloc[:2]
    
    parser = MMCIFParser(QUIET = True)
    pdb_list = PDBList(pdb = './pdbs')

    loops_coord = generate_coordinate_dataset(df, parser, pdb_list)
    filepath = './loopsCoordinates/bs_loops_coords'+str(args.file_num)+'.npz'
    np.savez(filepath, id = loops_coord['id'], coords = loops_coord['coords'].to_numpy())
