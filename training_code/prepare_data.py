# Prepare Data
import os
import numpy as np
import pandas as pd
from MLAb.data_storer import *
import argparse
from Bio.PDB import PDBParser

def get_loop_structure(seq, conf_index, data_path):
    filepath = data_path + '/' + seq + '_' + str(conf_index) + '.pdb'
    parser = PDBParser()
    with open(filepath, "r") as loop_file:
        structure = parser.get_structure(seq, loop_file)
        loop_file.close()
    return structure

def generate_data(seq, conf_index, data_path):
    try:
        loop_struc = get_loop_structure(seq, conf_index, data_path)
    except:
        print("Structure", seq, "conformation", conf_index, "couldn't be found. Ignoring and continuing.")
        return np.nan, np.nan
    if list(loop_struc.get_chains()) == []:
        return np.nan, np.nan
    
    struc_res = [res.get_resname() for res in list(loop_struc.get_chains())[0].get_residues()]     # only 1 chain
    string_rep = "".join([restype_3to1[resname] for resname in struc_res])      # includes the main loop and the anchoring residues

    residues, sequence = {}, {"Lo":string_rep}
    atom_positions = {}

    for chain in loop_struc.get_chains():
        chain_type = "Lo"
        chain_residues = [r for r in chain.get_residues() if r.get_id()[0] == " "]
        for r in chain_residues:
            r.detach_parent()
        residues[chain_type] = chain_residues

    for chain_type in ["Lo"]:
        atom_pos = []
        for res in residues[chain_type]:
            pos = np.zeros((37, 3))
            mask = np.zeros((37,))
            for atom in res:
                if atom.name not in atom_types:
                    continue
                pos[atom_order[atom.name]] = atom.coord
                mask[atom_order[atom.name]] = 1.
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            atom_pos.append(pos)
        atom_positions[chain_type] = np.array(atom_pos)
        
    atom14 = np.concatenate([convert_all_atoms_37_to_14(atom_positions[x], sequence[x]) for x in ["Lo"]], axis = 0)
    return atom14, string_rep

def extract_all_data(directory):
    loops_path = os.path.join(directory, "loops_dataset.pkl")
    data_dir = os.path.join(directory, "data")
    pdb_dir = os.path.join(directory, "pdbs")
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    with open(os.path.join(data_dir, "all_data.csv"), "w+") as file:
        file.write("ID,Seq,Resolution\n")

    df = pd.read_pickle(loops_path)

    for i in range(len(df)):
        structure = df.iloc[i]
        aa_seq = structure.aa_seq
        n_conformations = structure.conformation_indxs
        for j in range(len(n_conformations)):
            
            atom14, string_rep = generate_data(aa_seq, n_conformations[j], pdb_dir)
            if np.isnan(atom14).all() and pd.isnull(string_rep):
                print(aa_seq, j)
                continue
            save_to = os.path.join(data_dir, "{}{}".format(aa_seq, n_conformations[j]))
            #print(save_to)
            np.save(save_to, atom14)
            with open(os.path.join(data_dir, "all_data.csv"), "a+") as file:
                # Resolution is set to 0 for now because a. I don't know what it is b. It may not be relevant to us
                file.write("{}{},{},{}\n".format(aa_seq, n_conformations[j], string_rep, 0.0))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Prepare data for training')
    argparser.add_argument('--directory', type=str, default='/data/localhost/not-backed-up/chsweeney/training_code')
    args = argparser.parse_args()

    extract_all_data(args.directory)
