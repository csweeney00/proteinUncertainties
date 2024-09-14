import pandas as pd
import numpy as np
import os

usual_dir = os.path.dirname(os.path.realpath(__file__))
#train_data = pd.read_csv(os.path.join(usual_dir, "train_data.csv"))
#test_data = pd.read_csv(os.path.join(usual_dir, "test_data.csv"))
#val_data = pd.read_csv(os.path.join(usual_dir, "val_data.csv"))

train_data = pd.read_csv(os.path.join(usual_dir, "train_data_len_11.csv"))
val_data = pd.read_csv(os.path.join(usual_dir, "val_data_len_11.csv"))

# CDRs include 2 anchor residues at either side
region_definitions = {
    "H1": range(25, 41),
    "H2": range(54, 68),
    "H3": range(103, 120),
    "L1": range(153, 169),
    "L2": range(182, 196),
    "L3": range(231, 248),
    "H": range(0, 129),
    "L": range(129, 256)
}

restypes = 'ARNDCQEGHILKMFPSTWYV'

residue_atoms = {
    'A': ['CA', 'N', 'C', 'CB', 'O'],
    'C': ['CA', 'N', 'C', 'CB', 'O', 'SG'],
    'D': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'OD1', 'OD2'],
    'E': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'OE1', 'OE2'],
    'F': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'G': ['CA', 'N', 'C', 'CA', 'O'],  # G has no CB so I am padding it with CA so the Os are aligned
    'H': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD2', 'CE1', 'ND1', 'NE2'],
    'I': ['CA', 'N', 'C', 'CB', 'O', 'CG1', 'CG2', 'CD1'],
    'K': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'CE', 'NZ'],
    'L': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2'],
    'M': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CE', 'SD'],
    'N': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'ND2', 'OD1'],
    'P': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD'],
    'Q': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'NE2', 'OE1'],
    'R': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD', 'CZ', 'NE', 'NH1', 'NH2'],
    'S': ['CA', 'N', 'C', 'CB', 'O', 'OG'],
    'T': ['CA', 'N', 'C', 'CB', 'O', 'CG2', 'OG1'],
    'V': ['CA', 'N', 'C', 'CB', 'O', 'CG1', 'CG2'],
    'W': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE1'],
    'Y': ['CA', 'N', 'C', 'CB', 'O', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']}

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

atom_order = {x: i for i, x in enumerate(atom_types)}

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

chain_lens = {"H": 128, "L": 127}

r2n = {x: i for i, x in enumerate(restypes)}
res_to_num = lambda x: r2n[x] if x in r2n else len(r2n)


def convert_all_atoms_37_to_14(all_atoms_37, seq):
    N = len(seq)
    all_atoms_14 = np.zeros((N, 14, 3))
    for i in range(N):
        atoms = [atom_order[x] for x in residue_atoms[seq[i]]]
        all_atoms_14[i] = np.pad(all_atoms_37[i, :][atoms], ((0, 14 - len(atoms)), (0, 0)),
                                 constant_values=float("Nan"))
    return all_atoms_14


def convert_all_atoms_14_to_37(all_atoms_14, seq):
    N = len(seq)
    all_atoms_37 = np.zeros((N, 37, 3))
    for i in range(N):
        atoms = [atom_order[x] for x in residue_atoms[seq[i]]]
        all_atoms_37[i, atoms] = all_atoms_14[i, :len(atoms)]
    return all_atoms_37


def stringrep_to_sequence(stringrep):
    return stringrep.translate({45: None, 40: None, 41: None})  # Assumes unicode


def stringrep_to_numbers(stringrep):
    numbs = []
    i = 1
    insertion = False
    for x in stringrep:
        if x == "-":
            i += 1
        elif x == "(":
            insertion = True
        elif x == ")":
            insertion = False
        else:
            numbs.append(i)
            i += not insertion
    return numbs


def get_one_hot(targets, nb_classes=21):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


class DataStorer:
    def __init__(self, data, regions=["H1", "H2", "H3", "L1", "L2", "L3"], data_path=None):
        self.data = data.reset_index(drop=True)
        self.regions = regions

        self.geoms = []
        self.seqs = []
        self.encodings = []
        self.IDs = []
        self.data_path = "MLAb" if data_path is None else data_path

        self.numbered_regs = (25*[0] + (41-25)*[1] +
                              (54-41)*[2] + (68-54)*[3] +
                              (103-68)*[4] + (120-103)*[5] +
                              (129-120)*[6] + (153-129)*[7] +
                              (169-153)*[8] + (182-169)*[9] +
                              (196-182)*[10] + (231-196)*[11] +
                              (248-231)*[12] + (256-248)*[13])
        
        if len(self.regions) > 1:
            if self.regions[0] == "H" and self.regions[1] == "L":
                self.region_classification = []

        self.__populate()

    def __populate(self):
        for i in range(len(self.data)):
            seq_rep = self.data.Seq[i]
            full_numb = stringrep_to_numbers(seq_rep)
            full_seq = stringrep_to_sequence(seq_rep)
            id = "".join(self.data.ID[i].split("_"))
            try:
                full_coords = np.load(os.path.join(self.data_path, "data", id + ".npy"))
            except:
                full_coords = np.load(os.path.join(self.data_path, "data", id + '0' + ".npy"))

            all_reg_encodings = []
            all_reg_geoms = []
            all_reg_seqs = ""

            for j, reg in enumerate(self.regions):
                #numb = [x for x in range(len(full_numb)) if full_numb[x] in region_definitions[reg]]
                #coords = full_coords[numb]
                #seq = [full_seq[x] for x in numb]
                coords = full_coords
                seq = full_seq
                #if len(seq) < 5:
                #    continue
                one_hot_amino = get_one_hot(np.array([res_to_num(x) for x in seq]), 22)
                #one_hot_region = get_one_hot(j * np.ones(len(seq), dtype=int), len(self.regions))
                #encoding = np.concatenate([one_hot_amino, one_hot_region], axis=-1)
                encoding = one_hot_amino

                all_reg_encodings.append(encoding)
                all_reg_geoms.append(coords)
                all_reg_seqs += "".join(seq)
                
            if len(self.regions) > 1:
                if self.regions[0] == "H" and self.regions[1] == "L":
                    region_classification = [self.numbered_regs[x] for x in full_numb]
                    self.region_classification.append(np.array(region_classification))

            self.geoms.append(np.concatenate(all_reg_geoms, axis=0))
            self.encodings.append(np.concatenate(all_reg_encodings, axis=0))
            self.seqs.append(all_reg_seqs)
            self.IDs.append(self.data.ID[i])

    def __len__(self):
        return len(self.seqs)
