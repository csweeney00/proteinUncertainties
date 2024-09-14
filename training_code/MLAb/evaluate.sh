# Evaluates the performance of both the ensemble of models and the mc dropout model
import torch
import os
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser
from Bio.PDB.StructureBuilder import StructureBuilder
from collections import Counter

from util import *
from data_storer import DataStorer
from metrics import exposed_accuracy, sidechain_accuracy, count_clashes, refine, parser, CDR_rmsds

def get_loop_structure(aa_seq, i):
    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/Loops'
    filepath = vols_home + '/loopsPDBs/' + aa_seq + '_' + str(i) + '.pdb'
    parser = PDBParser()
    return parser.get_structure(aa_seq, filepath)

def get_all_confs(aa_seq, conformation_idxs):
    loops = []
    for i in range(len(conformation_idxs)):
        loop = get_loop_structure(aa_seq, conformation_idxs[i])
        loops.append(loop)
    return loops

# Just backbone coords
def structure_from_coords(coords):
    builder = StructureBuilder()
    builder.init_structure("mean_conf")
    builder.init_model(1)
    builder.init_chain("A")
    for i in range(len(coords)):
        coord = coords[i]
        builder.init_residue("A", "", i, "( ,1, )")
        # Everything is CA. It's not, but it will register everything as a backbone atom
        builder.init_atom('CA', coord, 0, 0, 0, " CA ")
    structure = builder.get_structure()
    return structure

def align_anchors(x, y, loop_aa_seq, return_backbone = False):
    # We align the two loops via the anchoring residues
    x_res = [res for res in x.get_residues()]
    y_res = [res for res in y.get_residues()]
    
    x_anchor_start_split = "".join([restype_3to1[res.get_resname()] for res in x_res]).find(loop_aa_seq)
    y_anchor_start_split = "".join([restype_3to1[res.get_resname()] for res in y_res]).find(loop_aa_seq)
    x_atoms = ([res.get_atoms() for res in x_res[:x_anchor_start_split]], 
               [res.get_atoms() for res in x_res[x_anchor_start_split:x_anchor_start_split+len(loop_aa_seq)]],
               [res.get_atoms() for res in x_res[x_anchor_start_split+len(loop_aa_seq):]])
    y_atoms = ([res.get_atoms() for res in y_res[:y_anchor_start_split]], 
               [res.get_atoms() for res in y_res[y_anchor_start_split:y_anchor_start_split+len(loop_aa_seq)]],
               [res.get_atoms() for res in y_res[y_anchor_start_split+len(loop_aa_seq):]])

    x_anchor_backbone = []
    for anchor_section in x_atoms:
        section_coords = []
        for res in anchor_section:
            for atom in res:
                if atom.name in atom_types[:4]:
                    section_coords.append(atom.get_coord())
        x_anchor_backbone.append(section_coords)
    y_anchor_backbone = []
    for anchor_section in y_atoms:
        section_coords = []
        for res in anchor_section:
            for atom in res:
                if atom.name in atom_types[:4]:
                    section_coords.append(atom.get_coord())
        y_anchor_backbone.append(section_coords)

    si = SVDSuperimposer()
    si.set(np.array(x_anchor_backbone[0] + x_anchor_backbone[2]), np.array(y_anchor_backbone[0] + y_anchor_backbone[2]))
    si.run()
    rot, tran = si.get_rotran()

    if return_backbone:
        x_coord = x_anchor_backbone[0] + x_anchor_backbone[1] + x_anchor_backbone[2]
        y_coord = y_anchor_backbone[0] + y_anchor_backbone[1] + y_anchor_backbone[2]
    else:
        x_coord = get_structure_coords(x)
        y_coord = get_structure_coords(y)
    y_coord = np.dot(y_coord, rot) + tran

    return np.array(x_coord), np.array(y_coord), si

def get_conf_mean(loops, loop_aa_seq):
    base_loop = loops[0]
    aligned_loops = []
    for loop in loops:
        aligned_anchors = align_anchors(base_loop, loop, loop_aa_seq, return_backbone=True)
        al = aligned_anchors[1]
        if al.shape == aligned_anchors[0].shape:
            aligned_loops.append(al)
    aligned_loops = np.array(aligned_loops)

    #mean = structure_from_coords(np.mean(aligned_loops, axis = 0))
    return np.mean(aligned_loops, axis = 0)

def load_ensemble(paths):
    ensemble = []
    for path in paths:
        model = StructureModule(22, 8,
                        rel_pos_dim=16,
                        embed_dim=32, dropout=0.1,
                        heads=12, head_dim=16,
                        n_query_points=4,
                        n_value_points=8,
                        error_estimation=0,
                        n_seq_blocks=0,
                        edge_update=0,
                        extra_losses=1,
                        rotation_prop=0).to(device)
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        ensemble.append(model)
    return ensemble

def load_test_data():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data = pd.concat([pd.read_csv(os.path.join(current_dir, "test_data_len_11.csv")),
                      pd.read_csv(os.path.join(current_dir, "val_data_len_11.csv")),
                      pd.read_csv(os.path.join(current_dir, "train_data_len_11.csv"))])

    #data = data[data.ID.isin(test_points)]
    data = data[data.ID.str.contains("".join([point + '|' for point in test_points])[:-1])]
    print(data)

    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code'
    data_path = vols_home
    data = DataStorer(data, data_path=data_path, regions=["Lo"])
    return data

def compute_bb_rmsd(pred_structure, true_structure):
    pred_bb = get_structure_coords(pred_structure, True, False)
    true_bb = get_structure_coords(true_structure, True, False)
    pred_bb = np.array(pred_bb)
    true_bb = np.array(true_bb)
    
    imposer = SVDSuperimposer()
    imposer.set(true_bb, pred_bb)
    imposer.run()
    return imposer.get_rms()

# save both the unrefined and refined structures and save the metrics of the refined structures
def eval_model_ensemble(model, ens_idx, data, filepath):
    if not os.path.exists(os.path.join(filepath, "preds", "ens_" + str(ens_idx))):
        os.mkdir(os.path.join(filepath, "preds", "ens_" + str(ens_idx)))
        os.mkdir(os.path.join(filepath, "refined", "ens_" + str(ens_idx)))
        os.mkdir(os.path.join(filepath, "covars", "ens_" + str(ens_idx)))

    true_files, pred_files, refined_files, covar_files = [], [], [], []
    with torch.no_grad():
        model.eval()
        
        for j in range(len(data)):
            encoding, geom, seq = data.encodings[j], torch.tensor(data.geoms[j]).to(device), data.seqs[j]
            rigid_in, moveable_mask = rigid_in_from_geom_out(data.geoms[j], encoding)
            rigid_true = rigid_from_tensor(geom)
            torsion_true = torsion_angles_from_geom(geom, seq)
            missing_atom_mask = (geom != 0.0).all(-1)
            try:
                new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                        rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                        torsion_true=torsion_true,
                                                                        update_mask=moveable_mask.to(device))
            except:
                print(seq)
                continue

            var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)

            pred_txt = to_pdb(all_atoms, seq, encoding)
            true_txt = to_pdb(geom, seq, encoding)

            true_files.append(os.path.join(filepath, "truth", data.IDs[j] + ".pdb"))
            pred_files.append(os.path.join(filepath, "preds", "ens_" + str(ens_idx), data.IDs[j] + ".pdb"))
            #refined_files.append(os.path.join(filepath, "refined", "ens_" + str(ens_idx), data.IDs[j] + ".pdb"))

            covar_files.append(os.path.join(filepath, "covars", "ens_" + str(ens_idx), data.IDs[j] + ".pt"))

            with open(true_files[-1], "w+") as file:
                file.write(true_txt)
            with open(pred_files[-1], "w+") as file:
                file.write(pred_txt)
            torch.save(var, covar_files[-1])

    results = []
    for j in range(len(true_files)):
        true_file = true_files[j]
        #out_files = {"unrefined": pred_files[j], "refined": refined_files[j]}
        #refine(out_files["unrefined"], out_files["refined"])
        out_files = {"unrefined": pred_files[j]}
        result = {}
        true_ab = parser.get_structure(true_file, true_file)

        for ref in out_files:
            pred_ab = parser.get_structure(out_files[ref], out_files[ref])
            rmsd = compute_bb_rmsd(pred_ab, true_ab)
            result.update({ref + "_target_rmsd": rmsd})

            sequence = out_files[ref].split("/")[-1].split("_")[0]
            idxs = [entry.split("_")[1] for entry in data.IDs if entry.split("_")[0] == sequence]
            conformations = get_all_confs(sequence, idxs)
            try:
                mean = get_conf_mean(conformations, sequence)
                rmsd = compute_bb_rmsd(pred_ab, mean)
                result.update({ref + "_mean_conf_rmsd": rmsd})
            except:
                print("had issues with get mean conf for", sequence)

        results.append(result)

    return true_files, pred_files, refined_files, covar_files, results

# Computes the results dictionary where the prediction made is the mean of all the model
# predictions
def eval_whole_ensemble(data, true_files, ensemble_pred_files):
    parser = PDBParser()
    ensemble_size = len(ensemble_pred_files)
    results = []
    for j in range(len(true_files)):
        result = {}
        true_file = true_files[j]
        true_structure = parser.get_structure(str(j), true_file)
        ensemble_pred_structures = [parser.get_structure(str(i), ensemble_pred_files[i][j]) for i in range(ensemble_size)]

        true_coords = np.array(get_structure_coords(true_structure, True, False))
        mean_pred_coords = np.array(np.mean([get_structure_coords(pred_structure, True, False) for pred_structure in ensemble_pred_structures], axis=0))

        imposer = SVDSuperimposer()
        imposer.set(true_coords, mean_pred_coords)
        imposer.run()
        rmsd = imposer.get_rms()
        result.update({"unrefined" + "_target_rmsd": rmsd})

        sequence = true_file.split("/")[-1].split("_")[0]
        idxs = [entry.split("_")[1] for entry in data.IDs if entry.split("_")[0] == sequence]
        conformations = get_all_confs(sequence, idxs)
        mean = get_conf_mean(conformations, sequence)
        try:
            imposer = SVDSuperimposer()
            print(mean_pred_coords.shape, mean.shape)
            imposer.set(mean, mean_pred_coords)
            imposer.run()
            rmsd = imposer.get_rms()
            result.update({"unrefined" + "_mean_conf_rmsd": rmsd})
        except:
            print("had issues with get mean conf for", sequence)
        results.append(result)
    return results

def get_structure_coords(structure, bb = False, ca=False):
    structure_coords = []
    for chain in structure.get_chains():
        for res in chain.get_residues():
            for atom in res.get_atoms():
                if bb:
                    if ca:
                        if atom.name == 'CA':
                            structure_coords.append(atom.get_coord())
                    else:
                        if atom.name in atom_types[:4]:
                            structure_coords.append(atom.get_coord())
                else:
                    structure_coords.append(atom.get_coord())
    return structure_coords

def get_structures_coords(structures, bb=False, ca=False):
    structures_coords = []
    for structure in structures:
        structure_coords = get_structure_coords(structure, bb, ca)
        structures_coords.append(structure_coords)
    return np.array(structures_coords)

def get_structure_variance(structures):
    num_data_points = len(structures)
    vars = []
    for i in range(num_data_points):
        point_structures = structures[i]
        structures_coords = get_structures_coords(point_structures, True, True)
        vars.append(np.var(structures_coords, axis=0))
    return vars        

def get_ensemble_means_vars(mean_files, var_files):
    num_models = mean_files.shape[0]
    num_data_points = mean_files.shape[1]
    ensemble_means, ensemble_vars = [], []
    for i in range(num_data_points):
        mean_file = mean_files[:,i] # File for the predicted structure of a point for all models in ensemble
        var_file = var_files[:,i]
        structures = []
        vars = []
        for j in range(num_models):
            mean = parser.get_structure(mean_file[j], mean_file[j])     # The predicted structure for the jth model in the ensemble
            structures.append(mean)
            var = torch.load(var_file[j])
            vars.append(var.cpu())
        ensemble_means.append(structures)
        ensemble_vars.append(vars)
    return ensemble_means, ensemble_vars

def get_dropout_means_vars(mean_files, var_files, num_models, num_samples):
    dropout_means, dropout_vars = [], []
    for data_point in mean_files.keys():
        mean_file = mean_files[data_point] # File for the predicted structure of a point for all models in ensemble
        var_file = var_files[data_point]
        structures = []
        vars = []
        for j in range(num_models):
            for k in range(num_samples):
                mean = parser.get_structure(mean_file[j*num_samples + k], mean_file[j*num_samples + k])     # The predicted structure for the jth model in the ensemble
                structures.append(mean)
                var = torch.load(var_file[j])
                vars.append(var.cpu())
        dropout_means.append(structures)
        dropout_vars.append(vars)
    return dropout_means, dropout_vars

def zero_rate(params):
    num_zeros = 0
    total_params = 0
    for row in params:
        for col in row:
            total_params += 1
            if col == 0:
                num_zeros += 1
    return num_zeros / total_params

def eval_model_dropout(model, ens_idx, data, filepath, num_samples):
    if not os.path.exists(os.path.join(filepath, "preds", "ens_" + str(ens_idx))):
        os.mkdir(os.path.join(filepath, "preds", "ens_" + str(ens_idx)))
        os.mkdir(os.path.join(filepath, "refined", "ens_" + str(ens_idx)))
        os.mkdir(os.path.join(filepath, "covars", "ens_" + str(ens_idx)))


    true_files, pred_files, refined_files, covar_files = [], [], [], []
    with torch.no_grad():
        model.train()           # We keep the model in train model as we want the dropout masks to be on
        for j in range(len(data)):
            encoding, geom, seq = data.encodings[j], torch.tensor(data.geoms[j]).to(device), data.seqs[j]

            rigid_in, moveable_mask = rigid_in_from_geom_out(data.geoms[j], encoding)
            rigid_true = rigid_from_tensor(geom)
            torsion_true = torsion_angles_from_geom(geom, seq)
            missing_atom_mask = (geom != 0.0).all(-1)
            for k in range(num_samples):
                new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                            rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                            torsion_true=torsion_true,
                                                                            update_mask=moveable_mask.to(device))

                var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)

                pred_txt = to_pdb(all_atoms, seq, encoding)

                pred_files.append(os.path.join(filepath, "preds", "ens_" + str(ens_idx), data.IDs[j] + "_" + str(k) + ".pdb"))
                #refined_files.append(os.path.join(filepath, "refined", "ens_" + str(ens_idx), data.IDs[j] + "_" + str(k) + ".pdb"))
                covar_files.append(os.path.join(filepath, "covars", "ens_" + str(ens_idx), data.IDs[j] + "_" + str(k) + ".pt"))

                with open(pred_files[-1], "w+") as file:
                    file.write(pred_txt)
                torch.save(var, covar_files[-1])

            true_txt = to_pdb(geom, seq, encoding)
            true_files.append(os.path.join(filepath, "truth", data.IDs[j] + ".pdb"))
            with open(true_files[-1], "w+") as file:
                file.write(true_txt)

    # results will be over all predictions - including the mc dropout samples
    results = []
    for j in range(len(pred_files)):
        true_file = true_files[j // num_samples]

        #out_files = {"unrefined": pred_files[j], "refined": refined_files[j]}
        #refine(out_files["unrefined"], out_files["refined"])
        out_files = {"unrefined": pred_files[j]}
        result = {}
        true_ab = parser.get_structure(true_file, true_file)

        for ref in out_files:
            pred_ab = parser.get_structure(out_files[ref], out_files[ref])

            rmsd = compute_bb_rmsd(pred_ab, true_ab)
            result.update({ref + "_" + "Loop": rmsd})
            
            sequence = out_files[ref].split("/")[-1].split("_")[0]
            idxs = [entry.split("_")[1] for entry in data.IDs if entry.split("_")[0] == sequence]
            conformations = get_all_confs(sequence, idxs)
            mean = get_conf_mean(conformations, sequence)
            rmsd = compute_bb_rmsd(pred_ab, mean)
            result.update({ref + "_mean_conf_rmsd": rmsd})

        results.append(result)

    return true_files, pred_files, refined_files, covar_files, results

def run_ensemble(ensemble):
    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code'
    model_dir = vols_home + '/models/'
    model_paths = [model_dir + 'train_data_ens_' + ensemble[i] + '_len_11' for i in range(len(ensemble))]

    models = load_ensemble(model_paths)
    #print("weights", [zero_rate(p.data) for n,p in models[0].named_parameters() if n == "variance_layers.3.weight" or n == "variance_layers.2.weight" or n == "variance_layers.1.weight" or n == "variance_layers.0.weight"])
    data = load_test_data()

    if not os.path.exists("ensemble_results"):
        os.mkdir("ensemble_results")
    if not os.path.exists(os.path.join("ensemble_results", "truth")):
        os.mkdir(os.path.join("ensemble_results", "truth"))
    if not os.path.exists(os.path.join("ensemble_results", "preds")):
        os.mkdir(os.path.join("ensemble_results", "preds"))
    if not os.path.exists(os.path.join("ensemble_results", "refined")):
        os.mkdir(os.path.join("ensemble_results", "refined"))
    if not os.path.exists(os.path.join("ensemble_results", "covars")):
        os.mkdir(os.path.join("ensemble_results", "covars"))

    models_files = {'pred':[], 'covar':[]}
    ensemble_results = []
    for i in range(len(ensemble)):
        model = models[i]
        ens_idx = ensemble[i]
        true_files, pred_files, _, covar_files, results = eval_model_ensemble(model, ens_idx, data, 'ensemble_results')
        models_files['pred'].append(pred_files)
        models_files['covar'].append(covar_files)
        #ensemble_results.append(results)
    
    ensemble_results = eval_whole_ensemble(data, true_files, models_files['pred'])

    models_files['pred'] = np.array(models_files['pred'])
    models_files['covar'] = np.array(models_files['covar'])
    ensemble_results = np.array(ensemble_results)
    
    ensemble_means, ensemble_vars = get_ensemble_means_vars(models_files['pred'], models_files['covar'])
    epistemic_uncertainty = get_structure_variance(ensemble_means)
    aleatoric_uncertainty = np.mean(ensemble_vars, axis=1)
    return ensemble_results, epistemic_uncertainty, aleatoric_uncertainty
        
# Make sure the length of the ensemble is definitely 1
def run_dropout(ensemble):
    vols_home = '/vols/teaching/msc-projects/2023-2024/chsweeney/My Documents/training_code'
    model_dir = vols_home + '/models/'
    model_paths = [model_dir + 'train_data_ens_' + ensemble[i] + '_len_11' for i in range(len(ensemble))]

    models = load_ensemble(model_paths)
    data = load_test_data()

    if not os.path.exists("dropout_results"):
        os.mkdir("dropout_results")
    if not os.path.exists(os.path.join("dropout_results", "truth")):
        os.mkdir(os.path.join("dropout_results", "truth"))
    if not os.path.exists(os.path.join("dropout_results", "preds")):
        os.mkdir(os.path.join("dropout_results", "preds"))
    if not os.path.exists(os.path.join("dropout_results", "refined")):
        os.mkdir(os.path.join("dropout_results", "refined"))
    if not os.path.exists(os.path.join("dropout_results", "covars")):
        os.mkdir(os.path.join("dropout_results", "covars"))

    models_files = {'refined':{data.IDs[i] : [] for i in range(len(data.IDs))}, 'covar':{data.IDs[i] : [] for i in range(len(data.IDs))}}
    dropout_results = []
    num_samples = 10
    for i in range(len(ensemble)):
        model = models[i]
        ens_idx = ensemble[i]
        _, _, refined_files, covar_files, results = eval_model_dropout(model, ens_idx, data, 'dropout_results', num_samples)
        for j in range(len(data.IDs)):
            models_files['refined'][data.IDs[j]] = models_files['refined'][data.IDs[j]] + refined_files[j*num_samples : (j+1)*num_samples]
            models_files['covar'][data.IDs[j]] = models_files['covar'][data.IDs[j]] + covar_files[j*num_samples : (j+1)*num_samples]
        #average_results = {x: np.mean([result[x] for result in results]) for x in results[0]}
        dropout_results.append(results)

    dropout_results = np.array(dropout_results)
    
    dropout_means, dropout_vars = get_dropout_means_vars(models_files['refined'], models_files['covar'], len(ensemble), num_samples)
    epistemic_uncertainty = get_structure_variance(dropout_means)
    aleatoric_uncertainty = np.mean(dropout_vars, axis=1)
    return dropout_results, epistemic_uncertainty, aleatoric_uncertainty

if __name__ == "__main__":
    assert torch.cuda.is_available(), "You are not using a GPU for training?"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dropout = False

    # List of the names of the points that we want to use for the evaluation. This includes both points in the test set and train set
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_data = pd.read_csv(os.path.join(current_dir, "val_data_len_11.csv"))
    test_points = list(set([entry.split("_")[0] for entry in train_data.ID]))
    ensemble = ['1', '2', '3', '4', '5']

    banned_points = ['WDHPRMPWEGT', 'LSSRDKYGRVV', 'TASHNPMDYNG']
    test_points = [point for point in test_points if point not in banned_points]

    print("Length training data", len(test_points), len(train_data))

    if dropout:
        results, epistemic_uncertainty, aleatoric_uncertainty = run_dropout(ensemble)
    else:
        results, epistemic_uncertainty, aleatoric_uncertainty = run_ensemble(ensemble)

    epistemic_uncertainty = {test_points[i] : epistemic_uncertainty[i] for i in range(len(test_points))}
    aleatoric_uncertainty = {test_points[i] : aleatoric_uncertainty[i] for i in range(len(test_points))}

    if not os.path.exists(os.path.join('text_results')):
        os.mkdir(os.path.join('text_results'))

    if dropout:
        np.save('text_results/dropout_results_' + str(ensemble[0]) + '.npy', results, allow_pickle=True)
    else:
        #np.save('text_results/ensemble_results_' + str(ensemble[0]) + '_val' + '.npy', results, allow_pickle=True)
        np.save('text_results/ensemble_results_' + 'val' + '.npy', results, allow_pickle=True)

    #np.save('text_results/epi_uncertainty_' + str(ensemble[0]) + '_val' + '.npy', epistemic_uncertainty, allow_pickle=True)
    #np.save('text_results/ale_uncertainty_' + str(ensemble[0]) + '_val' + '.npy', aleatoric_uncertainty, allow_pickle=True)
    np.save('text_results/epi_uncertainty_' + 'val' + '.npy', epistemic_uncertainty, allow_pickle=True)
    np.save('text_results/ale_uncertainty_' + 'val' + '.npy', aleatoric_uncertainty, allow_pickle=True)
