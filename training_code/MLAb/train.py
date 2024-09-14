#from MLAb import data, test, DataStorer
from data_storer import DataStorer, train_data, val_data
from util import *
import torch
import numpy as np
import copy
import pandas as pd
import os
import sys

import wandb
import cProfile

########################
# Sort out Config File #
########################

#profiler = cProfile.Profile()
#profiler.enable()

# 65 68 70 72
ens_idx = 5
seeds = [65, 68, 70, 71, 74, 100]
np.random.seed(seeds[ens_idx-1])
torch.manual_seed(seeds[ens_idx-1])
#np.random.seed(seeds[-1])
#torch.manual_seed(seeds[-1])

epochs = 500 # instead of 1000 (500 for others)
just_var = True
just_mean = False
var_loss_weighting = 1

torch.set_default_tensor_type(torch.DoubleTensor)
assert torch.cuda.is_available(), "You are not using a GPU for training?"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Should be MLAb
current_dir = os.path.dirname(os.path.realpath(__file__))

# Load the configuration file
config_path = os.path.join(current_dir, 'config.csv')
configs = pd.read_csv(config_path)
for run_number_i in range(len(configs)):
    row = configs.iloc[run_number_i]
    if row["state"] == "unstarted":
        config = row
        break

if len(sys.argv) > 1:
    config = configs[configs.run_id == sys.argv[1]].iloc[0]

data_path = os.path.dirname(current_dir) # Should be training_code
model_path = os.path.join(data_path, "models", config["run_id"])
optim_path = os.path.join(data_path, "optims", config["run_id"])
benchmark_path = os.path.join(data_path, "benchmarks")

# Mark it as started
configs.at[run_number_i, "state"] = "Started"
configs.to_csv(config_path, index=False)

print("\nSettings for this run:\n")
print(config)

wandb.init(
    # set the wandb project where this run will be logged
    project="Quick-Length-11-Loops-ABB2-Training-Run",
    entity = 'csweeney00',
    # track hyperparameters and run metadata
    config={
    "architecture": "ABB2",
    "dataset": "Loops_Length_11",
    "epochs": 1000,
    }
)

##########################################
# Load Data (and model if already saved) #
##########################################

# Data is already split into train test
a = DataStorer(train_data, data_path=data_path, regions=["Lo"])
b = DataStorer(val_data, data_path=data_path, regions=["Lo"])

model = StructureModule(22, int(config["number_of_layers"]),
                        rel_pos_dim=int(config["rel_pos_dim"]),
                        embed_dim=int(config["embed_dim"]), dropout=config["dropout"],
                        heads=int(config["IPA_attention_heads"]), head_dim=int(config["IPA_attention_head_dim"]),
                        n_query_points=int(config["IPA_query_points"]),
                        n_value_points=int(config["IPA_value_points"]),
                        error_estimation=int(config["error_estimation"]),
                        n_seq_blocks=int(config["n_seq_blocks"]),
                        edge_update=int(config["edge_update"]),
                        extra_losses=int(config["loss_every_layer"]),
                        rotation_prop=int(config["rotation_propagation"])).to(device)

if just_var == False:
    restart = False
    if len(sys.argv) > 1:
        config = configs[configs.run_id == sys.argv[1]].iloc[0]
        print(os.path.exists(os.path.join(model_path)))
        if os.path.exists(os.path.join(model_path)):
            restart = True
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            print("Successfully loaded {}".format(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print("Successfully loaded {}".format(model_path))
    #Re-Initialise the variance weights
    model.reinit()
    restart=False       # We don't want it to load up the optimiser again

###################
# Defining Losses #
###################

mean_val_losses = []
mean_train_losses = []

def main_loss(all_reference_frames, true_reference_frames, all_atoms, geom, rigids_mask, mask, regions, var=None):
    if config["loss_type"] == "AF2":
        return AF2_generic_fape(all_reference_frames, true_reference_frames, all_atoms, geom, rigids_mask, mask)
    elif config["loss_type"] == "CDR":
        CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
        return AFM_generic_fape(all_reference_frames, true_reference_frames, all_atoms, geom, rigids_mask, mask, CDR_mask)
    elif config["loss_type"] == "AFM":
        H_mask = torch.tensor([x < 8 for x in regions], device=device)
        return AFM_generic_fape(all_reference_frames, true_reference_frames, all_atoms, geom, rigids_mask, mask, H_mask)

def inv_lower_triangular_mat(mat, band_size, eps):
    L = to_chol_matrix(mat, band_size, device)
    eps_mat = eps*torch.eye(mat.shape[0]).to(device)
    return torch.cholesky_inverse(L + eps_mat)

def inv_lower_triangular_mat_batched(mat, band_size, eps):
    L = to_chol_matrix(mat, band_size, device) + eps*torch.eye(mat[0].shape[0]).to(device)
    return torch.cholesky_inverse(L + eps_mat)

#def var_loss(all_reference_frames, true_reference_frames, var, band_size):
    eps = 10e-5     # Included to avoid a singluar matrix
    eps_mat = eps*torch.ones(var.shape[0]).to(device)

    # Determinant of a positive def matrix with a given cholesky decomposition is the product of the square of the diagonals of the 
    # cholesky matrix
    log_det = torch.sum(2*torch.log(var[:,-1] + eps_mat))
    
    inv = inv_lower_triangular_mat(var, band_size, eps)
    pred_mean_diff = AF2_BB_squared_error(all_reference_frames.detach(), true_reference_frames)
    
    loss = 0.5 * log_det + 0.5 * torch.matmul(torch.matmul(pred_mean_diff.T, inv), pred_mean_diff)
    return loss

# version for diagonal covariance
def var_loss(all_reference_frames, true_reference_frames, var, band_size):
    eps = 10e-5
    pred_mean_diff = AF2_BB_squared_error(all_reference_frames.detach(), true_reference_frames)
    #pred_mean_diff = AF2_BB_squared_error(all_reference_frames, true_reference_frames)
    log_det = torch.sum(torch.log(var + eps))
    det = torch.prod(var + eps)

    loss = 0.5*log_det + 0.5 * torch.inner(torch.pow(var+eps,-1).squeeze(-1), torch.pow(pred_mean_diff, 2))
    return loss

def var_loss_batched(all_reference_frames, true_reference_frames, vars, band_size):
    eps = 10e-5     # Included to avoid a singluar matrix
    eps_mat = eps*torch.ones(vars.shape[1]).to(device)

    # Determinant of a positive def matrix with a given cholesky decomposition is the product of the square of the diagonals of the 
    # cholesky matrix
    log_det = torch.sum(2*torch.log(vars[:,:,-1] + eps_mat), dim=1) # 284
    
    inv = inv_lower_triangular_mat_batched(vars, band_size, eps) # 284 9 9

    # This isn't batched and I'm not sure if it can be. Comp cost should be minimal though.
    pred_mean_diffs = torch.empty(0, requires_grad=True, device=device) # 284 9
    for i in range(vars.shape[0]):
        pred_mean_diff = AF2_BB_squared_error(all_reference_frames[i].detach(), true_reference_frames[i])
        pred_mean_diffs = torch.cat([pred_mean_diffs, pred_mean_diff.unsqueeze(0)])

    loss = 0.5 * log_det + 0.5 * torch.matmul(torch.matmul(torch.transpose(pred_mean_diffs.unsqueeze(-1), 1, 2), inv), pred_mean_diffs.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return torch.sum(loss, dim=0)

###############################
# Initial Evaluation of Model #
###############################

epoch_val_loss = []
model.eval()
with torch.no_grad():
    for j in range(len(b)):
        encoding, geom, seq = b.encodings[j], torch.tensor(b.geoms[j]).to(device), b.seqs[j]
        #regions = b.region_classification[j]
        #CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
        regions = None

        rigid_in, moveable_mask = rigid_in_from_geom_out(b.geoms[j], encoding)
        rigid_true = rigid_from_tensor(geom)
        torsion_true = torsion_angles_from_geom(geom, seq)
        missing_atom_mask = (geom != 0.0).all(-1)
        new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                  rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                  torsion_true=torsion_true,
                                                                  update_mask=moveable_mask.to(device))

        mask = missing_atom_mask * all_atom_mask
        H_mask = torch.tensor(encoding[:, -2] == 1).to(device)
        rigids_mask = torch.tensor(np.array([valid_rigids[x] * [1] + (6 - valid_rigids[x]) * [0] for x in seq]),
                                   device=device, dtype=bool)

        var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
        #var = model.variance_head(new_node_features, encoding)

        if just_var == False and just_mean == False:
            loss = model.aux_loss

            mean_head_loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
            loss = mean_head_loss + var_loss_weighting*var_loss(bb_frame, rigid_true, var, model.band_size)
        elif just_mean:
            loss = model.aux_loss
            loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
        else:
            loss = var_loss(bb_frame, rigid_true, var, model.band_size)

        epoch_val_loss.append(loss.item())
    mean_val_losses.append(np.mean(epoch_val_loss))

########################
# Setting up Optimiser #
########################

# need to change optimisation parameters to include a stop gradient from variance to trunk network

if config["optimizer"] == "radam":
    if just_var == False: 
        optim = RAdam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optim = RAdam([p for n,p in model.named_parameters() if "variance" in n], lr=config["lr"], weight_decay=config["weight_decay"])
else:
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=50, T_mult=1, eta_min=config["lr_min"],
                                                                last_epoch=-1)

if len(sys.argv) > 1:
    config = configs[configs.run_id == sys.argv[1]].iloc[0]
    if os.path.exists(os.path.join(optim_path)) and restart:
        optim.load_state_dict(torch.load(optim_path, map_location=torch.device(device)))

######################
# Beginning Training #
######################

print("Starting first stage of training")
print("{epoch} {train_loss:.4f} {val_loss:.4f}".format(epoch=0, train_loss=10, val_loss=np.mean(epoch_val_loss)))

optim.zero_grad()

batch = int(config["batch"])
#epochs = 10 # instead of 1000 (500 for others)
patience = 0

for i in range(epochs):
    epoch_loss = []
    epoch_val_loss = []
    mean_head_epoch_loss = []
    mean_head_epoch_val_loss = []
    var_head_epoch_loss = []
    var_head_epoch_val_loss = []
    order = np.random.permutation(range(len(a)))
    model.train()
    for k in range(len(a) // batch):
        for l in range(batch):
            j = order[k * batch + l]
            try:
                encoding, geom, seq = a.encodings[j], torch.tensor(a.geoms[j]).to(device), a.seqs[j]
                #regions = b.region_classification[j]
                #CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
                regions = None

                rigid_in, moveable_mask = rigid_in_from_geom_out(a.geoms[j], encoding)
                rigid_true = rigid_from_tensor(geom)
                torsion_true = torsion_angles_from_geom(geom, seq)
                missing_atom_mask = (geom != 0.0).all(-1)
                new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                          rigid_in.to(device), seq,
                                                                          rigid_true=rigid_true,
                                                                          torsion_true=torsion_true,
                                                                          update_mask=moveable_mask.to(device))
                mask = missing_atom_mask * all_atom_mask
                H_mask = torch.tensor(encoding[:, -2] == 1).to(device)
                rigids_mask = torch.tensor(np.array([valid_rigids[x] * [1] + (6 - valid_rigids[x]) * [0] for x in seq]),
                                           device=device, dtype=bool)

                var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
                #var = model.variance_head(new_node_features, encoding)

                if just_var == False and just_mean == False:
                    loss = model.aux_loss
                    mean_head_loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                    v_loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                    loss = mean_head_loss + var_loss_weighting*v_loss
                elif just_mean:
                    loss = model.aux_loss
                    loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                    mean_head_loss = loss
                else:
                    loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                    v_loss = loss
            except IndexError:
                continue
            else:
                if loss.item() != loss.item():
                    print(a.IDs[j])
                    raise Exception("Shit went to Nan.")
                (loss / batch).backward()
                #print([p.grad for name, p in model.named_parameters() if name == "variance_layers.3.weight"])
                #print([p.data for name, p in model.named_parameters() if name == "variance_layers.3.weight"])

        optim.step()
        optim.zero_grad()
        epoch_loss.append(loss.item())
        if just_var == False:
            mean_head_epoch_loss.append(mean_head_loss.item())
        if just_mean == False:
            var_head_epoch_loss.append(v_loss.item())
    model.eval()
    with torch.no_grad():
        for j in range(len(b)):
            encoding, geom, seq = b.encodings[j], torch.tensor(b.geoms[j]).to(device), b.seqs[j]
            #regions = b.region_classification[j]
            #CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
            regions = None

            rigid_in, moveable_mask = rigid_in_from_geom_out(b.geoms[j], encoding)
            rigid_true = rigid_from_tensor(geom)
            torsion_true = torsion_angles_from_geom(geom, seq)
            missing_atom_mask = (geom != 0.0).all(-1)
            new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                      rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                      torsion_true=torsion_true,
                                                                      update_mask=moveable_mask.to(device))
            mask = missing_atom_mask * all_atom_mask
            H_mask = torch.tensor(encoding[:, -2] == 1).to(device)
            rigids_mask = torch.tensor(np.array([valid_rigids[x] * [1] + (6 - valid_rigids[x]) * [0] for x in seq]),
                                       device=device, dtype=bool)

            var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
            #var = model.variance_head(new_node_features, encoding)

            if just_var == False and just_mean == False:
                loss = model.aux_loss
                mean_head_loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                v_loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                loss = mean_head_loss + var_loss_weighting*v_loss
            elif just_mean:
                loss = model.aux_loss
                loss = loss + config["all_atom_fape_loss"] * main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                mean_head_loss = loss
            else:
                loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                v_loss = loss

            epoch_val_loss.append(loss.item())
            if just_var == False:
                mean_head_epoch_val_loss.append(mean_head_loss.item())
            if just_mean == False:
                var_head_epoch_val_loss.append(v_loss.item())

        mean_val_losses.append(np.mean(epoch_val_loss))
        mean_train_losses.append(np.mean(epoch_loss))
        lr_sched.step()

    wandb.log({"train_loss": np.mean(epoch_loss)})
    wandb.log({"val_loss": np.mean(epoch_val_loss)})
    wandb.log({"var_loss_train":np.mean(var_head_epoch_loss)})
    wandb.log({"var_loss_val":np.mean(var_head_epoch_val_loss)})
    wandb.log({"mean_loss_train":np.mean(mean_head_epoch_loss)})
    wandb.log({"mean_loss_val":np.mean(mean_head_epoch_val_loss)})
    wandb.log({"lr": lr_sched.get_last_lr()[0]})

    print("{epoch} {train_loss:.4f} {val_loss:.4f}".format(epoch=i, train_loss=np.mean(epoch_loss),
                                                           val_loss=np.mean(epoch_val_loss)))

    #print(np.min(mean_val_losses), mean_val_losses[-1])
    if np.min(mean_val_losses) == mean_val_losses[-1]:
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), model_path)
        torch.save(optim.state_dict(), optim_path)
        patience = 0
        print("model is saved")

    else:
        # if it doesn't get better after 60 epochs, move on
        # I will reduce it to 40
        if patience < 60:
            patience += 1
        else:
            break

###############################################
# For some reason we train again. Finetuning? #
###############################################

print("Starting second stage of training")
mean_val_losses = []
mean_train_losses = []

if config["optimizer"] == "radam":
    if just_var == False:
        optim = RAdam(model.parameters(), lr=config["lr_min"])
    else:
        optim = RAdam([p for n,p in model.named_parameters() if "variance" in n], lr=config["lr_min"])
else:
    optim = torch.optim.Adam(model.parameters(), lr=config["lr_min"])

patience = 0
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

if os.path.exists(os.path.join(optim_path)):
    optim.load_state_dict(torch.load(optim_path, map_location=torch.device(device)))

optim.zero_grad()

for i in range(epochs):
    epoch_loss = []
    epoch_val_loss = []
    mean_head_epoch_loss = []
    mean_head_epoch_val_loss = []
    var_head_epoch_loss = []
    var_head_epoch_val_loss = []
    order = np.random.permutation(range(len(a)))
    model.eval()
    for k in range(len(a) // batch):
        for l in range(batch):
            j = order[k * batch + l]
            try:
                encoding, geom, seq = a.encodings[j], torch.tensor(a.geoms[j]).to(device), a.seqs[j]
                #regions = b.region_classification[j]
                #CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
                regions = None
                CDR_mask = None

                rigid_in, moveable_mask = rigid_in_from_geom_out(a.geoms[j], encoding)
                rigid_true = rigid_from_tensor(geom)
                torsion_true = torsion_angles_from_geom(geom, seq)
                missing_atom_mask = (geom != 0.0).all(-1)
                new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                          rigid_in.to(device), seq,
                                                                          rigid_true=rigid_true,
                                                                          torsion_true=torsion_true,
                                                                          update_mask=moveable_mask.to(device),
                                                                          CDR_mask=CDR_mask)
                mask = missing_atom_mask * all_atom_mask
                H_mask = torch.tensor(encoding[:, -2] == 1).to(device)
                rigids_mask = torch.tensor(np.array([valid_rigids[x] * [1] + (6 - valid_rigids[x]) * [0] for x in seq]),
                                           device=device, dtype=bool)

                var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
                #var = model.variance_head(new_node_features, encoding)

                if just_var == False and just_mean == False:
                    loss = model.aux_loss
                    loss = loss + config["all_atom_fape_loss"] * \
                        main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                    loss = loss + min(i / 50 + restart, 1) * config["bond_ideality_loss"] * (
                            dist_check(all_atoms, geom) + angle_check(all_atoms, geom))
                    mean_head_loss = loss + config["clash_loss"] * clash_check(all_atoms, all_atom_mask, seq)
                    v_loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                    loss = mean_head_loss + v_loss
                elif just_mean:
                    loss = model.aux_loss
                    loss = loss + config["all_atom_fape_loss"] * \
                        main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                    loss = loss + min(i / 50 + restart, 1) * config["bond_ideality_loss"] * (
                            dist_check(all_atoms, geom) + angle_check(all_atoms, geom))
                    loss = loss + config["clash_loss"] * clash_check(all_atoms, all_atom_mask, seq)
                    mean_head_loss = loss
                else:
                    loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                    v_loss = loss

            except IndexError:
                continue
            else:
                if loss.item() != loss.item():
                    print(a.IDs[j])
                (loss / batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optim.step()
        optim.zero_grad()

        epoch_loss.append(loss.item())
        if just_var == False:
            mean_head_epoch_loss.append(mean_head_loss.item())
        if just_mean == False:
            var_head_epoch_loss.append(v_loss.item())

    model.eval()
    with torch.no_grad():
        for j in range(len(b)):
            encoding, geom, seq = b.encodings[j], torch.tensor(b.geoms[j]).to(device), b.seqs[j]
            #regions = b.region_classification[j]
            #CDR_mask = torch.tensor([x in [1, 3, 5, 8, 10, 12] for x in regions], device=device)
            regions = None
            CDR_mask = None

            rigid_in, moveable_mask = rigid_in_from_geom_out(b.geoms[j], encoding)
            rigid_true = rigid_from_tensor(geom)
            torsion_true = torsion_angles_from_geom(geom, seq)
            missing_atom_mask = (geom != 0.0).all(-1)
            new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = model(torch.tensor(encoding).to(device),
                                                                      rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                      torsion_true=torsion_true,
                                                                      update_mask=moveable_mask.to(device),
                                                                      CDR_mask=CDR_mask)
            mask = missing_atom_mask * all_atom_mask
            H_mask = torch.tensor(encoding[:, -2] == 1).to(device)
            rigids_mask = torch.tensor(np.array([valid_rigids[x] * [1] + (6 - valid_rigids[x]) * [0] for x in seq]),
                                       device=device, dtype=bool)

            var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
            #var = model.variance_head(new_node_features, encoding)

            if just_var == False and just_mean == False:
                loss = model.aux_loss
                loss = loss + config["all_atom_fape_loss"] * \
                    main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                loss = loss + min(i / 50 + restart, 1) * config["bond_ideality_loss"] * (
                        dist_check(all_atoms, geom) + angle_check(all_atoms, geom))
                mean_head_loss = loss + config["clash_loss"] * clash_check(all_atoms, all_atom_mask, seq)
                loss = mean_head_loss + var_loss(bb_frame, rigid_true, var, model.band_size)
            elif just_mean:
                loss = model.aux_loss
                loss = loss + config["all_atom_fape_loss"] * \
                    main_loss(all_reference_frames, global_frames_from_geom(geom, seq), all_atoms, geom, rigids_mask, mask, regions)
                loss = loss + min(i / 50 + restart, 1) * config["bond_ideality_loss"] * (
                        dist_check(all_atoms, geom) + angle_check(all_atoms, geom))
                loss = loss + config["clash_loss"] * clash_check(all_atoms, all_atom_mask, seq)
                mean_head_loss = loss
            else:
                loss = var_loss(bb_frame, rigid_true, var, model.band_size)
                v_loss = loss

            epoch_val_loss.append(loss.item())
            if just_var == False:
                mean_head_epoch_val_loss.append(mean_head_loss.item())
            if just_mean == False:
                var_head_epoch_val_loss.append(v_loss.item())

        mean_val_losses.append(np.mean(epoch_val_loss))
        mean_train_losses.append(np.mean(epoch_loss))

    wandb.log({"lr": lr_sched.get_last_lr()[0]})
    wandb.log({"train_loss": np.mean(epoch_loss)})
    wandb.log({"val_loss": np.mean(epoch_val_loss)})
    wandb.log({"var_loss_train":np.mean(var_head_epoch_loss)})
    wandb.log({"var_loss_val":np.mean(var_head_epoch_val_loss)})
    wandb.log({"mean_loss_train":np.mean(mean_head_epoch_loss)})
    wandb.log({"mean_loss_val":np.mean(mean_head_epoch_val_loss)})

    print("{epoch} {train_loss:.4f} {val_loss:.4f}".format(epoch=i, train_loss=np.mean(epoch_loss),
                                                           val_loss=np.mean(epoch_val_loss)))
    if np.min(mean_val_losses) == mean_val_losses[-1]:
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), model_path)
        torch.save(optim.state_dict(), optim_path)
        patience = 0

    else:
        # adamend is 0 in config so we never switch over
        if (patience < 30) and (int(config["adamend"]) > 0):
            if just_var == False:
                optim = torch.optim.Adam(model.parameters(), lr=1e-4)
            else:
                optim = torch.optim.Adam([p for n,p in model.named_parameters() if "variance" in n], lr=1e-4)
        if patience < 75:#75#120
            patience += 1
        else:
            break

######################
# Evaluate the Model #
######################

print("Model evaluation")

true_files, pred_files, refined_files = [], [], []
covar_files = []

if not os.path.exists(os.path.join(benchmark_path, "preds", "ens_" + str(ens_idx))):
    os.mkdir(os.path.join(benchmark_path, "preds", "ens_" + str(ens_idx)))
    os.mkdir(os.path.join(benchmark_path, "refined", "ens_" + str(ens_idx)))
    
if not os.path.exists(os.path.join(benchmark_path, "covars", "ens_" + str(ens_idx))):
    os.mkdir(os.path.join(benchmark_path, "covars", "ens_" + str(ens_idx)))

with torch.no_grad():
    best_model.eval()
    for j in range(len(b)):
        encoding, geom, seq = b.encodings[j], torch.tensor(b.geoms[j]).to(device), b.seqs[j]

        rigid_in, moveable_mask = rigid_in_from_geom_out(b.geoms[j], encoding)
        rigid_true = rigid_from_tensor(geom)
        torsion_true = torsion_angles_from_geom(geom, seq)
        missing_atom_mask = (geom != 0.0).all(-1)
        new_node_features, all_reference_frames, all_atoms, all_atom_mask, bb_frame = best_model(torch.tensor(encoding).to(device),
                                                                       rigid_in.to(device), seq, rigid_true=rigid_true,
                                                                       torsion_true=torsion_true,
                                                                       update_mask=moveable_mask.to(device))

        var = model.variance_head(new_node_features.detach().requires_grad_(), encoding)
        #var = model.variance_head(new_node_features, encoding)

        pred_txt = to_pdb(all_atoms, seq, encoding)
        true_txt = to_pdb(geom, seq, encoding)

        true_files.append(os.path.join(benchmark_path, "truth", b.IDs[j] + ".pdb"))
        pred_files.append(os.path.join(benchmark_path, "preds", "ens_" + str(ens_idx), b.IDs[j] + ".pdb"))
        refined_files.append(os.path.join(benchmark_path, "refined", "ens_" + str(ens_idx), b.IDs[j] + ".pdb"))

        covar_files.append(os.path.join(benchmark_path, "covars", "ens_" + str(ens_idx), b.IDs[j] + ".pt"))

        with open(true_files[-1], "w+") as file:
            file.write(true_txt)
        with open(pred_files[-1], "w+") as file:
            file.write(pred_txt)
        torch.save(var, covar_files[-1])

from metrics import abangle_errors, exposed_accuracy, sidechain_accuracy, count_clashes, refine, parser, CDR_rmsds

results = []

for j in range(len(true_files)):
    true_file = true_files[j]
    #out_files = {"unrefined": pred_files[j], "refined": refined_files[j]}
    out_files = {"unrefined": pred_files[j]}

    #refine(out_files["unrefined"], out_files["refined"])
    result = {}
    true_ab = parser.get_structure(true_file, true_file)

    for ref in out_files:
        pred_ab = parser.get_structure(out_files[ref], out_files[ref])

        rmsds = CDR_rmsds(true_ab, pred_ab)
        result.update({ref + "_" + x: rmsds[x] for x in rmsds})

        # Calculates the errors of angles between the chains in the protein - we don't have chains so we don't calculate this
        #abangles = abangle_errors(true_ab, pred_ab)
        #result.update({ref + "_" + x: abangles[x] for x in abangles})

        side_chain = sidechain_accuracy(true_ab, pred_ab)
        result.update({ref + "_" + x: side_chain[x] for x in side_chain})

        result[ref + "_exposed_accuracy"] = exposed_accuracy(true_ab, pred_ab)
        result[ref + "_clashes"] = count_clashes(pred_ab)

    results.append(result)

average_results = {x: np.mean([result[x] for result in results]) for x in results[0]}
average_results["run_id"] = config["run_id"]

####################
# Save the results #
####################

print("Saving the results")

results_file = os.path.join(data_path, "results_ens_" + str(ens_idx) + ".csv")

if os.path.exists(results_file):
    res = pd.read_csv(results_file)
else:
    res = pd.DataFrame()

res = res.append(average_results, ignore_index=True)
res.to_csv(results_file, index=False)



# Mark run as finished
configs = pd.read_csv(config_path)
configs.at[run_number_i, "state"] = "DONE"
configs.to_csv(config_path, index=False)



print("All done! :)")
wandb.finish()

#profiler.disable()
#profiler.dump_stats("profiler_results.prof")
