import torch
import torch.nn as nn
import numpy as np
import wandb

class FFNetMeanVar3D(nn.Module):
    """
    Parametric curve regression model
    FeedForward neural network with Leaky-ReLU activations.
    Trunk network with depth 2, then a mean head of
    depth 2 and a variance head of depth 4.
    """
    def __init__(self, input, output, n_points, batch, band, inner_width=150):
        super(FFNetMeanVar3D, self).__init__()
        assert input == n_points
        assert output == input*3
        self.band_size = band
        self.activation = nn.functional.leaky_relu
        self.batch = batch
        self.fc1 = nn.Linear(input, 70)
        self.inner_layers = nn.ModuleList([nn.Linear(70, inner_width)])
        self.mean_output = nn.Linear(inner_width, output)
        self.var_output = nn.Linear(inner_width, n_points * self.band_size)
        self.mean_layers = nn.ModuleList([nn.Linear(inner_width, inner_width) for i in range(2)])
        self.var_layers = nn.ModuleList([nn.Linear(inner_width, inner_width) for i in range(4)])

        torch.nn.init.kaiming_normal_(self.fc1.weight, a=0.01)
        torch.nn.init.kaiming_normal_(self.mean_output.weight, a=0.01)
        torch.nn.init.kaiming_normal_(self.var_output.weight, a=0.01)

        for layer in self.inner_layers:
            torch.nn.init.kaiming_normal_(layer.weight, a=0.01)
        for layer in self.mean_layers:
            torch.nn.init.kaiming_normal_(layer.weight, a=0.01)
        for layer in self.var_layers:
            torch.nn.init.kaiming_normal_(layer.weight, a=0.01)

    """
    Produces a mean prediction of size (batch size, data size x 3) and 
    a variance prediction of size (batch size, data size x band size)
    """
    def forward(self, x):
        x1 = self.activation(self.fc1(x.flatten(start_dim=1)))
        for layer in self.inner_layers:
            x1 = self.activation(layer(x1))
        var_x = torch.clone(x1).detach().requires_grad_()
        for layer in self.mean_layers:
            x1 = self.activation(layer(x1))
        for layer in self.var_layers:
            var_x = self.activation(layer(var_x))
        mean_out = self.mean_output(x1)
        var_out = self.var_output(var_x).reshape(1, -1, self.band_size)
        var_out = torch.cat([var_out[:,:,:-1], torch.nn.functional.softplus(var_out[:,:,-1]).unsqueeze(dim=-1)], dim=2).reshape(1,-1)
        return mean_out, var_out

    """
    Reshape the network predictions to be of size (batch size, data size, 3)
    for the mean and (batch size, data size x band size) for the variance
    """
    def reshape_preds(self, pred, var, batch):
        return pred.reshape(batch, -1, 3), var.reshape(batch, -1, self.band_size)

    """
    Converts the banded output of the variance head to a lower
    triangular positive semi definite matrix L (cholesky matrix
    for the cholesky decomposition of the covariance matrix)
    """
    def to_chol_matrix(self, var_output):
        batch = var_output.shape[0]
        n = var_output.shape[1]
        L = torch.zeros(batch, n, n)
        half_band_size = (self.band_size + 1) // 2
        for b in range(batch):
            for i in range(n):
                for j in range(half_band_size):
                    if i - (half_band_size - 1 - j) >= 0:
                        L[b, i, i - (half_band_size - 1 - j)] = var_output[b, i, j]
        L.requires_grad_()
        return L

    def train_one_epoch(self, device, train_loader, optimiser, loss_fn, scheduler, epoch):
        iters = len(train_loader)
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device).requires_grad_(), target.to(device).requires_grad_()
            mean, var = self.forward(data)

            loss = loss_fn(mean, target, var)
            optimiser.zero_grad()
            loss.backward()
            for p in self.parameters():
                torch.nn.utils.clip_grad_norm_(p, 2)   # clamp the gradient to prevent explosion
            optimiser.step()
            scheduler.step(epoch + i / iters)

        wandb.log({"train_loss": loss})

    def gaussian_log_likelihood(self, pred, target, var):
        eps = 0.01
        if self.band_size != 1:
            var = var.reshape(1,-1,self.band_size)

        log_det = torch.sum(torch.log(var[:,:,-1]**2 + eps), dim=-1)    # Determinant of a lower triangular matrix is the
                                                                        # product of squared diagonal entries
        L = self.to_chol_matrix(var) + eps                              # Add epsilon to guarantee large enough determinant
                                                                        # for inversion
        part_quad_term = torch.linalg.solve(L, pred-target)             # stable inversion of L

        # computation of log likelihood
        loss = -0.5 * log_det - 0.5 * 1/3 * torch.pow(torch.norm(part_quad_term),2) # Take the mean over the coordinates
        return loss
    
    def mean_var_loss(self, pred, target, var):
        pred, var = self.reshape_preds(pred, var, pred.shape[0])
        mean_loss = 0.5 * torch.nn.MSELoss(reduction = 'mean')(pred, target)
        var_loss = torch.mean(self.gaussian_log_likelihood(pred.detach(), target, var))
        return mean_loss - var_loss
    
    def train_model(self, args, train_loader, val_loader):
        optimiser = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay=0.0001)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=50, T_mult=1, eta_min=0.0001)
        loss_fn = self.mean_var_loss
      
        best_val_loss = np.inf
        best_val_mse_loss = np.inf
        patience = 0
        for epoch in range(args.num_epochs):
            self.train(True)
            self.train_one_epoch(args.device, train_loader, optimiser, loss_fn, lr_sched, epoch)

            wandb.log({"mse_loss": self.eval_model(args, train_loader, torch.nn.MSELoss())})

            if epoch % args.epoch_print_rate == 0:
                print('Train Epoch: {} \tLoss: {:.6f} \tVal Loss: {:.6f}'.format(
                    epoch, self.eval_model(args, train_loader, torch.nn.MSELoss()), self.eval_model(args, val_loader, torch.nn.MSELoss())))
                if args.dry_run:
                    break
            
            wandb.log({"lr": lr_sched.get_last_lr()[0]})

            val_loss = self.eval_model(args, val_loader, loss_fn, need_var=True)
            val_mse_loss = self.eval_model(args, val_loader, torch.nn.MSELoss())
            wandb.log({"val_loss": val_loss})
            wandb.log({"val_mse_loss": val_mse_loss})

            # Early stopping
            if epoch > 500:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    if patience < 500:
                        patience += 1
                    else:
                        break

    def eval_model(self, args, data_loader, loss_fn, need_var = False):
        self.eval()
        loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data, target = data.to(args.device), target.to(args.device)
                mean, var = self.forward(data)
                if need_var:
                    loss += loss_fn(mean, target, var)
                else:
                    mean, var = self.reshape_preds(mean, var, mean.shape[0])
                    loss += loss_fn(mean, target)

        loss /= len(data_loader.dataset)
        return loss

    """
    Helper function for plots and evaluation to return all predictions made by the
    model and the input and target values they correspond to.
    """
    def get_all_predictions(self, args, data_loader):
        self.eval()
        t_values, z_values, means, vars = [], [], [], []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(args.device), target.to(args.device)
                mean, var = self.forward(data)
                if len(z_values) == 0:
                    t_values = data.detach().numpy()
                    z_values = target.detach().numpy()
                    means = mean.detach().numpy()
                else:
                    t_values = np.vstack([t_values, data.detach().numpy()])
                    z_values = np.vstack([z_values, target.detach().numpy()])
                    means = np.vstack([means, mean.detach().numpy()])
                vars.extend(var.detach().numpy())
        t_values, z_values, means, vars = np.array(t_values), np.array(z_values), np.array(means), np.array(vars)
        return t_values, z_values, means, vars
    
class FFNetMeanVar(nn.Module):
    """
    Simple regression problem
    FeedForward neural network with ReLU activations
    The network outputs a mean and a variance over its prediction
    """
    def __init__(self, input, output):
        super(FFNetMeanVar, self).__init__()
        self.fc1 = nn.Linear(input, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 400)
        self.fc4 = nn.Linear(400, 500)
        self.fc5 = nn.Linear(500, 300)
        self.fc6 = nn.Linear(300, 100)
        self.mean_output = nn.Linear(100, output)
        self.var_output_1 = nn.Linear(100, 50)
        self.var_output_2 = nn.Linear(50, output**2)
        self.activation = nn.functional.relu
        self.output_size = output

    def forward(self, x):
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        x3 = self.activation(self.fc3(x2))
        x4 = self.activation(self.fc4(x3))
        x5 = self.activation(self.fc5(x4))
        x6 = self.activation(self.fc6(x5))
        mean_out = self.mean_output(x6)
        var_out_1 = self.activation(self.var_output_1(x6.detach().requires_grad_()))
        var_out = torch.nn.functional.softplus(self.var_output_2(var_out_1))
        return mean_out, var_out
    
    def train_one_epoch(self, device, train_loader, optimiser, loss_fn):
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device).requires_grad_(), target.to(device).requires_grad_().reshape(-1,1)
            mean, var = self.forward(data)

            loss = loss_fn(mean, target, var)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    def gaussian_log_likelihood(self, pred, target, var, reduction = 'sum'):
        loss = torch.sum(-0.5 * torch.log(var) - 0.5 * torch.pow(pred - target, 2) / var)
        return loss
    
    def mean_var_loss(self, pred, target, var):
        batch_size = len(pred)
        mean_loss = 0.5 * torch.nn.MSELoss(reduction = 'sum')(pred, target)
        var_loss = self.gaussian_log_likelihood(pred.detach(), target, var)
        return (1/batch_size) * (mean_loss - var_loss)
    
    def train_model(self, args, train_loader):
        optimiser = torch.optim.SGD(self.parameters(), lr = args.lr)
        loss_fn = self.mean_var_loss
        for epoch in range(args.num_epochs):
            self.train(True)
            self.train_one_epoch(args.device, train_loader, optimiser, loss_fn)

            if epoch % args.epoch_print_rate == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, self.eval_model(args, train_loader, torch.nn.MSELoss())))
                if args.dry_run:
                    break

    def eval_model(self, args, data_loader, loss_fn, need_var = False):
        self.eval()
        loss = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(args.device), target.to(args.device)
                mean, var = self.forward(data)
                if need_var:
                    loss += loss_fn(mean, target, var)
                else:
                    loss += loss_fn(mean, target)

        loss /= len(data_loader.dataset)
        return loss
    
    def get_all_predictions(self, args, data_loader):
        self.eval()
        x_values, y_values, means, vars = [], [], [], []
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(args.device), target.to(args.device)
                mean, var = self.forward(data)
                x_values.extend(data.detach().numpy().flatten())
                y_values.extend(target.detach().numpy().flatten())
                means.extend(mean.detach().numpy().flatten())
                vars.extend(var.detach().numpy().flatten())
        x_values, y_values, means, vars = np.array(x_values), np.array(y_values), np.array(means), np.array(vars)
        return x_values, y_values, means, vars
