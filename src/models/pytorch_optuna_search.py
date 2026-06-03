import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import rank_IC_torch

class RollingOptunaSearchPyTorch():

    def __init__(self, model_builder, input_features, device, min_train_size=24, n_trials=30, sampler=None, pruner=None, verbose=1, hpt_epochs=100):
        self.model_builder = model_builder
        self.input_features = input_features
        self.device = device
        self.min_train_size = min_train_size
        self.n_trials = n_trials
        self.verbose = verbose
        self.sampler = sampler or optuna.samplers.TPESampler(multivariate=True, seed=42)
        self.pruner = pruner or optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=10)

        self.hpt_epochs = hpt_epochs

    def _build_objective(self, Xs, y, lens):


        def objective(trial):

            scores = []

            for i, (train_len,t) in enumerate(lens):

                if len(Xs[t]) <= train_len:
                    continue

                model = self.model_builder(trial, self.input_features).to(self.device)

                train_dataloader = DataLoader(TensorDataset(Xs[t][:train_len], y[:train_len]),batch_size=model.batch_size,shuffle=False)
                val_dataloader = DataLoader(TensorDataset(Xs[t][train_len:], y[train_len:len(Xs[t])]),batch_size=model.batch_size,shuffle=False)

                score = train_model(model, {"train" : train_dataloader, "val" : val_dataloader},optim.AdamW(model.parameters(),lr=model.lr,weight_decay=model.wd),nn.HuberLoss(delta=0.5),self.hpt_epochs,0)
                scores.append(score)

                current_mean = np.mean(scores)

                trial.report(current_mean, step=i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            mu = np.mean(scores)

            return mu
        
        return objective

    def fit(self, X, y):

        X = [torch.from_numpy(x).float().to(self.device, non_blocking=True) for x in X]
        y = torch.from_numpy(y).float().to(self.device, non_blocking=True)

        start_t = max(1, self.min_train_size)
        lens = [(len(X[t-1]), t) for t in range(start_t, len(X))]

        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner
        )

        study.optimize(self._build_objective(X, y, lens), n_trials=self.n_trials)

        self.study_ = study
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value

        self.completed_trials = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        self.pruned_trials = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)

        if self.verbose:
            print(f"Trials: {len(study.trials)} total, {self.completed_trials} complete, {self.pruned_trials} pruned")
            print("Best params:", self.best_params_)
            print("Best score:", self.best_score_)

        self.best_model_ = self.model_builder(optuna.trial.FixedTrial(self.best_params_), self.input_features).to(self.device)
        train_dataloader = DataLoader(TensorDataset(X[-1],y),batch_size=self.best_model_.batch_size,shuffle=False)
        self.best_model_ = train_model(self.best_model_,{"train": train_dataloader},optim.AdamW(self.best_model_.parameters(),lr=self.best_model_.lr,weight_decay=self.best_model_.wd),nn.HuberLoss(delta=0.5),300,1)

        return self

    def predict(self, X):
        self.best_model_.eval()
        with torch.no_grad():
            return self.best_model_(X).cpu().numpy()    
        
def train_model(model,dataloaders,optimizer,criterion,epochs,verbose=0):
    if verbose:
        run_train_loss, run_val_loss, run_bids_score = [], [], []

    ema_bids = None
    max_bids = -np.inf
    alpha = 0.9

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    for e in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in dataloaders["train"]:

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x),batch_y)
            loss.backward()
            optimizer.step()

            if verbose:
                train_loss += loss.item() * batch_y.size(0)               
        
        if "val" in dataloaders:
            model.eval()
            with torch.no_grad():

                preds = torch.cat([model(batch_x) for batch_x, _ in dataloaders["val"]])
                targets = torch.cat([batch_y for _, batch_y in dataloaders["val"]])

                current_bids = rank_IC_torch(targets, preds).item()
                ema_bids = current_bids if ema_bids is None else (alpha * ema_bids) + ((1 - alpha) * current_bids)

                if ema_bids > max_bids:
                    max_bids = ema_bids
        
        scheduler.step()     
        
        if verbose:
            train_loss /= len(dataloaders["train"].dataset)  
            print(f"Epoch {e+1}/{epochs}.. ")
            print(f"Train loss: {train_loss:.6f}.. ")
            if "val" in dataloaders:
                val_loss = criterion(preds,targets).item()
                print(f"Validation Loss: {val_loss}")
                print(f"BIDS Score: {ema_bids}")
        
            run_train_loss.append(train_loss)
            if "val" in dataloaders:
                run_val_loss.append(val_loss)
                run_bids_score.append(ema_bids)
            
    
    if verbose:
        epochs_range = range(1, e+2)
        plt.plot(epochs_range, run_train_loss, label='Train Loss')
        if "val" in dataloaders:
            plt.plot(epochs_range,run_val_loss, label="Validation Loss")
            plt.plot(epochs_range,run_bids_score, label="BIDS EMA Score")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        plt.show()

    
    return max_bids if "val" in dataloaders else model