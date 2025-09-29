"""
At the moment this contains a class for fully-connected DNNs.
"""

import os
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
import torch.nn as nn
import torch.nn.functional as F

from qsprpred.tasks import ModelTasks
from .base_torch import QSPRModelPyTorchGPU, DEFAULT_TORCH_GPUS
from ....data.sampling.splits import DataSplit
from ....data.tables.qspr import QSPRDataset
#from ....extra.gpu.models.graph_neural_network import GGNN
#STFullyConnected, Base
from ....models.early_stopping import EarlyStoppingMode, early_stopping
from ....models.monitors import BaseMonitor, FitMonitor


import dgl
from dgl.nn import GatedGraphConv, GlobalAttentionPooling
from dgl.dataloading import GraphDataLoader
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import math
from rdkit import Chem



class GGNN(nn.Module):
    def __init__(
        self,
        n_dim,
        device,
        gpus,
        is_reg,
        patience,
        tol,
        parameters,
        n_class=2
    ):
    
        super().__init__()
        print("GGNN updated")

        self.n_dim = n_dim
        self.n_hidden_layers = parameters['n_hidden_layers']
        self.dropout = nn.Dropout(parameters['dropout_rate'])
        self.in_feats = parameters['in_feats']
        self.n_steps = parameters['n_steps']
        self.n_etypes = parameters['n_etypes']
        self.out_feats = parameters['n_dim']
        self.layers = nn.ModuleList()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.gpus = gpus
        self.n_cls = n_class
        self.is_reg = False
        self.patience = patience
        self.tol = tol
        
        for i in range(self.n_hidden_layers):
            if i == 0:
                set_in_feats = self.in_feats
            else:
                set_in_feats = self.out_feats

            layer = GatedGraphConv(
                in_feats=set_in_feats,
                out_feats=self.out_feats,
                n_steps=self.n_steps,
                n_etypes=self.n_etypes
            )
            self.layers.append(layer)        
        pooling_gate_nn = nn.Linear(self.out_feats, 1)
        self.pooling = GlobalAttentionPooling(pooling_gate_nn) 
        self.output_layer = nn.Linear(self.out_feats, self.n_cls)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, graph, features):
        

        features = features.to(graph.device) 
        
        h = F.relu(self.layers[0](graph, features))
        h = self.dropout(h)
        for i in range(self.n_hidden_layers):

            if i == 0:
                continue
            else:
                h = F.relu(self.layers[i](graph, h))  
                if i < self.n_hidden_layers - 1:
                    h = self.dropout(h)
        h = self.pooling(graph, h)
        h = self.output_layer(h)
        return h
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def set_parameters(self, parameters):
        ...
        
    
    def collate(self, samples): 
        graphs = [s[0] for s in samples]
        batched_graph = dgl.batch(graphs)
        return batched_graph

    def getLoader(self, X, y, batch_size, schuffle=True, include_labels=False):
        graphs, labels = [], []
        for i in range(len(X)):

            smiles = X[i][0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[WARNING] Skipping invalid SMILES at index {i}: {smiles}")
                continue
            graph = mol_to_bigraph( 
                mol,              
                node_featurizer=CanonicalAtomFeaturizer(),
                edge_featurizer=CanonicalBondFeaturizer(),
                explicit_hydrogens=False
            )
            graph = dgl.add_self_loop(graph)
            graphs.append(graph)
            if include_labels:
                label = y[i]
                labels.append(label)
        if include_labels:
            labels = torch.tensor(labels).unsqueeze(1).squeeze()

            loader = GraphDataLoader(
                list(zip(graphs, labels)),
                batch_size=batch_size,
                shuffle=True if schuffle else False,
                collate_fn=self.collate,
                num_workers=0)
        else:
            loader = GraphDataLoader(
                list(zip(graphs)),
                batch_size=batch_size,
                shuffle=True if schuffle else False,
                collate_fn=self.collate,
                num_workers=0)
        return loader
    
    class EarlyStopping():
        def __init__(
            self,
            patience_loss=10,
            patience_mcc=10,
            verbose=True,
            delta_loss=0.001,
            delta_mcc=0.001,
        ):
            self.patience_loss = patience_loss
            self.patience_mcc = patience_mcc
            self.verbose = verbose
            self.loss_counter = 0
            self.mcc_counter = 0
            self.best_loss = np.inf
            self.best_mcc = -1     
            self.early_stop = False           
            self.delta_loss = 0.001
            self.delta_mcc = 0.001
            self.best_epoch = 0
        def __call__(self, val_loss, val_acc, val_mcc, model, epoch):   
            improved_loss = False
            improved_mcc = False

            if val_loss < self.best_loss - self.delta_loss:  
                self.best_loss = val_loss   
                self.loss_counter = 0  
                improved_loss = True
            else:
                self.loss_counter += 1

            if val_mcc > self.best_mcc + self.delta_mcc:
                self.best_mcc = val_mcc
                self.mcc_counter = 0
                improved_mcc = True
            else:
                self.mcc_counter += 1

            if improved_loss or improved_mcc:  
                self.save_checkpoint(model, val_loss, val_mcc, val_acc)
                self.best_epoch = epoch

            if self.loss_counter >= self.patience_loss and self.mcc_counter >= self.patience_mcc: #
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

        def save_checkpoint(self, model, val_loss, val_mcc, val_acc):
            self.best_weights = self.state_dict()
            torch.save(model.state_dict(), "ggnn_checkpoint.pt")
            if self.verbose:
                print(f"Checkpoint saved, mcc: {val_mcc}, loss: {val_loss}, accuracy: {val_acc}")

            
    def fit(self, X, y, Xval=None, yval=None, monitor=None, num_epochs=0, optimizer=None, criterion=None, scheduler=None, accumulation_steps=2):
        print("Fitting...")

        #X = X.flatten().tolist()
        #have_val = True
        train_loader = self.getLoader(X, y, batch_size=128, schuffle=True, include_labels=True)
        if (Xval is not None and yval is not None):
            #have_val = False
            val_loader = self.getLoader(Xval, yval, batch_size=128, schuffle=False, include_labels=True)
        
        
        train_losses, val_losses = [], []
        scaler = torch.GradScaler()    
        self.best_weights = self.state_dict()
        best_epoch = 0
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            optimizer.zero_grad() 
            for batch_idx, (batched_graph, labels) in enumerate(train_loader):
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device) 
                batched_graph.ndata['h'] = batched_graph.ndata['h'].float().to(self.device)
                with torch.autocast():    
                    logits = self(batched_graph, batched_graph.ndata['h'].float())
                    loss = criterion(logits, labels) / accumulation_steps 
                scaler.scale(loss).backward() 
                train_loss += loss.item() * accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1: 
                    scaler.step(optimizer) 
                    scaler.update()  
                    optimizer.zero_grad() 
            train_loss = train_loss/len(train_loader)
            train_losses.append(train_loss)

            validation_loss = 0.0
            validation_accuraccy = 0.0
            num_val_correct = 0   
            num_total = 0            
            TP, TN, FP, FN = 0, 0, 0, 0
            if val_loader is not None:  
                self.eval()
                with torch.no_grad():  
                    for batched_graph, labels in val_loader:  
                        batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)  
                        batched_graph.ndata['h'] = batched_graph.ndata['h'].to(self.device) 
                        with torch.autocast(): 
                            logits = self(batched_graph, batched_graph.ndata['h'].float())
                            loss = criterion(logits, labels)    # we compute the loss
                        validation_loss += loss          
                        _, predicted = torch.max(logits.data, 1)    
                        num_total += labels.size(0)           
                        num_val_correct += (predicted == labels).sum().item()   
                        TP += ((predicted == 1) & (labels == 1)).sum().item()
                        TN += ((predicted == 0) & (labels == 0)).sum().item()
                        FP += ((predicted == 1) & (labels == 0)).sum().item()
                        FN += ((predicted == 0) & (labels == 1)).sum().item()
                    num = TP * TN - FP * FN
                    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                    validation_mcc = num / den if den > 0 else 0
                    validation_loss = validation_loss/len(val_loader)   # we get the average loss
                    val_losses.append(validation_loss)
                    validation_accuraccy = num_val_correct/num_total    # saving for early stopping
                    if early_stopping:  # checking if early stopping is not None
                        early_stopping(validation_loss, validation_accuraccy, validation_mcc, self, epoch + 1)
                        best_epoch = early_stopping.best_epoch
                        if early_stopping.early_stop:
                            print(f"Early stopping triggered at epoch {epoch + 1}")
                            break
                    if (epoch + 1) % 5 == 0 or epoch == 0:
                        print(f'Epoch {epoch + 1}/{num_epochs} '
                              f'Train loss: {train_loss:.4f} '
                              f'Val loss: {validation_loss:.4f} '
                              f'Val accuracy: {100 * validation_accuraccy:.2f}% '
                              f'MCC: {validation_mcc}')
                        
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            elif scheduler is not None: 
                scheduler.step()
                       
        self.load_state_dict(self.best_weights)
        return self, best_epoch
    
    def predict(self, X):
        test_loader = self.getLoader(X, y=None, batch_size=128, schuffle=False, include_labels=False)
        self.eval()
        all_proba = []
        with torch.no_grad():      
            for batched_graph in test_loader:       
                batched_graph = batched_graph.to(self.device)   
                node_features = batched_graph.ndata['h'].to(self.device).float()
                logits = self(batched_graph, node_features)

                proba = torch.softmax(logits, dim=1)

                all_proba.append(proba.cpu().numpy())
        all_proba = np.vstack(all_proba)
        return all_proba




class DNNModel(QSPRModelPyTorchGPU):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is performed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
        patience (int):
            number of epochs to wait before early stop if no progress
            on validation set score
        tol (float):
            minimum absolute improvement of loss necessary to count as
            progress on best validation score
        nClass (int): number of classes
        nDim (int): number of features
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
    """

    def getGPUs(self):
        return self.gpus

    def setGPUs(self, gpus: list[int]):
        self.gpus = gpus
        if not isinstance(self.estimator, str):
            self.estimator.gpus = gpus
        if torch.cuda.is_available() and gpus:
            self.setDevice(f"cuda:{gpus[0]}")
        else:
            self.setDevice("cpu")

    def getDevice(self) -> torch.device:
        return self.device

    def setDevice(self, device: str):
        self.device = torch.device(device)
        if isinstance(self.estimator, Base):
            self.estimator.device = self.device

    def __init__(
            self,
            base_dir: str,
            alg: Type = GGNN,
            name: str | None = None,
            parameters: dict | None = None,
            random_state: int | None = None,
            autoload: bool = True,
            gpus: list[int] = DEFAULT_TORCH_GPUS,
            patience: int = 50,
            tol: float = 0,
    ):
        """Initialize a DNNModel model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_TORCH_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_TORCH_GPUS`.
            patience (int, optional):
                number of epochs to wait before early stop if no progress
                on validation set score. Defaults to 50.
            tol (float, optional):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score. Defaults to 0.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.gpus = None
        self.patience = patience
        self.tol = tol
        self.nClass = 2
        self.nDim = parameters['n_dim']
        super().__init__(
            base_dir,
            alg,
            name,
            parameters,
            autoload=autoload,
            random_state=random_state,
        )
        self.setGPUs(gpus)

    def initRandomState(self, random_state):
        """Set random state if applicable.
        Defaults to random state of dataset if no random state is provided by the constructor.

        Args:
            random_state (int): Random state to use for shuffling and other random operations.
        """
        super().initRandomState(random_state)
        if random_state is not None:
            torch.manual_seed(random_state)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return True

    def initFromDataset(self, data: QSPRDataset | None):
        super().initFromDataset(data)
        if self.targetProperties[0].task.isRegression():
            self.nClass = 1
        elif data is not None:
            self.nClass = self.targetProperties[0].nClasses
        if data is not None:
            self.nDim = data.getFeatures()[0].shape[1]

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.nClass is None or self.nDim is None:
            return "Uninitialized model."
        # initialize model - GGNN here
        estimator = self.alg(
            n_dim=self.nDim,
            n_class=self.nClass,
            device=str(self.device),
            gpus=self.gpus,
            is_reg=False,#self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol,
            parameters=params,
        ).to(self.device)
        # set parameters if available and return
        #new_parameters = self.getParameters(params)
        #if new_parameters is not None:
        #    estimator.set_params(**new_parameters)
        return estimator

    def loadEstimatorFromFile(
            self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from `alg` and `params` if no estimator
                found at path

        Returns:
            estimator (object): estimator instance
        """
        path = f"{self.outPrefix}_weights.pkg"
        estimator = self.loadEstimator(params)
        if estimator == "Uninitialized model.":
            return estimator
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(
                f"No estimator found at {path}, "
                f"loading estimator weights from file failed."
            )
        return estimator

    def saveEstimator(self) -> str:
        """Save the DNNModel model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        if not isinstance(self.estimator, str):
            torch.save(self.estimator.state_dict(), path)
        else:
            # just save the estimator message
            with open(path, "w") as f:
                f.write(self.estimator)
        return path

    @early_stopping
    def fit(
            self,
            X: pd.DataFrame | np.ndarray,
            y: pd.DataFrame | np.ndarray,
            estimator: Any | None = None,
            mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
            split: DataSplit | None = None,
            monitor: FitMonitor | None = None,
            **kwargs,
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray): data matrix to fit
            y (pd.DataFrame, np.ndarray): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            split (DataSplit): data split to use for early stopping,
                if None, a ShuffleSplit with 10% validation set size is used
            monitor (FitMonitor): fit monitor instance, if None, a BaseMonitor is used
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for this model."
            )
        monitor = BaseMonitor() if monitor is None else monitor
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        split = split or ShuffleSplit(
            n_splits=1, test_size=0.1, random_state=self.randomState
        )
        X, y = self.convertToNumpy(X, y)
        # fit with early stopping
        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            train_index, val_index = next(split.split(X, y))
            monitor.onFitStart(
                self, X[train_index, :], y[train_index], X[val_index, :], y[val_index]
            )
            estimator_fit = estimator.fit(
                X[train_index, :],
                y[train_index],
                X[val_index, :],
                y[val_index],
                monitor=monitor,
                **kwargs,
            )
            monitor.onFitEnd(estimator_fit[0], estimator_fit[1])
            return estimator_fit
        monitor.onFitStart(self, X, y)
        # set fixed number of epochs if early stopping is not used
        estimator.n_epochs = self.earlyStopping.getEpochs()
        estimator_fit = estimator.fit(X, y, monitor=monitor, **kwargs)
        monitor.onFitEnd(estimator_fit[0])
        return estimator_fit

    def predict(
            self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        scores = self.predictProba(X, estimator)
        # return class labels for classification
        if self.task.isClassification():
            return np.argmax(scores[0], axis=1, keepdims=True)
        else:
            return scores[0]

    def predictProba(
            self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        estimator.device = self.device
        estimator.gpus = self.gpus
        X = self.convertToNumpy(X)

        return [estimator.predict(X)]
    


# KOD PRO INSPIRACI - CO MUSI!!! BYT IMPLEMENTOVANO!!!


"""This module holds the base class for DNN models
as well as fully connected NN subclass.
"""

import inspect
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader, TensorDataset

from ....logs import logger
from ....models.monitors import BaseMonitor, FitMonitor


class Base(nn.Module):
    """Base structure for all classification/regression DNN models.

    Mainly, it provides the general methods for training, evaluating model and
    predicting the given data.

    Attributes:
        n_epochs (int):
            (maximum) number of epochs to train the model
        lr (float):
            learning rate
        batch_size (int):
            batch size
        patience (int):
            number of epochs to wait before early stop if no progress on validation
            set score, if patience = -1, always train to `n_epochs`
        tol (float):
            minimum absolute improvement of loss necessary to count as progress
            on best validation score
        device (torch.device):
            device to run the model on
        gpus (list):
            list of gpus to run the model on
    """

    def __init__(  # nejak done
        self,
        device: str,
        gpus: list[int],
        n_epochs: int = 1000,
        lr: float = 1e-4,
        batch_size: int = 256,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize the DNN model.

        Args:
            device (str):
                device to run the model on
            gpus (list):
                list of gpus to run the model on
            n_epochs (int):
                (maximum) number of epochs to train the model
            lr (float):
                learning rate
            batch_size (int):
                batch size
            patience (int):
                number of epochs to wait before early stop if no progress on validation
                set score, if patience = -1, always train to `n_epochs`
            tol (float):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.device = torch.device(device)
        self.gpus = gpus
        if len(self.gpus) > 1:
            logger.warning(
                f"At the moment multiple gpus is not possible: "
                f"running DNN on gpu: {gpus[0]}."
            )

    def fit(   # nejak done
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        monitor: FitMonitor | None = None,
    ) -> int:
        """Training the DNN model.

        Training is, similar to the scikit-learn or Keras style.
        It saves the optimal value of parameters.

        Args:
            X_train (np.ndarray or pd.Dataframe):
                training data (m X n), m is the No. of samples, n is the No. of features
            y_train (np.ndarray or pd.Dataframe):
                training target (m X l), m is the No. of samples, l is
                the No. of classes or tasks
            X_valid (np.ndarray or pd.Dataframe):
                validation data (m X n), m is the No. of samples, n is
                the No. of features
            y_valid (np.ndarray or pd.Dataframe):
                validation target (m X l), m is the No. of samples, l is
                the No. of classes or tasks
            monitor (FitMonitor):
                monitor to use for training, if None, use base monitor

        Returns:
            int:
                the epoch number when the optimal model is saved
        """
        self.to(self.device)
        monitor = BaseMonitor() if monitor is None else monitor
        train_loader = self.getDataLoader(X_train, y_train)
        valid_loader = None
        # if validation data is provided, use early stopping
        if X_valid is not None and y_valid is not None:
            valid_loader = self.getDataLoader(X_valid, y_valid)
            patience = self.patience
        else:
            patience = -1
        if "optim" in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # record the minimum loss value based on the calculation of the
        # loss function by the current epoch
        best_loss = np.inf
        best_weights = self.state_dict()
        last_save = 0  # record the epoch when optimal model is saved.
        for epoch in range(self.n_epochs):
            monitor.onEpochStart(epoch)
            loss = None
            # decrease learning rate over the epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr * (1 - 1 / self.n_epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(train_loader):
                monitor.onBatchStart(i)
                # Batch of target tenor and label tensor
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                # predicted probability tensor
                y_ = self(Xb, is_train=True)
                # ignore all the NaN values
                ix = yb == yb
                if self.n_class > 1:
                    yb, y_ = yb[ix], y_[ix[:, -1], :]
                else:
                    yb, y_ = yb[ix], y_[ix]
                # loss function calculation based on predicted tensor and label tensor
                if self.n_class > 1:
                    loss = self.criterion(y_, yb.long())
                else:
                    loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
                monitor.onBatchEnd(i, float(loss))
            if patience == -1:
                monitor.onEpochEnd(epoch, loss.item())
            else:
                # loss value on validation set based on which optimal model is saved.
                loss_valid = self.evaluate(valid_loader)
                if loss_valid + self.tol < best_loss:
                    best_weights = self.state_dict()
                    best_loss = loss_valid
                    last_save = epoch
                elif epoch - last_save > patience:  # early stop
                    break
                monitor.onEpochEnd(epoch, loss.item(), loss_valid)
        if patience == -1:
            best_weights = self.state_dict()
        self.load_state_dict(best_weights)
        return self, last_save

    def evaluate(self, loader) -> float:  # provest!
        """Evaluate the performance of the DNN model.

        Args:
            loader (torch.util.data.DataLoader):
                data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the
                No. of classes or tasks)

        Return:
            loss (float):
                the average loss value based on the calculation of loss
                function with given test set.
        """
        self.to(self.device)
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            y_ = self.forward(Xb)
            ix = yb == yb
            if self.n_class > 1:
                yb, y_ = yb[ix], y_[ix[:, -1], :]
            else:
                yb, y_ = yb[ix], y_[ix]
            if self.n_class > 1:
                loss += self.criterion(y_, yb.long()).item()
            else:
                loss += self.criterion(y_, yb).item()
        loss = loss / len(loader)
        return loss

    def predict(self, X_test) -> np.ndarray:
        """Predicting the probability of each sample in the given dataset.

        Args:
            X_test (ndarray):
                m X n target array (m is the No. of sample,
                n is the No. of features)

        Returns:
            score (ndarray):
                probability of each sample in the given dataset,
                it is an m X l FloatTensor (m is the No. of sample, l is the
                No. of classes or tasks.)
        """
        self.to(self.device)
        loader = self.getDataLoader(X_test)
        score = []
        for X_b in loader:
            X_b = X_b.to(self.device)
            y_ = self.forward(X_b)
            score.append(y_.detach().cpu())
        score = torch.cat(score, dim=0).numpy()
        return score

    @classmethod
    def _get_param_names(cls) -> list:
        """Get the class parameter names.

        Function copied from sklearn.base_estimator!

        Returns:
            parameter names (list): list of the class parameter names.
        """
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True) -> dict:
        """Get parameters for this estimator.

        Function copied from sklearn.base_estimator!

        Args:
            deep (bool): If True, will return the parameters for this estimator

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "Base":
        """Set the parameters of this estimator.

        Function copied from sklearn.base_estimator!
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Args:
            **params : dict Estimator parameters.

        Returns:
            self : estimator instance
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # grouped by prefix
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def getDataLoader(self, X, y=None):
        """Convert data to tensors and get generator over dataset with dataloader.

        Args:
            X (numpy 2d array): input dataset
            y (numpy 1d column vector): output data
        """
        # if pandas dataframe is provided, convert it to numpy array
        if hasattr(X, "values"):
            X = X.values
        if y is not None and hasattr(y, "values"):
            y = y.values
        if y is None:
            tensordataset = torch.Tensor(X)
        else:
            tensordataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        return DataLoader(tensordataset, batch_size=self.batch_size)


class STFullyConnected(Base):
    """Single task DNN classification/regression model.

    It contains four fully connected layers between which are
    dropout layers for robustness.

    Attributes:
        n_dim (int): the No. of columns (features) for input tensor
        n_class (int): the No. of columns (classes) for output tensor.
        device (torch.cude): device to run the model on
        gpus (list): list of gpu ids to run the model on
        n_epochs (int): max number of epochs
        lr (float): neural net learning rate
        batch_size (int): batch size for training
        patience (int): early stopping patience
        tol (float): early stopping tolerance
        is_reg (bool): whether the model is for regression or classification
        neurons_h1 (int): No. of neurons in the first hidden layer
        neurons_hx (int): No. of neurons in the second hidden layer
        extra_layer (bool): whether to add an extra hidden layer
        dropout_frac (float): dropout fraction
        criterion (torch.nn.Module): the loss function
        dropout (torch.nn.Module): the dropout layer
        fc0 (torch.nn.Module): the first fully connected layer
        fc1 (torch.nn.Module): the second fully connected layer
        fc2 (torch.nn.Module): the third fully connected layer
        fc3 (torch.nn.Module): the fourth fully connected layer
        activation (torch.nn.Module): the activation function
    """

    def __init__(
        self,
        n_dim,
        n_class,
        device,
        gpus,
        n_epochs=100,
        lr=None,
        batch_size=256,
        patience=50,
        tol=0,
        is_reg=True,
        neurons_h1=256,
        neurons_hx=128,
        extra_layer=False,
        dropout_frac=0.25,
    ):
        """Initialize the STFullyConnected model.

        Args:
            n_dim (int):
                the No. of columns (features) for input tensor
            n_class (int):
                the No. of columns (classes) for output tensor.
            device (torch.cude):
                device to run the model on
            gpus (list):
                list of gpu ids to run the model on
            n_epochs (int):
                max number of epochs
            lr (float):
                neural net learning rate
            batch_size (int):
                batch size
            patience (int):
                number of epochs to wait before early stop if no progress on
                validation set score, if patience = -1, always train to n_epochs
            tol (float):
                minimum absolute improvement of loss necessary to
                count as progress on best validation score
            is_reg (bool, optional):
                Regression model (True) or Classification model (False)
            neurons_h1 (int):
                number of neurons in first hidden layer
            neurons_hx (int):
                number of neurons in other hidden layers
            extra_layer (bool):
                add third hidden layer
            dropout_frac (float):
                dropout fraction
        """
        if not lr:
            lr = 1e-4 if is_reg else 1e-5
        super().__init__(
            device=device,
            gpus=gpus,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            tol=tol,
        )
        self.n_dim = n_dim
        self.is_reg = is_reg
        self.n_class = n_class if not self.is_reg else 1
        self.neurons_h1 = neurons_h1
        self.neurons_hx = neurons_hx
        self.extra_layer = extra_layer
        self.dropout_frac = dropout_frac
        self.dropout = None
        self.fc0 = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.activation = None
        self.criterion = None
        self.initModel()

    def initModel(self):
        """Define the layers of the model."""
        self.dropout = nn.Dropout(self.dropout_frac)
        self.fc0 = nn.Linear(self.n_dim, self.neurons_h1)
        self.fc1 = nn.Linear(self.neurons_h1, self.neurons_hx)
        if self.extra_layer:
            self.fc2 = nn.Linear(self.neurons_hx, self.neurons_hx)
        self.fc3 = nn.Linear(self.neurons_hx, self.n_class)
        if self.is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        elif self.n_class == 1:
            # loss and activation function of output layer for binary classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            # loss and activation function of output layer for multiple classification
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax(dim=1)

    def set_params(self, **params) -> "STFullyConnected":
        """Set parameters and re-initialize model.

        Args:
            **params: parameters to be set

        Returns:
            self (STFullyConnected): the model itself
        """
        super().set_params(**params)
        self.initModel()
        return self

    def forward(self, X, is_train=False) -> torch.Tensor:
        """Invoke the class directly as a function.

        Args:
            X (FloatTensor):
                m X n FloatTensor, m is the No. of samples, n is
                the No. of features.
            is_train (bool, optional):
                is it invoked during training process (True) or
                just for prediction (False)
        Returns:
            y (FloatTensor): m X n FloatTensor, m is the No. of samples,
                n is the No. of classes
        """
        y = f.relu(self.fc0(X))
        if is_train:
            y = self.dropout(y)
        y = f.relu(self.fc1(y))
        if self.extra_layer:
            if is_train:
                y = self.dropout(y)
            y = f.relu(self.fc2(y))
        if is_train:
            y = self.dropout(y)
        if self.is_reg:
            y = self.fc3(y)
        else:
            y = self.activation(self.fc3(y))
        return y
