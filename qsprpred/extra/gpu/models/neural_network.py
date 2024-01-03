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

from ....extra.gpu import DEFAULT_TORCH_DEVICE, DEFAULT_TORCH_GPUS
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

    def __init__(
        self,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        gpus: list[int] = DEFAULT_TORCH_GPUS,
        n_epochs: int = 1000,
        lr: float = 1e-4,
        batch_size: int = 256,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize the DNN model.

        Args:
            device (torch.device):
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
        if device.type == "cuda":
            self.device = torch.device(f"cuda:{gpus[0]}")
        else:
            self.device = device
        self.gpus = gpus
        if len(self.gpus) > 1:
            logger.warning(
                f"At the moment multiple gpus is not possible: "
                f"running DNN on gpu: {gpus[0]}."
            )

    def fit(
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

    def evaluate(self, loader) -> float:
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
        n_class=1,
        device=DEFAULT_TORCH_DEVICE,
        gpus=DEFAULT_TORCH_GPUS,
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
        self.to(self.device)

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
