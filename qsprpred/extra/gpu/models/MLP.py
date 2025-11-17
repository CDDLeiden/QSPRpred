"""
This module holds the alg class for DNN models.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Model reproducibility
import inspect
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader, TensorDataset


#from .base_torch import QSPRModelPyTorchGPU, DEFAULT_TORCH_GPUS
#from ....logs import logger
from qsprpred.models.monitors import BaseMonitor, FitMonitor
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import matthews_corrcoef, accuracy_score
import copy


class MultiLayerPerceptron(nn.Module):
    """
    A combined class inheriting from `nn.Module` to implement a Deep Neural Network (DNN)
    for classification or regression tasks, specifically configured for binary classification.

    It includes general methods for training, evaluating, and predicting with the model,
    as well as the specific architecture for a fully connected neural network.

    Attributes:
        n_dim (int): The number of input features (columns) for the input tensor.
        n_class (int): The number of output classes.
                       (Note: In the current `initModel` implementation, the output layer
                       is fixed to 1 neuron, suitable for binary classification or regression).
        device (torch.device): The device (CPU or GPU) on which the model will run.
        act_fun (torch.nn.functional): The activation function for hidden layers (e.g., f.relu).
        n_epochs (int): The (maximum) number of epochs for training the model.
        patience (int): The number of epochs to wait for an improvement in validation score
                        before early stopping. If -1, training runs for `n_epochs`.
        tol (float): The minimum absolute improvement in the metric required to count as progress
                     in the best validation score.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        neuron_layers (list): A list of integers, where each integer represents the number of neurons
                              in a hidden layer.
        dropout_frac (float): The dropout probability.
        weight_decay (float): Weight decay (L2 regularization).
        optimizer (torch.optim.Optimizer): The optimizer class to use for training.
        seed (int): The random seed for reproducibility.
        print_outputs (int): The verbosity level for training output.
                             (0: silent, 1: epoch summary, >1: detailed batch output).
        criterion (torch.nn.Module): The loss function, typically set in the `fit` method.
        dropout (torch.nn.Module): The dropout layer.
        layers (nn.ModuleList): A list containing all the linear layers of the neural network.
    """

    def __init__(
            self,
            n_dim: int,
            gpus,
            is_reg: bool,
            n_class: int = 1,
            device: str = "cpu",
            act_fun=f.relu,
            n_epochs: int = 100,
            lr: float = 1e-4,
            batch_size: int = 256,
            patience: int = 50,
            tol: float = 0,
            neuron_layers=None,
            dropout_frac: float = 0.25,
            weight_decay: float = 0,
            optimizer=optim.AdamW,
            seed: int = 42,
            print_outputs: int = 0, 
    ):
        """
        Initializes the DNN model with training configuration and its architectural parameters.
        """
        super().__init__()
        if neuron_layers is None:
            self.neuron_layers = [2048, 1024]
        else:
            self.neuron_layers = neuron_layers
        # Set reproducibility seed early
        self.seed = seed
        self.set_seed(seed=seed)

        # Assign training parameters
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.device = torch.device(device)
        self.weight_decay = weight_decay
        self.optimizer = optimizer # Store the optimizer class
        self.print_outputs = print_outputs

        # Assign model architecture parameters
        self.n_dim = n_dim
        self.n_class = n_class # This attribute exists but the output layer is fixed to 1 in initModel
        self.dropout_frac = dropout_frac
        self.act_fun = act_fun

        # Initialize model components, actual layers built in initModel
        self.layers = nn.ModuleList()
        self.dropout = None
        self.criterion = None
        
        self.gpus = gpus #Not implemented
        self.is_reg = False # Not implemented - only binary classification 
        self.n_class = n_class # Not implemented - only binary classification 
        
        self.initModel() # Build the neural network layers

    def initModel(self):
        """
        Defines the layers of the neural network.
        This method constructs `nn.Linear` layers based on `n_dim` and `neuron_layers`,
        and initializes the `nn.Dropout` layer.
        """
        self.layers = nn.ModuleList()

        if not self.neuron_layers: # If no hidden layers are specified
            # Direct connection from input to a single output neuron for binary classification
            self.layers.append(nn.Linear(self.n_dim, 1))
        else:
            self.layers.append(nn.Linear(self.n_dim, self.neuron_layers[0]))
            for i in range(1, len(self.neuron_layers)):
                self.layers.append(nn.Linear(self.neuron_layers[i - 1], self.neuron_layers[i]))
            self.layers.append(nn.Linear(self.neuron_layers[-1], 1))

        self.dropout = nn.Dropout(self.dropout_frac)

    def forward(self, X: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        """
        Defines the forward pass logic of the neural network.

        Args:
            X (torch.Tensor): The input tensor, typically of shape (num_samples, num_features).
            is_train (bool, optional): A flag indicating if the model is in training mode.
                                       When `True`, dropout is applied. Defaults to `False`.
        Returns:
            torch.Tensor: The output tensor of shape (num_samples, 1), representing raw logits
                          (pre-sigmoid) for binary classification or regression values.
        """
        # Handles the case where the network has only one linear layer (input directly to output)
        if len(self.layers) == 1:
            y = self.layers[0](X)
        else: # Standard multi-layer perceptron path
            y = self.act_fun(self.layers[0](X)) # Apply activation after the first hidden layer
            for i in range(1, len(self.layers) - 1): # Iterate through subsequent hidden layers
                y = self.act_fun(self.layers[i](y))
                if is_train: # Apply dropout only during training mode
                    y = self.dropout(y)
            y = self.layers[-1](y) # Output layer, no explicit activation here as BCEWithLogitsLoss expects logits
        return y

    def fit(
            self,
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            monitor: FitMonitor | None = None, # Commented out, external monitoring system
    ) -> tuple["MultiLayerPerceptron", int]:
        """
        Trains the DNN model using the provided training data, with optional validation
        and early stopping functionality.

        Args:
            X_train (pandas.DataFrame or torch.Tensor): Training features.
            y_train (pandas.Series or torch.Tensor): Training labels (for binary classification, typically 0s and 1s).
            X_valid (pandas.DataFrame or torch.Tensor, optional): Validation features for performance monitoring.
            y_valid (pandas.Series or torch.Tensor, optional): Validation labels corresponding to `X_valid`.

        Returns:
            tuple:
                - self: The trained model instance.
                - last_save (int): The epoch index (0-based) where the model achieved its best
                                   validation loss (or last epoch if no validation set/early stopping).
        """
        self.to(self.device) # Move model to the specified device (CPU/GPU)
        #print("Epochs",self.n_epochs)
        #print("Patience",self.patience)
        monitor = BaseMonitor() if monitor is None else monitor 
        
        # Prepare DataLoaders for training and validation sets
        train_loader = self.getDataLoader(X_train, y_train)
        valid_loader = self.getDataLoader(X_valid, y_valid) if X_valid is not None and y_valid is not None else None

        # Optimizer initialization logic
        if "optim" in self.__dict__ and self.optim is not None:
            optimizer_instance = self.optim
        else:
            optimizer_instance = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Prepare y_train for pos_weight calculation
        y_tensor_values = y_train.values if hasattr(y_train, "values") else y_train
        try:
            y_tensor = torch.tensor(y_tensor_values, dtype=torch.float32)

            num_zeros = (y_tensor == 0).sum().item()
            num_ones = (y_tensor == 1).sum().item()

            if num_ones == 0 or num_zeros == 0:
                # If only one class is present, pos_weight calculation is ill-defined
                # or can lead to problematic training behavior.
                # In this specific context, it implies a problem with the dataset for binary classification.
                print("Warning: Training data contains only one class. `pos_weight` cannot be calculated meaningfully for binary classification. Using unweighted loss.")
                self.criterion = torch.nn.BCEWithLogitsLoss()
            else:
                pos_weight_val = num_zeros / num_ones
                pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        except Exception as e:
            # Fallback for unexpected issues during pos_weight calculation
            print(f"Warning: Could not calculate pos_weight for BCEWithLogitsLoss due to: {e}. Using unweighted loss.")
            self.criterion = torch.nn.BCEWithLogitsLoss()

        best_loss = np.inf
        best_acc = -1
        bes_mcc = -2
        best_weights = copy.deepcopy(self.state_dict()) # Store the initial best weights
        last_save = 0
        epochs_no_improve = 0

        # Learning rate scheduler (OneCycleLR) setup
        scheduler = OneCycleLR(
            optimizer_instance,
            max_lr=self.lr * 10,
            total_steps=self.n_epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e4,
            div_factor=25.0,
        )

        for epoch in range(self.n_epochs):
            # monitor.onEpochStart(epoch) # Monitoring system related code
            epoch_train_loss = 0.0
            num_batches_train = 0

            self.train() # Set model to training mode for this epoch
            for i, (Xb, yb) in enumerate(train_loader):
                # monitor.onBatchStart(i) # Monitoring system related code
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer_instance.zero_grad() # Zero gradients before backward pass
                y_pred = self(Xb, is_train=True) # Perform forward pass

                if y_pred.ndim == 1:
                    yb = yb.squeeze()
                if y_pred.shape != yb.shape:
                    yb = yb.view_as(y_pred)

                # Filter out NaN values from both targets and predictions to prevent loss calculation errors.
                valid_mask = ~torch.isnan(yb) & ~torch.isnan(y_pred)
                yb_clean, y_pred_clean = yb[valid_mask], y_pred[valid_mask]

                if yb_clean.numel() > 0: # Only calculate loss if there are valid samples in the batch
                    loss = self.criterion(y_pred_clean, yb_clean)
                    loss.backward() # Backpropagate the loss
                    optimizer_instance.step() # Update model weights
                    scheduler.step() # Update learning rate
                    # monitor.onBatchEnd(i, float(loss)) # Monitoring system related code
                    epoch_train_loss += loss.item()
                    num_batches_train += 1
                else: # If all samples in a batch are NaN, skip loss calculation for this batch
                    pass
                    # monitor.onBatchEnd(i, float('nan')) # Monitoring system related code

            # Calculate average training loss for the current epoch
            avg_epoch_train_loss = epoch_train_loss / num_batches_train if num_batches_train > 0 else float('nan')

            # Validation phase if a validation loader is provided
            if valid_loader is not None:
                loss_valid = self.evaluate(valid_loader) # Evaluate on validation set
                mcc_valid = 0
                acc_valid = 0 
                if X_valid is not None: # Ensure X_valid was originally provided
                    # Get predictions (probabilities) using the model's predict method
                    y_pred_proba_valid = self.predict(X_valid)[:,1:]
                    # Convert probabilities to binary (0 or 1) based on a 0.5 threshold
                    binary_predictions_valid = (y_pred_proba_valid > 0.5).astype(int)

                    # Ensure y_valid is a numpy array and 1D for `matthews_corrcoef`
                    y_valid_np = y_valid.values if hasattr(y_valid, 'values') else np.array(y_valid)
                    if y_valid_np.ndim > 1 and y_valid_np.shape[1] == 1:
                        y_valid_np = y_valid_np.squeeze()


                    mcc_valid = matthews_corrcoef(y_valid_np, binary_predictions_valid)
                    acc_valid = accuracy_score(y_valid_np, binary_predictions_valid)
                if self.print_outputs > 1:
                    print(
                        f"Epoch {epoch + 1} | Train Loss: {avg_epoch_train_loss:.4f} | Valid Loss: {loss_valid:.4f} | MCC: {mcc_valid:.4f} | ACC: {acc_valid:.4f}")

                # monitor.onEpochEnd(epoch, avg_epoch_train_loss, loss_valid) # Monitoring system related code

                # Early stopping logic: check if validation loss improved
                if loss_valid + self.tol < best_loss:
                    best_loss = loss_valid
                    best_mcc = mcc_valid
                    best_acc = acc_valid
                    best_weights = copy.deepcopy(self.state_dict()) # Save model state if performance improved
                    last_save = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Trigger early stopping if no improvement for 'patience' epochs
                if self.patience > 0 and epochs_no_improve >= self.patience:
                    if self.print_outputs > 0:
                        print(f"Early stopping at epoch {epoch + 1} | Best Valid Loss: {best_loss:.4f} | Best Valid ACC: {best_acc:.4f} | Best Valid MCC: {best_mcc:.4f} ")
                    break
            else: # If no validation loader is provided, track "best" as last epoch's weights
                if self.print_outputs > 1:
                    print(f"Epoch {epoch + 1} | Train Loss: {avg_epoch_train_loss:.4f}")
                # monitor.onEpochEnd(epoch, avg_epoch_train_loss) # Monitoring system related code
                best_weights = copy.deepcopy(self.state_dict()) # If no validation, best is the last state
                last_save = epoch

        self.load_state_dict(best_weights) # Load the weights of the best performing model
        # Final training summary output
        if self.print_outputs > 0 and valid_loader is None:
            print(f"Training finished after {epoch + 1} epochs. Final train loss: {avg_epoch_train_loss:.4f}")
        elif self.print_outputs > 0 and valid_loader is not None and not (
                self.patience > 0 and epochs_no_improve >= self.patience):
            # Message if training completed all epochs or was stopped for other reasons than patience
            print(f"Training finished after {self.n_epochs} epochs. Best validation loss: {best_loss:.4f} at epoch {last_save + 1}.")

        return self, last_save


    def evaluate(self, loader: DataLoader) -> float:
        """
        Evaluates the MLP model's performance on a given dataset using its configured loss function.

        Args:
            loader (DataLoader): A PyTorch DataLoader containing the data to be evaluated.

        Returns:
            float: The average loss calculated over the entire dataset. Returns `inf` if no valid
                   samples are present in the loader.
        """
        self.to(self.device)
        self.eval() # Set model to evaluation mode (e.g., disables dropout layers)
        total_loss = 0.0
        total_samples = 0

        if self.criterion is None:
            # Fallback for evaluation if `fit` (which sets criterion) was not called first.
            print("Warning: self.criterion is not set in evaluate. Using BCEWithLogitsLoss as default.")
            self.criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad(): # Disable gradient calculations for faster evaluation
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                y_pred = self.forward(Xb, is_train=False) # Perform forward pass in inference mode

                # Ensure target (yb) and prediction (y_pred) shapes match for loss calculation
                if y_pred.ndim == 1:
                    yb = yb.squeeze()
                if y_pred.shape != yb.shape:
                    yb = yb.view_as(y_pred)

                # Filter out NaN values from both targets and predictions
                valid_mask = ~torch.isnan(yb) & ~torch.isnan(y_pred)
                yb_clean, y_pred_clean = yb[valid_mask], y_pred[valid_mask]

                if yb_clean.numel() > 0:
                    batch_size = yb_clean.size(0)
                    loss = self.criterion(y_pred_clean, yb_clean)
                    total_loss += loss.item() * batch_size # Accumulate total loss (sum of losses per sample)
                    total_samples += batch_size
                # If no clean samples in batch, nothing is added

        return total_loss / total_samples if total_samples > 0 else float("inf")



    def predict(self, X_test) -> np.ndarray:
        """
        Generates probability predictions for each sample in the input dataset.

        Args:
            X_test (pandas.DataFrame or numpy.ndarray or torch.Tensor or DataLoader):
                The input features for which to generate predictions.

        Returns:
            numpy.ndarray: A 1D NumPy array containing the predicted probabilities (values between 0 and 1).
        """
        self.to(self.device)
        self.eval() # Set model to evaluation mode for inference

        # Use existing DataLoader or create one if raw data is provided
        if isinstance(X_test, DataLoader):
            loader = X_test
        else:
            # Create a DataLoader for prediction; labels are not needed here
            loader = self.getDataLoader(X_test, y=None, shuffle_override=False)

        predictions = []
        with torch.no_grad(): # Disable gradient calculations for prediction
            for batch in loader:
                # Extract features from the batch; DataLoader might return (X_b,) or just X_b
                X_b = batch[0] if isinstance(batch, (tuple, list)) else batch
                X_b = X_b.to(self.device)
                y_logits = self.forward(X_b, is_train=False) # Get raw logits from the model
                #y_proba = torch.sigmoid(y_logits) # Apply sigmoid to convert logits to probabilities
                #y_proba_class_0 = 1.0 - y_proba
                #predictions.append(y_proba.detach().cpu()) # Store predictions on CPU
                # --- ZAČÁTEK ÚPRAVY ---

                # Vypočítáme P(třída 1) pomocí sigmoidu
                # y_proba_class_1 bude mít tvar [batch_size, 1]
                y_proba_class_1 = torch.sigmoid(y_logits)
                
                # Vypočítáme P(třída 0) = 1 - P(třída 1)
                # y_proba_class_0 bude mít také tvar [batch_size, 1]
                y_proba_class_0 = 1.0 - y_proba_class_1
                
                # Spojíme oba sloupce dohromady (podél dimenze 1)
                # combined_probs bude mít tvar [batch_size, 2]
                combined_probs = torch.cat((y_proba_class_0, y_proba_class_1), dim=1)
                
                # Uložíme 2-sloupcové predikce
                predictions.append(combined_probs.detach().cpu()) 

                # --- KONEC ÚPRAVY ---

        # Concatenate predictions from all batches and convert to a NumPy array
        return torch.cat(predictions, dim=0).numpy()



    @classmethod
    def _get_param_names(cls) -> list:
        """
        Retrieves the names of parameters in the class's `__init__` method.
        This function is adapted from `sklearn.base_estimator` for compatibility
        with scikit-learn's parameter handling conventions.
        """
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD # Exclude 'self' and variable keyword arguments
        ]
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> dict:
        """
        Retrieves the parameters for this estimator.
        This function is adapted from `sklearn.base_estimator` for compatibility
        with scikit-learn's parameter handling conventions.

        Args:
            deep (bool, optional): If `True`, returns the parameters for this estimator and
                                   any contained sub-objects that are also estimators (e.g., in a pipeline).
                                   Defaults to `True`.

        Returns:
            dict: A dictionary mapping parameter names to their current values.
        """
        out = {}
        for key in self._get_param_names(): # Use the class method to get parameter names
            value = getattr(self, key)
            # If deep is True and the value is a sub-estimator, recursively get its parameters
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "MultiLayerPerceptron":
        """
        Sets the parameters of this estimator.
        This method is partially adapted from `sklearn.base_estimator`.
        After parameters are set, the model's layers are re-initialized by calling `initModel`.

        Args:
            **params: Keyword arguments where keys are parameter names (or nested parameter names
                      like 'optimizer__lr') and values are the new parameter values.

        Returns:
            MultiLayerPerceptron: The estimator instance itself, allowing for method chaining.

        Raises:
            ValueError: If an invalid parameter name is provided.
        """
        if not params: # If no parameters are provided, return self unchanged
            return self

        valid_params = self.get_params(deep=True) # Get current valid parameters to validate input
        nested_params = defaultdict(dict)

        for key, value in params.items():
            simple_key, delim, sub_key = key.partition("__") # Split for nested parameters (e.g., 'optimizer__lr')
            if simple_key not in valid_params and simple_key not in self._get_param_names():
                # Critical: Validate input parameter names to prevent typos or invalid assignments.
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {simple_key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(local_valid_params)!r}."
                )

            if delim: # If it's a nested parameter
                nested_params[simple_key][sub_key] = value
            else: # If it's a direct parameter
                setattr(self, simple_key, value) # Set the attribute directly

        for key, sub_params in nested_params.items():
            # If the parameter corresponds to a sub-estimator that has its own `set_params` method
            if hasattr(getattr(self, key), 'set_params'):
                getattr(self, key).set_params(**sub_params)
            # No `else` block is needed here as this is primarily for handling sub-estimators.

        # This ensures that changes to architectural parameters (e.g., n_dim, neuron_layers)
        # properly rebuild the network structure.
        self.initModel()
        return self



    def getDataLoader(self, X, y=None, shuffle_override: bool = None) -> DataLoader:
        """
        Converts input data (features and optionally labels) into PyTorch Tensors
        and creates a `DataLoader` for efficient batch processing during training,
        validation, or prediction.

        Args:
            X: Features. Can be a `pandas.DataFrame`, `numpy.ndarray`, or `torch.Tensor`.
            y: Labels (optional). Can be a `pandas.Series`, `numpy.ndarray`, or `torch.Tensor`.
               If `None`, only features `X` are used to create the `DataLoader` (e.g., for prediction).
            shuffle_override (bool, optional): Overrides the default shuffling behavior.
                                               If `True`, data is shuffled. If `False`, it's not.
                                               If `None`, data is shuffled if labels `y` are provided
                                               (typically for training), and not shuffled otherwise.

        Returns:
            DataLoader: A PyTorch `DataLoader` instance.
        """
        # Convert X to a NumPy array for consistent tensor conversion
        if hasattr(X, "values"):
            X_np = X.values
        elif isinstance(X, torch.Tensor):
            X_np = X.numpy()
        else:
            X_np = np.asarray(X)

        if y is not None:
            # Convert y to a NumPy array
            if hasattr(y, "values"):
                y_np = y.values
            elif isinstance(y, torch.Tensor):
                y_np = y.numpy()
            else:
                y_np = np.asarray(y)

            # by PyTorch loss functions like `BCEWithLogitsLoss` which expect target shape (N, 1).
            if y_np.ndim == 1:
                y_np = y_np.reshape(-1, 1)

            tensor_dataset = TensorDataset(torch.Tensor(X_np), torch.Tensor(y_np))
            current_shuffle = True # Default for datasets with labels (e.g., training)
        else:
            tensor_dataset = torch.Tensor(X_np) # Create dataset with only features for prediction/evaluation
            current_shuffle = False # Default for prediction/evaluation (no labels)

        # Apply shuffle override if explicitly specified
        if shuffle_override is not None:
            current_shuffle = shuffle_override

        # Create a PyTorch Generator for reproducibility if shuffling is enabled
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            tensor_dataset,
            batch_size=self.batch_size,
            shuffle=current_shuffle,
            generator=generator if current_shuffle else None, # Use generator only when shuffling
            num_workers=0 # Set to 0 for full reproducibility and simpler debugging, avoids multiprocessing issues
        )



    @staticmethod
    def set_seed(seed: int):
        """
        Sets the random seed for various libraries (`numpy`, `torch` CPU, and `torch` CUDA)
        to ensure reproducibility of results.

        Args:
            seed (int): The integer random seed to apply.
        """
        # Set CuDNN to deterministic mode for reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # Disable CuDNN benchmarking for reproducibility, might slightly impact performance
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed) # Set NumPy seed
        torch.manual_seed(seed) # Set PyTorch CPU seed
        if torch.cuda.is_available(): # Set PyTorch CUDA (GPU) seeds if GPU is available
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU setups