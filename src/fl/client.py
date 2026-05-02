from __future__ import annotations
import flwr as fl
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.data.dataset import TabularDataset
from src.models.train_utils import train_one_epoch, predict
from src.eval.metrics import compute_metrics
from src.fl.common import (
    load_metadata,
    load_client_df,
    load_global_scaler,
    make_model,
    get_model_parameters,
    set_model_parameters,
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        partition_name: str,
        local_epochs: int = 1,
        batch_size: int = 512,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.partition_name = partition_name
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        meta = load_metadata()
        self.feature_cols = meta["feature_cols"]
        self.label_names = meta["label_names"]
        self.num_classes = meta["num_classes"]

        self.df = load_client_df(partition_name, client_id)
        scaler = load_global_scaler()
        self.df[self.feature_cols] = scaler.transform(self.df[self.feature_cols])

        self.dataset = TabularDataset(self.df, self.feature_cols, "LabelId")
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model = make_model(len(self.feature_cols), self.num_classes).to(self.device)

    def get_parameters(self, config):
        return get_model_parameters(self.model)
    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        y = self.df["LabelId"].values

        # Use only classes actually present on this client
        present_classes = np.unique(y)
        present_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=y,
        )

        # Build full class-weight vector for all global classes
        full_weights = np.ones(self.num_classes, dtype=np.float32)
        for cls, w in zip(present_classes, present_weights):
            full_weights[int(cls)] = float(w)

        # Reduce BENIGN weight slightly to lower BENIGN -> ATTACK false positives
        benign_class_index = 0
        benign_ratio = np.mean(y == benign_class_index)
        if benign_ratio>0.5:
            full_weights[benign_class_index] *= 0.7
        

        class_weights = torch.tensor(full_weights, dtype=torch.float32).to(self.device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.loader, optimizer, criterion, self.device)

        return get_model_parameters(self.model), len(self.df), {}
    # def fit(self, parameters, config):
    #     set_model_parameters(self.model, parameters)

    #     y = self.df["LabelId"].values
    #     classes = np.arange(self.num_classes)
    #     class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    #     class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    #     criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    #     for _ in range(self.local_epochs):
    #         train_one_epoch(self.model, self.loader, optimizer, criterion, self.device)

    #     return get_model_parameters(self.model), len(self.df), {}

    # def evaluate(self, parameters, config):
    #     set_model_parameters(self.model, parameters)
    #     y_true, y_pred = predict(self.model, self.loader, self.device)
    #     metrics = compute_metrics(y_true, y_pred)
    #     return float(1.0 - metrics["f1_macro"]), len(self.df), metrics
    def evaluate(self, parameters, config):
        try:
            set_model_parameters(self.model, parameters)

            y_true, y_pred = predict(self.model, self.loader, self.device)
            metrics = compute_metrics(y_true, y_pred)

            # Ensure all metrics are Python floats (not numpy types)
            metrics = {k: float(v) for k, v in metrics.items()}

            loss = float(1.0 - metrics["f1_macro"])

            return loss, len(self.df), metrics

        except Exception as e:
            print(f"[Client {self.client_id}] Evaluation error:", e)
            return 1.0, len(self.df), {}