from __future__ import annotations
import json
from pathlib import Path

import flwr as fl
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TabularDataset
from src.fl.common import (
    prepare_global_metadata,
    load_metadata,
    load_global_scaler,
    make_model,
)
from src.models.train_utils import predict
from src.eval.metrics import compute_metrics, per_class_report

PARTITION_NAME = "label_skew_alpha_0_3"
NUM_CLIENTS = 10
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(f"results/fl/{PARTITION_NAME}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_evaluate_fn():
    meta = load_metadata()
    feature_cols = meta["feature_cols"]
    label_names = meta["label_names"]
    num_classes = meta["num_classes"]

    val_df = pd.read_csv("data/processed/splits_cicids2017_fl/val.csv", low_memory=False)
    test_df = pd.read_csv("data/processed/splits_cicids2017_fl/test.csv", low_memory=False)

    scaler = load_global_scaler()
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    val_ds = TabularDataset(val_df, feature_cols, "LabelId")
    test_ds = TabularDataset(test_df, feature_cols, "LabelId")

    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

    history_rows = []

    def evaluate(server_round, parameters, config):
        model = make_model(len(feature_cols), num_classes).to(DEVICE)

        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        model.load_state_dict(new_state, strict=True)

        y_val, y_val_pred = predict(model, val_loader, DEVICE)
        val_metrics = compute_metrics(y_val, y_val_pred)

        history_rows.append({"round": server_round, **{f"val_{k}": v for k, v in val_metrics.items()}})
        pd.DataFrame(history_rows).to_csv(RESULTS_DIR / "round_history.csv", index=False)

        print(
            f"[Round {server_round}] "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if server_round == NUM_ROUNDS:
            y_test, y_test_pred = predict(model, test_loader, DEVICE)
            test_metrics = compute_metrics(y_test, y_test_pred)
            report = per_class_report(y_test, y_test_pred, label_names)

            with open(RESULTS_DIR / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)
            with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report)

            torch.save(model.state_dict(), RESULTS_DIR / "global_model.pt")

        return float(1.0 - val_metrics["f1_macro"]), val_metrics

    return evaluate


def client_fn(context: fl.common.Context):
    from src.fl.client import FlowerClient
    cid = int(context.node_config["partition-id"])
    return FlowerClient(
        client_id=cid,
        partition_name=PARTITION_NAME,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
    ).to_client()


def main():
    prepare_global_metadata()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(),
    )

    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=config,
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )


if __name__ == "__main__":
    main()