from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion(
    pred_csv: str,
    out_png: str,
    normalize: str | None = None,  # None, "true", "pred", "all"
    title: str = "Confusion Matrix",
) -> None:
    df = pd.read_csv(pred_csv)

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    # preserve label order from ids
    id_to_name = (
        df[["y_true", "y_true_label"]]
        .drop_duplicates()
        .sort_values("y_true")
    )
    labels = id_to_name["y_true"].tolist()
    display_labels = id_to_name["y_true_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)

    ax.set_title(title)
    plt.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Centralized
    plot_confusion(
        pred_csv="results/centralized/test_predictions.csv",
        out_png="results/centralized/confusion_matrix_raw.png",
        normalize=None,
        title="Centralized - Confusion Matrix (Raw Counts)",
    )
    plot_confusion(
        pred_csv="results/centralized/test_predictions.csv",
        out_png="results/centralized/confusion_matrix_normalized.png",
        normalize="true",
        title="Centralized - Confusion Matrix (Row Normalized)",
    )

    # FL-IID
    plot_confusion(
        pred_csv="results/fl/iid/test_predictions.csv",
        out_png="results/fl/iid/confusion_matrix_raw.png",
        normalize=None,
        title="FL-IID - Confusion Matrix (Raw Counts)",
    )
    plot_confusion(
        pred_csv="results/fl/iid/test_predictions.csv",
        out_png="results/fl/iid/confusion_matrix_normalized.png",
        normalize="true",
        title="FL-IID - Confusion Matrix (Row Normalized)",
    )

    # FL-Non-IID
    plot_confusion(
        pred_csv="results/fl/label_skew_alpha_0_3/test_predictions.csv",
        out_png="results/fl/label_skew_alpha_0_3/confusion_matrix_raw.png",
        normalize=None,
        title="FL-Non-IID - Confusion Matrix (Raw Counts)",
    )
    plot_confusion(
        pred_csv="results/fl/label_skew_alpha_0_3/test_predictions.csv",
        out_png="results/fl/label_skew_alpha_0_3/confusion_matrix_normalized.png",
        normalize="true",
        title="FL-Non-IID - Confusion Matrix (Row Normalized)",
    )