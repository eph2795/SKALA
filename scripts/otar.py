from pprint import pprint
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import numpy as np
from pathlib import Path
import json


def main():
    data = pd.read_csv("results/final_clf_preds.csv")

    experts: list[str] = [data[f"e{i}"] for i in range(3)]

    metrics = {
        "precision": precision_score(y_true=data.true_label, y_pred=data.predicted_label, average="macro"),
        "recall": recall_score(y_true=data.true_label, y_pred=data.predicted_label, average="macro"),
    }
    model_expert: list[float] = [
        precision_score(y_true=data[expert], y_pred=data.predicted_label, average="macro") for expert in experts
    ]
    expert_expert: list[float] = []
    for i, expert in enumerate(experts):
        for j, other_expert in enumerate(experts):
            if i < j:
                expert_expert.append(precision_score(y_true=data[expert], y_pred=data[other_expert], average="macro"))
    metrics["otar"]: float = np.average(model_expert) / np.average(expert_expert)

    model_expert: list[float] = [
        recall_score(y_true=data[expert], y_pred=data.predicted_label, average="macro") for expert in experts
    ]
    expert_expert: list[float] = []
    for i, expert in enumerate(experts):
        for j, other_expert in enumerate(experts):
            if i < j:
                expert_expert.append(recall_score(y_true=data[expert], y_pred=data[other_expert], average="macro"))
    metrics["opar"]: float = np.average(model_expert) / np.average(expert_expert)

    Path("results/clf.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    pprint(metrics)


if __name__ == "__main__":
   main()