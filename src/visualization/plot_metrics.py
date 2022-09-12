import os
from pathlib import Path

import pandas as pd
import plotly.express as px

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

results_path = os.path.join(RESULTS_DIR, "metrics_results.csv")
df = pd.read_csv(results_path)


def get_plot(metric, name):
    fig = px.bar(
        df,
        x="Dataset",
        y=metric,
        color="Model",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=500,
        width=500,
        labels={metric: name, "Model": "Model", "Dataset": "Dataset"},
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20), font=dict(size=18), showlegend=False
    )
    fig.write_image(os.path.join(RESULTS_DIR, f"images/{metric}.svg"))


metrics = ["pos_f1", "pAcc", "rAcc", "sdiv", "hdiv", "gsdiv"]
names = [
    "Position Score",
    "Pitch Accuracy",
    "Rhythm Accuracy",
    "Silence Divergence",
    "Pitch Class Divergence",
    "Groove Similarity Divergence",
]
[get_plot(m, n) for m, n in zip(metrics, names)]
