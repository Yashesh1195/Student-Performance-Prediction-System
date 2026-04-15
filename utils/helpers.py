from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def plot_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=col,
        nbins=30,
        color_discrete_sequence=["#7B5E2B"],
        marginal="box",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual GPA", "y": "Predicted GPA"},
        color_discrete_sequence=["#C89B3C"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    residuals = y_true - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted GPA", "y": "Residual"},
        color_discrete_sequence=["#7B5E2B"],
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#C89B3C")
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig
