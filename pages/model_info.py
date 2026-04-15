import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.helpers import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
)
from utils.preprocessing import split_features_target


def render() -> None:
    st.title("Model Information")

    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    reference_df = st.session_state.get("reference_df")
    splits = st.session_state.get("splits", {})
    splits_sample = st.session_state.get("splits_sample", splits)
    encoders = st.session_state.get("encoders")

    if model is None or reference_df is None:
        st.error("Model or reference data not available.")
        return

    st.subheader("Model Summary")
    st.markdown(f"**Model Type:** {model.__class__.__name__}")

    params = model.get_params() if hasattr(model, "get_params") else {}
    if params:
        st.dataframe(pd.DataFrame.from_dict(params, orient="index", columns=["Value"]))

    insight_df = st.session_state.get("data_sample")
    if insight_df is None:
        insight_df = splits_sample.get("train") or splits.get("train") or reference_df

    if insight_df is not None and "GPA" in insight_df.columns:
        st.subheader("Dataset Insights")
        try:
            numeric_cols = insight_df.select_dtypes(include=[np.number]).columns.tolist()
            if "GPA" in numeric_cols and len(numeric_cols) > 1:
                corr = (
                    insight_df[numeric_cols]
                    .corr()["GPA"]
                    .drop("GPA")
                    .sort_values(ascending=False)
                )
                top_pos = corr.head(2)
                top_neg = corr.tail(2)
                st.markdown(
                    "- Strongest positive signals: "
                    + ", ".join([f"{idx} ($r$={val:.2f})" for idx, val in top_pos.items()])
                )
                st.markdown(
                    "- Strongest negative signals: "
                    + ", ".join([f"{idx} ($r$={val:.2f})" for idx, val in top_neg.items()])
                )

            for col in ["Gender", "SchoolType", "Locale", "ParentalEducation"]:
                if col in insight_df.columns:
                    means = insight_df.groupby(col, dropna=True)["GPA"].mean().sort_values(
                        ascending=False
                    )
                    if not means.empty:
                        st.markdown(
                            f"- {col} trend: highest avg GPA in **{means.index[0]}** "
                            f"({means.iloc[0]:.2f}); lowest in **{means.index[-1]}** "
                            f"({means.iloc[-1]:.2f})."
                        )
            gpa_stats = insight_df["GPA"].describe()
            st.markdown(
                f"- GPA distribution: median {gpa_stats['50%']:.2f}, "
                f"IQR {gpa_stats['25%']:.2f}–{gpa_stats['75%']:.2f}."
            )
        except Exception as exc:
            st.info(f"Insights unavailable: {exc}")

    test_df = splits_sample.get("test")
    if test_df is None:
        test_df = splits.get("test")
    if test_df is not None:
        try:
            X_test, y_test, feature_cols = split_features_target(
                test_df, reference_df, scaler, encoders=encoders
            )
            y_pred = model.predict(X_test)
            metrics = compute_regression_metrics(y_test, y_pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("R2 Score", f"{metrics['r2']:.3f}")
            col2.metric("MAE", f"{metrics['mae']:.3f}")
            col3.metric("RMSE", f"{metrics['rmse']:.3f}")

            st.subheader("Actual vs Predicted")
            st.plotly_chart(plot_actual_vs_predicted(y_test, y_pred), use_container_width=True)

            st.subheader("Residual Plot")
            st.plotly_chart(plot_residuals(y_test, y_pred), use_container_width=True)

            if hasattr(model, "coef_"):
                st.subheader("Feature Importance (Coefficients)")
                coefs = pd.DataFrame(
                    {"Feature": feature_cols, "Coefficient": model.coef_}
                ).sort_values(by="Coefficient", ascending=False)
                st.dataframe(coefs, use_container_width=True)
                fig = px.bar(coefs, x="Feature", y="Coefficient")
                st.plotly_chart(fig, use_container_width=True)
            elif hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance")
                importances = pd.DataFrame(
                    {
                        "Feature": feature_cols,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values(by="Importance", ascending=False)
                fig = px.bar(importances, x="Feature", y="Importance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance is not available for this model.")

        except Exception as exc:
            st.error(f"Model analysis failed: {exc}")
    else:
        st.warning("Test split not found. Metrics and plots are unavailable.")


if __name__ == "__main__":
    render()
