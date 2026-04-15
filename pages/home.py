import streamlit as st

from utils.helpers import compute_regression_metrics
from utils.preprocessing import split_features_target


def render() -> None:
    st.title("Student Performance Prediction System")
    st.markdown(
        "A classic, data-driven experience for understanding the academic and socio-economic "
        "signals that shape GPA outcomes."
    )

    reference_df = st.session_state.get("reference_df")
    splits = st.session_state.get("splits", {})
    splits_sample = st.session_state.get("splits_sample", splits)
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    encoders = st.session_state.get("encoders")

    metrics = st.session_state.get("metrics")
    if metrics is None and model is not None and reference_df is not None:
        test_df = splits_sample.get("test")
        if test_df is None:
            test_df = splits.get("test")
        if test_df is not None:
            try:
                X_test, y_test, _ = split_features_target(
                    test_df, reference_df, scaler, encoders=encoders
                )
                y_pred = model.predict(X_test)
                metrics = compute_regression_metrics(y_test, y_pred)
                st.session_state.metrics = metrics
            except Exception as exc:
                st.warning(f"Metrics unavailable: {exc}")

    if metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("R2 Score", f"{metrics['r2']:.3f}")
        col2.metric("MAE", f"{metrics['mae']:.3f}")
        col3.metric("RMSE", f"{metrics['rmse']:.3f}")

    st.markdown("---")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Problem Statement")
        st.markdown(
            "Predict student GPA from academic performance indicators and socio-demographic "
            "context. The model helps educators and analysts identify key drivers of success."
        )

        st.subheader("ML Pipeline Overview")
        st.markdown(
            "**EDA → Preprocessing → Feature Scaling → Model Training → Evaluation → Prediction**"
        )
        st.markdown(
            "The pipeline mirrors the original notebook: categorical encoding, standardized features, "
            "and regression modeling with diagnostic review."
        )

    with right:
        st.markdown(
            """
            <div class="accent-card">
                <h4>What You Can Do</h4>
                <ul>
                    <li>Explore dataset patterns and correlations</li>
                    <li>Interact with visual insights</li>
                    <li>Generate real-time GPA predictions</li>
                    <li>Run batch inference for CSV uploads</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render()
