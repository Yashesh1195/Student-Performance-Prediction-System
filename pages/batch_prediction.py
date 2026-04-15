import pandas as pd
import streamlit as st

from utils.preprocessing import preprocess_input, validate_columns, TARGET_COL


def render() -> None:
    st.title("Batch Prediction")

    reference_df = st.session_state.get("reference_df")
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    encoders = st.session_state.get("encoders")

    if reference_df is None or model is None:
        st.error("Model or reference data not available.")
        return

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file with the same feature columns used in training.")
        return

    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return

    feature_columns = [col for col in reference_df.columns if col != TARGET_COL]
    missing, _ = validate_columns(df_input, feature_columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    with st.spinner("Running predictions..."):
        try:
            X_pred, _ = preprocess_input(
                df_input, reference_df, scaler, encoders=encoders
            )
            predictions = model.predict(X_pred)
            df_output = df_input.copy()
            df_output["Predicted_GPA"] = predictions

            st.dataframe(df_output.head(200), use_container_width=True)

            csv_data = df_output.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")


if __name__ == "__main__":
    render()
