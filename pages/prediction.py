import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocessing import preprocess_input


def _numeric_range(series: pd.Series) -> tuple:
    series = pd.to_numeric(series.dropna(), errors="coerce")
    return float(series.min()), float(series.max())


def render() -> None:
    st.title("GPA Prediction")

    reference_df = st.session_state.get("reference_df")
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    encoders = st.session_state.get("encoders")

    if reference_df is None or model is None:
        st.error("Model or reference data not available.")
        return

    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = reference_df.select_dtypes(include=["object", "category"]).columns.tolist()

    with st.form("prediction_form"):
        st.subheader("Student Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            age_min, age_max = _numeric_range(reference_df["Age"])
            age = st.slider("Age", int(age_min), int(age_max), int(np.median(reference_df["Age"])))

            grade_min, grade_max = _numeric_range(reference_df["Grade"])
            grade = st.slider(
                "Grade",
                int(grade_min),
                int(grade_max),
                int(np.median(reference_df["Grade"])),
            )

            gender = st.selectbox(
                "Gender", sorted(reference_df["Gender"].dropna().unique())
            )

        with col2:
            race = st.selectbox("Race", sorted(reference_df["Race"].dropna().unique()))
            parental = st.selectbox(
                "Parental Education",
                sorted(reference_df["ParentalEducation"].dropna().unique()),
            )
            school_type = st.selectbox(
                "School Type", sorted(reference_df["SchoolType"].dropna().unique())
            )

        with col3:
            locale = st.selectbox(
                "Locale", sorted(reference_df["Locale"].dropna().unique())
            )
            ses_min, ses_max = _numeric_range(reference_df["SES_Quartile"])
            ses_quartile = st.slider(
                "SES Quartile",
                int(ses_min),
                int(ses_max),
                int(np.median(reference_df["SES_Quartile"])),
            )

        st.subheader("Academic Indicators")
        col4, col5, col6 = st.columns(3)

        with col4:
            math_min, math_max = _numeric_range(reference_df["TestScore_Math"])
            test_math = st.slider(
                "Test Score - Math",
                float(math_min),
                float(math_max),
                float(np.median(reference_df["TestScore_Math"])),
            )

        with col5:
            read_min, read_max = _numeric_range(reference_df["TestScore_Reading"])
            test_reading = st.slider(
                "Test Score - Reading",
                float(read_min),
                float(read_max),
                float(np.median(reference_df["TestScore_Reading"])),
            )

        with col6:
            sci_min, sci_max = _numeric_range(reference_df["TestScore_Science"])
            test_science = st.slider(
                "Test Score - Science",
                float(sci_min),
                float(sci_max),
                float(np.median(reference_df["TestScore_Science"])),
            )

        st.subheader("Lifestyle & Support")
        col7, col8, col9 = st.columns(3)

        with col7:
            att_min, att_max = _numeric_range(reference_df["AttendanceRate"])
            attendance = st.slider(
                "Attendance Rate",
                float(att_min),
                float(att_max),
                float(np.median(reference_df["AttendanceRate"])),
            )

            study_min, study_max = _numeric_range(reference_df["StudyHours"])
            study_hours = st.slider(
                "Study Hours",
                float(study_min),
                float(study_max),
                float(np.median(reference_df["StudyHours"])),
            )

        with col8:
            internet = st.selectbox(
                "Internet Access",
                sorted(reference_df["InternetAccess"].dropna().unique()),
            )
            extracurricular = st.selectbox(
                "Extracurricular",
                sorted(reference_df["Extracurricular"].dropna().unique()),
            )
            part_time = st.selectbox(
                "Part-Time Job",
                sorted(reference_df["PartTimeJob"].dropna().unique()),
            )

        with col9:
            parent_support = st.selectbox(
                "Parent Support",
                sorted(reference_df["ParentSupport"].dropna().unique()),
            )
            romantic = st.selectbox(
                "Romantic",
                sorted(reference_df["Romantic"].dropna().unique()),
            )
            free_time = st.selectbox(
                "Free Time",
                sorted(reference_df["FreeTime"].dropna().unique()),
            )
            go_out = st.selectbox(
                "Go Out",
                sorted(reference_df["GoOut"].dropna().unique()),
            )

        submitted = st.form_submit_button("Predict GPA")

    if submitted:
        payload = {
            "Age": age,
            "Grade": grade,
            "Gender": gender,
            "Race": race,
            "SES_Quartile": ses_quartile,
            "ParentalEducation": parental,
            "SchoolType": school_type,
            "Locale": locale,
            "TestScore_Math": test_math,
            "TestScore_Reading": test_reading,
            "TestScore_Science": test_science,
            "AttendanceRate": attendance,
            "StudyHours": study_hours,
            "InternetAccess": internet,
            "Extracurricular": extracurricular,
            "PartTimeJob": part_time,
            "ParentSupport": parent_support,
            "Romantic": romantic,
            "FreeTime": free_time,
            "GoOut": go_out,
        }

        input_df = pd.DataFrame([payload])
        with st.spinner("Predicting..."):
            try:
                X_pred, _ = preprocess_input(
                    input_df, reference_df, scaler, encoders=encoders
                )
                prediction = float(model.predict(X_pred)[0])
                st.success(f"Predicted GPA: {prediction:.2f}")

                metrics = st.session_state.get("metrics")
                if metrics and "rmse" in metrics:
                    lower = prediction - metrics["rmse"]
                    upper = prediction + metrics["rmse"]
                    st.markdown(
                        f"Estimated range: {lower:.2f} to {upper:.2f} (based on RMSE)"
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    render()
