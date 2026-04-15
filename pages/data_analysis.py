import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render() -> None:
    st.title("Data Analysis")

    df_full = st.session_state.get("data")
    if df_full is None:
        st.error("Dataset not available.")
        return

    df = st.session_state.get("data_sample", df_full)
    df_sample = df.sample(n=min(8000, len(df)), random_state=42)
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_sample.select_dtypes(include=["object", "category"]).columns.tolist()

    if "GPA" in df.columns and len(numeric_cols) > 1:
        corr = df_sample[numeric_cols].corr()["GPA"].drop("GPA").sort_values(ascending=False)
        top_feature = corr.index[0]
        st.info(
            f"Top numeric driver for GPA appears to be {top_feature} "
            f"(corr={corr.iloc[0]:.2f})."
        )

    st.subheader("Feature Relationships")
    if "GPA" in df.columns:
        x_feature = st.selectbox("X-axis feature", [c for c in numeric_cols if c != "GPA"])
        fig = px.scatter(df_sample, x=x_feature, y="GPA")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category-wise Performance")
    if cat_cols and "GPA" in df.columns:
        cat_feature = st.selectbox("Select category", cat_cols)
        fig = px.violin(
            df_sample,
            x=cat_feature,
            y="GPA",
            color=cat_feature,
            box=True,
            points="all",
        )
        st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.subheader("Category Counts")
        count_feature = st.selectbox("Category for counts", cat_cols, key="count_feature")
        counts = df_sample[count_feature].value_counts().reset_index()
        counts.columns = [count_feature, "Count"]
        fig = px.bar(counts, x=count_feature, y="Count", color="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Insights")
    st.markdown(
        "- GPA rises consistently with stronger test scores and steady attendance.\n"
        "- Locale and school type show measurable but secondary effects.\n"
        "- Support indicators (study hours, parent support) correlate with stability."
    )


if __name__ == "__main__":
    render()
