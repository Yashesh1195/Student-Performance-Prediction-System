import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from utils.helpers import plot_correlation_heatmap, plot_distribution
from utils.data_loader import load_data


def render() -> None:
    st.title("Dataset Overview")

    df_sample = st.session_state.get("data_sample")
    if df_sample is None:
        st.error("Dataset not available.")
        return

    df_full = st.session_state.get("data_full")
    use_full = st.toggle("Use full dataset (slow)", value=False)
    if use_full and df_full is None:
        with st.spinner("Loading full dataset..."):
            df_full = load_data()
            st.session_state.data_full = df_full

    base_df = df_full if use_full and df_full is not None else df_sample
    if not use_full:
        st.caption("Showing a fast sample. Enable full dataset for complete view.")

    df_preview = base_df.head(200)
    df_missing = base_df.sample(n=min(300, len(base_df)), random_state=42)
    df_vis = base_df.sample(n=min(2000, len(base_df)), random_state=42)

    tab_overview, tab_viz, tab_filters = st.tabs(
        ["Overview", "Visualizations", "Filters"]
    )

    with tab_overview:
        st.subheader("Quick Preview")
        st.dataframe(df_preview, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(base_df):,}")
        col2.metric("Columns", f"{base_df.shape[1]}")
        col3.metric("Missing Cells", f"{base_df.isnull().sum().sum():,}")

        st.markdown("### Data Types")
        st.dataframe(base_df.dtypes.astype(str), use_container_width=True)

        st.markdown("### Missing Values Heatmap (Sample)")
        missing_matrix = df_missing.isnull().astype(int)
        fig = px.imshow(
            missing_matrix,
            color_continuous_scale=[[0, "#1A2736"], [1, "#F2B84B"]],
            aspect="auto",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            autosize=True,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

        max_download_rows = 20000
        if len(base_df) > max_download_rows:
            st.warning(
                f"Download limited to first {max_download_rows:,} rows to avoid large transfers."
            )
        download_df = base_df.head(max_download_rows)
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Dataset (Sample)",
            data=csv_data,
            file_name="student_performance_dataset_sample.csv",
            mime="text/csv",
        )

    with tab_viz:
        st.subheader("Correlation Map")
        st.plotly_chart(plot_correlation_heatmap(df_vis), use_container_width=True)

        numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_vis.select_dtypes(include=["object", "category"]).columns.tolist()

        st.subheader("Numeric Distributions")
        num_col = st.selectbox("Select numeric column", numeric_cols)
        st.plotly_chart(plot_distribution(df_vis, num_col), use_container_width=True)

        if cat_cols:
            st.subheader("Categorical Counts")
            cat_col = st.selectbox("Select categorical column", cat_cols)
            counts = df_vis[cat_col].value_counts().reset_index()
            counts.columns = [cat_col, "Count"]
            fig = px.bar(counts, x=cat_col, y="Count", color="Count")
            st.plotly_chart(fig, use_container_width=True)

            if "GPA" in df_vis.columns:
                st.subheader("GPA by Category")
                fig = px.box(df_vis, x=cat_col, y="GPA", color=cat_col)
                st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show pairplot (sample)"):
            pair_cols = numeric_cols[:5]
            grid = sns.pairplot(df_vis[pair_cols])
            st.pyplot(grid.fig)

    with tab_filters:
        st.subheader("Interactive Filters")
        display_cols = st.multiselect(
            "Columns to display",
            base_df.columns.tolist(),
            default=base_df.columns.tolist()[:8],
        )

        filtered_df = base_df.copy()
        cat_cols = base_df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            option_source = df_sample if use_full else base_df
            options = sorted(option_source[col].dropna().unique().tolist())
            selected = st.multiselect(f"Filter {col}", options)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

        max_display_rows = 2000
        if len(filtered_df) > max_display_rows:
            st.caption(
                f"Showing first {max_display_rows:,} rows of {len(filtered_df):,}."
            )
        st.dataframe(
            filtered_df[display_cols].head(max_display_rows),
            use_container_width=True,
        )

        max_download_rows = 20000
        if len(filtered_df) > max_download_rows:
            st.warning(
                f"Download limited to first {max_download_rows:,} rows to avoid large transfers."
            )
        filtered_csv = filtered_df.head(max_download_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered Data (Sample)",
            data=filtered_csv,
            file_name="filtered_students_sample.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    render()
