
from datetime import date
import os
import glob

import streamlit as st
import pandas as pd
import plotly.express as px

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Train Delay Prediction System",
    page_icon="🚆",
    layout="wide"
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "C:/TrainDelayProject/train_delay.csv"
    )

    df["train_no"] = df["train_no"].astype(str)

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df["delay"] = pd.to_numeric(
        df["delay"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "delay", "train_no", "station_name"]
    )

    df["month"] = df["date"].dt.strftime("%Y-%m")

    return df


df = load_data()


# ==========================================
# 3. LOAD MODEL EVALUATION DATA
# ==========================================

@st.cache_data
def load_evaluation_data():

    base_path = "C:/TrainDelayProject/model_evaluation"

    def read_spark_csv(folder):

        folder_path = os.path.join(
            base_path,
            folder
        )

        files = glob.glob(
            os.path.join(folder_path, "part-*.csv")
        )

        if not files:
            return pd.DataFrame()

        dataframes = []

        for file in files:

            temp_df = pd.read_csv(file)

            dataframes.append(temp_df)

        return pd.concat(
            dataframes,
            ignore_index=True
        )

    predictions = read_spark_csv(
        "model_predictions"
    )

    residuals = read_spark_csv(
        "model_residuals"
    )

    metrics = read_spark_csv(
        "model_metrics"
    )

    return predictions, residuals, metrics


predictions_df, residuals_df, metrics_df = load_evaluation_data()


# ==========================================
# 4. LOAD SPARK AND MLlib MODEL
# ==========================================

@st.cache_resource
def load_spark():

    return SparkSession.builder \
        .appName("TrainDelayStreamlit") \
        .master("local[2]") \
        .getOrCreate()


@st.cache_resource
def load_prediction_model():

    return PipelineModel.load(
        "C:/TrainDelayProject/train_delay_rf_model_v2"
    )


# ==========================================
# 5. GRAPHX RESULTS
# ==========================================

# Actual results obtained from GraphX execution

top_stations = pd.DataFrame({

    "Station": [
        "PUNE", "BRC", "RTM", "RGS", "PNVL",
        "BSL", "BSR", "ET", "GKP", "MMR"
    ],

    "Connections": [
        27, 26, 25, 25, 24,
        24, 23, 23, 23, 22
    ]

})


top_routes = pd.DataFrame({

    "Source Station": [
        "DR", "TNA", "DR", "CSMT", "KYN",
        "TNA", "BNC", "SGR", "KKW", "RN"
    ],

    "Destination Station": [
        "TNA", "KYN", "CSMT", "DR", "TNA",
        "DR", "SBC", "RN", "SNDD", "SGR"
    ],

    "Frequency": [
        2759, 2595, 2285, 2284, 2063,
        1831, 1319, 1314, 1313, 1313
    ]

})


components = pd.DataFrame({

    "Component ID": [0, 100, 46, 22],

    "Number of Stations": [1507, 9, 6, 5]

})


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🚆 Train Delay System")

page = st.sidebar.radio(

    "Navigation",

    [
        "Dashboard",
        "Train Analysis",
        "Station Analysis",
        "Monthly Analysis",
        "Network Analysis",
        "Delay Prediction",
        "Model Evaluation"
    ]

)


# ==========================================
# 7. MAIN HEADER
# ==========================================

st.title("🚆 Train Delay Prediction System")

st.caption(
    "Cluster Computing Project | Apache Spark | GraphX | MLlib"
)


# ==========================================
# 8. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("System Overview")

    total_records = len(df)

    avg_delay = df["delay"].mean()

    delayed_records = (df["delay"] > 0).sum()

    on_time = (df["delay"] == 0).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    c2.metric(
        "Average Delay",
        f"{avg_delay:.2f} min"
    )

    c3.metric(
        "Delayed Records",
        f"{delayed_records:,}"
    )

    c4.metric(
        "On-Time Records",
        f"{on_time:,}"
    )


    st.subheader("Delay Distribution")

    delay_counts = pd.DataFrame({

        "Category": ["Early", "On Time", "Delayed"],

        "Records": [
            (df["delay"] < 0).sum(),
            (df["delay"] == 0).sum(),
            (df["delay"] > 0).sum()
        ]

    })


    fig = px.pie(

        delay_counts,

        names="Category",

        values="Records",

        hole=0.4,

        title="Train Delay Categories"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ==========================================
# 9. TRAIN ANALYSIS
# ==========================================

elif page == "Train Analysis":

    st.header("Train-wise Delay Analysis")

    train_stats = (

        df.groupby("train_no")["delay"]

        .agg(["mean", "count"])

        .reset_index()

    )


    train_stats.columns = [

        "Train Number",

        "Average Delay",

        "Records"

    ]


    train_stats = train_stats[

        train_stats["Records"] >= 30

    ]


    train_stats = train_stats.sort_values(

        "Average Delay",

        ascending=False

    ).head(15)


    fig = px.bar(

        train_stats,

        x="Train Number",

        y="Average Delay",

        title="Top 15 Trains by Average Delay",

        hover_data=["Records"],

        color="Average Delay",

        text_auto=".1f"

    )


    fig.update_xaxes(type="category")


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.dataframe(

        train_stats,

        use_container_width=True

    )


# ==========================================
# 10. STATION ANALYSIS
# ==========================================

elif page == "Station Analysis":

    st.header("Station-wise Delay Analysis")


    station_stats = (

        df.groupby("station_name")["delay"]

        .agg(["mean", "count"])

        .reset_index()

    )


    station_stats.columns = [

        "Station",

        "Average Delay",

        "Records"

    ]


    station_stats = station_stats[

        station_stats["Records"] >= 30

    ]


    station_stats = station_stats.sort_values(

        "Average Delay",

        ascending=False

    ).head(15)


    fig = px.bar(

        station_stats,

        x="Station",

        y="Average Delay",

        title="Top 15 Stations by Average Delay",

        hover_data=["Records"],

        color="Average Delay",

        text_auto=".1f"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.dataframe(

        station_stats,

        use_container_width=True

    )


# ==========================================
# 11. MONTHLY ANALYSIS
# ==========================================

elif page == "Monthly Analysis":

    st.header("Monthly Delay Analysis")


    monthly_stats = (

        df.groupby("month")["delay"]

        .agg(["mean", "count"])

        .reset_index()

        .sort_values("month")

    )


    monthly_stats.columns = [

        "Month",

        "Average Delay",

        "Records"

    ]


    fig = px.line(

        monthly_stats,

        x="Month",

        y="Average Delay",

        markers=True,

        title="Monthly Average Train Delay"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )


    st.dataframe(

        monthly_stats,

        use_container_width=True

    )


# ==========================================
# 12. GRAPHX NETWORK ANALYSIS
# ==========================================

elif page == "Network Analysis":

    st.header("🚉 Railway Network Analysis")


    st.write(

        "Graph-based railway station connectivity analysis "

        "using Apache Spark GraphX."

    )


    # GraphX summary

    st.subheader("Graph Network Overview")


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Total Stations",
        "1,527"
    )

    c2.metric(
        "Total Connections",
        "44,448"
    )

    c3.metric(
        "Connected Components",
        "4"
    )


    st.divider()


    # Top connected stations

    st.subheader("Top 10 Connected Stations")


    fig1 = px.bar(

        top_stations.sort_values(

            "Connections",

            ascending=True

        ),

        x="Connections",

        y="Station",

        orientation="h",

        text="Connections",

        title="Stations Ranked by Network Connections",

        color="Connections"

    )


    st.plotly_chart(

        fig1,

        use_container_width=True

    )


    st.dataframe(

        top_stations,

        use_container_width=True,

        hide_index=True

    )


    st.divider()


    # Most frequent routes

    st.subheader("Top 10 Railway Routes")


    route_display = top_routes.copy()

    route_display["Route"] = (

        route_display["Source Station"]

        + " → "

        + route_display["Destination Station"]

    )


    fig2 = px.bar(

        route_display.sort_values(

            "Frequency",

            ascending=True

        ),

        x="Frequency",

        y="Route",

        orientation="h",

        text="Frequency",

        title="Most Frequently Observed Station Connections",

        color="Frequency"

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )


    st.dataframe(

        top_routes[

            [

                "Source Station",

                "Destination Station",

                "Frequency"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )


    st.divider()


    # Connected components

    st.subheader("Connected Components Analysis")


    fig3 = px.bar(

        components,

        x="Component ID",

        y="Number of Stations",

        text="Number of Stations",

        title="Stations in Each Connected Component"

    )


    st.plotly_chart(

        fig3,

        use_container_width=True

    )


    st.dataframe(

        components,

        use_container_width=True,

        hide_index=True

    )


    st.info(

        "The largest connected component contains "

        "1,507 of the 1,527 stations. GraphX connected "

        "components identify groups of connected stations, "

        "ignoring edge direction for connectivity."

    )


# ==========================================
# 13. TRAIN DELAY PREDICTION
# ==========================================

elif page == "Delay Prediction":

    st.header("🚆 Train Delay Prediction")


    st.write(

        "Predict train delay using the trained Apache Spark MLlib model."

    )


    with st.form("prediction_form"):


        train_no = st.text_input(

            "Train Number",

            placeholder="Example: 7085"

        )


        station_name = st.text_input(

            "Station Name",

            placeholder="Example: PUNE"

        )


        station_no = st.number_input(

            "Station Position",

            min_value=1,

            max_value=64,

            value=1

        )


        journey_date = st.date_input(

            "Journey Date",

            value=date.today()

        )


        previous_delay = st.number_input(

            "Previous Station Delay (minutes)",

            min_value=-120.0,

            max_value=7018.0,

            value=0.0

        )


        submitted = st.form_submit_button(

            "Predict Train Delay"

        )


    if submitted:


        if not train_no.strip() or not station_name.strip():

            st.error(

                "Please enter both train number and station name."

            )


        else:


            try:


                with st.spinner("Loading MLlib model and predicting..."):


                    spark = load_spark()

                    model = load_prediction_model()


                    # Spark dayofweek:
                    # Sunday = 1, Monday = 2, ..., Saturday = 7

                    spark_day = (

                        (journey_date.weekday() + 1) % 7

                    ) + 1


                    input_data = [(

                        train_no.strip(),

                        station_name.strip().upper(),

                        float(station_no),

                        float(journey_date.month),

                        float(spark_day),

                        float(previous_delay)

                    )]


                    input_columns = [

                        "train_no",

                        "station_name",

                        "station_no",

                        "month",

                        "day_of_week",

                        "previous_station_delay"

                    ]


                    input_df = spark.createDataFrame(

                        input_data,

                        input_columns

                    )


                    result = model.transform(input_df)


                    predicted_delay = result.select(

                        "prediction"

                    ).first()["prediction"]


                st.success("Prediction completed!")


                st.metric(

                    "Predicted Train Delay",

                    f"{predicted_delay:.2f} minutes"

                )


                if predicted_delay > 0:

                    st.warning(

                        "The model predicts a delay."

                    )


                elif predicted_delay < 0:

                    st.info(

                        "The model predicts an early arrival."

                    )


                else:

                    st.success(

                        "The model predicts an on-time arrival."

                    )


            except Exception as e:

                st.error(

                    f"Prediction failed: {e}"

                )


# ==========================================
# 14. MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.header("📊 Model Evaluation")


    st.write(

        "Evaluation of the Random Forest Regression model "

        "developed using Apache Spark MLlib."

    )


    if predictions_df.empty or metrics_df.empty:

        st.error(

            "Evaluation files not found. "

            "Please check the model_evaluation folder."

        )


    else:

        # ----------------------------------
        # 1. LOAD MODEL METRICS
        # ----------------------------------

        metric_columns = {

            str(column).strip().upper(): column

            for column in metrics_df.columns

        }


        required_metrics = ["MAE", "RMSE", "R2"]


        if not all(

            name in metric_columns

            for name in required_metrics

        ):

            st.error(

                "Metrics CSV must contain MAE, RMSE and R2 columns."

            )


        else:

            metric_row = metrics_df.iloc[0]


            mae = float(

                metric_row[metric_columns["MAE"]]

            )


            rmse = float(

                metric_row[metric_columns["RMSE"]]

            )


            r2 = float(

                metric_row[metric_columns["R2"]]

            )


            # ----------------------------------
            # 2. MODEL PERFORMANCE
            # ----------------------------------

            st.subheader("Model Performance Metrics")


            c1, c2, c3 = st.columns(3)


            c1.metric(

                "MAE",

                f"{mae:.2f} minutes"

            )


            c2.metric(

                "RMSE",

                f"{rmse:.2f} minutes"

            )


            c3.metric(

                "R² Score",

                f"{r2:.4f}"

            )


            st.divider()


            # ----------------------------------
            # 3. PREPARE PREDICTION DATA
            # ----------------------------------

            plot_df = predictions_df.copy()


            plot_df["delay"] = pd.to_numeric(

                plot_df["delay"],

                errors="coerce"

            )


            plot_df["prediction"] = pd.to_numeric(

                plot_df["prediction"],

                errors="coerce"

            )


            plot_df = plot_df.dropna(

                subset=["delay", "prediction"]

            )


            # ----------------------------------
            # 4. EVALUATION DATASET
            # ----------------------------------

            st.subheader("Evaluation Dataset")


            c1, c2 = st.columns(2)


            c1.metric(

                "Prediction Records",

                f"{len(plot_df):,}"

            )


            c2.metric(

                "Metrics Available",

                "MAE | RMSE | R²"

            )


            expected_test_records = 53427


            if len(plot_df) != expected_test_records:

                st.warning(

                    f"Record-count mismatch: expected "

                    f"{expected_test_records:,} testing records "

                    f"from the training log, but found "

                    f"{len(plot_df):,} prediction records. "

                    "Check whether old Spark output files are "

                    "present in the model_evaluation folders."

                )


            else:

                st.success(

                    "Prediction record count matches the training log."

                )


            st.divider()


            # ----------------------------------
            # 5. ACTUAL VS PREDICTED
            # ----------------------------------

            st.subheader("Actual vs Predicted Delay")


            scatter_df = plot_df.copy()


            if len(scatter_df) > 5000:

                scatter_df = scatter_df.sample(

                    5000,

                    random_state=42

                )


            fig1 = px.scatter(

                scatter_df,

                x="delay",

                y="prediction",

                title="Actual Delay vs Predicted Delay",

                labels={

                    "delay": "Actual Delay (minutes)",

                    "prediction": "Predicted Delay (minutes)"

                },

                opacity=0.5

            )


            # Perfect prediction reference line

            minimum_value = min(

                plot_df["delay"].min(),

                plot_df["prediction"].min()

            )


            maximum_value = max(

                plot_df["delay"].max(),

                plot_df["prediction"].max()

            )


            fig1.add_shape(

                type="line",

                x0=minimum_value,

                y0=minimum_value,

                x1=maximum_value,

                y1=maximum_value,

                line=dict(

                    color="red",

                    width=2,

                    dash="dash"

                )

            )


            fig1.add_annotation(

                x=maximum_value,

                y=maximum_value,

                text="Perfect Prediction",

                showarrow=False,

                xanchor="right",

                yanchor="bottom"

            )


            st.plotly_chart(

                fig1,

                use_container_width=True

            )


            st.caption(

                "The red dashed line represents perfect predictions. "

                "Points closer to this line indicate smaller errors."

            )


            st.divider()


            # ----------------------------------
            # 6. RESIDUAL ANALYSIS
            # ----------------------------------

            st.subheader("Prediction Error Distribution")


            if (

                not residuals_df.empty

                and "residual" in residuals_df.columns

            ):


                residual_plot = residuals_df.copy()


                residual_plot["residual"] = pd.to_numeric(

                    residual_plot["residual"],

                    errors="coerce"

                )


                residual_plot = residual_plot.dropna(

                    subset=["residual"]

                )


                # Error statistics

                residual_plot["absolute_error"] = (

                    residual_plot["residual"].abs()

                )


                mean_residual = residual_plot["residual"].mean()


                median_residual = residual_plot["residual"].median()


                max_error = residual_plot["absolute_error"].max()


                e1, e2, e3 = st.columns(3)


                e1.metric(

                    "Mean Residual",

                    f"{mean_residual:.2f} min"

                )


                e2.metric(

                    "Median Residual",

                    f"{median_residual:.2f} min"

                )


                e3.metric(

                    "Maximum Absolute Error",

                    f"{max_error:.2f} min"

                )


                # Full residual distribution

                st.markdown(

                    "**Full Residual Distribution**"

                )


                fig2 = px.histogram(

                    residual_plot,

                    x="residual",

                    nbins=50,

                    title="Distribution of All Prediction Residuals",

                    labels={

                        "residual": "Residual (minutes)"

                    }

                )


                st.plotly_chart(

                    fig2,

                    use_container_width=True

                )


                # Zoomed residual distribution

                st.markdown(

                    "**Zoomed Residual Distribution "

                    "(Central 98% of Records)**"

                )


                lower_limit = residual_plot["residual"].quantile(0.01)


                upper_limit = residual_plot["residual"].quantile(0.99)


                zoom_df = residual_plot[

                    (residual_plot["residual"] >= lower_limit)

                    &

                    (residual_plot["residual"] <= upper_limit)

                ]


                fig3 = px.histogram(

                    zoom_df,

                    x="residual",

                    nbins=50,

                    title="Residual Distribution - Central 98%",

                    labels={

                        "residual": "Residual (minutes)"

                    }

                )


                st.plotly_chart(

                    fig3,

                    use_container_width=True

                )


                st.caption(

                    f"The zoomed chart displays residuals between "

                    f"{lower_limit:.2f} and {upper_limit:.2f} minutes. "

                    "The full chart above retains all records."

                )


            else:

                st.warning(

                    "Residual data is unavailable."

                )


            st.divider()


            # ----------------------------------
            # 7. PREDICTION RESULTS TABLE
            # ----------------------------------

            st.subheader("Prediction Results")


            st.dataframe(

                plot_df.head(100),

                use_container_width=True,

                hide_index=True

            )


            # ----------------------------------
            # 8. MODEL INTERPRETATION
            # ----------------------------------

            st.divider()


            st.subheader("Model Interpretation")


            st.write(

                f"**MAE:** The average absolute prediction error "

                f"is {mae:.2f} minutes."

            )


            st.write(

                f"**RMSE:** The root mean squared error is "

                f"{rmse:.2f} minutes. Larger prediction errors "

                "have a greater influence on this metric."

            )


            st.write(

                f"**R² Score:** The model explains approximately "

                f"{r2 * 100:.2f}% of the variation in test-set delays."

            )


            st.divider()


            # ----------------------------------
            # 9. MODEL LIMITATIONS
            # ----------------------------------

            st.subheader("Model Limitations")


            st.warning(

                """
                1. The model may underestimate extremely high train delays.

                2. Random Forest predictions can form clusters because
                   records may fall into the same decision-tree leaves.

                3. The model depends on input features such as train
                   number, station, station position, month, day of week
                   and previous station delay.

                4. Unexpected operational disruptions may not be
                   represented adequately by these features.

                5. Prediction performance should be interpreted using
                   the chronological test dataset, not only training results.
                """

            )


            st.info(

                "Residual = Actual Delay − Predicted Delay. "

                "A positive residual means the model underestimated "

                "the actual delay; a negative residual means it "

                "overestimated the delay."

            )
