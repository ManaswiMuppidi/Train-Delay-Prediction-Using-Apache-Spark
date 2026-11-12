\# Train Delay Prediction System Using Apache Spark



A Big Data and Machine Learning project that analyzes railway delays, discovers station connectivity patterns, and predicts train delays using Apache Spark, GraphX, Spark MLlib, and Streamlit.



\## Project Overview



The Train Delay Prediction System processes large-scale railway data using distributed computing techniques. It performs data preprocessing, statistical analysis, graph-based network analysis, and machine learning to understand and predict train delays.



The project also includes an interactive Streamlit dashboard for exploring railway delay patterns and viewing model predictions.



\## Technologies Used



\- \*\*Python\*\* – Data processing and machine learning

\- \*\*Apache Spark\*\* – Distributed data processing

\- \*\*Spark SQL\*\* – Data manipulation and analysis

\- \*\*Spark GraphX\*\* – Railway station network analysis

\- \*\*Spark MLlib\*\* – Machine learning

\- \*\*Random Forest Regression\*\* – Delay prediction algorithm

\- \*\*Scala\*\* – GraphX implementation

\- \*\*Streamlit\*\* – Interactive dashboard

\- \*\*Pandas and Plotly\*\* – Data analysis and visualization



\## System Architecture



```text

Railway Dataset (CSV)

&#x20;       |

&#x20;       v

Data Ingestion

&#x20;       |

&#x20;       v

Apache Spark

&#x20;       |

&#x20;       v

Data Cleaning and Preprocessing

&#x20;       |

&#x20;       v

Feature Engineering

&#x20;       |

&#x20;       +-----------------------+

&#x20;       |                       |

&#x20;       v                       v

&#x20;  GraphX Analysis         Spark MLlib

&#x20;       |                       |

&#x20;       v                       v

Station Connectivity     Random Forest

and Route Patterns       Regression

&#x20;       |                       |

&#x20;       +-----------+-----------+

&#x20;                   |

&#x20;                   v

&#x20;            Model Evaluation

&#x20;                   |

&#x20;                   v

&#x20;            Streamlit Dashboard

```



\## Project Modules



\### 1. Data Ingestion and Preprocessing



\- Loaded railway data using Apache Spark.

\- Processed missing values and invalid records.

\- Converted date columns into appropriate formats.

\- Prepared clean data for further analysis.



\### 2. Distributed Data Analysis



Used Spark DataFrame operations to analyze:



\- Average train delays.

\- Station-wise delay patterns.

\- Train-wise delay patterns.

\- Monthly delay trends.

\- Early, on-time and delayed arrivals.



\### 3. Graph Analysis Using GraphX



Implemented graph-based railway network analysis using Scala and Apache Spark GraphX.



\- Railway stations represented as vertices.

\- Consecutive station connections represented as directed edges.

\- Analyzed station connectivity and route frequency.

\- Identified connected components in the railway network.



\*\*GraphX Results:\*\*



| Parameter | Result |

|---|---:|

| Total vertices (stations) | 1,527 |

| Total edges (connections) | 44,448 |

| Connected components | 4 |

| Largest component | 1,507 stations |



\### 4. Machine Learning Using Spark MLlib



Implemented a Random Forest Regression model to predict train delays.



\*\*Feature Engineering:\*\*

\- Train number

\- Station name

\- Station number

\- Month

\- Day of the week

\- Previous station delay



\*\*Model Configuration:\*\*

\- Algorithm: Random Forest Regression

\- Number of trees: 10

\- Maximum depth: 5

\- Training and testing: Chronological split

\- Data sampling: 20%



\*\*Model Evaluation Results:\*\*



| Metric | Result |

|---|---:|

| Mean Absolute Error (MAE) | 33.19 minutes |

| Root Mean Squared Error (RMSE) | 77.51 minutes |

| R² Score | 0.6583 |



The model explains approximately 65.83% of the variation in the test-set delay values.



\### 5. Interactive Dashboard



Developed an interactive Streamlit dashboard containing:



\- Overall Railway Delay Dashboard

\- Train Analysis

\- Station Analysis

\- Monthly Analysis

\- Railway Network Analysis

\- Delay Prediction

\- Model Evaluation



\## Project Structure



```text

Train-Delay-Prediction-Using-Apache-Spark/

│

├── dashboard/

│   └── app.py

│

├── graphx/

│   ├── build.sbt

│   ├── project/

│   │   └── build.properties

│   └── src/

│       └── main/

│           └── scala/

│               └── TrainDelayGraph.scala

│

├── check\_stations.py

├── clean\_data.py

├── investigate\_delay.py

├── load\_data.py

├── preprocess.py

├── spark\_analysis.py

├── spark\_test.py

├── test\_saved\_model.py

├── train\_delay\_mllib.py

├── .gitignore

└── README.md

```



\## Installation and Execution



\### Prerequisites



\- Python 3.10

\- Java 17

\- Apache Spark 3.5.x

\- Scala 2.12.x

\- SBT

\- Git



\### 1. Clone the Repository



```bash

git clone https://github.com/ManaswiMuppidi/Train-Delay-Prediction-Using-Apache-Spark.git

cd Train-Delay-Prediction-Using-Apache-Spark

```



\### 2. Create a Virtual Environment



```bash

python -m venv venv

venv\\Scripts\\activate

```



\### 3. Install Python Dependencies



```bash

pip install pyspark==3.5.6 pandas streamlit plotly

```



\### 4. Add the Dataset



Place the railway dataset in the project root directory with the filename:



```text

train\_delay.csv

```



The dataset is not included in this repository.



\### 5. Run Data Preprocessing



```bash

python clean\_data.py

```



\### 6. Run Spark Analysis



```bash

python spark\_analysis.py

```



\### 7. Run the MLlib Model



```bash

python train\_delay\_mllib.py

```



The trained model is saved locally for use by the dashboard.



\### 8. Run GraphX Analysis



Navigate to the GraphX directory:



```bash

cd graphx

sbt run

```



\### 9. Launch the Dashboard



Return to the project root and run:



```bash

cd ..

python -m streamlit run dashboard/app.py

```



Make sure the dataset and trained model are available at the paths expected by the dashboard.



\## Key Highlights



\- Distributed processing of approximately 980,283 valid delay records.

\- Graph-based analysis of 1,527 railway stations.

\- Identification of 44,448 station connections.

\- Machine learning using Apache Spark MLlib.

\- Chronological model evaluation.

\- Interactive dashboard for railway delay analysis and prediction.



\## Future Enhancements



\- Integrate real-time railway data.

\- Experiment with additional regression algorithms.

\- Improve prediction performance for extreme delays.

\- Develop route-based delay propagation analysis.

\- Deploy the dashboard to a cloud platform.



\## Author



\*\*Manaswi Muppidi\*\*



GitHub: \[ManaswiMuppidi](https://github.com/ManaswiMuppidi)



\---



\*Developed as an academic Big Data and Cluster Computing project using Apache Spark, GraphX, and Machine Learning.\*

