# flight_delay_prediction
## Flight Delay Prediction with MLOps

## Overview
This project develops a machine learning model to predict flight delays (arrival delays > 15 minutes) using historical flight data from the United States Department of Transportation's Bureau of Transportation Statistics. The goal is to create a robust, interpretable, and scalable solution that demonstrates key data science and MLOps practices, including data preprocessing, model training, and explainability. The project is designed to support optimizing flight schedules, improving operational efficiency, and enhancing customer satisfaction through data-driven insights.

## Dataset
- **Source**: United States Department of Transportation's Bureau of Transportation Statistics ([link](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)).
- **Features**: Includes `DAY_OF_WEEK`, `FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DEP_DELAY`, `DISTANCE`, and more.
- **Target**: Binary `delayed` variable (1 if `ARR_DELAY > 15`, else 0).
- **Acknowledgements**: Inspired by Farzad Nekouei's Kaggle notebook ([link](https://www.kaggle.com/code/farzadnekouei/flight-data-eda-to-preprocessing)).

## Approach
The project is structured into four main sections, implemented in `flight_delay_prediction.ipynb`:

1. **Data Exploration and Preprocessing**:
   - Used `pandas` and `numpy` for data manipulation and cleaning.
   - Performed exploratory data analysis (EDA) with `matplotlib` and `seaborn`, creating high-quality visualizations to uncover patterns (e.g., delay distributions, trends by day or airport).
   - Engineered features like `dep_hour` and applied target encoding to `ORIGIN` and `DEST`, Yeo-Johnson transformation to `DEP_DELAY`, and feature scaling for numerical features.
<img width="1190" height="590" alt="output 1" src="https://github.com/user-attachments/assets/189a567e-1e96-4c9f-9233-6bc509e92b4b" />
<img width="858" height="775" alt="output 2" src="https://github.com/user-attachments/assets/735f410b-58ee-421a-90ff-096697e25a6d" />

2. **Model Building**:
   - Built a machine learning pipeline using `sklearn` and `category_encoders` to streamline preprocessing (imputation, target encoding, Yeo-Johnson transformation, scaling) and modeling.
   - Trained an `XGBoost` classifier, optimized via `RandomizedSearchCV`, to predict flight delays.
   - Evaluated performance using F1-score and AUC-ROC to handle potential class imbalance.

3. **Model Explainability**:
   - Utilized `SHAP` to analyze feature importance, revealing `DEP_DELAY` as the dominant predictor and identifying delay-prone airports.
   - Visualized SHAP results to communicate insights effectively to non-technical stakeholders.
<img width="755" height="259" alt="output 3" src="https://github.com/user-attachments/assets/08be376d-cfea-41f4-8438-90e11bfc08c8" />

4. **Discussion**:
   - Translated model insights into actionable strategies for Virgin Atlantic, such as optimizing schedules, implementing delay mitigation protocols, and enhancing passenger communication.

## Key Technical Aspects
- **Data Manipulation**: Leveraged `pandas` and `numpy` for efficient data handling and preprocessing.
- **Visualization**: Created clear, accessible plots with `matplotlib` and `seaborn` to convey insights to non-technical audiences, a critical data science skill.
- **Modeling**: Used `sklearn`, `category_encoders`, and `xgboost` for robust classification, with a focus on a standardized pipeline to reduce code redundancy and errors.
- **MLOps**: Implemented reusable functions and a pipeline for scalability and maintainability, versioned with Git for reproducibility.

## Results
- **Model Performance**: The optimized XGBoost model achieved an AUC-ROC of approximately 0.93, indicating strong predictive capability.
- **Key Insights**:
  - `DEP_DELAY` is the primary driver of arrival delays, with higher departure delays strongly correlated with late arrivals.
  - Certain airports (`ORIGIN`, `DEST`) show consistent delay patterns, suggesting opportunities for targeted operational improvements.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Includes `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `category_encoders`, `xgboost`, `shap`, `streamlit`, `joblib`).

2. **Run the Notebook**:
   ```bash
   jupyter notebook flight_delay_prediction.ipynb
   ```
   Execute cells sequentially to perform EDA, train the model, and generate SHAP visualizations.

3. **Optional Deployment Simulation**:
   - A Streamlit app (`flight_delay_prediction_streamlit_app.py`) simulates model deployment for real-time delay predictions.
   - Run with:
     ```bash
     streamlit run flight_delay_prediction_streamlit_app.py
     ```

## Summary
This project demonstrates how machine learning can drive operational excellence and customer satisfaction:
- **Scheduling Optimization**: Adjust schedules based on delay-prone airports and times identified by SHAP analysis.
- **Proactive Delay Management**: Flag high-risk flights for preemptive measures like crew rotations or passenger rerouting.
- **Customer Experience**: Use model predictions to provide timely delay alerts and rebooking options, fostering loyalty.
- **Transparency**: Present SHAP-driven insights in reports for regulators and stakeholders, showcasing data-driven decision-making.

This work highlights technical proficiency in machine learning, MLOps, and actionable analytics, making it directly applicable to efficiency and customer-centric innovation.
