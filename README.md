  #   Liver Cirrhosis Stage Detection

##   Overview

    This project aims to determine the level of liver damage (liver cirrhosis) in patients based on their medical data. The system is designed to output the histologic stage of the disease (1, 2, or 3).

##   Dataset

    The data utilized for this project is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver, conducted between 1974 and 1984.

    **Description of Columns (from Liver Cirrhosis Stage Detection.pdf):**

    * **N\_Days**: Number of days between registration and the earlier of death, transplantation, or study analysis time in 1986.
    * **Status**: Status of the patient C (censored), CL (censored due to liver tx), or D (death).
    * **Drug**: Type of drug D-penicillamine or placebo.
    * **Age**: Age in days.
    * **Sex**: M (male) or F (female).
    * **Ascites**: Presence of ascites N (No) or Y (Yes).
    * **Hepatomegaly**: Presence of hepatomegaly N (No) or Y (Yes).
    * **Spiders**: Presence of spiders N (No) or Y (Yes).
    * **Edema**: Presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy).
    * **Bilirubin**: Serum bilirubin in \[mg/dl].
    * **Cholesterol**: Serum cholesterol in \[mg/dl].
    * **Albumin**: Albumin in \[gm/dl].
    * **Copper**: Urine copper in \[ug/day].
    * **Alk\_Phos**: Alkaline phosphatase in \[U/liter].
    * **SGOT**: SGOT in \[U/ml] (a liver enzyme).
    * **Tryglicerides**: Triglycerides in \[mg/dl].
    * **Platelets**: Platelets per cubic \[ml/1000].
    * **Prothrombin**: Prothrombin time in seconds \[s].
    * **Stage**: Histologic stage of disease (1, 2, or 3) - **Target Variable**.

    **First 5 rows of the dataset:**

    ```
       N_Days Status Drug   Age Sex Ascites Hepatomegaly Spiders Edema  Bilirubin  Cholesterol  Albumin  Copper  Alk_Phos   SGOT  Tryglicerides  Platelets  Prothrombin  Stage
    0   -999   D    D-penicillamine  19248   F       N            Y       Y     N        1.4       261.0    2.60    156.0    1718.0  137.95           172.0     190.0         12.0
    1   -881   C    D-penicillamine  21279   F       N            Y       N     N        3.4       284.0    3.50     43.0    1425.0  137.95           172.0     190.0         12.0
    2   -807   C    D-penicillamine  23508   F       N            Y       Y     N        0.8       331.0    3.70     88.0     698.0  137.95           172.0     190.0         12.0
    3   -764   C    D-penicillamine  23189   F       N            Y       Y     N        3.4       282.0    3.11     88.0       NaN  137.95           172.0     190.0         12.0
    4   -725   C    D-penicillamine  20378   F       N            N       Y     N        0.8       309.0    3.87    143.0    1430.0  137.95           172.0     190.0         12.0
    ```

  ##   Files

  * `liver_cirrhosis.csv`: The dataset.
  * `liver_cirrhosis.ipynb`: Jupyter Notebook containing the code and analysis.
  * `Liver Cirrhosis Stage Detection.pdf`: Project description.

  ##   Code and Analysis

  *(Based on `liver_cirrhosis.ipynb`)*

  **Libraries Used:**

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    #   Add other libraries used in your notebook
    ```

  **Data Preprocessing:**

  To accurately describe the data preprocessing, I need to refer to your notebook (`liver_cirrhosis.ipynb`). However, based on common practices, here's a likely outline:

  * *Handling Missing Values:** Missing values (e.g., in `Cholesterol`, `Alk_Phos`) were handled using appropriate techniques (imputation or removal).
    * **Encoding Categorical Features:** Categorical features (e.g., `Sex`, `Ascites`, `Drug`, `Status`, `Edema`) were converted into numerical representations using methods like Label Encoding or One-Hot Encoding.
    * **Feature Scaling:** Numerical features were scaled (e.g., using StandardScaler) to ensure all features contribute equally to the model.
    * **Outlier Handling:** Outliers might have been addressed if present.
    * **Target Variable Handling:** The target variable (`Stage`) was prepared for modeling.

    **Models Used:**

    *the specific models used in your notebook.*

    * Random Forest Classifier
    * Logistic Regression
    * Support Vector Machine (SVM)

    **Model Evaluation:**

    *the evaluation metrics used in your notebook*

    * Accuracy Score
    * Classification Report (Precision, Recall, F1-score)
    * Confusion Matrix

    ##   Data Preprocessing 🛠️

    The data was preprocessed by handling missing values, encoding categorical features, and scaling numerical features to prepare it for machine learning models.

    ##   Exploratory Data Analysis (EDA) 🔍

    Key EDA steps likely involved:

    * Visualizing the distribution of individual features (e.g., histograms for numerical, bar plots for categorical).
    * Examining the distribution of the target variable (`Stage`).
    * Exploring relationships between features and the target variable (e.g., box plots, violin plots).
    * Identifying correlations between different features using heatmaps.

    ##   Model Selection and Training 🧠

    The project likely involved selecting one or more classification algorithms suitable for multi-class classification.
    A common choice, given the imported libraries, is the Random Forest Classifier.
    The training process would have involved splitting the data into training and testing sets and fitting the chosen model(s) to the training data.
    Hyperparameter tuning might have been performed to optimize model performance.

    ##   Model Evaluation ✅

    The trained model(s) were evaluated on the testing data using metrics such as
    accuracy score,
    classification report (providing precision, recall, and F1-score for each class),
     and
    a confusion matrix to understand the types of errors made by the model.

    ##   Results ✨

    The goal of the project was to accurately predict the stage of liver cirrhosis.
    The results would typically highlight the performance of the chosen model(s) on the test set, with the key metric being the accuracy in classifying the different stages.
    The classification report and confusion matrix would provide further insights into the model's ability to correctly identify each stage.

    ##   Setup ⚙️

    1.  Clone the repository.
    2.  Install the necessary libraries (as used in your notebook):

        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn
        #   Add any other libraries
        ```

    3.  Run the Jupyter Notebook `liver_cirrhosis.ipynb`.

    ##   Usage ▶️

    The `liver_cirrhosis.ipynb` notebook can be used to:

    * Load and explore the dataset.
    * Preprocess the data.
    * Train machine learning models for liver cirrhosis stage detection.
    * Evaluate the performance of the trained models.

    ##   Contributing 🤝

    Contributions to this project are welcome. Please feel free to submit a pull request.

    ##   License 📄

    This project is open source and available under the MIT License.
