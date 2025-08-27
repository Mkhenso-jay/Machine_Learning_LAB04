# Lab 4 – Data Preprocessing

This lab explores common data preprocessing techniques: handling missing values, encoding categorical variables, splitting datasets, scaling features, and feature selection. Each section has code cells (run in VS Code with `# %%`) and saves figures into the `outputs/` folder.

---

## 1. Handling Missing Data

We created a small dataset with missing values:

```text
      A     B     C     D
0   1.0   2.0   NaN   4.0
1   5.0   6.0   NaN   8.0
2  10.0  11.0  12.0   NaN
```

The missing value counts per column were:

```
A    0
B    0
C    1
D    1
```

### Methods tried:
- **Drop rows with missing values**  
- **Fill with column mean**  
- **Fill with column median**  
- **Fill with column mode**  

📌 **Observation:**  
Dropping rows reduced the dataset size but lost information. Mean and median imputation preserved all rows and worked well for numeric data. Mode imputation was less useful here since the data is numeric.

---

## 2. Categorical Variables

We tested two methods:

- **Ordinal Encoding** (assign numbers to ordered categories, e.g. `cold=0, warm=1, hot=2`).  
- **One-Hot Encoding** (create binary columns for each category).  

Figures:  
- `outputs/ordinal_mapping.png`  
- `outputs/one_hot_encoding.png`  

📌 **Observation:**  
Ordinal encoding is useful when categories have a natural order. One-hot encoding avoids introducing a false order but increases the number of columns.

---

## 3. Train/Test Split

We used the Wine dataset and split into training (70%) and testing (30%).  

Figure:  
- `outputs/class_distribution.png`  

📌 **Observation:**  
The train/test split preserved class balance fairly well, which is important for classification tasks.

---

## 4. Feature Scaling

We compared raw features vs MinMax scaling vs Standardization.  

Figure:  
- `outputs/scaling_histogram.png`  

📌 **Observation:**  
MinMax scaling compressed all values between 0 and 1. Standardization centered features around 0 with unit variance. Scaling ensures fair comparison of features during model training.

---

## 5. Feature Selection

We applied three techniques:

- **L1-Regularized Logistic Regression**  
- **Sequential Backward Selection (SBS) with kNN**  
- **Random Forest Feature Importances**  

Figures:  
- `outputs/l1_logreg_coefficients.png`  
- `outputs/sbs_plot.png`  
- `outputs/rf_feature_importances.png`  

📌 **Observation:**  
Logistic regression highlighted only a few strong features. SBS showed how model performance changes as features are removed. Random Forest provided a clear ranking of feature importance.

---

# How to Run

1. Clone or copy files into a folder.  
2. Create a virtual environment and install requirements:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the script in VS Code using `# %%` cell execution.  
4. Check results in the `outputs/` folder.
