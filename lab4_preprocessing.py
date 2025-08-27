 
# %% [markdown]
# ## Section 0 — Setup

# %%
# Imports
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Make printing a little wider
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 100)

# Where to save figures
import os
os.makedirs("outputs", exist_ok=True)

# %% [markdown]
# ## Section 1 — Handling Missing Data
# ### Exercise 1.1: Identifying Missing Values

# %%
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print("Original DataFrame:")
print(df)
print("\nMissing values per column:")
print(df.isnull().sum())

 

# %% [markdown]
# ### Exercise 1.2: Eliminating Missing Values

# %%
print("Drop rows with any missing values:")
print(df.dropna(axis=0))

print("\nDrop columns with any missing values:")
print(df.dropna(axis=1))

print("\nDrop rows where all values are missing (none here):")
print(df.dropna(how='all'))

print("\nDrop rows with fewer than 4 non-missing values:")
print(df.dropna(thresh=4))

print("\nDrop rows where 'C' is missing:")
print(df.dropna(subset=['C']))

 

# %% [markdown]
# ### Exercise 1.3: Imputing Missing Values
# %%
print("Original DataFrame:\n", df)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
print("Imputed data with 'mean' strategy:")
print(df_imputed)


 
# %% [markdown]
# ## Section 2 — Handling Categorical Data
# ### Exercise 2.1: Mapping Ordinal Features

# %%
df_cat = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red',   'L', 13.5, 'class1'],
    ['blue',  'XL', 15.3, 'class2']
], columns=['color', 'size', 'price', 'classlabel'])

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df_cat['size'] = df_cat['size'].map(size_mapping)
print(df_cat)

# Inverse mapping
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print("Inverse-mapped sizes:")
print(df_cat['size'].map(inv_size_mapping))

 

# %% [markdown]
# ### Exercise 2.2: Encoding Class Labels

# %%
class_le = LabelEncoder()
y = class_le.fit_transform(df_cat['classlabel'].values)
print("Encoded class labels:", y)
print("Inverse transform:", class_le.inverse_transform(y))

 
# %% [markdown]
# ### Exercise 2.3: One-Hot Encoding for Nominal Features

# %%
X = df_cat[['color', 'size', 'price']].values.copy()

# Label-encode color first (to transform into integers)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("After LabelEncoding 'color':\n", X)

# One-hot encode only the 'color' column
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # sklearn >= 1.2
except TypeError:
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')         # older sklearn

color_ohe = ohe.fit_transform(X[:, [0]])
print("One-hot for 'color':\n", color_ohe)

# Using pandas get_dummies on multiple columns at once
print("pd.get_dummies on ['price', 'color', 'size']:")
print(pd.get_dummies(df_cat[['price', 'color', 'size']]))

print("pd.get_dummies with drop_first=True (avoid multicollinearity):")
print(pd.get_dummies(df_cat[['price', 'color', 'size']], drop_first=True))

 

# %% [markdown]
# ## Section 3 — Partitioning the Wine Dataset
# We attempt to load from the UCI URL. If that fails (e.g., offline),
# we fall back to sklearn's built-in wine dataset for convenience.

# %%
uci_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
cols = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

def load_wine_dataframe():
    try:
        df_wine = pd.read_csv(uci_url, header=None, names=cols)
        source = "UCI"
    except Exception as e:
        from sklearn.datasets import load_wine
        data = load_wine()
        df_wine = pd.DataFrame(data=np.c_[data['target'] + 1, data['data']],  # match class labels 1..3
                               columns=['Class label'] + data['feature_names'])
        # Rename sklearn feature names to match the book's columns where possible
        rename_map = {
            'alcohol': 'Alcohol',
            'malic_acid': 'Malic acid',
            'ash': 'Ash',
            'alcalinity_of_ash': 'Alcalinity of ash',
            'magnesium': 'Magnesium',
            'total_phenols': 'Total phenols',
            'flavanoids': 'Flavanoids',
            'nonflavanoid_phenols': 'Nonflavanoid phenols',
            'proanthocyanins': 'Proanthocyanins',
            'color_intensity': 'Color intensity',
            'hue': 'Hue',
            'od280/od315_of_diluted_wines': 'OD280/OD315 of diluted wines',
            'proline': 'Proline'
        }
        df_wine = df_wine.rename(columns=rename_map)
        source = "sklearn"
    return df_wine, source

df_wine, source = load_wine_dataframe()
print(f"Wine dataset loaded from: {source}")
print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)
print('Train shape:', X_train.shape, 'Test shape:', X_test.shape)

# Check class proportions
def proportions(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return dict(zip(vals, (counts / counts.sum()).round(3)))

print("Class proportions (y_train):", proportions(y_train))
print("Class proportions (y_test):", proportions(y_test))

 

# %% [markdown]
# ## Section 4 — Feature Scaling
# ### Exercise 4.1: Normalization (Min-Max Scaling)

# %%
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print("First two rows of normalized X_train:\n", X_train_norm[:2])

 
plt.figure()
plt.hist(X_train[:, 0], bins=20)
plt.title("Alcohol — Before MinMax Scaling")
plt.xlabel("Value"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/alcohol_before_minmax.png")

plt.figure()
plt.hist(X_train_norm[:, 0], bins=20)
plt.title("Alcohol — After MinMax Scaling")
plt.xlabel("Value"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/alcohol_after_minmax.png")

# %% [markdown]
# ### Exercise 4.2: Standardization

# %%
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print("First two rows of standardized X_train:\n", X_train_std[:2])

# Compare mean and std (should be ~0 and ~1)
print("Means (train, rounded):", np.mean(X_train_std, axis=0).round(4))
print("Stds  (train, rounded):", np.std(X_train_std, axis=0).round(4))

  
# %% [markdown]
# ## Section 5 — Selecting Meaningful Features
# ### Exercise 5.1: L1 Regularization for Sparsity (Logistic Regression, OvR)

# %%
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr', max_iter=1000)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print('Coefficients (per class):')
print(pd.DataFrame(lr.coef_, columns=df_wine.columns[1:]))

 

# Example: coefficient paths
Cs = np.logspace(-2, 2, 10)
coefs = []
for C in Cs:
    lrC = LogisticRegression(penalty='l1', C=C, solver='liblinear', multi_class='ovr', max_iter=1000)
    lrC.fit(X_train_std, y_train)
    coefs.append(np.mean(np.abs(lrC.coef_), axis=0))  # mean absolute coef across classes

coefs = np.array(coefs)

plt.figure()
for j in range(coefs.shape[1]):
    plt.plot(Cs, coefs[:, j], marker='o')
plt.xscale('log')
plt.xlabel('C (log scale)'); plt.ylabel('Mean |coef| across classes')
plt.title('L1 coefficient magnitude vs C')
plt.tight_layout()
plt.savefig("outputs/l1_coeffs_vs_C.png")

# %% [markdown]
# ### Exercise 5.2: Sequential Backward Selection (SBS)
# Implementation adapted from the book.

# %%
from itertools import combinations
from copy import deepcopy

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.estimator = deepcopy(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        n_features = X_train.shape[1]
        self.indices_ = tuple(range(n_features))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while n_features > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=n_features - 1):
                s = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(s)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            n_features -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        return self.scoring(y_test, y_pred)

# Run SBS with k-NN
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.figure()
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.savefig("outputs/sbs_accuracy_vs_k.png")

# Example: pick 3-feature subset (index depends on path length)
k3_index = min(10, len(sbs.subsets_) - 1)
k3 = list(sbs.subsets_[k3_index])
print("Selected 3-feature indices (example):", k3)
print("Corresponding feature names:", df_wine.columns[1:][k3].tolist())

# Evaluate k-NN with selected features
knn.fit(X_train_std[:, k3], y_train)
print('k-NN Test accuracy with selected features:', knn.score(X_test_std[:, k3], y_test))

 
# %% [markdown]
# ### Exercise 5.3: Feature Importance with Random Forests

# %%
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("Random Forest feature importances (descending):")
for rank, idx in enumerate(indices, start=1):
    print(f"{rank:2d}) {feat_labels[idx]:30s} {importances[idx]:.6f}")

plt.figure()
plt.title('Feature Importance (Random Forest)')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.savefig("outputs/rf_feature_importance.png")

# SelectFromModel with threshold=0.1
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_train_sfm = sfm.transform(X_train)
X_test_sfm = sfm.transform(X_test)
print("Selected features by SelectFromModel (threshold=0.1):",
      feat_labels[sfm.get_support()].tolist())

# Train a simple classifier on selected features
clf_sfm = LogisticRegression(max_iter=1000)
clf_sfm.fit(X_train_sfm, y_train)
print("LogReg (SFM-selected) Train acc:", clf_sfm.score(X_train_sfm, y_train))
print("LogReg (SFM-selected) Test  acc:", clf_sfm.score(X_test_sfm, y_test))

 

# %%
