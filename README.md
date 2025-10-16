# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"E:\income(1) (1).csv")
print(data.head())
print(data.info())

```

<img width="802" height="712" alt="image" src="https://github.com/user-attachments/assets/00cf822c-9d82-46fd-8f63-6a69eecce4e1" />

```
print(data.isnull().sum())

```

<img width="390" height="367" alt="image" src="https://github.com/user-attachments/assets/bc9e5407-79fb-431f-b59f-6edcb86b189e" />

```
data = data.dropna()
categorical_cols = data.select_dtypes(include=['object']).columns
print( list(categorical_cols))

```

<img width="1312" height="72" alt="image" src="https://github.com/user-attachments/assets/79fff651-62c8-4abe-b74b-1ecb381bbb4e" />

```
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
print(data.head())
```

<img width="777" height="342" alt="image" src="https://github.com/user-attachments/assets/0a6caac0-c33e-4f42-9232-547858c939ba" />

```
target_col = data.columns[-1]
X = data.drop(columns=[target_col])
y = data[target_col]
print(X.head())
print(y.head())
```

<img width="757" height="472" alt="image" src="https://github.com/user-attachments/assets/eb6631d5-f295-4f71-a6e1-441ce34f1609" />

```
scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
print(X_standardized.head())
```

<img width="845" height="367" alt="image" src="https://github.com/user-attachments/assets/c7c9908e-8a6d-4df8-b4cb-e1a78b61a4f7" />

```
scaler_mm = MinMaxScaler()
X_minmax = scaler_mm.fit_transform(X)
X_minmax = pd.DataFrame(X_minmax, columns=X.columns)
print(X_minmax.head())

```

<img width="931" height="365" alt="image" src="https://github.com/user-attachments/assets/6c5b6574-c796-42a6-84ef-5f1bb89f25e5" />

```
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
X_robust = pd.DataFrame(X_robust, columns=Xanova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X_train, y_train)
anova_scores = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': anova_selector.scores_
}).sort_values(by='F-Score', ascending=False)

print(anova_scores).columns)
print(X_robust.head())
```

<img width="842" height="352" alt="image" src="https://github.com/user-attachments/assets/648f5b0e-5514-40f0-9704-f0ce9e39a8d9" />

```
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)
X_chi = X_minmax.copy()  # Chi2 requires non-negative values
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X_chi, y)
chi2_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_selector.scores_
}).sort_values(by='Chi2 Score', ascending=False)

print(chi2_scores)
```

<img width="580" height="330" alt="image" src="https://github.com/user-attachments/assets/66fb014c-3269-44c9-863d-47ad477dd824" />

```
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X_train, y_train)
mi_scores = pd.DataFrame({
    'Feature': X.columns,
    'Mutual Info Score': mi_selector.scores_
}).sort_values(by='Mutual Info Score', ascending=False)
print(mi_scores)

```

<img width="407" height="326" alt="image" src="https://github.com/user-attachments/assets/2b4e29c7-7ad7-4473-a05f-6bf75947da17" />

```
X_standardized.to_csv("Income_Standardized.csv", index=False)
X_minmax.to_csv("Income_MinMax_Normalized.csv", index=False)
X_robust.to_csv("Income_Robust_Scaled.csv", index=False)
chi2_scores.to_csv("Income_Chi2_FeatureScores.csv", index=False)
anova_scores.to_csv("Income_ANOVA_FeatureScores.csv", index=False)
mi_scores.to_csv("Income_MutualInfo_FeatureScores.csv", index=False)

```
```
```

# RESULT:
The above code  and output  for the Feature Scaling and Selection
       
