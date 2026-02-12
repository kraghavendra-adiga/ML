import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



print(f"Lab 2")

df = pd.read_csv("Lab 2\housing.csv")


#1 print(df.describe())
# print(df.info())

#2 df.hist(bins=50, figsize=(20,15))
# plt.show()


#3 train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# print(f"Train set: {train_set}")
# print(f"Test Set is: {test_set}")

#4 plt.figure(figsize=(10,7))
# plt.scatter(df["longitude"], df["latitude"],
#             c=df["median_house_value"], cmap="jet", alpha=0.4)
# plt.colorbar(label="House Price")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.show()

#5
# corr_matrix = df.corr(numeric_only=True)
# print(corr_matrix)
# corr_matrix["median_house_value"].sort_values(ascending=False)
# plt.scatter(df["median_income"], df["median_house_value"], alpha=0.2)
# plt.xlabel("Median Income")
# plt.ylabel("Median House Value")
# plt.show()

#6
# df["rooms_per_household"] = df["total_rooms"] / df["households"]
# df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
# df["population_per_household"] = df["population"] / df["households"]

# print(df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False))

 #7
# print(df.isnull().sum())

#8
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(sparse_output=False)
# ocean_encoded = encoder.fit_transform(df[["ocean_proximity"]])

# encoded_df = pd.DataFrame(ocean_encoded,
#                           columns=encoder.get_feature_names_out())
# print(encoded_df.head())

#10
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

print("Lab 2 - Q10")

# Load dataset (FIXED PATH)
df = pd.read_csv(r"Lab 2\housing.csv")

# Separate features
housing = df.drop("median_house_value", axis=1)

# -------------------------------
# CUSTOM TRANSFORMER
# -------------------------------
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        bedrooms_per_room = X[:, 4] / X[:, 3]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

# Separate numerical & categorical attributes
num_attribs = housing.select_dtypes(include=["int64", "float64"]).columns
cat_attribs = ["ocean_proximity"]

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("scaler", StandardScaler())
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# Apply pipeline
housing_prepared = full_pipeline.fit_transform(housing)

print("Final shape of processed data:")
print(housing_prepared.shape)
