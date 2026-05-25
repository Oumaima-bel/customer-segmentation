import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import FeatureEngineer
from utils import load_df




df = load_df()


kmeans_final = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
    max_iter=1000
)




seg_pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("scaler", StandardScaler()),
])


seg_pipeline.fit(df)
X = seg_pipeline.transform(df)


kmeans_final.fit_predict(X)


joblib.dump(seg_pipeline,"models/seg_pipeline.pkl")
joblib.dump(kmeans_final,"models/kmeans_model.pkl")