import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


from preprocessing import FeatureEngineer
from utils import load_df

RANDOM_STATE = 42




df = load_df()

pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
])

df_processed = pipeline.fit_transform(df)



X = df_processed.drop("Response",axis=1)
y = df_processed['Response']


scale_pos_weight = (y == 0).sum() / (y == 1).sum()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --- Logistic Regression ---

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    ))
])

lr_param_grid = {
    'clf__C'       : [0.01, 0.1, 1, 10],
    'clf__penalty' : ['l1', 'l2'],
    'clf__solver'  : ['liblinear']
}

gs_lr = GridSearchCV(
    lr_pipeline, lr_param_grid,
    scoring='f1', cv=cv,
    n_jobs=-1, verbose=0
)

gs_lr.fit(X,y)


# --- Random Forest ---
rf_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf' : [1, 2]
}

gs_rf = GridSearchCV(
    rf_model, rf_param_grid,
    scoring='f1', cv=cv,
    n_jobs=-1, verbose=0
)

gs_rf.fit(X,y)


# --- XGBoost ---
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_param_grid = {
    'n_estimators'  : [100, 200, 300],
    'max_depth'     : [3, 5, 7],
    'learning_rate' : [0.01, 0.05, 0.1],
    'subsample'     : [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

gs_xgb = GridSearchCV(
    xgb_model, xgb_param_grid,
    scoring='f1', cv=cv,
    n_jobs=-1, verbose=0
)

gs_xgb.fit(X,y)


# Logistic Regression — dans un pipeline avec StandardScaler
best_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', gs_lr.best_estimator_.named_steps['clf'])
])

# Random Forest
best_rf = gs_rf.best_estimator_

# XGBoost
best_xgb = gs_xgb.best_estimator_



soft_voting = VotingClassifier(
    estimators=[
        ('lr',  best_lr),
        ('rf',  best_rf),
        ('xgb', best_xgb)
    ],
    voting='soft',
    n_jobs=-1
)


soft_voting.fit(X,y)


joblib.dump(soft_voting,"response_model.pkl")