import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin




class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X,y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Suppression des valeurs manquantes résiduelles
        X.dropna(inplace=True)

        # Suppression des colonnes inutiles pour la segmentation
        cols_to_drop = ['ID', 'Dt_Customer']
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

        try:
            X["Marital_Status"] = X["Marital_Status"].replace({
                "Alone": "Single"
            }) # remplacer alone avec single

            X = X[~X["Marital_Status"].isin(["YOLO", "Absurd"])] # supprimer yolo et absurd
        except:
            print("")


        spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

        cmp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                    'AcceptedCmp4', 'AcceptedCmp5']

        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                    'NumStorePurchases', 'NumDealsPurchases']

        # Feature engineering
        spend_cols = [c for c in spend_cols if c in X.columns]
        if spend_cols:
            X['TotalSpend'] = X[spend_cols].sum(axis=1)

        
        cmp_cols = [c for c in cmp_cols if c in X.columns]
        if cmp_cols:
            X['TotalCampaignsAccepted'] = X[cmp_cols].sum(axis=1)

        
        purchase_cols = [c for c in purchase_cols if c in X.columns]
        if purchase_cols:
            X['TotalPurchases'] = X[purchase_cols].sum(axis=1)



        cat_cols = X.select_dtypes(include='object').columns.tolist()


        # Encodage One-Hot avec drop_first pour éviter la multicolinéarité
        X = pd.get_dummies(
            X,
            columns=cat_cols,
            drop_first=True
        )

        # Conversion des booléens en int 
        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)


        return X