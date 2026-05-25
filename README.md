# 📊 Marketing Intelligence Dashboard

Application de **Machine Learning appliquée au Marketing Digital** développée avec **Streamlit**, permettant :

- la segmentation automatique des clients
- la prédiction de réponse aux campagnes marketing
- l’analyse des comportements d’achat
- la génération d’insights business
- l’export des données enrichies

---

# 🚀 Objectif du projet

Ce projet a pour objectif de montrer comment le **Machine Learning** peut être utilisé dans le domaine du **marketing digital** afin d’aider les entreprises à :

- mieux comprendre leurs clients
- identifier les segments les plus rentables
- prédire les clients susceptibles de répondre à une campagne
- améliorer le ciblage marketing
- optimiser le ROI des campagnes

L’application propose une **vue business interactive** accessible aux profils non techniques.

---

# 🧠 Fonctionnalités principales

## ✅ Segmentation clients

Utilisation de l’algorithme :

- **K-Means Clustering**

pour regrouper automatiquement les clients selon leurs comportements :

- dépenses
- revenus
- achats web
- campagnes acceptées
- habitudes d’achat

---

## ✅ Prédiction de réponse marketing

Utilisation d’un modèle supervisé pour prédire :

- la probabilité qu’un client réponde à une campagne marketing

Le dashboard permet :

- d’identifier les clients à fort potentiel
- de filtrer les clients prioritaires
- de créer des campagnes ciblées

---

## ✅ Dashboard Business Interactif

L’application contient plusieurs vues :

### 📊 Vue d’ensemble
- KPI business
- insights automatiques
- répartition des segments
- potentiel marketing

### 👥 Segments clients
- profils clients
- comportement moyen
- recommandations marketing

### 🎯 Probabilité de réponse
- scoring clients
- clients prioritaires
- recommandations stratégiques

### 📦 Analyse produits
- catégories les plus achetées
- dépenses par segment

### 📣 Campagnes marketing
- taux d’acceptation
- performance des campagnes

### 🏆 Top clients
- clients les plus rentables
- top revenus
- top probabilités de réponse

### 📤 Export
- export CSV
- rapport texte automatique

---

# 🛠️ Technologies utilisées

## Frontend / Dashboard
- Streamlit
- Plotly

## Data Science / ML
- Pandas
- NumPy
- Scikit-Learn
- Joblib

## Machine Learning
- K-Means
- Classification supervisée
- Feature Engineering
- Pipelines Scikit-Learn

---

# 📂 Structure du projet

```bash
CUSTOMER-SEGMENTATION/
│
├── dataset/
│   ├── .complete
│   └── marketing_campaign.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── response_model.pkl
│   └── seg_pipeline.pkl
│
├── notebooks/
│   ├── eda&pretraitement.ipynb
│   ├── prediction_response_comparaison_modeles.ipynb
│   └── segmentation_clients.ipynb
│
├── venv/
│
├── app.py
├── preprocessing.py
├── requirements.txt
├── response_model.py
├── segmentation_model.py
├── utils.py
│
└── README.md
```

---

# 📥 Dataset utilisé

Le projet utilise le dataset :

## Marketing Campaign Dataset

Variables principales :

| Variable | Description |
|---|---|
| Income | Revenu annuel |
| Recency | Jours depuis dernier achat |
| MntWines | Dépenses en vins |
| MntMeatProducts | Dépenses en viande |
| NumWebPurchases | Achats web |
| AcceptedCmp1 | Campagne acceptée |
| Education | Niveau d’étude |
| Marital_Status | Situation familiale |

---

# ⚙️ Installation

## 1️⃣ Cloner le projet

```bash
git clone https://github.com/your-username/customer-segmentation.git

cd customer-segmentation
```

---

## 2️⃣ Créer un environnement virtuel

### Windows

```bash
python -m venv venv
```

Activation :

```bash
venv\Scripts\activate
```

---

## 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# ▶️ Lancer l’application

```bash
streamlit run app.py
```

L’application sera disponible sur :

```bash
http://localhost:8501
```

---

# 📋 Colonnes attendues dans le fichier CSV

Le fichier importé doit contenir les colonnes suivantes :

```text
Year_Birth
Education
Marital_Status
Income
Kidhome
Teenhome
Recency
MntWines
MntFruits
MntMeatProducts
MntFishProducts
MntSweetProducts
MntGoldProds
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
AcceptedCmp1
AcceptedCmp2
AcceptedCmp3
AcceptedCmp4
AcceptedCmp5
```

---

# 📊 Pipeline Machine Learning

## 🔹 Feature Engineering

Création automatique de nouvelles variables :

- TotalSpend
- TotalPurchases
- Age
- FamilySize
- TotalCampaignsAccepted

---

## 🔹 Segmentation

Pipeline :

```text
Preprocessing
→ Scaling
→ KMeans Clustering
```

---

## 🔹 Prediction

Pipeline :

```text
Feature Engineering
→ Preprocessing
→ Classification Model
```

---

# 💡 Exemples d’insights générés

- Clients VIP à forte valeur
- Clients à risque de départ
- Produits les plus performants
- Segments à fort potentiel
- Campagnes les plus efficaces

---

# 📤 Exports disponibles

L’application permet d’exporter :

✅ base clients enrichie  
✅ profils segments  
✅ clients prioritaires  
✅ rapport marketing texte

---

# 🎯 Cas d’utilisation Business

Cette application peut être utilisée pour :

- CRM intelligent
- ciblage marketing
- fidélisation clients
- recommandations campagnes
- analyse comportementale
- aide à la décision

---

# 📈 Perspectives d’amélioration

Améliorations possibles :

- déploiement cloud
- recommandations produits
- analyse temps réel
- intégration Power BI
- deep learning
- NLP marketing
- emailing automatique

---

# 👩‍💻 Auteur

Projet réalisé dans le cadre d’un projet de :

## Machine Learning appliqué au Marketing Digital

École Mohammadia d’Ingénieurs (EMI)

---

# 📜 Licence

Projet académique — usage éducatif uniquement.
