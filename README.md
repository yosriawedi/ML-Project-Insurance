🏥 ML-Project-Insurance

Projet de Machine Learning avancé appliqué à l’assurance santé.
Nous utilisons des méthodes supervisées (régression, classification) et non supervisées (clustering, détection d’anomalies) afin de :

prédire les charges médicales,

détecter les assurés atypiques,

segmenter la clientèle pour adapter les services,

identifier les profils à risque pour la prévention.

Le tout est développé en suivant la méthodologie CRISP-DM.

🎯 Objectifs Business (BO) et Data Science (DSO)
BO1 : Prédire les coûts médicaux pour anticiper les dépenses

Prédire le coût médical des assurés afin d’optimiser la tarification, réduire les risques et automatiser le processus.

DSO1 : Construire un modèle de régression prédictive avec une MAE < 3 000 $ pour estimer les charges médicales à partir des données clients.

Méthodes envisagées : Régression linéaire, Random Forest, XGBoost, Réseaux de neurones.

BO2 : Identifier les assurés « atypiques »

Détecter les profils dont les caractéristiques diffèrent fortement de la majorité.

DSO2 : Détection d’anomalies globales (variables numériques + catégorielles).

Méthodes envisagées :

Isolation Forest (robuste et scalable)

One-Class SVM (frontière inliers/outliers)

BO3 : Évaluer et répartir les charges de soins

Analyser la répartition des charges de soins pour optimiser les ressources et améliorer la qualité de prise en charge.

DSO3 : Analyse statistique et exploration visuelle des charges (distribution, outliers, corrélations).

BO4 : Segmenter et catégoriser la clientèle

Identifier et classer les clients selon leurs besoins, habitudes et profils pour personnaliser l’accompagnement et améliorer l’expérience assurée.

DSO4 : Mettre en place un clustering (K-Means, DBSCAN, GMM) pour créer des segments homogènes de clients.

BO5 : Détecter les jeunes assurés à risque élevé

Identifier les jeunes assurés présentant un profil de risque afin de les intégrer dans des programmes de prévention ciblés.

DSO5 : Construire un modèle de classification supervisée multi-classes pour détecter ces profils.

⚙️ Méthodologie CRISP-DM

Business Understanding → Définition des objectifs business (BO).

Data Understanding → Exploration et nettoyage des données (statistiques, visualisations).

Data Preparation → Feature engineering, encodage des variables, normalisation, gestion des valeurs manquantes.

Modeling → Mise en place de modèles supervisés (régression, classification) et non supervisés (clustering, anomalies).

Evaluation → Validation des modèles (MAE, RMSE, Accuracy, F1-score, Silhouette score, etc.).

Deployment → Industrialisation et intégration des modèles (API, dashboard, notebooks documentés).

📊 Résultats attendus

Régression : MAE < 3 000 $ sur la prédiction des charges médicales.

Anomalies : Identification automatique des assurés atypiques.

Clustering : Segmentation claire de la clientèle en groupes homogènes.

Classification : Détection des jeunes assurés à risque avec une précision > 80 %.
