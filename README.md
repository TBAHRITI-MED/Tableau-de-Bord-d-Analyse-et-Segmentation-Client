# 📊 PersonaInsight™: Tableau de Bord d'Analyse et Segmentation Client

![Customer Segmentation Dashboard](https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.gif?raw=true)

## 📝 Description

PersonaInsight™ est un tableau de bord interactif conçu pour l'analyse de données clients et la segmentation marketing. Cette application Streamlit permet de visualiser, d'analyser et de segmenter les données clients à travers plusieurs dimensions, offrant des insights précieux pour les stratégies marketing et la prise de décision.

## ✨ Fonctionnalités

### 📋 Navigation par Modules
- **Accueil**: Vue d'ensemble des données et statistiques principales
- **Données Démographiques**: Analyse des caractéristiques démographiques des clients
- **Comportement d'Achat**: Exploration des habitudes et préférences d'achat
- **Analyse des Campagnes**: Évaluation de l'efficacité des campagnes marketing
- **Segmentation des Clients**: Exploration des différents segments de clientèle
- **Analyse des Clusters**: Visualisation et interprétation des clusters de clients
- **Visualisation 3D**: Représentation tridimensionnelle des clusters
- **Analyses Avancées**: Visualisations supplémentaires et analyses RFM

### 📈 Visualisations Interactives
- Graphiques à secteurs pour la répartition démographique
- Graphiques à barres pour les comportements d'achat
- Heatmaps pour les corrélations et patterns
- Graphiques 3D pour l'exploration multidimensionnelle
- Graphiques radar pour les profils de segments
- Boxplots pour l'analyse de distribution

### 🧠 Algorithmes d'Analyse Avancée
- Clustering K-Means pour la segmentation automatique
- Analyse en Composantes Principales (PCA)
- Segmentation RFM (Récence, Fréquence, Montant)
- Analyse de corrélation multi-variables

## 🚀 Installation

1. Clonez ce dépôt:
```bash
git clone https://github.com/votre-utilisateur/personainsight.git
cd personainsight
```

2. Installez les dépendances:
```bash
pip install -r requirements.txt
```

3. Lancez l'application:
```bash
streamlit run app.py
```

## 📦 Dépendances

```
streamlit==1.32.0
pandas==2.1.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.18.0
scikit-learn==1.3.0
pillow==10.0.0
datetime
requests==2.31.0
```

## 📊 Utilisation

1. Téléchargez vos propres données clients au format CSV ou Excel via l'interface
2. Explorez les différentes sections du tableau de bord via la barre latérale
3. Interagissez avec les visualisations pour découvrir des insights
4. Utilisez les options de personnalisation pour adapter les visualisations à vos besoins

## 📁 Format de Données

L'application accepte les fichiers CSV et Excel avec les colonnes suivantes (recommandées):
- Données démographiques: Age, Education, Marital_Status, Income, Kidhome, Teenhome
- Comportement d'achat: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
- Comportement des canaux: NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth
- Campagnes: AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Response

## 🖼️ Captures d'écran

![Dashboard Overview](https://via.placeholder.com/800x450)
![Segmentation View](https://via.placeholder.com/800x450)
![3D Visualization](https://via.placeholder.com/800x450)

## 🔍 Cas d'Utilisation

- **Marketing Personnalisé**: Identifiez les segments de clients pour des campagnes ciblées
- **Analyse de Fidélité**: Comprenez quels clients sont les plus fidèles et pourquoi
- **Optimisation Produit**: Découvrez quelles catégories de produits sont populaires auprès de quels segments
- **Prévention d'Attrition**: Identifiez les clients à risque pour mettre en place des stratégies de rétention
- **Allocation Budgétaire**: Optimisez les dépenses marketing en fonction des segments les plus rentables

## 👥 Public Cible

- Analystes Marketing
- Responsables CRM
- Chefs de Produit
- Directeurs Commerciaux
- Data Scientists

## 🤝 Contribution

Les contributions à ce projet sont les bienvenues! N'hésitez pas à:
1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/amazing-feature`)
3. Commit vos changements (`git commit -m 'Add some amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 📧 Contact

Pour toute question ou support: [tbahritimohamed8@gmail.com](mailto:tbahritimohamed8@gmail.com)

---

Développé avec ❤️ pour l'Analyse de Personnalité Client | 2024-2025