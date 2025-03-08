import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.figure_factory as ff
from PIL import Image
import io
import base64
import requests
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Définir une palette de couleurs personnalisée
palette = ["#682F2F", "#B6666F", "#9C9990", "#FFF9ED", "#D9BFB0"]

# Configuration de la page
st.set_page_config(
    page_title="PersonaInsight™",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application avec HTML personnalisé
st.markdown("""
# <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">Customer Segmentation</p>
""", unsafe_allow_html=True)

# Ajouter une image d'en-tête
st.markdown("""
<img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.gif?raw=true" width="100%">
""", unsafe_allow_html=True)

# Introduction du projet
st.markdown("""
In this project, I will be performing an unsupervised clustering of data on the customer's records from a groceries firm's database. Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in each cluster. I will divide customers into segments to optimize the significance of each customer to the business. To modify products according to distinct needs and behaviours of the customers. It also helps the business to cater to the concerns of different types of customers.
""")

# Fonction pour définir les labels des graphiques matplotlib
def set_labels(x=None, y=None, title=None):
    if x:
        plt.xlabel(x, fontsize=14)
    if y:
        plt.ylabel(y, fontsize=14)
    if title:
        plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

# Fonction pour créer une figure matplotlib et la convertir en image pour Streamlit
def plt_to_streamlit(fig, clear_figure=True):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    if clear_figure:
        plt.clf()
    return buf

# Fonction pour charger les données
@st.cache_data
def charger_donnees(fichier_telecharge=None):
    if fichier_telecharge is not None:
        try:
            # Vérifier le type de fichier
            extension = fichier_telecharge.name.split('.')[-1].lower()
            
            if extension == 'csv':
                df = pd.read_csv(fichier_telecharge)
            elif extension in ['xls', 'xlsx']:
                df = pd.read_excel(fichier_telecharge)
            else:
                df = pd.read_csv(fichier_telecharge, sep='\t')
            
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            return None
    
    # Si pas de fichier téléchargé, créer des données synthétiques
    try:
        df = pd.read_csv("marketing_campaign.csv", sep='\t')
        return df
    except FileNotFoundError:
        # Créer des données synthétiques
        np.random.seed(42)
        n_samples = 200
        
        data = []
        for i in range(n_samples):
            data.append({
                'ID': 1000 + i,
                'Year_Birth': np.random.randint(1940, 2000),
                'Education': np.random.choice(['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']),
                'Marital_Status': np.random.choice(['Single', 'Married', 'Together', 'Divorced', 'Widow']),
                'Income': np.random.randint(20000, 150000),
                'Kidhome': np.random.randint(0, 3),
                'Teenhome': np.random.randint(0, 3),
                'Dt_Customer': f"{np.random.randint(1, 29)}-{np.random.randint(1, 13)}-{np.random.randint(2012, 2015)}",
                'Recency': np.random.randint(0, 100),
                'MntWines': np.random.randint(0, 1000),
                'MntFruits': np.random.randint(0, 200),
                'MntMeatProducts': np.random.randint(0, 500),
                'MntFishProducts': np.random.randint(0, 300),
                'MntSweetProducts': np.random.randint(0, 200),
                'MntGoldProds': np.random.randint(0, 300),
                'NumDealsPurchases': np.random.randint(0, 15),
                'NumWebPurchases': np.random.randint(0, 15),
                'NumCatalogPurchases': np.random.randint(0, 15),
                'NumStorePurchases': np.random.randint(0, 15),
                'NumWebVisitsMonth': np.random.randint(0, 10),
                'AcceptedCmp1': np.random.randint(0, 2),
                'AcceptedCmp2': np.random.randint(0, 2),
                'AcceptedCmp3': np.random.randint(0, 2),
                'AcceptedCmp4': np.random.randint(0, 2),
                'AcceptedCmp5': np.random.randint(0, 2),
                'Response': np.random.randint(0, 2),
                'Complain': np.random.randint(0, 2),
                'Z_CostContact': 3,
                'Z_Revenue': 11
            })
        
        return pd.DataFrame(data)

# Fonction pour ajouter des colonnes dérivées
def ajouter_colonnes_derivees(df):
    df_processed = df.copy()
    
    # Ajouter l'âge basé sur l'année de naissance
    if 'Year_Birth' in df.columns:
        annee_courante = datetime.now().year
        df_processed['Age'] = annee_courante - df['Year_Birth']
    
    # Nombre total d'enfants (enfants + adolescents)
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
        df_processed['Children'] = df['Kidhome'] + df['Teenhome']
    
    # Dépenses totales (somme de toutes les colonnes Mnt*)
    mnt_columns = [col for col in df.columns if col.startswith('Mnt')]
    if mnt_columns:
        df_processed['TotalDepenses'] = df[mnt_columns].sum(axis=1)
        # Renommer pour compatibilité avec certains graphiques
        df_processed['Spent'] = df_processed['TotalDepenses']
    
    # Total des promotions acceptées
    cmp_columns = [col for col in df.columns if col.startswith('AcceptedCmp')]
    if cmp_columns:
        df_processed['TotalPromotions'] = df[cmp_columns].sum(axis=1)
    
    # Total des achats tous canaux confondus
    purchase_columns = [col for col in df.columns if 'Purchases' in col]
    if purchase_columns:
        df_processed['TotalAchats'] = df[purchase_columns].sum(axis=1)
    
    return df_processed

# Fonction pour créer un histogramme interactif avec Plotly
def plot_histogram(df, column, title, color="#682F2F", bins=30):
    fig = px.histogram(
        df, x=column, 
        title=title,
        nbins=bins,
        color_discrete_sequence=[color],
        opacity=0.8,
        marginal="box"
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        bargap=0.05
    )
    return fig

# Fonction pour créer un graphique à secteurs interactif avec Plotly
def plot_pie(df, column, title):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    
    fig = px.pie(
        counts, 
        values='Count', 
        names=column,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.3
    )
    fig.update_layout(
        title_font_size=20,
        legend_title_font_size=14
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# Fonction pour créer un graphique à barres interactif avec Plotly
def plot_bar(df, x, y, title, color=None, color_discrete_sequence=None):
    fig = px.bar(
        df, x=x, y=y,
        title=title,
        color=color,
        color_discrete_sequence=color_discrete_sequence if color_discrete_sequence else px.colors.qualitative.Pastel
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        bargap=0.2
    )
    return fig
# Sidebar pour téléchargement de fichier
st.sidebar.title("Paramètres")
fichier_telecharge = st.sidebar.file_uploader("Télécharger vos données (CSV ou Excel)", type=["csv", "xlsx", "xls"])

# Charger les données
df = charger_donnees(fichier_telecharge)

# Ajouter des colonnes dérivées
if df is not None:
    df = ajouter_colonnes_derivees(df)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sélectionner une page", 
    ["Accueil", "Données Démographiques", "Comportement d'Achat", "Analyse des Campagnes", 
     "Segmentation des Clients", "Analyse des Clusters", "Visualisation 3D", "Analyses Avancées"]
)

# Afficher les informations de base
if df is not None:
    st.sidebar.write(f"Données: {df.shape[0]} clients, {df.shape[1]} caractéristiques")

# Page d'accueil
if page == "Accueil":
    st.header("Bienvenue sur PersonaInsight™")
    st.write("""
    Ce tableau de bord interactif vous permet d'explorer les données clients et d'analyser leurs comportements d'achat.
    
    ### Présentation du Projet
    L'objectif de cette étude est de mieux comprendre les tendances d'achat des clients et d'identifier différents modèles 
    de comportement parmi les groupes de clients. En analysant la fidélité et le comportement des clients, les entreprises 
    peuvent développer des stratégies marketing ciblées.
    
    ### Fonctionnalités Principales
    - **Données Démographiques**: Visualiser les caractéristiques démographiques des clients
    - **Comportement d'Achat**: Analyser les habitudes d'achat et les préférences
    - **Analyse des Campagnes**: Évaluer l'efficacité des campagnes marketing
    - **Segmentation des Clients**: Explorer différents segments de clientèle
    - **Visualisation 3D**: Explorer les clusters en trois dimensions
    - **Analyses Avancées**: Visualisations additionnelles et analyses prédictives
    
    ### Utilisez la barre latérale pour naviguer entre les différentes sections!
    """)
    
    # Afficher un aperçu des données
    st.subheader("Aperçu des Données")
    st.write(df.head())
    
    # Afficher des statistiques de base
    st.subheader("Statistiques Descriptives")
    col1, col2, col3 = st.columns(3)
    
    if 'Income' in df.columns:
        col1.metric("Nombre Total de Clients", f"{len(df)}")
        col2.metric("Revenu Moyen", f"${df['Income'].mean():.2f}")
        
        if 'TotalDepenses' in df.columns:
            col3.metric("Dépenses Moyennes", f"${df['TotalDepenses'].mean():.2f}")
    
    # Résumé des données numériques
    st.subheader("Résumé des Données Numériques")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Tableau des statistiques descriptives avec formatage amélioré
    stats_df = df[numeric_cols].describe().T
    stats_df = stats_df.style.background_gradient(cmap='YlOrRd', axis=1)
    st.write(stats_df)
    
    # Matrice de corrélation pour les variables numériques
    st.subheader("Matrice de Corrélation")
    
    # Sélectionner un sous-ensemble de colonnes numériques pour la corrélation
    corr_cols = st.multiselect(
        "Sélectionner des variables pour la matrice de corrélation",
        options=numeric_cols,
        default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
    )
    
    if corr_cols:
        corr = df[corr_cols].corr()
        
        # Créer une heatmap interactive avec Plotly
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Matrice de Corrélation des Variables Sélectionnées",
            zmin=-1, zmax=1
        )
        fig.update_layout(
            width=800,
            height=800,
            title_font_size=18
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des variables principales
    st.subheader("Distribution des Variables Principales")
    
    # Sélectionner une variable pour l'histogramme
    hist_var = st.selectbox(
        "Sélectionner une variable pour voir sa distribution",
        options=numeric_cols,
        index=numeric_cols.index('Age') if 'Age' in numeric_cols else 0
    )
    
    if hist_var:
        # Créer un histogramme avec KDE pour la variable sélectionnée
        fig = px.histogram(
            df, x=hist_var,
            color_discrete_sequence=[palette[0]],
            marginal="box",
            title=f"Distribution de la variable {hist_var}",
            opacity=0.7,
            nbins=30
        )
        fig.update_layout(
            xaxis_title=hist_var,
            yaxis_title="Fréquence",
            title_font_size=18
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Afficher un graphique à secteurs pour une variable catégorielle
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        st.subheader("Répartition des Variables Catégorielles")
        
        pie_var = st.selectbox(
            "Sélectionner une variable catégorielle pour voir sa répartition",
            options=cat_cols,
            index=cat_cols.index('Education') if 'Education' in cat_cols else 0
        )
        
        if pie_var:
            fig = plot_pie(df, pie_var, f"Répartition de {pie_var}")
            st.plotly_chart(fig, use_container_width=True)
# Page Données Démographiques
elif page == "Données Démographiques":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Analyse Démographique des Clients</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous permet d'explorer les caractéristiques démographiques des clients, y compris leur âge, 
    niveau d'éducation, statut marital, revenu et composition du foyer. Ces informations permettent de mieux 
    comprendre qui sont vos clients.
    """)
    
    # Questions spécifiques sur les données démographiques
    st.markdown("### What is the distribution of the years of birth for our customers?")
    
    if 'Year_Birth' in df.columns:
        fig = plot_histogram(df, 'Year_Birth', "Distribution des Années de Naissance", palette[0], bins=25)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### What is the education level distribution?")
    
    if 'Education' in df.columns:
        # Graphique à secteurs pour l'éducation
        fig = plot_pie(df, 'Education', "Distribution des Niveaux d'Éducation")
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique à barres pour l'éducation avec comptage
        education_counts = df['Education'].value_counts().reset_index()
        education_counts.columns = ['Niveau d\'Éducation', 'Nombre']
        
        fig = plot_bar(
            education_counts,
            x='Niveau d\'Éducation', y='Nombre',
            title='Nombre de Clients par Niveau d\'Éducation',
            color='Niveau d\'Éducation'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### What is the martial status of the majority of customers?")
    
    if 'Marital_Status' in df.columns:
        # Graphique à secteurs pour le statut marital
        fig = plot_pie(df, 'Marital_Status', "Distribution des Statuts Maritaux")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des revenus
    if 'Income' in df.columns:
        st.subheader("Distribution des Revenus des Clients")
        
        # Créer la figure matplotlib
        plt.figure(figsize=(10, 6))
        sns.displot(df['Income'], kde=True, height=6, aspect=2.6)
        plt.xlim([0, 200000])
        set_labels(x="Income", y="Frequency", title="Customer Income Distribution")
        
        # Convertir en image pour Streamlit
        buf = plt_to_streamlit(plt.gcf())
        st.image(buf)
    
    # Distribution des âges avec Seaborn
    if 'Age' in df.columns:
        st.subheader("Distribution des Âges des Clients")
        
        plt.figure(figsize=(20, 8))
        p = sns.histplot(df["Age"], color="#682F2F", kde=True, bins=50, alpha=1, fill=True, edgecolor="black")
        p.axes.lines[0].set_color(palette[0])
        p.axes.set_title("\nCustomer's Age Distribution\n", fontsize=25)
        plt.ylabel("Count", fontsize=20)
        plt.xlabel("\nAge", fontsize=20)
        sns.despine(left=True, bottom=True)
        
        # Convertir en image pour Streamlit
        buf = plt_to_streamlit(plt.gcf())
        st.image(buf)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution de la récence avec violin plot
        if 'Recency' in df.columns:
            plt.figure(figsize=(12, 6))
            set_labels(x="Number of Days Since Customer's Last Purchase", title="Distribution of Customer Recency")
            sns.violinplot(x=df['Recency'], palette=palette)
            
            # Convertir en image pour Streamlit
            buf = plt_to_streamlit(plt.gcf())
            st.image(buf)
    
    with col2:
        # Relation entre âge et revenu avec nuage de points interactif
        if 'Age' in df.columns and 'Income' in df.columns:
            fig = px.scatter(
                df, x='Age', y='Income',
                title='Relation entre Âge et Revenu',
                opacity=0.7,
                color_discrete_sequence=[palette[0]],
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Composition du foyer
    st.subheader("Composition du Foyer")
    
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
        # Visualisation plus avancée de la composition du foyer
        household_data = df.groupby(['Kidhome', 'Teenhome']).size().reset_index()
        household_data.columns = ['Enfants', 'Adolescents', 'Nombre']
        
        # Créer une heatmap 2D
        heatmap_data = household_data.pivot(index='Enfants', columns='Adolescents', values='Nombre')
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Nombre d'Adolescents", y="Nombre d'Enfants", color="Nombre de Clients"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='YlOrRd',
            title="Répartition des Clients par Composition du Foyer"
        )
        fig.update_layout(
            title_font_size=18,
            width=700,
            height=500
        )
        # Ajouter les valeurs dans la heatmap
        for i in range(len(household_data)):
            fig.add_annotation(
                x=household_data.iloc[i]['Adolescents'],
                y=household_data.iloc[i]['Enfants'],
                text=str(household_data.iloc[i]['Nombre']),
                showarrow=False,
                font=dict(size=14, color="black")
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique à barres empilées pour la composition du foyer
        kid_teen_df = pd.DataFrame(columns=['Composition', 'Nombre'])
        
        for kid in range(3):
            for teen in range(3):
                count = len(df[(df['Kidhome'] == kid) & (df['Teenhome'] == teen)])
                if count > 0:
                    kid_teen_df = pd.concat([kid_teen_df, pd.DataFrame({
                        'Composition': [f"{kid} enfant(s), {teen} ado(s)"],
                        'Nombre': [count]
                    })], ignore_index=True)
        
        fig = px.bar(
            kid_teen_df,
            x='Composition', y='Nombre',
            title='Nombre de Clients par Composition du Foyer',
            color_discrete_sequence=[palette[0]]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenu moyen par niveau d'éducation avec graphique à barres
    if 'Education' in df.columns and 'Income' in df.columns:
        income_by_education = df.groupby('Education')['Income'].mean().reset_index()
        
        fig = px.bar(
            income_by_education,
            x='Education', y='Income',
            title='Revenu Moyen par Niveau d\'Éducation',
            color='Education',
            text_auto='.2s'
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Niveau d'Éducation",
            yaxis_title="Revenu Moyen"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pairplot pour visualiser les relations entre plusieurs variables
    st.subheader("Relations entre les Principales Variables Démographiques")
    
    if all(col in df.columns for col in ['Income', 'Age', 'Recency']):
        if 'Spent' not in df.columns and 'TotalDepenses' in df.columns:
            df['Spent'] = df['TotalDepenses']
        
        if 'Spent' in df.columns and 'Marital_Status' in df.columns:
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[['Income', 'Age', 'Recency', 'Spent', 'Marital_Status']], 
                        hue='Marital_Status', 
                        palette='Set2',
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6})
            plt.tight_layout()
            
            # Convertir en image pour Streamlit
            buf = plt_to_streamlit(plt.gcf())
            st.image(buf)
# Page Comportement d'Achat
elif page == "Comportement d'Achat":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Analyse du Comportement d'Achat</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous permet d'analyser les comportements d'achat des clients, y compris leurs dépenses par catégorie de produits,
    leurs préférences de canaux d'achat, et les relations entre leurs caractéristiques démographiques et leurs habitudes d'achat.
    """)
    
    # Dépenses par catégorie de produits avec graphique à secteurs
    st.subheader("Dépenses par Catégorie de Produits")
    
    mnt_columns = [col for col in df.columns if col.startswith('Mnt')]
    if mnt_columns:
        # Graphique à secteurs interactif pour les dépenses moyennes par catégorie
        avg = df[mnt_columns].mean(axis=0)
        
        # Nettoyer les noms des catégories pour l'affichage
        clean_names = [col.replace('Mnt', '').replace('Products', '') for col in mnt_columns]
        
        fig = px.pie(
            values=avg.values, 
            names=clean_names,
            title='Average Amount Spent on Product Categories',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            title_font_size=18,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique à barres pour les dépenses par catégorie et niveau d'éducation
        if 'Education' in df.columns:
            cat = df.groupby('Education')[mnt_columns].mean().T
            
            # Convertir en format pour Plotly
            cat_reset = cat.reset_index()
            cat_melted = pd.melt(cat_reset, id_vars='index', value_vars=cat.columns)
            cat_melted.columns = ['Category', 'Education', 'Amount']
            
            # Nettoyer les noms des catégories
            cat_melted['Category'] = cat_melted['Category'].apply(lambda x: x.replace('Mnt', '').replace('Products', ''))
            
            fig = px.bar(
                cat_melted,
                x='Category', 
                y='Amount',
                color='Education',
                title='Average Amount Spent On Products for Each Education Level',
                barmode='group'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Catégorie de Produit",
                yaxis_title="Montant Moyen Dépensé"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique à barres empilées pour les dépenses par catégorie et groupe d'âge
        if 'Age' in df.columns:
            # Créer des groupes d'âge si pas déjà fait
            if 'CategorieAge' not in df.columns:
                df['CategorieAge'] = pd.cut(
                    df['Age'], 
                    bins=[0, 30, 40, 50, 60, 100], 
                    labels=['<30', '30-40', '40-50', '50-60', '>60']
                )
            
            age_cat = df.groupby('CategorieAge')[mnt_columns].mean().T
            
            # Convertir en format pour Plotly
            age_cat_reset = age_cat.reset_index()
            age_cat_melted = pd.melt(age_cat_reset, id_vars='index', value_vars=age_cat.columns)
            age_cat_melted.columns = ['Category', 'Age Group', 'Amount']
            
            # Nettoyer les noms des catégories
            age_cat_melted['Category'] = age_cat_melted['Category'].apply(lambda x: x.replace('Mnt', '').replace('Products', ''))
            
            fig = px.bar(
                age_cat_melted,
                x='Category', 
                y='Amount',
                color='Age Group',
                title='Average Amount Spent On Products for Each Age Group',
                barmode='group'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Catégorie de Produit",
                yaxis_title="Montant Moyen Dépensé"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des dépenses totales avec histogramme
        if 'TotalDepenses' in df.columns:
            fig = px.histogram(
                df, x='TotalDepenses',
                title='Distribution des Dépenses Totales',
                nbins=30,
                color_discrete_sequence=[palette[1]],
                opacity=0.7,
                marginal="box"
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Dépenses Totales",
                yaxis_title="Nombre de Clients"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Relation entre revenu et dépenses avec nuage de points
        if 'Income' in df.columns and 'TotalDepenses' in df.columns:
            fig = px.scatter(
                df, x='Income', y='TotalDepenses',
                title='Relation entre Revenu et Dépenses',
                opacity=0.7,
                trendline='ols',
                color_discrete_sequence=[palette[0]],
                hover_data=['Age', 'Education']
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Revenu",
                yaxis_title="Dépenses Totales"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Dépenses par canal d'achat
    purchase_columns = [col for col in df.columns if 'Purchases' in col and col.startswith('Num')]
    if purchase_columns:
        st.subheader("Comportement d'Achat par Canal")
        
        channel_data = pd.DataFrame({
            'Canal': [col.replace('Num', '').replace('Purchases', '') for col in purchase_columns],
            'Achats Moyens': [df[col].mean() for col in purchase_columns]
        })
        
        fig = px.pie(
            channel_data,
            values='Achats Moyens', names='Canal',
            title='Répartition des Achats par Canal',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_layout(
            title_font_size=18
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Ajouter un graphique à barres pour les achats par canal et niveau d'éducation
        if 'Education' in df.columns:
            purchase_by_edu = df.groupby('Education')[purchase_columns].mean().reset_index()
            purchase_by_edu_melted = pd.melt(purchase_by_edu, id_vars='Education', value_vars=purchase_columns)
            purchase_by_edu_melted.columns = ['Education', 'Canal', 'Moyenne']
            purchase_by_edu_melted['Canal'] = purchase_by_edu_melted['Canal'].apply(lambda x: x.replace('Num', '').replace('Purchases', ''))
            
            fig = px.bar(
                purchase_by_edu_melted,
                x='Canal', y='Moyenne',
                color='Education',
                barmode='group',
                title='Achats Moyens par Canal et Niveau d\'Éducation'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Canal d'Achat",
                yaxis_title="Nombre Moyen d'Achats"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des visites sur le site web
    if 'NumWebVisitsMonth' in df.columns:
        st.subheader("Analyse des Visites Web")
        
        fig = px.histogram(
            df, x='NumWebVisitsMonth',
            title='Distribution du Nombre de Visites Web par Mois',
            nbins=15,
            color_discrete_sequence=[palette[2]],
            opacity=0.7
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Nombre de Visites Web par Mois",
            yaxis_title="Nombre de Clients"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relation entre les visites web et les dépenses
        if 'TotalDepenses' in df.columns:
            fig = px.box(
                df, x='NumWebVisitsMonth', y='TotalDepenses',
                title='Relation entre Visites Web et Dépenses',
                color_discrete_sequence=[palette[0]]
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Nombre de Visites Web par Mois",
                yaxis_title="Dépenses Totales"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des remises et offres spéciales
    if 'NumDealsPurchases' in df.columns:
        st.subheader("Analyse des Achats avec Remise")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x='NumDealsPurchases',
                title='Distribution des Achats avec Remise',
                nbins=15,
                color_discrete_sequence=[palette[4]],
                opacity=0.7
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Nombre d'Achats avec Remise",
                yaxis_title="Nombre de Clients"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Relation entre achats avec remise et revenus
            if 'Income' in df.columns:
                fig = px.box(
                    df, x='NumDealsPurchases', y='Income',
                    title='Relation entre Achats avec Remise et Revenu',
                    color_discrete_sequence=[palette[3]]
                )
                fig.update_layout(
                    title_font_size=18,
                    xaxis_title="Nombre d'Achats avec Remise",
                    yaxis_title="Revenu"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap des corrélations entre comportements d'achat
    st.subheader("Corrélations entre Comportements d'Achat")
    
    # Sélectionner les colonnes liées au comportement d'achat
    purchase_vars = [col for col in df.columns if 'Purchases' in col or col.startswith('Mnt') or col in ['Recency', 'NumWebVisitsMonth']]
    
    if purchase_vars:
        corr = df[purchase_vars].corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Matrice de Corrélation des Comportements d'Achat",
            zmin=-1, zmax=1
        )
        fig.update_layout(
            title_font_size=18,
            width=800,
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)
# Page Analyse des Campagnes
elif page == "Analyse des Campagnes":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Analyse de l'Efficacité des Campagnes</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous permet d'analyser l'efficacité des différentes campagnes marketing, 
    d'identifier les facteurs qui influencent l'acceptation des campagnes et de comprendre 
    les caractéristiques des clients les plus réceptifs aux offres promotionnelles.
    """)
    
    # Taux d'acceptation par campagne
    campaign_cols = [col for col in df.columns if col.startswith('AcceptedCmp')]
    
    if campaign_cols:
        # Calculer le taux d'acceptation global
        df['TotalCampaigns'] = len(campaign_cols)
        if 'TotalPromotions' not in df.columns:
            df['TotalPromotions'] = df[campaign_cols].sum(axis=1)
        
        # Correction de l'erreur de formatage
        total_promotions = df['TotalPromotions'].sum()
        total_possible = df['TotalCampaigns'].iloc[0] * len(df)
        acceptance_rate = (total_promotions / total_possible) * 100
        
        # Métriques d'acceptation des campagnes
        st.subheader("Métriques d'Acceptation des Campagnes")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Taux d'Acceptation Global", f"{acceptance_rate:.2f}%")
        
        if 'NumDealsPurchases' in df.columns:
            col2.metric("Achats avec Remise (Moyenne)", f"{df['NumDealsPurchases'].mean():.2f}")
        
        # Proportion de clients ayant accepté au moins une campagne
        accepted_any = (df['TotalPromotions'] > 0).mean() * 100
        col3.metric("Clients Ayant Accepté Au Moins Une Campagne", f"{accepted_any:.2f}%")
        
        # Taux d'acceptation par campagne avec graphique à barres amélioré
        campaign_data = []
        for col in campaign_cols:
            campaign_num = col.replace('AcceptedCmp', '')
            acceptance = df[col].mean() * 100
            campaign_data.append({
                'Campagne': f'Campagne {campaign_num}',
                'Taux d\'Acceptation (%)': acceptance
            })
        
        campaign_df = pd.DataFrame(campaign_data)
        
        fig = px.bar(
            campaign_df,
            x='Campagne', y='Taux d\'Acceptation (%)',
            title='Taux d\'Acceptation par Campagne',
            color='Campagne',
            text_auto='.2f'
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Campagne",
            yaxis_title="Taux d'Acceptation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution du nombre de campagnes acceptées par client
        fig = px.histogram(
            df, x='TotalPromotions',
            title='Distribution du Nombre de Campagnes Acceptées par Client',
            nbins=len(campaign_cols) + 1,
            color_discrete_sequence=[palette[0]],
            opacity=0.7
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Nombre de Campagnes Acceptées",
            yaxis_title="Nombre de Clients",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des achats avec remise
    if 'NumDealsPurchases' in df.columns:
        st.subheader("Analyse des Achats avec Remise")
        
        fig = px.histogram(
            df, x='NumDealsPurchases',
            title='Distribution des Achats avec Remise',
            nbins=10,
            color_discrete_sequence=[palette[1]]
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Nombre d'Achats avec Remise",
            yaxis_title="Nombre de Clients"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relation entre achats avec remise et dépenses totales
        if 'TotalDepenses' in df.columns:
            # Créer des catégories pour les achats avec remise
            df['DealsCategory'] = pd.cut(
                df['NumDealsPurchases'],
                bins=[-1, 0, 3, 6, 15],
                labels=['Aucun', 'Faible (1-3)', 'Moyen (4-6)', 'Élevé (7+)']
            )
            
            # Graphique à barres des dépenses moyennes par catégorie d'achats avec remise
            deals_spending = df.groupby('DealsCategory')['TotalDepenses'].mean().reset_index()
            
            fig = px.bar(
                deals_spending,
                x='DealsCategory', y='TotalDepenses',
                title='Dépenses Moyennes par Nombre d\'Achats avec Remise',
                color='DealsCategory',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Catégorie d'Achats avec Remise",
                yaxis_title="Dépenses Moyennes"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Taux d'acceptation par caractéristiques démographiques
    st.subheader("Analyse de l'Acceptation des Campagnes par Caractéristiques Démographiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Taux d'acceptation par niveau d'éducation
        if 'Education' in df.columns and 'TotalPromotions' in df.columns:
            acceptance_by_education = df.groupby('Education').agg({
                'TotalPromotions': 'mean',
                'TotalCampaigns': 'first'
            }).reset_index()
            
            acceptance_by_education['AcceptanceRate'] = (acceptance_by_education['TotalPromotions'] / acceptance_by_education['TotalCampaigns']) * 100
            
            fig = px.bar(
                acceptance_by_education,
                x='Education', y='AcceptanceRate',
                title='Taux d\'Acceptation par Niveau d\'Éducation',
                labels={'AcceptanceRate': 'Taux d\'Acceptation (%)'},
                color='Education',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Niveau d'Éducation",
                yaxis_title="Taux d'Acceptation (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux d'acceptation par statut marital
        if 'Marital_Status' in df.columns and 'TotalPromotions' in df.columns:
            acceptance_by_marital = df.groupby('Marital_Status').agg({
                'TotalPromotions': 'mean',
                'TotalCampaigns': 'first'
            }).reset_index()
            
            acceptance_by_marital['AcceptanceRate'] = (acceptance_by_marital['TotalPromotions'] / acceptance_by_marital['TotalCampaigns']) * 100
            
            fig = px.bar(
                acceptance_by_marital,
                x='Marital_Status', y='AcceptanceRate',
                title='Taux d\'Acceptation par Statut Marital',
                labels={'AcceptanceRate': 'Taux d\'Acceptation (%)'},
                color='Marital_Status',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Statut Marital",
                yaxis_title="Taux d'Acceptation (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Relation entre visites sur le site web et acceptation des campagnes
    if 'NumWebVisitsMonth' in df.columns and 'TotalPromotions' in df.columns:
        st.subheader("Relation entre Comportement Web et Acceptation des Campagnes")
        
        # Calculer le taux d'acceptation moyen par nombre de visites
        web_acceptance = df.groupby('NumWebVisitsMonth').agg({
            'TotalPromotions': 'mean',
            'TotalCampaigns': 'first'
        }).reset_index()
        
        web_acceptance['AcceptanceRate'] = (web_acceptance['TotalPromotions'] / web_acceptance['TotalCampaigns']) * 100
        
        fig = px.line(
            web_acceptance,
            x='NumWebVisitsMonth', y='AcceptanceRate',
            title='Taux d\'Acceptation par Nombre de Visites Web Mensuelles',
            labels={'AcceptanceRate': 'Taux d\'Acceptation (%)', 'NumWebVisitsMonth': 'Nombre de Visites Web par Mois'},
            markers=True,
            color_discrete_sequence=[palette[0]]
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Nombre de Visites Web par Mois",
            yaxis_title="Taux d'Acceptation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Nuage de points pour explorer la relation individuelle
        fig = px.scatter(
            df, x='NumWebVisitsMonth', y='TotalPromotions',
            title='Relation entre Visites Web et Acceptation des Campagnes',
            opacity=0.7,
            color_discrete_sequence=[palette[2]],
            trendline='ols'
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Nombre de Visites Web par Mois",
            yaxis_title="Nombre de Campagnes Acceptées"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficacité des campagnes par groupe d'âge
    if 'Age' in df.columns and 'TotalPromotions' in df.columns:
        st.subheader("Efficacité des Campagnes par Groupe d'Âge")
        
        if 'CategorieAge' not in df.columns:
            df['CategorieAge'] = pd.cut(
                df['Age'], 
                bins=[0, 30, 40, 50, 60, 100], 
                labels=['<30', '30-40', '40-50', '50-60', '>60']
            )
        
        campaign_by_age = df.groupby('CategorieAge').agg({
            'TotalPromotions': 'mean',
            'TotalCampaigns': 'first'
        }).reset_index()
        
        campaign_by_age['AcceptanceRate'] = (campaign_by_age['TotalPromotions'] / campaign_by_age['TotalCampaigns']) * 100
        
        fig = px.bar(
            campaign_by_age,
            x='CategorieAge', y='AcceptanceRate',
            title='Taux d\'Acceptation par Groupe d\'Âge',
            labels={'AcceptanceRate': 'Taux d\'Acceptation (%)'},
            color='CategorieAge',
            text_auto='.2f'
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Groupe d'Âge",
            yaxis_title="Taux d'Acceptation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des âges pour les clients qui ont accepté vs. ceux qui n'ont pas accepté
        df['AnyAcceptance'] = df['TotalPromotions'] > 0
        
        fig = px.histogram(
            df, 
            x='Age', 
            color='AnyAcceptance',
            nbins=20,
            title="Distribution des Âges par Acceptation des Campagnes",
            opacity=0.7,
            barmode='overlay',
            color_discrete_map={True: palette[0], False: palette[3]},
            labels={'AnyAcceptance': 'A Accepté au Moins Une Campagne'}
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Âge",
            yaxis_title="Nombre de Clients",
            legend_title="A Accepté au Moins Une Campagne"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse de la récence et de son impact sur l'acceptation des campagnes
    if 'Recency' in df.columns and 'TotalPromotions' in df.columns:
        st.subheader("Impact de la Récence sur l'Acceptation des Campagnes")
        
        # Créer des catégories de récence
        df['RecencyCategory'] = pd.cut(
            df['Recency'],
            bins=[0, 15, 30, 60, 100],
            labels=['Très Récent (0-15j)', 'Récent (16-30j)', 'Modéré (31-60j)', 'Ancien (61j+)']
        )
        
        recency_acceptance = df.groupby('RecencyCategory').agg({
            'TotalPromotions': 'mean',
            'TotalCampaigns': 'first'
        }).reset_index()
        
        recency_acceptance['AcceptanceRate'] = (recency_acceptance['TotalPromotions'] / recency_acceptance['TotalCampaigns']) * 100
        
        fig = px.bar(
            recency_acceptance,
            x='RecencyCategory', y='AcceptanceRate',
            title='Taux d\'Acceptation par Catégorie de Récence',
            labels={'AcceptanceRate': 'Taux d\'Acceptation (%)'},
            color='RecencyCategory',
            text_auto='.2f'
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Catégorie de Récence",
            yaxis_title="Taux d'Acceptation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
# Page Segmentation des Clients
elif page == "Segmentation des Clients":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Segmentation des Clients</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous permet d'explorer différentes façons de segmenter vos clients.
    Vous pouvez visualiser les segments basés sur différentes caractéristiques et comprendre
    comment les comportements d'achat varient entre ces segments.
    """)
    
    # Segmentation par revenu
    if 'Income' in df.columns:
        st.subheader("Segmentation par Revenu")
        
        # Créer des segments de revenu
        if 'SegmentRevenu' not in df.columns:
            df['SegmentRevenu'] = pd.cut(
                df['Income'], 
                bins=[0, 30000, 60000, 90000, float('inf')], 
                labels=['Bas', 'Moyen', 'Élevé', 'Très élevé']
            )
        
        # Graphique à secteurs pour la répartition des segments de revenu
        income_segment_counts = df['SegmentRevenu'].value_counts().reset_index()
        income_segment_counts.columns = ['Segment de Revenu', 'Nombre']
        
        fig = px.pie(
            income_segment_counts,
            values='Nombre', names='Segment de Revenu',
            title='Répartition des Clients par Segment de Revenu',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            title_font_size=18
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Dépenses moyennes par segment de revenu
        if 'TotalDepenses' in df.columns:
            spending_by_income = df.groupby('SegmentRevenu')['TotalDepenses'].mean().reset_index()
            
            fig = px.bar(
                spending_by_income,
                x='SegmentRevenu', y='TotalDepenses',
                title='Dépenses Moyennes par Segment de Revenu',
                color='SegmentRevenu',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Segment de Revenu",
                yaxis_title="Dépenses Moyennes"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique radar des dépenses par catégorie pour chaque segment de revenu
            if all(col in df.columns for col in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']):
                # Calculer les dépenses moyennes par catégorie pour chaque segment
                cat_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']
                cat_names = [col.replace('Mnt', '').replace('Products', '') for col in cat_cols]
                
                income_cat_spend = df.groupby('SegmentRevenu')[cat_cols].mean().reset_index()
                
                # Préparer les données pour le graphique radar
                fig = go.Figure()
                
                for i, segment in enumerate(income_cat_spend['SegmentRevenu']):
                    fig.add_trace(go.Scatterpolar(
                        r=income_cat_spend.iloc[i][cat_cols].values,
                        theta=cat_names,
                        fill='toself',
                        name=segment
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, income_cat_spend[cat_cols].max().max() * 1.1]
                        )
                    ),
                    showlegend=True,
                    title="Profil de Dépenses par Segment de Revenu"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation par âge
    if 'Age' in df.columns:
        st.subheader("Segmentation par Âge")
        
        if 'CategorieAge' not in df.columns:
            df['CategorieAge'] = pd.cut(
                df['Age'], 
                bins=[0, 30, 40, 50, 60, 100], 
                labels=['<30', '30-40', '40-50', '50-60', '>60']
            )
        
        age_segment_counts = df['CategorieAge'].value_counts().reset_index()
        age_segment_counts.columns = ['Catégorie d\'Âge', 'Nombre']
        
        fig = px.pie(
            age_segment_counts,
            values='Nombre', names='Catégorie d\'Âge',
            title='Répartition des Clients par Catégorie d\'Âge',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            title_font_size=18
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Relation entre l'âge et les comportements d'achat
        if 'TotalDepenses' in df.columns:
            age_spending = df.groupby('CategorieAge')['TotalDepenses'].mean().reset_index()
            
            fig = px.bar(
                age_spending,
                x='CategorieAge', y='TotalDepenses',
                title='Dépenses Moyennes par Catégorie d\'Âge',
                color='CategorieAge',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Catégorie d'Âge",
                yaxis_title="Dépenses Moyennes"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation par composition du foyer
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
        st.subheader("Segmentation par Composition du Foyer")
        
        # Créer des segments basés sur la présence d'enfants
        if 'SegmentFoyer' not in df.columns:
            df['SegmentFoyer'] = 'Sans enfants'
            df.loc[df['Kidhome'] > 0, 'SegmentFoyer'] = 'Avec jeunes enfants'
            df.loc[df['Teenhome'] > 0, 'SegmentFoyer'] = 'Avec adolescents'
            df.loc[(df['Kidhome'] > 0) & (df['Teenhome'] > 0), 'SegmentFoyer'] = 'Avec enfants et adolescents'
        
        household_segment_counts = df['SegmentFoyer'].value_counts().reset_index()
        household_segment_counts.columns = ['Segment de Foyer', 'Nombre']
        
        fig = px.pie(
            household_segment_counts,
            values='Nombre', names='Segment de Foyer',
            title='Répartition des Clients par Composition du Foyer',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            title_font_size=18
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Dépenses moyennes par segment de foyer
        if 'TotalDepenses' in df.columns:
            spending_by_household = df.groupby('SegmentFoyer')['TotalDepenses'].mean().reset_index()
            
            fig = px.bar(
                spending_by_household,
                x='SegmentFoyer', y='TotalDepenses',
                title='Dépenses Moyennes par Segment de Foyer',
                color='SegmentFoyer',
                text_auto='.2f'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Segment de Foyer",
                yaxis_title="Dépenses Moyennes"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Profil de dépenses par segment de foyer
            if all(col in df.columns for col in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']):
                st.subheader("Profil de Dépenses par Segment de Foyer")
                
                # Préparation des données
                cat_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']
                household_spending = df.groupby('SegmentFoyer')[cat_cols].mean().reset_index()
                
                # Convertir en format long pour Plotly
                household_spending_long = pd.melt(
                    household_spending, 
                    id_vars='SegmentFoyer', 
                    value_vars=cat_cols,
                    var_name='Catégorie',
                    value_name='Dépenses Moyennes'
                )
                
                # Nettoyer les noms des catégories
                household_spending_long['Catégorie'] = household_spending_long['Catégorie'].apply(
                    lambda x: x.replace('Mnt', '').replace('Products', '')
                )
                
                fig = px.bar(
                    household_spending_long,
                    x='Catégorie', y='Dépenses Moyennes',
                    color='SegmentFoyer',
                    barmode='group',
                    title='Profil de Dépenses par Segment de Foyer'
                )
                fig.update_layout(
                    title_font_size=18,
                    xaxis_title="Catégorie de Produit",
                    yaxis_title="Dépenses Moyennes"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation par comportement d'achat
    if 'TotalDepenses' in df.columns:
        st.subheader("Segmentation par Comportement d'Achat")
        
        # Créer des segments basés sur les dépenses
        if 'SegmentAchat' not in df.columns:
            df['SegmentAchat'] = pd.qcut(
                df['TotalDepenses'], 
                q=[0, 0.25, 0.5, 0.75, 1], 
                labels=['Faible', 'Modéré', 'Élevé', 'Premium']
            )
        
        purchase_segment_counts = df['SegmentAchat'].value_counts().reset_index()
        purchase_segment_counts.columns = ['Segment d\'Achat', 'Nombre']
        
        fig = px.pie(
            purchase_segment_counts,
            values='Nombre', names='Segment d\'Achat',
            title='Répartition des Clients par Comportement d\'Achat',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(
            title_font_size=18
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualiser les segments d'achat en fonction du revenu
        if 'Income' in df.columns:
            fig = px.box(
                df, x='SegmentAchat', y='Income',
                title='Distribution des Revenus par Segment d\'Achat',
                color='SegmentAchat'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Segment d'Achat",
                yaxis_title="Revenu"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Caractéristiques moyennes par segment d'achat
            if 'Age' in df.columns and 'NumWebVisitsMonth' in df.columns and 'Recency' in df.columns:
                st.subheader("Caractéristiques Moyennes par Segment d'Achat")
                
                segment_avg = df.groupby('SegmentAchat')[['Age', 'Income', 'NumWebVisitsMonth', 'Recency']].mean().reset_index()
                
                # Préparer les données pour le radar chart
                segment_avg_scaled = segment_avg.copy()
                for col in ['Age', 'Income', 'NumWebVisitsMonth', 'Recency']:
                    min_val = segment_avg[col].min()
                    max_val = segment_avg[col].max()
                    segment_avg_scaled[col] = (segment_avg[col] - min_val) / (max_val - min_val)
                
                # Créer le radar chart
                fig = go.Figure()
                
                for i, segment in enumerate(segment_avg_scaled['SegmentAchat']):
                    fig.add_trace(go.Scatterpolar(
                        r=segment_avg_scaled.iloc[i][['Age', 'Income', 'NumWebVisitsMonth', 'Recency']].values,
                        theta=['Âge', 'Revenu', 'Visites Web', 'Récence'],
                        fill='toself',
                        name=segment
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="Caractéristiques des Segments d'Achat (Normalisées)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation multi-critères (2D)
    st.subheader("Segmentation Multi-critères")
    
    # Sélection des axes pour la visualisation
    col1, col2 = st.columns(2)
    
    with col1:
        available_numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x_axis = st.selectbox(
            "Sélectionner l'axe X",
            options=[col for col in available_numeric_columns if col not in ['ID']],
            index=available_numeric_columns.index('Income') if 'Income' in available_numeric_columns else 0
        )
    
    with col2:
        y_axis = st.selectbox(
            "Sélectionner l'axe Y",
            options=[col for col in available_numeric_columns if col not in ['ID', x_axis]],
            index=available_numeric_columns.index('TotalDepenses') if 'TotalDepenses' in available_numeric_columns else 0
        )
    
    # Sélection de la variable de couleur
    color_variable = st.selectbox(
        "Colorer par",
        options=['Education', 'Marital_Status', 'CategorieAge', 'SegmentRevenu', 'SegmentFoyer', 'SegmentAchat'],
        index=0
    )
    
    # Vérifier si la variable de couleur existe
    if color_variable in df.columns:
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color=color_variable,
            title=f'Segmentation: {x_axis} vs {y_axis} par {color_variable}',
            opacity=0.7,
            size='TotalDepenses' if 'TotalDepenses' in df.columns else None,
            hover_data=['Age', 'Education', 'Marital_Status'] if all(col in df.columns for col in ['Age', 'Education', 'Marital_Status']) else None
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title=x_axis,
            yaxis_title=y_axis
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"La variable {color_variable} n'est pas disponible. Veuillez sélectionner une autre variable.")
# Page Analyse des Clusters
elif page == "Analyse des Clusters":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Analyse des Clusters de Clients</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous permet d'explorer les clusters de clients identifiés dans le cadre de l'étude.
    Les clients sont regroupés en fonction de leurs caractéristiques et comportements similaires,
    ce qui permet de développer des stratégies marketing ciblées pour chaque groupe.
    """)
    
    # Afficher les profils de clusters déjà identifiés dans l'étude
    st.subheader("Profils de Clusters Identifiés")
    
    cluster_profiles = {
        0: {
            "nom": "Familles Économes",
            "description": "Ces clients ont un revenu modéré, des dépenses moyennes et utilisent fréquemment les remises. Ils ont généralement 1 à 2 enfants et sont majoritairement mariés.",
            "revenu_moyen": "34 865 €",
            "depenses_moyennes": "500 €",
            "taille": "895 clients (40.0%)",
            "couleur": "#3498db"  # Bleu
        },
        1: {
            "nom": "Couples Aisés",
            "description": "Clients à revenu élevé avec des dépenses importantes. Ils utilisent moins fréquemment les remises, ont généralement un enfant et sont principalement mariés sans être parents à temps plein.",
            "revenu_moyen": "65 463 €",
            "depenses_moyennes": "1 275 €",
            "taille": "578 clients (25.8%)",
            "couleur": "#2ecc71"  # Vert
        },
        2: {
            "nom": "Seniors Fortunés",
            "description": "Le groupe avec le revenu le plus élevé et les dépenses les plus importantes. Ils n'ont pas d'enfants à la maison mais sont considérés comme parents, suggérant des enfants plus âgés ayant quitté le foyer.",
            "revenu_moyen": "78 413 €",
            "depenses_moyennes": "1 500 €",
            "taille": "383 clients (17.1%)",
            "couleur": "#e74c3c"  # Rouge
        },
        3: {
            "nom": "Familles à la Recherche de Valeur",
            "description": "Revenu modéré avec des dépenses relativement faibles. Ils ont plusieurs enfants et utilisent très fréquemment les remises, montrant un comportement conscient des prix.",
            "revenu_moyen": "45 902 €",
            "depenses_moyennes": "500 €",
            "taille": "360 clients (16.1%)",
            "couleur": "#f39c12"  # Orange
        }
    }
    
    # Sélectionner un cluster à afficher
    selected_cluster = st.selectbox(
        "Sélectionner un cluster à visualiser",
        options=list(cluster_profiles.keys()),
        format_func=lambda x: f"Cluster {x}: {cluster_profiles[x]['nom']}"
    )
    
    # Afficher les détails du cluster sélectionné avec un style amélioré
    profile = cluster_profiles[selected_cluster]
    
    # Utiliser des colonnes pour un affichage plus attrayant
    st.markdown(f"<h3 style='color:{profile['couleur']};'>{profile['nom']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='background-color:#f9f9f9;padding:15px;border-radius:5px;'>{profile['description']}</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    col1.markdown(f"<div style='background-color:{profile['couleur']}20;padding:10px;border-radius:5px;text-align:center;'>"
                 f"<h4>Revenu Moyen</h4>"
                 f"<h2>{profile['revenu_moyen']}</h2>"
                 f"</div>", unsafe_allow_html=True)
    
    col2.markdown(f"<div style='background-color:{profile['couleur']}20;padding:10px;border-radius:5px;text-align:center;'>"
                 f"<h4>Dépenses Moyennes</h4>"
                 f"<h2>{profile['depenses_moyennes']}</h2>"
                 f"</div>", unsafe_allow_html=True)
    
    col3.markdown(f"<div style='background-color:{profile['couleur']}20;padding:10px;border-radius:5px;text-align:center;'>"
                 f"<h4>Taille du Cluster</h4>"
                 f"<h2>{profile['taille']}</h2>"
                 f"</div>", unsafe_allow_html=True)
    
    # Caractéristiques des clusters
    st.subheader("Caractéristiques des Clusters")
    
    # Créer des données simulées pour les caractéristiques des clusters
    cluster_data = pd.DataFrame({
        'Caractéristique': ['Revenu', 'Dépenses', 'Âge Moyen', 'Nombre d\'Enfants', 'Achats avec Remise'],
        'Cluster 0': [34865, 500, 45, 1.5, 4.5],
        'Cluster 1': [65463, 1275, 50, 0.8, 2.2],
        'Cluster 2': [78413, 1500, 55, 0, 3.8],
        'Cluster 3': [45902, 500, 48, 2.1, 5.2]
    })
    
    # Normaliser les données pour le graphique radar
    normalized_data = cluster_data.copy()
    for col in normalized_data.columns:
        if col != 'Caractéristique':
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    
    # Créer le graphique radar avec des couleurs personnalisées
    fig = go.Figure()
    
    colors = [cluster_profiles[i]['couleur'] for i in range(4)]
    
    for i in range(4):
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[f'Cluster {i}'],
            theta=normalized_data['Caractéristique'],
            fill='toself',
            name=f"Cluster {i}: {cluster_profiles[i]['nom']}",
            line_color=colors[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Comparaison des Caractéristiques des Clusters (Normalisée)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualisation de la distribution des clusters
    st.subheader("Distribution des Clusters")
    
    # Données pour le graphique à secteurs avec des couleurs personnalisées
    cluster_sizes = pd.DataFrame({
        'Cluster': [f"Cluster {i}: {profile['nom']}" for i, profile in cluster_profiles.items()],
        'Taille': [895, 578, 383, 360],
        'Couleur': [profile['couleur'] for i, profile in cluster_profiles.items()]
    })
    
    fig = px.pie(
        cluster_sizes,
        values='Taille',
        names='Cluster',
        title='Répartition des Clients par Cluster',
        color='Cluster',
        color_discrete_map={cluster_sizes['Cluster'][i]: cluster_sizes['Couleur'][i] for i in range(len(cluster_sizes))}
    )
    fig.update_layout(
        title_font_size=18
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparaison des caractéristiques d'achat par cluster
    st.subheader("Comportement d'Achat par Cluster")
    
    # Créer des données simulées pour les comportements d'achat
    purchase_data = pd.DataFrame({
        'Catégorie': ['Vins', 'Fruits', 'Viandes', 'Poissons', 'Sucreries'],
        'Cluster 0': [150, 30, 80, 40, 30],
        'Cluster 1': [400, 50, 200, 70, 50],
        'Cluster 2': [600, 40, 250, 100, 40],
        'Cluster 3': [120, 80, 100, 30, 50]
    })
    
    # Préparer les données pour un graphique à barres groupées
    purchase_data_melted = pd.melt(
        purchase_data, 
        id_vars='Catégorie', 
        value_vars=[f'Cluster {i}' for i in range(4)],
        var_name='Cluster',
        value_name='Dépenses Moyennes'
    )
    
    # Ajouter les noms des clusters
    purchase_data_melted['Nom Cluster'] = purchase_data_melted['Cluster'].apply(
        lambda x: f"{x}: {cluster_profiles[int(x.split(' ')[1])]['nom']}"
    )
    
    # Créer un graphique à barres groupées
    fig = px.bar(
        purchase_data_melted,
        x='Catégorie', y='Dépenses Moyennes',
        color='Nom Cluster',
        barmode='group',
        title='Dépenses Moyennes par Catégorie de Produit et Cluster',
        color_discrete_map={
            f"Cluster {i}: {profile['nom']}": profile['couleur'] 
            for i, profile in cluster_profiles.items()
        }
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title="Catégorie de Produit",
        yaxis_title="Dépenses Moyennes (€)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations marketing
    st.subheader("Recommandations Marketing par Cluster")
    
    # Recommandations pour chaque cluster
    recommendations = {
        0: [
            "Mettre l'accent sur les promotions orientées famille",
            "Souligner la valeur et les économies",
            "Offrir des programmes de fidélité avec des avantages immédiats",
            "Cibler avec des offres groupées pour les produits familiaux"
        ],
        1: [
            "Mettre en avant les produits et services premium",
            "Créer des offres exclusives qui font appel au statut",
            "Se concentrer moins sur les remises et plus sur la qualité/l'unicité",
            "Développer des programmes de partenariat avec des marques de luxe"
        ],
        2: [
            "Commercialiser des produits et expériences haut de gamme",
            "Créer des offres personnalisées axées sur la qualité et l'exclusivité",
            "Cibler avec des messages orientés retraite ou loisirs",
            "Mettre en œuvre des programmes VIP avec des services spéciaux"
        ],
        3: [
            "Fournir une valeur maximale grâce à des offres spéciales et des remises",
            "Créer des promotions groupées pour familles",
            "Mettre en place des programmes qui récompensent les achats fréquents",
            "Se concentrer sur les produits essentiels avec des prix compétitifs"
        ]
    }
    
    # Afficher les recommandations pour le cluster sélectionné avec un style amélioré
    st.markdown(f"<div style='background-color:{profile['couleur']}10;padding:15px;border-radius:5px;'>", unsafe_allow_html=True)
    for rec in recommendations[selected_cluster]:
        st.markdown(f"<p>✓ {rec}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Stratégie d'engagement
    st.subheader("Stratégie d'Engagement Client")
    
    engagement_strategies = {
        0: """
        **Canaux de Communication**: Newsletters par e-mail avec des conseils pour la famille et des idées d'économies.  
        **Fréquence**: Fréquence moyenne (communications hebdomadaires).  
        **Messages Clés**: "Économisez plus pour ce qui compte le plus", "Offres familiales économiques".  
        **Périodes Optimales**: Promotions de week-end pour les courses familiales.
        """,
        1: """
        **Canaux de Communication**: E-mails personnalisés et notifications d'application mobile sur des offres exclusives.  
        **Fréquence**: Faible fréquence (offres premium bimensuelles).  
        **Messages Clés**: "Exclusivement organisé pour vous", "Sélection premium".  
        **Périodes Optimales**: Promotions en soirée pour les achats après le travail.
        """,
        2: """
        **Canaux de Communication**: E-mails personnalisés, courrier direct et appels téléphoniques pour les offres de grande valeur.  
        **Fréquence**: Faible fréquence avec des communications de grande valeur.  
        **Messages Clés**: "Expérience de luxe", "Avantages exclusifs aux membres".  
        **Périodes Optimales**: Promotions en milieu de semaine lorsque les magasins sont moins fréquentés.
        """,
        3: """
        **Canaux de Communication**: SMS, e-mail avec des informations claires sur les remises et notifications d'application.  
        **Fréquence**: Haute fréquence avec des offres et promotions.  
        **Messages Clés**: "Les plus grandes économies de la semaine", "Offres familiales à durée limitée".  
        **Périodes Optimales**: Promotions de jour de paie alignées sur le calendrier des revenus.
        """
    }
    
    # Afficher la stratégie d'engagement pour le cluster sélectionné avec un style amélioré
    st.markdown(f"<div style='background-color:{profile['couleur']}10;padding:15px;border-radius:5px;'>", unsafe_allow_html=True)
    st.markdown(engagement_strategies[selected_cluster], unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
# Page Visualisation 3D
elif page == "Visualisation 3D":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Visualisation 3D des Clusters</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section vous propose une visualisation interactive en 3D des clusters de clients. 
    Cette représentation vous permet de mieux comprendre comment les clients sont regroupés 
    dans un espace tridimensionnel basé sur leurs caractéristiques principales.
    """)
    
    # Exécuter l'algorithme de clustering
    if 'Income' in df.columns and 'Age' in df.columns and 'TotalDepenses' in df.columns:
        st.subheader("Clustering K-Means en 3D")
        
        # Sélection des variables pour le clustering
        st.markdown("### Sélection des Variables")
        
        # Colonnes numériques disponibles
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclure les ID et autres colonnes non pertinentes
        exclude_cols = ['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue']
        available_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Sélection des variables X, Y et Z pour la visualisation 3D
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox(
                "Variable X",
                options=available_cols,
                index=available_cols.index('Income') if 'Income' in available_cols else 0
            )
        
        with col2:
            y_var = st.selectbox(
                "Variable Y",
                options=[col for col in available_cols if col != x_var],
                index=0
            )
        
        with col3:
            z_var = st.selectbox(
                "Variable Z",
                options=[col for col in available_cols if col not in [x_var, y_var]],
                index=0
            )
        
        # Nombre de clusters
        n_clusters = st.slider("Nombre de Clusters", min_value=2, max_value=8, value=4)
        
        # Exécuter le clustering K-Means
        st.markdown("### Visualisation des Clusters en 3D")
        
        # Préparer les données pour le clustering
        cluster_data = df[[x_var, y_var, z_var]].copy()
        # Gérer les valeurs manquantes
        cluster_data = cluster_data.dropna()
        
        # Standardiser les données
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Appliquer K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Ajouter les labels de cluster au dataframe
        cluster_data['Cluster'] = cluster_labels
        
        # Créer une palette de couleurs pour les clusters
        colors = px.colors.qualitative.Plotly
        
        # Visualisation 3D des clusters
        fig = px.scatter_3d(
            cluster_data, x=x_var, y=y_var, z=z_var,
            color='Cluster',
            color_discrete_sequence=colors[:n_clusters],
            opacity=0.7,
            title=f"Visualisation 3D des Clusters ({x_var}, {y_var}, {z_var})"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=z_var
            ),
            title_font_size=18,
            legend_title="Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des clusters
        st.subheader("Analyse des Clusters")
        
        # Statistiques descriptives par cluster
        cluster_stats = cluster_data.groupby('Cluster').agg({
            x_var: ['mean', 'std'],
            y_var: ['mean', 'std'],
            z_var: ['mean', 'std']
        }).reset_index()
        
        # Formater le tableau pour l'affichage
        cluster_stats.columns = ['Cluster', 
                                f'{x_var} (Moyenne)', f'{x_var} (Écart-type)',
                                f'{y_var} (Moyenne)', f'{y_var} (Écart-type)',
                                f'{z_var} (Moyenne)', f'{z_var} (Écart-type)']
        
        st.write("Statistiques par Cluster:")
        st.write(cluster_stats)
        
        # Taille des clusters
        cluster_sizes = cluster_data['Cluster'].value_counts().sort_index().reset_index()
        cluster_sizes.columns = ['Cluster', 'Nombre de Clients']
        
        fig = px.bar(
            cluster_sizes,
            x='Cluster', y='Nombre de Clients',
            title='Taille des Clusters',
            color='Cluster',
            color_discrete_sequence=colors[:n_clusters],
            text_auto=True
        )
        fig.update_layout(
            title_font_size=18,
            xaxis_title="Cluster",
            yaxis_title="Nombre de Clients"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualisation des centroïdes
        st.subheader("Centroïdes des Clusters")
        
        # Transformer les centroïdes pour les ramener à l'échelle d'origine
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)
        
        # Créer un dataframe pour les centroïdes
        centroids_df = pd.DataFrame(centroids, columns=[x_var, y_var, z_var])
        centroids_df['Cluster'] = range(n_clusters)
        
        # Créer un graphique 3D des centroïdes
        fig = px.scatter_3d(
            centroids_df, x=x_var, y=y_var, z=z_var,
            color='Cluster',
            color_discrete_sequence=colors[:n_clusters],
            size_max=20, size=[15] * n_clusters,
            title="Centroïdes des Clusters"
        )
        
        # Ajouter les points de données avec une faible opacité
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data[x_var],
                y=cluster_data[y_var],
                z=cluster_data[z_var],
                mode='markers',
                marker=dict(
                    size=3,
                    color=cluster_data['Cluster'],
                    colorscale='Rainbow',
                    opacity=0.1
                ),
                showlegend=False
            )
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=z_var
            ),
            title_font_size=18
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Certaines variables nécessaires pour le clustering 3D ne sont pas disponibles dans le jeu de données.")

# Page Analyses Avancées
elif page == "Analyses Avancées":
    st.markdown("## <p style='background-color:#682F2F;color:#FFF9ED;font-size:120%;padding:10px;border-radius:10px'>Analyses Avancées</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Cette section propose des visualisations et analyses supplémentaires plus avancées pour approfondir 
    la compréhension des données clients et découvrir des insights cachés.
    """)
    
    # Sélection du type d'analyse
    analysis_type = st.selectbox(
        "Sélectionner le type d'analyse",
        options=["Distribution des Revenus", "Relation Âge-Revenu-Dépenses", "Analyse PCA", 
                "Corrélation Heatmap", "Pattern d'Achat Temporel", "Segmentation RFM"]
    )
    
    if analysis_type == "Distribution des Revenus":
        st.subheader("Distribution des Revenus")
        
        if 'Income' in df.columns:
            # Créer la figure matplotlib
            plt.figure(figsize=(10, 6))
            sns.displot(df['Income'], kde=True, height=6, aspect=2.6)
            plt.xlim([0, 200000])
            set_labels(x="Income", y="Frequency", title="Customer Income Distribution")
            
            # Convertir en image pour Streamlit
            buf = plt_to_streamlit(plt.gcf())
            st.image(buf)
            
            # Statistiques sur la distribution des revenus
            stats = df['Income'].describe().reset_index()
            stats.columns = ['Statistique', 'Valeur']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Statistiques de Distribution:")
                st.write(stats)
            
            with col2:
                # Visualiser la distribution avec un boxplot
                fig = px.box(
                    df, y='Income',
                    title='Boxplot des Revenus',
                    points='all',
                    color_discrete_sequence=[palette[0]]
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Relation Âge-Revenu-Dépenses":
        st.subheader("Relation entre Âge, Revenu et Dépenses")
        
        if all(col in df.columns for col in ['Age', 'Income', 'TotalDepenses']):
            # Graphique 3D pour visualiser la relation
            fig = px.scatter_3d(
                df, x='Age', y='Income', z='TotalDepenses',
                color='TotalDepenses',
                opacity=0.7,
                color_continuous_scale='Viridis',
                title="Relation 3D entre Âge, Revenu et Dépenses"
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Âge",
                    yaxis_title="Revenu",
                    zaxis_title="Dépenses Totales"
                ),
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de scatter plots
            st.write("Matrice de Scatter Plots:")
            
            # Utiliser Seaborn pour créer une matrice de scatter plots
            plt.figure(figsize=(12, 10))
            scatter_cols = ['Age', 'Income', 'TotalDepenses', 'Recency'] if 'Recency' in df.columns else ['Age', 'Income', 'TotalDepenses']
            sns.pairplot(
                df[scatter_cols], 
                diag_kind='kde', 
                plot_kws={'alpha': 0.6, 'edgecolor': 'k', 'linewidth': 0.5},
                corner=True
            )
            plt.tight_layout()
            
            # Convertir en image pour Streamlit
            buf = plt_to_streamlit(plt.gcf())
            st.image(buf)
    
    elif analysis_type == "Analyse PCA":
        st.subheader("Analyse en Composantes Principales (PCA)")
        
        # Sélectionner les variables numériques pour PCA
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Exclure les ID et autres colonnes non pertinentes
        exclude_cols = ['ID', 'Z_CostContact', 'Z_Revenue', 'Year_Birth']
        pca_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(pca_cols) >= 3:  # Besoin d'au moins 3 variables pour PCA
            # Préparer les données pour PCA
            pca_data = df[pca_cols].copy().dropna()
            
            # Standardiser les données
            scaler = StandardScaler()
            pca_data_scaled = scaler.fit_transform(pca_data)
            
            # Appliquer PCA
            pca = PCA(n_components=3)  # Réduire à 3 dimensions pour visualisation
            principal_components = pca.fit_transform(pca_data_scaled)
            
            # Créer un dataframe avec les composantes principales
            pca_df = pd.DataFrame(
                data=principal_components,
                columns=['PC1', 'PC2', 'PC3']
            )
            
            # Afficher la variance expliquée
            explained_variance = pca.explained_variance_ratio_
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Variance Expliquée PC1", f"{explained_variance[0]:.2%}")
            col2.metric("Variance Expliquée PC2", f"{explained_variance[1]:.2%}")
            col3.metric("Variance Expliquée PC3", f"{explained_variance[2]:.2%}")
            
            st.write(f"Variance Cumulée Expliquée: {sum(explained_variance):.2%}")
            
            # Visualisation 3D des composantes principales
            fig = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3',
                opacity=0.7,
                color_discrete_sequence=[palette[0]],
                title="Visualisation 3D des Composantes Principales"
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"PC1 ({explained_variance[0]:.2%})",
                    yaxis_title=f"PC2 ({explained_variance[1]:.2%})",
                    zaxis_title=f"PC3 ({explained_variance[2]:.2%})"
                ),
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les loadings (contributions des variables)
            st.subheader("Contributions des Variables aux Composantes Principales")
            
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2', 'PC3'],
                index=pca_cols
            )
            
            # Visualiser les loadings avec un heatmap
            fig = px.imshow(
                loadings,
                labels=dict(x="Composante Principale", y="Variable", color="Contribution"),
                x=['PC1', 'PC2', 'PC3'],
                y=pca_cols,
                color_continuous_scale='RdBu_r',
                title="Matrice des Loadings de PCA"
            )
            fig.update_layout(
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Nombre insuffisant de variables numériques pour l'analyse PCA.")
    
    elif analysis_type == "Corrélation Heatmap":
        st.subheader("Matrice de Corrélation")
        
        # Sélectionner les variables numériques
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Permettre à l'utilisateur de sélectionner les variables
        selected_vars = st.multiselect(
            "Sélectionner les variables pour la matrice de corrélation",
            options=numeric_cols,
            default=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        )
        
        if selected_vars:
            # Calculer la matrice de corrélation
            corr = df[selected_vars].corr()
            
            # Créer un masque pour le triangle supérieur
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Créer un heatmap avec seaborn
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                corr, 
                mask=mask,
                cmap='coolwarm',
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8}
            )
            plt.title("Matrice de Corrélation", fontsize=16)
            plt.tight_layout()
            
            # Convertir en image pour Streamlit
            buf = plt_to_streamlit(plt.gcf())
            st.image(buf)
            
            # Afficher les corrélations les plus fortes
            st.subheader("Corrélations les Plus Fortes")
            
            # Créer un dataframe des corrélations
            corr_df = corr.unstack().reset_index()
            corr_df.columns = ['Variable 1', 'Variable 2', 'Corrélation']
            
            # Filtrer pour éviter les auto-corrélations et duplications
            corr_df = corr_df[corr_df['Variable 1'] != corr_df['Variable 2']]
            corr_df['Paire'] = corr_df.apply(lambda x: tuple(sorted([x['Variable 1'], x['Variable 2']])), axis=1)
            corr_df = corr_df.drop_duplicates('Paire')
            
            # Afficher les corrélations fortes (positives et négatives)
            corr_df['Corrélation Abs'] = corr_df['Corrélation'].abs()
            top_corr = corr_df.nlargest(10, 'Corrélation Abs')
            
            fig = px.bar(
                top_corr,
                x='Corrélation',
                y=[f"{row['Variable 1']} - {row['Variable 2']}" for _, row in top_corr.iterrows()],
                orientation='h',
                color='Corrélation',
                color_continuous_scale='RdBu_r',
                title="Top 10 des Corrélations (Absolues)"
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Coefficient de Corrélation",
                yaxis_title="Paire de Variables"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Pattern d'Achat Temporel":
        st.subheader("Patterns d'Achat Temporels")
        
        # Vérifier si les données temporelles sont disponibles
        if 'Dt_Customer' in df.columns:
            # Convertir la date en format datetime
            try:
                df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
                
                # Extraire le mois et l'année
                df['Mois'] = df['Dt_Customer'].dt.month
                df['Année'] = df['Dt_Customer'].dt.year
                
                # Agréger les dépenses par mois et année si disponibles
                if 'TotalDepenses' in df.columns:
                    # Grouper par mois et année
                    temp_df = df.groupby(['Année', 'Mois'])['TotalDepenses'].agg(['mean', 'count']).reset_index()
                    temp_df.columns = ['Année', 'Mois', 'Dépenses Moyennes', 'Nombre de Clients']
                    
                    # Créer une date combinée pour le graphique
                    temp_df['Date'] = pd.to_datetime(temp_df['Année'].astype(str) + '-' + temp_df['Mois'].astype(str) + '-01')
                    temp_df = temp_df.sort_values('Date')
                    
                    # Graphique des dépenses moyennes par mois
                    fig = px.line(
                        temp_df,
                        x='Date',
                        y='Dépenses Moyennes',
                        title="Évolution des Dépenses Moyennes par Mois",
                        markers=True
                    )
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title="Date",
                        yaxis_title="Dépenses Moyennes",
                        xaxis=dict(tickformat="%b %Y")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graphique du nombre de clients par mois
                    fig = px.bar(
                        temp_df,
                        x='Date',
                        y='Nombre de Clients',
                        title="Évolution du Nombre de Clients par Mois",
                        color_discrete_sequence=[palette[1]]
                    )
                    fig.update_layout(
                        title_font_size=18,
                        xaxis_title="Date",
                        yaxis_title="Nombre de Clients",
                        xaxis=dict(tickformat="%b %Y")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribution des clients par mois
                month_counts = df['Mois'].value_counts().sort_index().reset_index()
                month_counts.columns = ['Mois', 'Nombre de Clients']
                
                # Convertir les numéros de mois en noms
                month_names = {
                    1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
                    7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
                }
                month_counts['Nom du Mois'] = month_counts['Mois'].map(month_names)
                
                fig = px.bar(
                    month_counts,
                    x='Nom du Mois',
                    y='Nombre de Clients',
                    title="Distribution des Clients par Mois d'Inscription",
                    color='Nom du Mois',
                    category_orders={"Nom du Mois": [month_names[i] for i in range(1, 13) if i in month_counts['Mois'].values]}
                )
                fig.update_layout(
                    title_font_size=18,
                    xaxis_title="Mois",
                    yaxis_title="Nombre de Clients"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Erreur lors de l'analyse temporelle: {str(e)}")
        
        else:
            st.warning("Les données temporelles ne sont pas disponibles dans le jeu de données.")
    
    elif analysis_type == "Segmentation RFM":
        st.subheader("Segmentation RFM (Récence, Fréquence, Montant)")
        
        # Vérifier si les variables nécessaires sont disponibles
        if all(col in df.columns for col in ['Recency', 'TotalAchats', 'TotalDepenses']):
            # Créer les scores RFM
            # Pour la Récence, une valeur plus basse est meilleure (plus récent)
            df['R_Score'] = pd.qcut(df['Recency'], q=5, labels=range(5, 0, -1))
            
            # Pour la Fréquence, une valeur plus élevée est meilleure
            df['F_Score'] = pd.qcut(df['TotalAchats'], q=5, labels=range(1, 6))
            
            # Pour le Montant, une valeur plus élevée est meilleure
            df['M_Score'] = pd.qcut(df['TotalDepenses'], q=5, labels=range(1, 6))
            
            # Convertir en entiers
            df['R_Score'] = df['R_Score'].astype(int)
            df['F_Score'] = df['F_Score'].astype(int)
            df['M_Score'] = df['M_Score'].astype(int)
            
            # Calculer le score RFM combiné
            df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
            
            # Créer les segments en fonction du score RFM
            def rfm_segment(x):
                if x >= 13:
                    return "Champions"
                elif 10 <= x < 13:
                    return "Clients Fidèles"
                elif 7 <= x < 10:
                    return "Clients Potentiels"
                elif 5 <= x < 7:
                    return "À Risque"
                else:
                    return "Clients Perdus"
            
            df['RFM_Segment'] = df['RFM_Score'].apply(rfm_segment)
            
            # Distribution des segments RFM
            segment_counts = df['RFM_Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Nombre de Clients']
            
            # Définir l'ordre des segments
            segment_order = ["Champions", "Clients Fidèles", "Clients Potentiels", "À Risque", "Clients Perdus"]
            
            fig = px.bar(
                segment_counts,
                x='Segment', y='Nombre de Clients',
                title="Distribution des Clients par Segment RFM",
                color='Segment',
                category_orders={"Segment": segment_order}
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Segment RFM",
                yaxis_title="Nombre de Clients"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualisation des segments dans l'espace RFM
            fig = px.scatter_3d(
                df, x='R_Score', y='F_Score', z='M_Score',
                color='RFM_Segment',
                opacity=0.7,
                category_orders={"RFM_Segment": segment_order},
                title="Segments RFM dans l'Espace 3D"
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Récence (R)",
                    yaxis_title="Fréquence (F)",
                    zaxis_title="Montant (M)"
                ),
                title_font_size=18
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Caractéristiques des segments
            st.subheader("Caractéristiques des Segments RFM")
            
            # Agréger les statistiques par segment
            segment_stats = df.groupby('RFM_Segment').agg({
                'Recency': 'mean',
                'TotalAchats': 'mean',
                'TotalDepenses': 'mean',
                'Income': 'mean' if 'Income' in df.columns else 'count',
                'Age': 'mean' if 'Age' in df.columns else 'count'
            }).reset_index()
            
            # Formater le tableau pour l'affichage
            segment_stats.columns = ['Segment RFM', 'Récence Moyenne', 'Achats Moyens', 'Dépenses Moyennes', 
                                   'Revenu Moyen' if 'Income' in df.columns else 'Nombre', 
                                   'Âge Moyen' if 'Age' in df.columns else 'Nombre']
            
            # Appliquer un style au tableau
            st.write(segment_stats.style.background_gradient(cmap='YlGnBu', subset=segment_stats.columns[1:]))
            
            # Heatmap des scores moyens R, F, M par segment
            segment_scores = df.groupby('RFM_Segment').agg({
                'R_Score': 'mean',
                'F_Score': 'mean',
                'M_Score': 'mean'
            }).reset_index()
            
            # Convertir en format long pour la heatmap
            segment_scores_long = pd.melt(
                segment_scores, 
                id_vars='RFM_Segment',
                value_vars=['R_Score', 'F_Score', 'M_Score'],
                var_name='Score Type',
                value_name='Score Moyen'
            )
            
            fig = px.density_heatmap(
                segment_scores_long,
                x='RFM_Segment',
                y='Score Type',
                z='Score Moyen',
                title="Heatmap des Scores RFM par Segment",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                title_font_size=18,
                xaxis_title="Segment RFM",
                yaxis_title="Type de Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations par segment
            st.subheader("Recommandations Marketing par Segment RFM")
            
            rfm_recommendations = {
                "Champions": [
                    "Proposer des programmes de fidélité exclusifs",
                    "Offrir des récompenses pour les références",
                    "Solliciter des témoignages et avis",
                    "Tester de nouveaux produits avec ce segment"
                ],
                "Clients Fidèles": [
                    "Encourager les achats plus fréquents",
                    "Proposer des ventes croisées et montées en gamme",
                    "Inviter à des événements VIP",
                    "Offrir des programmes de fidélité premium"
                ],
                "Clients Potentiels": [
                    "Augmenter la fréquence d'achat avec des incitatifs",
                    "Envoyer des rappels d'achat personnalisés",
                    "Proposer des offres à durée limitée",
                    "Partager des témoignages de clients satisfaits"
                ],
                "À Risque": [
                    "Réactiver avec des offres spéciales",
                    "Proposer des remises pour retour",
                    "Demander des commentaires sur l'expérience précédente",
                    "Lancer des campagnes de ré-engagement"
                ],
                "Clients Perdus": [
                    "Offrir des réductions importantes pour réactiver",
                    "Présenter de nouveaux produits/services",
                    "Mener des enquêtes pour comprendre les raisons du départ",
                    "Proposer une offre de bienvenue pour un retour"
                ]
            }
            
            # Sélectionner un segment pour voir les recommandations
            selected_rfm_segment = st.selectbox(
                "Sélectionner un segment pour voir les recommandations",
                options=list(rfm_recommendations.keys())
            )
            
            # Afficher les recommandations pour le segment sélectionné
            st.markdown(f"<div style='background-color:#f0f0f0;padding:15px;border-radius:5px;'>", unsafe_allow_html=True)
            for rec in rfm_recommendations[selected_rfm_segment]:
                st.markdown(f"<p>✓ {rec}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.warning("Les variables nécessaires pour la segmentation RFM ne sont pas disponibles dans le jeu de données.")
            
# Ajouter un pied de page
st.markdown("---")
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center;'>
    <div>
        <p><strong>PersonaInsight™</strong> | Développé pour l'Analyse de Personnalité Client | 2024-2025</p>
    </div>
    <div>
        <p>Pour plus d'informations et de support: <a href="mailto:tbahritimohamed8@gmail.com">tbahritimohamed8@gmail.com</a></p>
    </div>
</div>
""", unsafe_allow_html=True)