import streamlit as st
import csv
import logging
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import re
from urllib.parse import urlparse
from googlesearch import search
from collections import Counter
import pandas as pd

# Configuration du logging
logging.basicConfig(filename='analyse_erreurs.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Téléchargement des ressources NLTK si nécessaire
try:
    stopwords.words('french')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Chargement du modèle Sentence Transformer
model_st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def nettoyer_texte(texte):
    if texte:
        return texte.replace("\n", " ").replace("\r", " ")
    return ""


def extraire_mots_cle(texte, mots_cles):
    mots = nltk.word_tokenize(texte)
    mots = [mot for mot in mots if mot.isalnum() and mot not in stopwords.words('french')]
    compteur = Counter(mots)
    mots_cles_principaux = [mot for mot in mots_cles.split() if mot in compteur]
    return compteur, mots_cles_principaux


def calculer_score_semantique(url, mots_cles):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        for nav in soup.find_all('nav'):
            nav.decompose()

        title = soup.find('title').text.strip().lower() if soup.find('title') else ""
        h1 = soup.find('h1').text.strip().lower() if soup.find('h1') else ""

        texte_page = soup.get_text(strip=True).replace('\n', ' ').lower()
        mots = nltk.word_tokenize(texte_page)
        mots = [mot for mot in mots if mot.isalnum() and mot not in stopwords.words('french')]
        texte_page_nettoye = " ".join(mots)

        nombre_mots = len(mots)

        embeddings_mots_cles = model_st.encode(mots_cles.lower())

        vectorizer = TfidfVectorizer()
        mots_cles_preprocesses = vectorizer.fit_transform([mots_cles.lower()])
        title_preprocesses = vectorizer.transform([title])
        h1_preprocesses = vectorizer.transform([h1])

        mots_cles_tfidf_dense = torch.tensor(mots_cles_preprocesses.toarray()).float()
        title_tfidf_dense = torch.tensor(title_preprocesses.toarray()).float()
        h1_tfidf_dense = torch.tensor(h1_preprocesses.toarray()).float()

        score_title_tfidf = util.pytorch_cos_sim(title_tfidf_dense, mots_cles_tfidf_dense).item() if title else 0
        score_h1_tfidf = util.pytorch_cos_sim(h1_tfidf_dense, mots_cles_tfidf_dense).item() if h1 else 0

        title_embedding = model_st.encode(title)
        h1_embedding = model_st.encode(h1)

        score_title_bert = util.pytorch_cos_sim(title_embedding, embeddings_mots_cles).item() if title else 0
        score_h1_bert = util.pytorch_cos_sim(h1_embedding, embeddings_mots_cles).item() if h1 else 0

        score_title = (score_title_tfidf + score_title_bert) / 2
        score_h1 = (score_h1_tfidf + score_h1_bert) / 2

        score_body = util.pytorch_cos_sim(model_st.encode(texte_page_nettoye), embeddings_mots_cles).item()
        score_total = (score_title + score_h1 + score_body) / 3

        return score_total, score_title, score_h1, score_body, nombre_mots, title, h1, texte_page

    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de requête HTTP pour l'URL {url}: {e}")
        return None, None, None, None, None, None, None, None
    except Exception as e:
        logging.error(f"Erreur lors du calcul du score sémantique pour l'URL {url}: {e}")
        return None, None, None, None, None, None, None, None



def analyser_csv(chemin_fichier, keyword, custom_url=None):
    urls = list(search(keyword, num_results=10))
    if custom_url:
        custom_url = f"http://{custom_url}" if not custom_url.startswith(('http://', 'https://')) else custom_url
        if custom_url not in urls:
            urls.append(custom_url)

    results = []

    for index, url in enumerate(urls, start=1):
        score_total, score_title, score_h1, score_body, nombre_mots, title, h1, texte_page = calculer_score_semantique(url, keyword)

        title = title.replace("\n", " ").replace("\r", " ") if title else ""
        h1 = h1.replace("\n", " ").replace("\r", " ") if h1 else ""


        results.append({
            'Position SERP': index,
            'URL': url,
            'Mot-clé': keyword,
            'Score Total': score_total,
            'Score Title': score_title,
            'Title': title,
            'Score H1': score_h1,
            'H1': h1,
            'Score Body': score_body,
            'Nombre de Mots': nombre_mots,
            'Recommandations': ""
        })


    if custom_url:
        score_total, score_title, score_h1, score_body, nombre_mots, title, h1, texte_page = calculer_score_semantique(custom_url, keyword)
        title = title.replace("\n", " ").replace("\r", " ") if title else ""
        h1 = h1.replace("\n", " ").replace("\r", " ") if h1 else ""

        compteur, mots_cles_principaux = extraire_mots_cle(texte_page, keyword)

        recommandations = []
        if len(mots_cles_principaux) < 2:
            recommandations.append("Ajouter plus de variantes de mots-clés dans le texte pour mieux couvrir le champ sémantique.")
        if compteur[keyword] < 3:
            recommandations.append(f"Le mot-clé '{keyword}' apparaît peu dans le texte. Augmentez sa fréquence.")

        termes_associes = [mot for mot, count in compteur.items() if count > 2 and mot != keyword]
        if termes_associes:
            recommandations.append(f"Ajoutez des termes associés comme : {', '.join(termes_associes)}.")

        for result in results:
            if result['URL'] == custom_url:
                result['Recommandations'] = "; ".join(recommandations)
                break


    with open(chemin_fichier, 'w', newline='', encoding='utf-8') as fichier_sortie:
        ecrivain_csv = csv.writer(fichier_sortie, quoting=csv.QUOTE_ALL)
        ecrivain_csv.writerow(['Position SERP', 'URL', 'Mot-clé', 'Score Total', 'Score Title', 'Title', 'Score H1', 'H1', 'Score Body', 'Nombre de Mots', 'Recommandations'])
        for result in results:
            ecrivain_csv.writerow([result.get('Position SERP'), result.get('URL'), result.get('Mot-clé'),
                                  result.get('Score Total'), result.get('Score Title'), result.get('Title'),
                                  result.get('Score H1'), result.get('H1'), result.get('Score Body'),
                                  result.get('Nombre de Mots'), result.get('Recommandations', '')])





# Interface Streamlit
st.title("Analyse Sémantique de Pages Web")

keyword = st.text_input("Entrez le mot-clé à analyser :")
custom_url = st.text_input("Entrez une URL personnalisée (facultatif):")

if st.button("Analyser"):
    if not keyword:
        st.error("Veuillez entrer un mot-clé.")
    else:
        with st.spinner('Analyse en cours...'):
            analyser_csv('resultats_analyse.csv', keyword, custom_url)

        # Afficher les résultats dans un tableau Streamlit
        try:
            with open('resultats_analyse.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                df = pd.DataFrame(reader, columns=['Position SERP', 'URL', 'Mot-clé', 'Score Total', 'Score Title', 'Title', 'Score H1', 'H1', 'Score Body', 'Nombre de Mots', 'Recommandations'])
            st.dataframe(df, use_container_width=True)

            # Option de téléchargement du CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les résultats en CSV",
                data=csv,
                file_name='resultats_analyse.csv',
                mime='text/csv',
            )

        except FileNotFoundError:
            st.error("Erreur lors de l'analyse. Veuillez vérifier les logs.")