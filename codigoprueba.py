import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import deque, defaultdict
import nltk
import re

# Descargar recursos necesarios para tokenización
nltk.download('punkt')
nltk.download('stopwords')

# Procesar la consulta y extraer palabras clave
def extract_keywords(query_text):
    stopwords_spanish = set(stopwords.words('spanish'))
    stopwords_english = set(stopwords.words('english'))
    all_stopwords = stopwords_spanish | stopwords_english
    tokens = word_tokenize(query_text.lower())
    return [word for word in tokens if word.isalnum() and word not in all_stopwords]

# Validar un token como término aceptable
def is_valid_token(token):
    return len(token) >= 4 and token.isalpha()

def main():
    # Solicitar parámetros al usuario
    starting_url = input("Ingrese la URL inicial para el rastreo: ").strip()
    max_links_per_page = int(input("Número máximo de enlaces a seguir por página: "))
    max_crawl_depth = int(input("Profundidad máxima del rastreo: "))
    search_query = input("¿Qué deseas buscar? ").strip()

    # Procesar la consulta del usuario
    keywords = extract_keywords(search_query)
    print(f"Palabras clave procesadas: {keywords}")

    # Inicializar estructuras de datos
    term_occurrences = defaultdict(int)
    processed_urls = set()
    pending_urls = deque([(starting_url, 0)])
    document_texts = []
    document_urls = []

    while pending_urls:
        current_url, current_depth = pending_urls.popleft()

        if current_depth > max_crawl_depth or current_url in processed_urls:
            continue
        processed_urls.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extraer texto útil de la página
            for element in soup(["script", "style"]):
                element.decompose()
            raw_text = soup.get_text(separator=' ')
            raw_text = re.sub(r'\s+', ' ', raw_text).strip()

            # Tokenizar texto extraído
            tokens = word_tokenize(raw_text.lower())
            if current_url != starting_url:
                document_texts.append(tokens)
                document_urls.append(current_url)

            # Actualizar frecuencias de términos
            for token in tokens:
                if is_valid_token(token):
                    term_occurrences[token] += 1

            # Encontrar enlaces relevantes en la página
            all_links = [
                (urljoin(current_url, a['href']), a.get_text(strip=True).lower())
                for a in soup.find_all('a', href=True)
            ]
            related_links = [
                urljoin(current_url, href) for href, anchor_text in all_links
                if urlparse(urljoin(current_url, href)).netloc == urlparse(starting_url).netloc
                and any(kw in anchor_text or kw in href.lower() for kw in keywords)
                and urljoin(current_url, href) not in processed_urls
            ]

            for link in related_links[:max_links_per_page]:
                pending_urls.append((link, current_depth + 1))

        except Exception as err:
            print(f"No se pudo procesar {current_url}: {err}")

    # Calcular puntuaciones BM25 para las páginas recolectadas
    ranking_model = BM25Okapi(document_texts)
    relevancy_scores = ranking_model.get_scores(keywords)

    # Ordenar resultados por relevancia
    sorted_results = sorted(set(zip(document_urls, relevancy_scores)), key=lambda x: x[1], reverse=True)

    # Mostrar resultados relevantes
    print("\nPáginas relevantes según BM25:")
    for page_url, score in sorted_results:
        print(f"{page_url}: Puntuación {score:.2f}")

    # Guardar términos y frecuencias en un archivo
    with open("frecuencias_terminos.txt", "w", encoding="utf-8") as output_file:
        for term, frequency in sorted(term_occurrences.items(), key=lambda x: x[1], reverse=True):
            output_file.write(f"{term}: {frequency}\n")

    print("\nFrecuencias de términos almacenadas en 'frecuencias_terminos.txt'.")

if __name__ == '__main__':
    main()
