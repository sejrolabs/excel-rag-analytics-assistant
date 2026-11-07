# Excel RAG Analytics Assistant

Assistant conversationnel pour explorer et analyser un fichier Excel de transactions bancaires grâce à une architecture hybride combinant Text-to-SQL, recherche sémantique (dense) et recherche lexicale (BM25).

## Fonctionnalités

- **Ingestion & Validation Excel** : vérification de schéma, détection des anomalies, formats et doublons.
- **Nettoyage & Normalisation robuste** : gestion automatique des devises (€, $, £), espaces insécables et formats de dates multiples (compatible Polars 2.0+).
- **Stockage analytique DuckDB** : persistance rapide zéro-copie (`to_arrow()`) et requêtes analytiques en lecture seule.
- **Recherche hybride (Qdrant + BM25)** : fusion par *Reciprocal Rank Fusion* (RRF) combinant la recherche sémantique (`multilingual-e5-small`) et lexicale.
- **Routage d'intentions intelligent** : orientation déterministe (`SQL_QUERY`, `VECTOR_SEARCH`, `HYBRID_SEARCH`, `GENERAL_QUESTION`) avec fallback heuristique de sécurité.
- **Text-to-SQL sécurisé** : exécution en lecture seule restreinte à la table `transactions` avec validation AST (`sqlglot`).
- **Visualisations interactives & Export** : génération automatique de graphiques (barres, courbes) et export CSV sous Streamlit.
- **Double interface** : Interface web Streamlit interactive et API REST FastAPI documentée.

> **Important** : Utiliser exclusivement des données synthétiques ou anonymisées. Ne jamais committer de données bancaires sensibles ou de clés API réelles.

---

## Démarrage local

```bash
# 1. Création et activation de l'environnement virtuel
python -m venv .venv
# Windows PowerShell :
.venv\Scripts\Activate.ps1
# macOS/Linux :
source .venv/bin/activate

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Configuration des variables d'environnement
cp .env.example .env
```

Configurer `.env` avec votre fournisseur LLM (`ollama` ou `openai`).

---

## Utilisation

### 1. Lancer l'interface Streamlit (Recommandé)

```bash
streamlit run app/streamlit_app.py
```
L'interface permet de charger vos fichiers Excel, de suivre l'état de synchronisation DuckDB/Qdrant, d'indexer en un clic et de converser avec vos données.

### 2. Ou utiliser les scripts en ligne de commande

```bash
# Ingestion du fichier Excel dans DuckDB
python scripts/ingest.py --file data/raw/transactions.xlsx

# Indexation vectorielle dans Qdrant
python scripts/index_text.py
```

### 3. Lancer l'API FastAPI

```bash
uvicorn api.main:app --reload
```
Swagger disponible sur `http://127.0.0.1:8000/docs`.

---

## Tests & Évaluation

```bash
# Lancer la suite de tests unitaires
pytest tests/

# Évaluer la précision du routeur sur le benchmark de référence
python scripts/run_eval.py
```

---

## Schéma de données attendu

Colonnes recommandées : `transaction_id`, `customer_id`, `transaction_date`, `merchant`, `category`, `amount`, `transaction_type`, `city`, `country`, `payment_method`, `description`.

*Le pipeline est flexible et s'adapte automatiquement aux colonnes disponibles dans votre fichier.*

---

## Architecture du projet

- `src/ingestion` : lecture et validation Excel.
- `src/processing` : nettoyage, normalisation et enrichissement textuel.
- `src/storage` : persistance et requêtes DuckDB.
- `src/retrieval` : embeddings, vector store Qdrant, BM25 et indexeur.
- `src/query_engine` : routage d'intentions, Text-to-SQL et exécuteur sécurisé.
- `src/llm` : adaptateurs LLM (Ollama, OpenAI).
- `src/rag` : orchestrateur et générateur de réponses analytiques.
- `src/evaluation` : métriques et benchmark.
- `app/` : application Streamlit.
- `api/` : API REST FastAPI.
- `tests/` : suite de tests automatisés pytest.
