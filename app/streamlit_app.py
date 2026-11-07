import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.config import settings
from src.ingestion.excel_loader import load_excel
from src.ingestion.data_validator import validate_dataframe
from src.processing.data_processor import clean_dataframe
from src.storage.duckdb_store import DuckDBStore
from src.retrieval.vector_store import VectorStore
from src.retrieval.indexer import index_transactions
from src.rag.assistant import RAGAssistant

st.set_page_config(
    page_title="Excel RAG Analytics Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_assistant() -> RAGAssistant:
    """Instancie l'assistant de manière persistante en cache."""
    return RAGAssistant()


def get_counts() -> tuple[int, int]:
    """Retourne le nombre de transactions dans DuckDB et de vecteurs dans Qdrant."""
    duck_count = 0
    qdrant_count = 0
    try:
        if Path(settings.duckdb_path).exists():
            duck_count = DuckDBStore(settings.duckdb_path).count_transactions()
    except Exception:
        duck_count = 0

    try:
        vs = VectorStore()
        qdrant_count = vs.count()
        vs.close()
    except Exception:
        qdrant_count = 0

    return duck_count, qdrant_count


def render_sources_and_visualizations(sources: list[dict], unique_key: str):
    """Génère automatiquement tableaux interactifs, graphiques et exports pour les résultats."""
    if not sources:
        return

    for idx, source in enumerate(sources):
        src_type = source.get("type")

        # Cas 1 : Données issues de DuckDB (SQL)
        if src_type == "duckdb" and "rows" in source:
            rows = source["rows"]
            if not rows or not isinstance(rows, list):
                continue

            df = pd.DataFrame(rows)
            if df.empty:
                continue

            # Si le résultat a plusieurs lignes, proposer tableau + graphique
            if len(df) > 1:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                dimension_cols = [c for c in df.columns if c not in numeric_cols]

                tab_chart, tab_table = st.tabs(["📊 Graphique interactif", "📋 Données tabulaires"])

                with tab_chart:
                    if numeric_cols and dimension_cols:
                        x_axis = dimension_cols[0]
                        y_axis = numeric_cols[0]
                        st.caption(f"Visualisation : **{y_axis}** en fonction de **{x_axis}**")
                        st.bar_chart(df, x=x_axis, y=y_axis, use_container_width=True)
                    elif numeric_cols:
                        st.line_chart(df[numeric_cols], use_container_width=True)
                    else:
                        st.info("Aucune colonne numérique pour générer un graphique.")

                with tab_table:
                    st.dataframe(df, use_container_width=True)
            else:
                # Une seule ligne (ex: COUNT(*), total, etc.)
                st.dataframe(df, use_container_width=True)

            # Bouton de téléchargement CSV
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger ce résultat (CSV)",
                data=csv_data,
                file_name="resultat_transactions.csv",
                mime="text/csv",
                key=f"dl_{unique_key}_{idx}",
            )

        # Cas 2 : Données issues de la recherche sémantique ou hybride
        elif src_type in ("dense_vector_search", "hybrid_retrieval") and "records" in source:
            records = source["records"]
            if records and isinstance(records, list):
                with st.expander(f"🔍 Transactions candidates extraites ({len(records)} résultats pertinents)", expanded=True):
                    preview_data = []
                    for r in records:
                        item = dict(r)
                        score = item.pop("score", None)
                        row_dict = {}
                        if score is not None:
                            row_dict["Pertinence"] = f"{score:.3f}" if isinstance(score, float) else score
                        for k in ["merchant", "category", "amount", "transaction_date", "description"]:
                            if k in item:
                                row_dict[k] = item[k]
                        preview_data.append(row_dict)

                    if preview_data:
                        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)


st.title("📊 Excel RAG Analytics Assistant")
st.caption("Assistant conversationnel analytique hybride : Text-to-SQL + Recherche Sémantique (Qdrant) + BM25")

# Initialisation de l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Barre latérale : Données & Indexation
with st.sidebar:
    st.header("1. Données & Indexation")

    duck_count, qdrant_count = get_counts()

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("DuckDB (SQL)", f"{duck_count:,}")
    col_s2.metric("Qdrant (Vecteurs)", f"{qdrant_count:,}")

    if duck_count != qdrant_count and duck_count > 0:
        st.warning("⚠️ L'index vectoriel Qdrant n'est pas synchronisé avec la base DuckDB.")

    uploaded = st.file_uploader("Charger un nouveau fichier Excel", type=["xlsx", "xls"])
    auto_index = st.checkbox("Indexer automatiquement pour le RAG sémantique après l'import", value=True)

    if uploaded:
        if st.button("Importer et charger dans DuckDB", type="primary", use_container_width=True):
            with st.spinner("Lecture et validation du fichier Excel..."):
                try:
                    temp_path = Path("data/raw/uploaded_transactions.xlsx")
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path.write_bytes(uploaded.getvalue())

                    raw_df = load_excel(temp_path)
                    report = validate_dataframe(raw_df)
                    df = clean_dataframe(raw_df)

                    store = DuckDBStore(settings.duckdb_path)
                    store.replace_transactions(df)
                    st.cache_resource.clear()

                    st.success(f"✅ {df.height:,} transactions enregistrées dans DuckDB.")

                    if auto_index:
                        progress_bar = st.progress(0.0, text="Indexation vectorielle dans Qdrant...")

                        def update_progress(indexed: int, total: int):
                            progress_bar.progress(indexed / total, text=f"Indexation Qdrant : {indexed:,}/{total:,}")

                        count_idx = index_transactions(progress_callback=update_progress)
                        progress_bar.empty()
                        st.success(f"✅ {count_idx:,} vecteurs indexés dans Qdrant.")

                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur lors de l'ingestion : {exc}")

    st.divider()

    # Bouton d'indexation manuelle
    st.subheader("Synchronisation Vectorielle")
    if duck_count > 0:
        if st.button("Reconstruire l'index sémantique Qdrant", use_container_width=True):
            progress_bar = st.progress(0.0, text="Démarrage de l'indexation...")

            def update_progress(indexed: int, total: int):
                progress_bar.progress(indexed / total, text=f"Indexation Qdrant : {indexed:,}/{total:,}")

            try:
                count_idx = index_transactions(progress_callback=update_progress)
                progress_bar.empty()
                st.cache_resource.clear()
                st.success(f"✅ {count_idx:,} vecteurs synchronisés avec succès !")
                st.rerun()
            except Exception as exc:
                st.error(f"Erreur pendant l'indexation : {exc}")

    st.divider()
    st.subheader("Configuration active")
    st.text(f"Fournisseur LLM : {settings.llm_provider}")
    if settings.llm_provider == "ollama":
        st.text(f"Modèle Ollama : {settings.ollama_model}")
    else:
        st.text(f"Modèle OpenAI : {settings.openai_model}")

    if st.button("Effacer l'historique du chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Zone d'interrogation
st.subheader("2. Posez vos questions à vos données")

# Suggestions rapides
suggested = [
    "Combien y a-t-il de transactions au total ?",
    "Quel est le montant moyen par catégorie ?",
    "Trouve les descriptions évoquant des voyages ou billets de train",
    "Donne-moi le top 5 des commerçants par dépenses",
]

col_btns = st.columns(len(suggested))
selected_suggestion = None
for i, prompt_text in enumerate(suggested):
    if col_btns[i].button(prompt_text, key=f"sug_{i}"):
        selected_suggestion = prompt_text

# Affichage des messages passés
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            if meta.get("route"):
                st.caption(f"🎯 **Stratégie :** `{meta['route']}`")
            if meta.get("sql"):
                with st.expander("Requête SQL exécutée"):
                    st.code(meta["sql"], language="sql")
            if meta.get("sources"):
                render_sources_and_visualizations(meta["sources"], unique_key=f"hist_{i}")

# Saisie utilisateur
chat_input = st.chat_input("Ex: Quel est le montant moyen par catégorie ?")
question_to_ask = selected_suggestion or chat_input

if question_to_ask:
    st.session_state.messages.append({"role": "user", "content": question_to_ask})
    with st.chat_message("user"):
        st.markdown(question_to_ask)

    with st.chat_message("assistant"):
        with st.spinner("Analyse et extraction des données en cours..."):
            try:
                assistant = get_assistant()
                result = assistant.ask(question_to_ask)

                st.markdown(result["answer"])
                st.caption(f"🎯 **Stratégie :** `{result['route']}`")

                if result.get("sql"):
                    with st.expander("Requête SQL exécutée"):
                        st.code(result["sql"], language="sql")

                if result.get("sources"):
                    render_sources_and_visualizations(result["sources"], unique_key="current")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "route": result["route"],
                        "sql": result.get("sql"),
                        "sources": result.get("sources"),
                    }
                })

            except Exception as exc:
                err_msg = f"❌ Impossible de traiter la question : {exc}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                })
