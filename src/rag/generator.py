import json
from src.llm.base import LLMProvider


ANSWER_PROMPT = """Tu es un analyste financier expert et rigoureux spécialisé dans l'analyse de transactions bancaires.

DIRECTIVES D'ANALYSE :
1. Réponds UNIQUEMENT et STRICTEMENT sur la base des données fournies ci-dessous (Evidence).
2. Ne jamais inventer de chiffres, de dates ou de commerçants. Si l'information n'est pas dans l'évidence, indique-le clairement.
3. Formate les montants monétaires de façon élégante en euros (ex: 1 250,50 €).
4. Si la route est "HYBRID_SEARCH" ou "VECTOR_SEARCH", rappelle avec nuance que les résultats correspondent aux transactions les plus pertinentes trouvées (top-K candidats) et ne constituent pas obligatoirement un total exhaustif de la base.
5. Structure ta réponse en français avec clarté, concision et professionnalisme (utilise des puces ou du gras pour les chiffres clés si pertinent).

Question de l'utilisateur :
{question}

Stratégie de résolution :
{route}

Données extraites (Evidence) :
{evidence}

Réponse analytique en français :
"""


class AnswerGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate(self, question: str, route: str, evidence) -> str:
        try:
            evidence_str = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
        except Exception:
            evidence_str = str(evidence)

        # Tronquer proprement si nécessaire pour respecter la fenêtre de contexte
        if len(evidence_str) > 12000:
            evidence_str = evidence_str[:12000] + "\n... [Résultats partiels tronqués pour la synthèse]"

        return self.llm.generate(
            ANSWER_PROMPT.format(
                question=question,
                route=route,
                evidence=evidence_str,
            ),
            temperature=0.1,
        ).strip()
