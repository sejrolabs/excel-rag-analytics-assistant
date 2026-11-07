import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from src.llm.factory import get_llm
from src.query_engine.query_router import QueryRouter
from src.evaluation.metrics import exact_match


def main():
    benchmark_path = ROOT_DIR / "data" / "sample" / "benchmark.json"
    if not benchmark_path.exists():
        print(f"Fichier benchmark introuvable : {benchmark_path}")
        return

    test_cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print("=" * 70)
    print("ÉVALUATION DU ROUTEUR D'INTENTIONS")
    print(f"Nombre de cas de test : {len(test_cases)}")
    print("=" * 70)

    try:
        llm = get_llm()
    except Exception as exc:
        print(f"Erreur d'initialisation du LLM : {exc}")
        return

    router = QueryRouter(llm)

    correct = 0
    predictions = []
    references = []

    print(f"\n{'Question':<50} | {'Attendu':<16} | {'Prédit':<16} | {'Résultat'}")
    print("-" * 95)

    for case in test_cases:
        question = case["question"]
        expected = case["intent"]

        decision = router.route(question)
        predicted = decision.intent

        is_match = exact_match(predicted, expected)
        if is_match:
            correct += 1
            status = "✅ OK"
        else:
            status = "❌ FAIL"

        predictions.append(predicted)
        references.append(expected)

        trunc_q = (question[:47] + "...") if len(question) > 50 else question
        print(f"{trunc_q:<50} | {expected:<16} | {predicted:<16} | {status}")

    accuracy = correct / len(test_cases) if test_cases else 0.0

    print("\n" + "=" * 70)
    print(f"RÉSULTAT GLOBAL : {correct}/{len(test_cases)} ({accuracy:.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
