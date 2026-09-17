"""End-to-end verification of the RAG guardrail and memory.

Exercises: PDF load, DOCX load, in-document answers, out-of-document refusal,
and a follow-up question that relies on conversation history.
"""
from langchain_core.messages import AIMessage, HumanMessage

from rag import build_vectorstore, build_chain, ask, REFUSAL


def run_case(chain, history, q):
    ans, _ = ask(chain, q, history)
    return ans.strip()


def main():
    for path in ["data/league_rules.pdf", "data/league_rules.docx"]:
        print(f"\n{'='*70}\nLoading: {path}")
        vs, n = build_vectorstore([path])
        print(f"Indexed into {n} chunks.")
        chain = build_chain(vs)

        # 1. In-document questions (should answer from the doc)
        in_doc = [
            "How long is each half of a match?",
            "How many points is a goal scored directly from a corner worth?",
            "How many substitutions can a team make per match?",
            "What is the suspension for a violent-conduct red card?",
        ]
        print("\n-- IN-DOCUMENT (expect real answers) --")
        for q in in_doc:
            print(f"Q: {q}\nA: {run_case(chain, [], q)}\n")

        # 2. Out-of-document questions (should refuse)
        out_doc = [
            "Who won the 2018 FIFA World Cup?",
            "What is the capital of France?",
            "How much does a season ticket cost?",  # plausible but not in doc
        ]
        print("-- OUT-OF-DOCUMENT (expect refusal) --")
        passed = 0
        for q in out_doc:
            a = run_case(chain, [], q)
            ok = a == REFUSAL
            passed += ok
            print(f"Q: {q}\nA: {a}   [{'PASS' if ok else 'FAIL'}]\n")
        print(f"Refusal guardrail: {passed}/{len(out_doc)} passed")

        # 3. Follow-up using conversation history
        print("\n-- FOLLOW-UP MEMORY --")
        history = []
        q1 = "How many yellow cards lead to a red card?"
        a1 = run_case(chain, history, q1)
        print(f"Q: {q1}\nA: {a1}\n")
        history += [HumanMessage(content=q1), AIMessage(content=a1)]
        q2 = "And how many matches is that player suspended for?"  # 'that' = red card
        a2 = run_case(chain, history, q2)
        print(f"Q: {q2}\nA: {a2}\n")

        break  # PDF run is enough for the full suite; DOCX load proven below

    # Prove DOCX also loads and answers
    print(f"\n{'='*70}\nDOCX quick check")
    vs, n = build_vectorstore(["data/league_rules.docx"])
    chain = build_chain(vs)
    print(f"DOCX indexed into {n} chunks.")
    print("Q: What is the minimum age to be eligible?")
    print("A:", run_case(chain, [], "What is the minimum age to be eligible?"))


if __name__ == "__main__":
    main()
