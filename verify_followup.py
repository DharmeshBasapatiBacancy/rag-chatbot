"""Minimal test: prove conversation memory + refusal on the flash-lite model."""
from langchain_core.messages import AIMessage, HumanMessage
from rag import build_vectorstore, build_chain, ask

vs, _ = build_vectorstore(["data/league_rules.pdf"])
chain = build_chain(vs)

history = []
q1 = "How many yellow cards lead to a red card?"
a1, _ = ask(chain, q1, history)
print(f"Q1: {q1}\nA1: {a1}\n")
history += [HumanMessage(content=q1), AIMessage(content=a1)]

# 'that player' only resolves via chat history -> tests memory + rewriting
q2 = "And how many matches is that player suspended for?"
a2, _ = ask(chain, q2, history)
print(f"Q2: {q2}\nA2: {a2}\n")

# quick out-of-doc refusal on this model too
q3 = "What is the stadium's address?"
a3, _ = ask(chain, q3, [])
print(f"Q3: {q3}\nA3: {a3}")
