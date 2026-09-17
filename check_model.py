"""Step 0: Verify the pinned Gemini models are LIVE for this API key
before we build anything on top of them. Models get retired without notice.

Lesson learned the hard way: list_models() can still SHOW a model that is
closed to new users (returns 404 on use). So this does a REAL generate call
and a REAL embed call -- the only true liveness test.

Run:  python check_model.py
"""
import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

if not API_KEY:
    sys.exit("ERROR: GOOGLE_API_KEY missing from .env")

genai.configure(api_key=API_KEY)


def check_generation(name):
    """Actually call the model. Returns (ok, message)."""
    try:
        model = genai.GenerativeModel(name)
        resp = model.generate_content("Reply with the single word: ok")
        return True, resp.text.strip()
    except Exception as e:
        return False, str(e).splitlines()[0]


def check_embedding(name):
    try:
        genai.embed_content(model=f"models/{name}", content="liveness test")
        return True, "embedding returned"
    except Exception as e:
        return False, str(e).splitlines()[0]


def list_available():
    print("\n  Models this key CAN currently use:")
    for m in genai.list_models():
        methods = m.supported_generation_methods
        if "generateContent" in methods or "embedContent" in methods:
            print(f"    - {m.name.split('/')[-1]}  ({', '.join(methods)})")


def main():
    print("Verifying pinned models with REAL API calls...\n")

    gen_ok, gen_msg = check_generation(GEN_MODEL)
    emb_ok, emb_msg = check_embedding(EMBED_MODEL)

    print(f"  generation  {GEN_MODEL}: {'[LIVE]' if gen_ok else '[DEAD]'}  {gen_msg}")
    print(f"  embeddings  {EMBED_MODEL}: {'[LIVE]' if emb_ok else '[DEAD]'}  {emb_msg}")

    if gen_ok and emb_ok:
        print("\nAll pinned models are LIVE. Safe to proceed.")
        sys.exit(0)

    list_available()
    print("\nFix GEN_MODEL / EMBED_MODEL in .env, then re-run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
