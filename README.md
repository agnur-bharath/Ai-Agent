# PantryChef — Simple AI Agent

PantryChef is a minimal CLI AI agent that suggests recipes, meal plans, and shopping lists based on ingredients you have. It's a demonstration scaffold that uses the OpenAI API.

Files created

- `agent.py` — main CLI and interactive REPL
- `requirements.txt` — Python dependencies
- `.gitignore` — common ignores

Quick setup (Windows PowerShell)

```powershell
# Create and activate a virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key (PowerShell)
$env:OPENAI_API_KEY = 'sk-REPLACE_WITH_YOURS'

# Run interactive REPL
python .\agent.py --interactive

# Or run a one-off query with ingredients
python .\agent.py --ingredients "eggs, milk, flour, tomato, onion"
```

Usage notes

- Commands inside REPL:
  - `/recipe <ingredients>` — request recipes for given comma-separated ingredients
  - `/plan <days>` — create a meal plan for N days
  - `/shopping <items>` — expand to a shopping list
  - `/help` — show help
  - `/exit` or `/quit` — quit

Customization

- To change the model, pass `--model` (defaults to `gpt-3.5-turbo`).
- You can extend `agent.py` to add embeddings, caching, or a vector index for a larger knowledge base.

Next steps I can do for you

- Add example unit tests or a simple integration test harness
- Add embeddings-based local doc search (requires more dependencies & a small indexer)
- Add Windows installer script or publish to PyPI

If you'd like any of those, tell me which one and I'll continue.

Ingesting docs, persistence, and tests

- To ingest markdown docs into the local embedding store (creates `store.db`):

```powershell
python .\agent.py --ingest .\docs
```

- Query using local docs as context:

```powershell
python .\agent.py --query "how do I make a quick tomato pasta?" --topk 3
```

- Use the REPL (supports `/ingest` and `/query`, plus caching for repeated prompts):

```powershell
python .\agent.py --interactive
# inside REPL:
# /ingest docs
# /query best quick dinners
# /recipe eggs, milk, cheese
```

- Run tests:

```powershell
pytest -q
```

Notes about persistence and caching

- The embedding store persists into `store.db` (SQLite). Ingested markdown files create per-chunk embeddings and are used for retrieval.
- Chat completions are cached in the same database to reduce repeated API calls for identical prompts.

Optional Faiss indexing

- If you install `faiss-cpu` (listed in `requirements.txt`), the embedding store will build an internal Faiss index for faster nearest-neighbor retrieval. If Faiss is not available, the store falls back to a SQLite-backed brute-force search.

PR automation

- Push to a branch matching `feature/*` and the workflow `.github/workflows/create-pr.yml` will automatically open a pull request against `main` using the repository's GitHub token.
