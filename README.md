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

# PantryChef — Simple AI Agent

PantryChef is a minimal CLI AI agent that suggests recipes, meal plans, and shopping lists based on available ingredients. It is a demonstration scaffold that integrates with the OpenAI API.

Files

- `agent.py` — main CLI and interactive REPL
- `requirements.txt` — Python dependencies
- `docs/` — sample recipe and documentation

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

Usage

- `/recipe <ingredients>` — request recipes for given comma-separated ingredients
- `/plan <days>` — create a meal plan for N days
- `/shopping <items>` — create a shopping list from items
- `/help` — show help
- `/exit` or `/quit` — quit the REPL

Customization

- To change the model, pass `--model` (defaults to `gpt-3.5-turbo`).
- The agent can be extended with embeddings, caching, or a vector index for larger datasets.

Ingesting docs, persistence, and tests

- Ingest markdown docs into the local embedding store (creates `store.db`):

```powershell
python .\agent.py --ingest .\docs
```

- Query using local docs as context:

```powershell
python .\agent.py --query "how do I make a quick tomato pasta?" --topk 3
```

- Run tests:

```powershell
pytest -q
```

Persistence and caching notes

- The embedding store persists into `store.db` (SQLite). Ingested markdown files are chunked and stored for retrieval.
- Chat completions may be cached to reduce repeated API calls for identical prompts.

Optional Faiss indexing

- Installing `faiss-cpu` (if desired) enables a Faiss index for faster nearest-neighbor retrieval. Without Faiss, the store falls back to a SQLite-backed search.

CI / PR automation

- Follow the repository's branching conventions (for example, `feature/*`) to trigger existing automation workflows.

License

This repository does not include a license file. Add a `LICENSE` to define project licensing.

