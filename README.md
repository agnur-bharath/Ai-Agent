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
