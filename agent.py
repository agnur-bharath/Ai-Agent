import os
import argparse
import sys
import json
import hashlib
import time

from dotenv import load_dotenv
import openai
import numpy as np
from rich.console import Console
from rich.panel import Panel

from agent_tools import EmbeddingStore

load_dotenv()
console = Console()

SYSTEM_PROMPT = (
    "You are PantryChef — a friendly assistant that suggests recipes, meal plans, and shopping lists "
    "based on available ingredients and any supporting documents provided. When context is available, cite the source IDs. "
    "Keep responses concise and actionable."
)
from utils import retry_openai

DEFAULT_MODEL = "gpt-3.5-turbo"


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        console.print("[red]Error:[/red] OPENAI_API_KEY environment variable not set.")
        console.print("Set it with: $env:OPENAI_API_KEY = 'sk-...' (PowerShell) or export OPENAI_API_KEY=... (Unix)")
        sys.exit(1)
    return key


def call_chat(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=600):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"[red]API error:[/red] {e}")
        return None


def cached_call_chat(prompt, model=DEFAULT_MODEL, temperature=0.7, max_tokens=600, store: EmbeddingStore = None):
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if store:
        cached = store.get_cache(key)
        if cached:
            return cached
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,

        )
        answer = response.choices[0].message.content.strip()
        if store:
            try:
                store.set_cache(key, answer)
            except Exception:
                pass
        return answer
    except Exception as e:
        console.print(f"[red]API error:[/red] {e}")
        return None


def interactive_loop(model, store=None, top_k=3):
    console.print(Panel("PantryChef — enter ingredients or commands. Type `/help` for commands."))
    history = []
    while True:
        try:
            user = console.input("[bold cyan]> [/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye — happy cooking!")
            break

        user = user.strip()
        if not user:
            continue
        if user in ("/exit", "/quit"):
            console.print("Goodbye — happy cooking!")
            break
        if user == "/help":
            console.print(
                "Commands:\n  /recipe <ingredients>  — request recipes for given comma-separated ingredients\n  /plan <days>           — create a meal plan for N days (use ingredients)\n  /shopping <items>      — expand to a shopping list\n  /ingest <dir>          — ingest markdown docs from directory\n  /query <question>      — ask a question and use ingested docs for context\n  /exit                  — quit\nJust type your ingredients or a question to get started."
            )
            continue

        # If user starts with a recognized command, shape the prompt
        if user.startswith("/recipe "):
            ingredients = user[len("/recipe "):].strip()
            prompt = f"I have the following ingredients: {ingredients}. Suggest 3 recipes I can make using most of these ingredients. For each recipe include: title, short description, ingredients (what I need), steps, and estimated time."
        elif user.startswith("/plan "):
            days = user[len("/plan "):].strip()
            prompt = f"Create a {days}-day meal plan for someone using common pantry items and the following ingredients (if any): . Provide meals and a consolidated shopping list."
        elif user.startswith("/shopping "):
            items = user[len("/shopping "):].strip()
            prompt = f"Create a shopping list and approximate quantities for the following items or to make meals from a typical pantry: {items}. Group by sections (produce, dairy, pantry)."
        elif user.startswith("/ingest "):
            d = user[len("/ingest "):].strip()
            store = EmbeddingStore()
            store.ingest_dir(d)
            console.print(f"Ingested docs from {d} into {store.path}")
            continue
        elif user.startswith("/query "):
            q = user[len("/query "):].strip()
            if not store:
                store = EmbeddingStore()
            ctx = store.query(q, top_k=top_k)
            ctx_text = "\n\n".join([f"Source:{c['id']}\n{c['text']}" for c in ctx])
            prompt = f"Use the following context documents to answer the question.\n\n{ctx_text}\n\nQuestion: {q}\n\nAnswer concisely and cite source IDs when relevant."
        else:
            # treat input as free-form ingredient list or question
            prompt = f"User input: {user}\n\nAct as a recipe recommender and provide helpful suggestions."

        console.print("[grey]Thinking...[/grey]")
        answer = cached_call_chat(prompt, model=model, store=store)
        if answer:
            console.print(Panel(answer, title="PantryChef"))
            history.append((user, answer))


def cli_main():
    parser = argparse.ArgumentParser(description="PantryChef — recipe agent powered by OpenAI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--ingredients", "-i", help="One-line comma-separated ingredients to ask about")
    parser.add_argument("--interactive", "-r", action="store_true", help="Start interactive REPL")
    parser.add_argument("--ingest", help="Ingest markdown docs from a directory into embeddings store")
    parser.add_argument("--query", help="Query with local docs context (one-line question)")
    parser.add_argument("--topk", type=int, default=3, help="Top-k docs to use for context")
    args = parser.parse_args()

    # ensure API key
    api_key = get_api_key()
    openai.api_key = api_key

    if args.ingest:
        store = EmbeddingStore()
        store.ingest_dir(args.ingest)
        console.print(f"Ingested docs from {args.ingest} into {store.path}")
        return

    if args.query:
        store = EmbeddingStore()
        ctx = store.query(args.query, top_k=args.topk)
        ctx_text = "\n\n".join([f"Source:{c['id']}\n{c['text']}" for c in ctx])
        prompt = f"Use the following context documents to answer the question.\n\n{ctx_text}\n\nQuestion: {args.query}\n\nAnswer concisely and cite source IDs when relevant."
        console.print("[grey]Querying agent with local context...[/grey]")
        store_local = store
        answer = cached_call_chat(prompt, model=args.model, store=store_local)
        if answer:
            console.print(Panel(answer, title="PantryChef"))
        return

    if args.interactive:
        # try to load store, but not required
        try:
            store = EmbeddingStore()
        except Exception:
            store = None
        interactive_loop(args.model, store=store, top_k=args.topk)
        return

    if args.ingredients:
        prompt = f"I have the following ingredients: {args.ingredients}. Suggest 3 recipes I can make using most of these ingredients. For each recipe include: title, short description, ingredients, steps, and estimated time."
        console.print("[grey]Querying agent...[/grey]")
        answer = call_chat(prompt, model=args.model)
        if answer:
            console.print(Panel(answer, title="PantryChef"))
        return

    # default help
    parser.print_help()


if __name__ == "__main__":
    cli_main()
