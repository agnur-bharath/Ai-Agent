import os
import sys
import builtins
from types import SimpleNamespace

import openai

import agent


def test_cli_ingredients_with_mock(monkeypatch, capsys):
    # Ensure API key check passes
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock ChatCompletion.create to return a predictable response
    def fake_chat_create(*args, **kwargs):
        resp = SimpleNamespace()
        resp.choices = [SimpleNamespace(message=SimpleNamespace(content="Hello from mock"))]
        return resp

    monkeypatch.setattr(openai.ChatCompletion, "create", fake_chat_create)

    # Simulate CLI args for a one-shot ingredients query
    monkeypatch.setattr(sys, "argv", ["agent.py", "--ingredients", "eggs, milk"], raising=False)

    # Run CLI main (should use the mocked ChatCompletion)
    agent.cli_main()

    captured = capsys.readouterr()
    assert "Hello from mock" in captured.out


def test_interactive_repl_simulated(monkeypatch, capsys):
    # Ensure API key check passes
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock ChatCompletion.create for REPL
    def fake_chat_create(*args, **kwargs):
        resp = SimpleNamespace()
        resp.choices = [SimpleNamespace(message=SimpleNamespace(content="Mocked REPL answer"))]
        return resp

    monkeypatch.setattr(openai.ChatCompletion, "create", fake_chat_create)

    # Replace console.input to simulate user entering a recipe command then exit
    inputs = iter(["/recipe eggs, milk", "/exit"])

    def fake_input(prompt=None):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    # Ensure Console.print outputs to stdout so capsys captures it
    monkeypatch.setattr(agent.console, "input", fake_input)
    monkeypatch.setattr(agent.console, "print", lambda *a, **k: builtins.print(*a, **k))

    # Run the interactive loop (it will exit after the simulated inputs)
    agent.interactive_loop(model=agent.DEFAULT_MODEL, store=None, top_k=3)

    captured = capsys.readouterr()
    assert "Mocked REPL answer" in captured.out
