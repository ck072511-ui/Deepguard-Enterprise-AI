import importlib
import streamlit as st


def test_resolve_api_base_url_prefers_explicit_env(monkeypatch):
    monkeypatch.setattr(st, "secrets", {}, raising=False)
    monkeypatch.delenv("DEEPGUARD_API_URL", raising=False)
    monkeypatch.delenv("DEEPGUARD_API_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPGUARD_API_BASE_URL", "https://api.example.com/api/v1")

    client_module = importlib.import_module("frontend.client")
    assert client_module.resolve_api_base_url() == "https://api.example.com/api/v1"


def test_resolve_api_base_url_uses_relative_default(monkeypatch):
    monkeypatch.setattr(st, "secrets", {}, raising=False)
    monkeypatch.delenv("DEEPGUARD_API_URL", raising=False)
    monkeypatch.delenv("DEEPGUARD_API_BASE_URL", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    client_module = importlib.import_module("frontend.client")
    assert client_module.resolve_api_base_url() == "https://deepguard-enterprise-ai.onrender.com/api/v1"
