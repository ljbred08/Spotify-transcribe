"""Shared pytest setup: isolate app data dir before any app import."""
import os

os.environ.setdefault("APP_DATA_DIR", ".pytest-data")
