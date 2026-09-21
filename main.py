"""Cloud entry point: the deployable front door for the app.

Cloud platforms (Render, Railway, Fly.io, …) inject the network settings
as environment variables, so they are read here at startup instead of
being hardcoded. Locally you don't need this file — run the dev server
with live reload as documented in the README:

    uvicorn app.server:app --reload
"""

import os

import uvicorn

from app.server import app  # noqa: E402  (the FastAPI application)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    # The app object is passed directly (not "main:app") so uvicorn never
    # re-imports this module; reload/multi-worker setups don't apply here.
    uvicorn.run(app, host=host, port=port)
