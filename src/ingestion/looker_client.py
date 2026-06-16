"""Looker API client for pulling feature extracts.

Uses the official Looker Python SDK (``looker-sdk``). Credentials are read
from environment variables (set in ``.env``, gitignored) rather than the
SDK's default ``looker.ini`` file, so secrets never need to be written to
disk in plaintext config.

Required env vars (see ``.env.example``):
    LOOKER_BASE_URL       e.g. https://your-instance.looker.com:19999
    LOOKER_CLIENT_ID
    LOOKER_CLIENT_SECRET
    LOOKER_VERIFY_SSL     optional, defaults to "true"
"""
from __future__ import annotations

import os

import looker_sdk
from dotenv import load_dotenv
from looker_sdk.rtl import api_settings

load_dotenv()

REQUIRED_ENV_VARS = ("LOOKER_BASE_URL", "LOOKER_CLIENT_ID", "LOOKER_CLIENT_SECRET")


class EnvApiSettings(api_settings.ApiSettings):
    """ApiSettings that reads Looker credentials from environment variables
    instead of a ``looker.ini`` file."""

    def read_config(self) -> api_settings.SettingsConfig:
        missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                f"Missing Looker credentials in environment: {', '.join(missing)}. "
                "Set them in .env (see .env.example)."
            )
        return {
            "base_url": os.environ["LOOKER_BASE_URL"],
            "client_id": os.environ["LOOKER_CLIENT_ID"],
            "client_secret": os.environ["LOOKER_CLIENT_SECRET"],
            "verify_ssl": os.environ.get("LOOKER_VERIFY_SSL", "true"),
        }


def get_sdk() -> looker_sdk.sdk.api40.methods.Looker40SDK:
    """Return an authenticated Looker 4.0 SDK client."""
    return looker_sdk.init40(config_settings=EnvApiSettings())
