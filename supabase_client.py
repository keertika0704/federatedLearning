import os

from dotenv import load_dotenv
from supabase import Client, create_client


def get_supabase_client() -> Client:
    """Create a Supabase client using credentials from the project .env file."""
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_KEY in a .env file or environment."
        )

    return create_client(supabase_url, supabase_key)