
import os
from dotenv import load_dotenv

load_dotenv()


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL not found in environment variables."
        )

    return database_url
