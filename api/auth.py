import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

# Comma-separated keys in your .env: API_KEYS=key1,key2,key3
VALID_KEYS = set(os.getenv("API_KEYS", "").split(","))

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Clients must pass header:  X-API-Key: <their_key>
    """
    if not x_api_key or x_api_key.strip() not in VALID_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. "
                   "Pass your key in the X-API-Key header."
        )
    return x_api_key