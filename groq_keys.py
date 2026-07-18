"""
groq_keys.py — shared multi-key rotation for Groq API calls.

IMPORTANT: Groq's daily token limit is scoped per ORGANIZATION, not per
key. Multiple keys from the SAME Groq account share one quota pool --
rotating between them does nothing. This only helps if each key comes
from a genuinely different Groq account (different email signup).

Setup:
    Set GROQ_API_KEYS as a comma-separated list of keys from DIFFERENT
    accounts (PowerShell):

        $env:GROQ_API_KEYS="key_from_account1,key_from_account2,key_from_account3"

    If GROQ_API_KEYS isn't set, falls back to the single GROQ_API_KEY
    you've been using all along -- so existing scripts don't break if
    you haven't set up multiple keys yet.

Usage (replaces direct OpenAI client calls):
    from groq_keys import call_groq

    response = call_groq(
        messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"},  # optional, omit for plain text
    )
    text = response.choices[0].message.content
"""
import os
import time
from openai import OpenAI

_keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
if not _keys:
    single = os.environ.get("GROQ_API_KEY")
    if single:
        _keys = [single]

if not _keys:
    raise RuntimeError(
        "No Groq API keys found. Set GROQ_API_KEYS (comma-separated, from "
        "DIFFERENT Groq accounts) or GROQ_API_KEY (single key)."
    )

_clients = [OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1") for k in _keys]
_current = 0

print(f"[groq_keys] Loaded {len(_clients)} key(s) for rotation.")


def call_groq(messages, model, response_format=None, max_cycles=1, **extra_kwargs):
    """
    Tries the current key. On a rate-limit error, rotates to the next
    key and retries immediately -- no waiting, since a different key
    (different org) isn't affected by the first one's limit.

    Any extra keyword arguments (e.g. temperature=0.0) are passed
    straight through to the underlying chat.completions.create call.

    If EVERY key is rate-limited in one pass (max_cycles=1), raises the
    last error rather than looping forever. Set max_cycles higher only
    if you want it to wait and re-try the whole key list again (rarely
    useful -- if all your accounts are limited, waiting a few seconds
    won't fix that).
    """
    global _current
    n = len(_clients)
    last_error = None

    for cycle in range(max_cycles):
        for _ in range(n):
            client = _clients[_current]
            key_num = _current + 1
            try:
                kwargs = dict(model=model, messages=messages, **extra_kwargs)
                if response_format:
                    kwargs["response_format"] = response_format
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e)
                last_error = e
                if "rate_limit" in msg.lower() or "429" in msg:
                    print(f"  [groq_keys] key #{key_num}/{n} rate-limited, rotating to next key...")
                    _current = (_current + 1) % n
                    continue
                else:
                    # not a rate-limit error -- don't rotate, just fail through
                    raise
        if cycle < max_cycles - 1:
            print(f"  [groq_keys] all {n} keys rate-limited, waiting 30s before next cycle...")
            time.sleep(30)

    raise last_error


def current_key_index():
    """For logging/debugging -- which key (1-indexed) is currently active."""
    return _current + 1