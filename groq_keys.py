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

# Explicit request timeout -- the OpenAI SDK's default is 600s (10 min).
# Without this, a single bad connection (dropped wifi, a hung Groq edge
# node, etc.) blocks for the full 10 minutes before failing, and since it
# doesn't look like a rate-limit error, it never even rotates to another
# key first. 30s is generous for a single chat completion; if a request
# hasn't come back by then, waiting longer isn't going to help --
# retrying (possibly on a different key) is more useful than waiting.
REQUEST_TIMEOUT_SECONDS = 30

_clients = [
    OpenAI(api_key=k, base_url="https://api.groq.com/openai/v1", timeout=REQUEST_TIMEOUT_SECONDS)
    for k in _keys
]
_current = 0

print(f"[groq_keys] Loaded {len(_clients)} key(s) for rotation (timeout={REQUEST_TIMEOUT_SECONDS}s).")


def _is_recoverable(error_msg: str) -> bool:
    """
    True if this looks like a transient/per-key issue worth rotating past
    (rate limit, connection drop, timeout) rather than a real bug in the
    request itself (bad model name, malformed messages, auth failure on
    ALL keys, etc.) where rotating and retrying would just waste time.
    """
    msg = error_msg.lower()
    recoverable_signals = [
        "rate_limit", "429",
        "connection error", "connection reset", "connection aborted",
        "timeout", "timed out",
        "temporarily unavailable", "service unavailable", "502", "503", "504",
    ]
    return any(sig in msg for sig in recoverable_signals)


def call_groq(messages, model, response_format=None, max_cycles=1, **extra_kwargs):
    """
    Tries the current key. On a rate-limit OR connection/timeout error,
    rotates to the next key and retries immediately -- a dropped
    connection or a slow edge node on one key doesn't mean the others
    are affected, so it's worth trying them before giving up.

    Any extra keyword arguments (e.g. temperature=0.0) are passed
    straight through to the underlying chat.completions.create call.

    If EVERY key fails in one pass (max_cycles=1), raises the last error
    rather than looping forever. Set max_cycles higher only if you want
    it to wait and re-try the whole key list again.
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
                if _is_recoverable(msg):
                    reason = "rate-limited" if ("rate_limit" in msg.lower() or "429" in msg) else "connection issue"
                    print(f"  [groq_keys] key #{key_num}/{n} {reason} ({msg[:80]}), rotating to next key...")
                    _current = (_current + 1) % n
                    continue
                else:
                    # Not a transient issue -- rotating won't help (e.g. a
                    # genuinely malformed request), so fail immediately
                    # rather than burning through every key pointlessly.
                    raise
        if cycle < max_cycles - 1:
            print(f"  [groq_keys] all {n} keys failed this pass, waiting 30s before next cycle...")
            time.sleep(30)

    raise last_error


def current_key_index():
    """For logging/debugging -- which key (1-indexed) is currently active."""
    return _current + 1