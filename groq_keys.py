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
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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


_WAIT_PATTERN = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s|m|h)?", re.IGNORECASE)
_DEFAULT_WAIT_SECONDS = 30
_MAX_WAIT_SECONDS = 6 * 60 * 60  # 6h safety cap -- if this undershoots a longer reset,
                                  # the next attempt just re-parses a fresh error and waits again.


def _parse_wait_seconds(error_msg: str) -> float:
    """
    Groq's 429 body includes a suggested wait, e.g. "...try again in 7.66s".
    Parse it so the retry sleeps roughly as long as needed instead of a
    blind fixed backoff. If the real message uses a compound duration this
    only catches the leading component -- harmless, since undershooting
    just means the next attempt hits another 429 and re-parses.
    """
    match = _WAIT_PATTERN.search(error_msg)
    if not match:
        return _DEFAULT_WAIT_SECONDS
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    multiplier = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}.get(unit, 1)
    return min(value * multiplier, _MAX_WAIT_SECONDS)


def call_groq(messages, model, response_format=None, max_cycles=None, **extra_kwargs):
    """
    Tries the current key. On a rate-limit OR connection/timeout error,
    rotates to the next key and retries immediately -- a dropped
    connection or a slow edge node on one key doesn't mean the others
    are affected, so it's worth trying them before giving up.

    Any extra keyword arguments (e.g. temperature=0.0) are passed
    straight through to the underlying chat.completions.create call.

    If EVERY key fails one pass with a RECOVERABLE error (rate limit,
    connection issue), this waits out Groq's own suggested retry time
    (parsed from the error body -- exact for per-minute limits, and for a
    full daily-quota exhaustion this can be hours) and tries the whole
    key list again, indefinitely, rather than raising -- an unattended
    multi-hour batch run should survive a quota reset without needing a
    manual restart. A genuinely non-recoverable error (bad model name,
    malformed request, permanent auth failure) still raises immediately,
    since no amount of waiting fixes those.

    max_cycles caps how many full passes to attempt before giving up;
    None (default) means retry forever on recoverable errors.
    """
    global _current
    n = len(_clients)
    last_error = None
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
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

        wait_seconds = _parse_wait_seconds(str(last_error))
        cycle += 1
        print(
            f"  [groq_keys] all {n} keys rate-limited/unreachable this pass, "
            f"waiting {wait_seconds:.0f}s (per Groq's own retry hint) before trying again..."
        )
        time.sleep(wait_seconds)

    raise last_error


def current_key_index():
    """For logging/debugging -- which key (1-indexed) is currently active."""
    return _current + 1