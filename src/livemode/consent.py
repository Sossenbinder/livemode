"""Turn a spoken answer to a permission question into allow, deny or unclear.

This is deliberately not done by the local model: a wrong "allow" is a security risk.
"""

import re

_ALLOW = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "allow",
    "allowed",
    "approve",
    "approved",
    "go",
    "do",
    "proceed",
    "fine",
    "affirmative",
}
_DENY = {
    "no",
    "nope",
    "nah",
    "deny",
    "denied",
    "don't",
    "dont",
    "stop",
    "cancel",
    "reject",
    "not",
    "never",
    "wait",
    "abort",
}


def parse(text: str) -> bool | None:
    """Return True for allow, False for deny, None if the answer is not clear.

    A deny word wins over an allow word, so "yes, no wait" is a deny.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))
    if words & _DENY:
        return False
    if words & _ALLOW:
        return True
    return None
