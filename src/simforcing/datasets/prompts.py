from typing import Optional

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
SIM_PROMPT = "A simulation-rendered video recorded from a robot's point of view executing the following instruction: {task}"
REAL_PROMPT = "A real-world video recorded from a robot's point of view executing the following instruction: {task}"
PROMPT_TEMPLATE_BY_HALF = {
    "left": SIM_PROMPT,
    "first_real_rest_sim": SIM_PROMPT,
    "right": REAL_PROMPT,
    "full": DEFAULT_PROMPT,
}
ALL_PROMPT_TEMPLATES = (DEFAULT_PROMPT, SIM_PROMPT, REAL_PROMPT)
PROMPT_TEMPLATE_BY_DOMAIN = {
    "sim": SIM_PROMPT,
    "real": REAL_PROMPT,
    "default": DEFAULT_PROMPT,
}


def build_prompt(task: str, half: Optional[str] = None) -> str:
    """Format ``task`` with the template matching the target-video domain.

    ``half`` is the dataset's window-domain tag (see
    ``PROMPT_TEMPLATE_BY_HALF``). ``None`` or an unknown value falls back to
    ``DEFAULT_PROMPT``, preserving the behaviour of every existing config /
    checkpoint that predates domain-tagged prompts.
    """
    template = PROMPT_TEMPLATE_BY_HALF.get(half, DEFAULT_PROMPT)
    return template.format(task=task)


def split_prompt(prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Inverse of ``build_prompt``: recover ``(template, task)``.

    Matching is done on the fixed prefix that precedes ``{task}``, longest
    first, so that no template can shadow another. Returns ``(None, None)``
    when ``prompt`` was not produced by any known template.
    """
    candidates = sorted(
        ALL_PROMPT_TEMPLATES, key=lambda t: len(t.split("{task}")[0]), reverse=True
    )
    for template in candidates:
        prefix = template.split("{task}")[0]
        if prompt.startswith(prefix):
            return (template, prompt[len(prefix) :])
    return (None, None)


def retarget_prompt(prompt: str, domain: str) -> Optional[str]:
    """Rewrite ``prompt`` to request a different target-video ``domain``.

    Keeps the task text intact and swaps only the template, which is what
    makes an A/B comparison valid: the two prompts differ exclusively in the
    domain tag. ``domain`` is one of ``'sim'`` / ``'real'`` / ``'default'``.
    Returns ``None`` when the task text cannot be recovered from ``prompt``
    (so callers can skip the comparison instead of guessing).
    """
    if domain not in PROMPT_TEMPLATE_BY_DOMAIN:
        raise ValueError(
            f"Unknown prompt domain '{domain}', expected one of {sorted(PROMPT_TEMPLATE_BY_DOMAIN)}."
        )
    _, task = split_prompt(prompt)
    if task is None:
        return None
    return PROMPT_TEMPLATE_BY_DOMAIN[domain].format(task=task)
