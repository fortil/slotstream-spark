"""Deterministic benchmark prompts (stdlib only), shared by bench/lb/client.py and the CLI."""
import random

_SUBJECTS = ["The river", "A librarian", "The old bridge", "Mount Arden", "The harbour", "A beekeeper",
             "The observatory", "A glassblower", "The northern railway", "The market"]
_VERBS = ["was founded in", "measures", "was rebuilt after", "is known for", "closed during",
          "employs", "was named after", "moved to", "was surveyed in", "reopened in"]
_OBJECTS = ["the spring of {y}", "{n} metres of stone", "the flood of {y}", "{n} regular visitors",
            "a copper roof", "the town of Velmar", "{n} wooden boats", "the winter of {y}",
            "a family from the coast", "{n} lamps"]


def build_prompt(spec):
    """spec: a literal string, or {"kind": "facts", "seed": S, "target_chars": N} (deterministic)."""
    if isinstance(spec, str):
        return spec
    if spec.get("kind") != "facts":
        raise ValueError(f"unknown prompt kind: {spec!r}")
    rng = random.Random(spec["seed"])
    target = int(spec["target_chars"])
    lines, size, i = [], 0, 1
    while size < target:
        s = (f"Fact {i}: {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
             f"{rng.choice(_OBJECTS).format(y=rng.randint(1700, 1990), n=rng.randint(12, 9000))}.")
        lines.append(s)
        size += len(s) + 1
        i += 1
    return "Read these facts, then list the three that seem most surprising.\n" + "\n".join(lines)
