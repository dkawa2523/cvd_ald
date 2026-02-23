"""Role enumeration for A/AI/AB/AIB classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RoleCandidate:
    A: str
    I: str | None
    B: str | None
    class_id: str


def class_id_from_roles(*, I: str | None, B: str | None) -> str:
    if I is None and B is None:
        return "A"
    if I is not None and B is None:
        return "AI"
    if I is None and B is not None:
        return "AB"
    return "AIB"


def _role_candidates(
    *,
    species: list[str],
    role_spec: Mapping[str, Any] | None,
    required: bool,
    allow_none: bool,
) -> list[str | None]:
    if role_spec is None:
        role_spec = {}
    raw = role_spec.get("candidates", "auto")
    species_set = set(species)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == "auto":
            base = list(species)
        elif text.startswith("[") and text.endswith("]"):
            items = [tok.strip().strip("'\"") for tok in text[1:-1].split(",") if tok.strip()]
            base = [name for name in items if name in species_set]
        elif text in species_set:
            base = [text]
        else:
            base = []
    elif isinstance(raw, Sequence):
        base = [str(s) for s in raw if str(s) in species_set]
    else:
        base = list(species)
    if required:
        return list(base)
    out: list[str | None] = list(base)
    if allow_none:
        out = [None, *out]
    return out


def enumerate_roles(
    species: Sequence[str],
    *,
    roles_spec: Mapping[str, Any] | None = None,
    constraints: Mapping[str, Any] | None = None,
    class_filter: Sequence[str] | None = None,
) -> list[RoleCandidate]:
    out: list[RoleCandidate] = []
    labels = [str(s) for s in species]
    role_cfg = dict(roles_spec or {})
    a_cfg = dict(role_cfg.get("A", {}) or {})
    i_cfg = dict(role_cfg.get("I", {}) or {})
    b_cfg = dict(role_cfg.get("B", {}) or {})
    cons = dict(constraints or {})
    enforce_disjoint = bool(cons.get("disjoint", True))
    class_allowed = {str(x) for x in class_filter} if class_filter else None

    a_candidates = _role_candidates(
        species=labels,
        role_spec=a_cfg,
        required=bool(a_cfg.get("required", True)),
        allow_none=bool(a_cfg.get("allow_none", False)),
    )
    i_candidates = _role_candidates(
        species=labels,
        role_spec=i_cfg,
        required=bool(i_cfg.get("required", False)),
        allow_none=bool(i_cfg.get("allow_none", True)),
    )
    b_candidates = _role_candidates(
        species=labels,
        role_spec=b_cfg,
        required=bool(b_cfg.get("required", False)),
        allow_none=bool(b_cfg.get("allow_none", True)),
    )

    max_i = int(i_cfg.get("max_size", 1))
    max_b = int(b_cfg.get("max_size", 1))
    if max_i > 1 or max_b > 1:
        raise ValueError("enumerate_roles currently supports max_size <= 1 for I/B roles")

    for s_a in a_candidates:
        if s_a is None:
            continue
        for s_i in i_candidates:
            for s_b in b_candidates:
                if enforce_disjoint:
                    selected = [x for x in (s_a, s_i, s_b) if x is not None]
                    if len(set(selected)) != len(selected):
                        continue
                cid = class_id_from_roles(I=s_i, B=s_b)
                if class_allowed is not None and cid not in class_allowed:
                    continue
                out.append(RoleCandidate(A=s_a, I=s_i, B=s_b, class_id=cid))
    return out


__all__ = ["RoleCandidate", "class_id_from_roles", "enumerate_roles"]
