"""Order enumeration helpers with total-order constraints."""

from __future__ import annotations

from typing import Any


def enumerate_orders(candidates: list[dict[str, Any]], *, has_b: bool, enforce_total_order_le: int = 3) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for cand in candidates:
        m_ads = int(cand["adsorption_site_order"])
        p_a = int(cand["reaction_site_order_A"])
        p_star = int(cand["reaction_site_order_star"])
        total = p_a + p_star + (1 if has_b else 0)
        if total > int(enforce_total_order_le):
            continue
        out.append(
            {
                "adsorption_site_order": m_ads,
                "reaction_site_order_A": p_a,
                "reaction_site_order_star": p_star,
            }
        )
    return out


__all__ = ["enumerate_orders"]
