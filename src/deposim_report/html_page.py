"""Small HTML page renderer for run/DOE reports."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape


def render_report_page(
    *,
    title: str,
    heading: str,
    sections: Sequence[str],
    back_href: str = "../../index.html",
    back_label: str = "Back to project index",
    style: str | None = None,
) -> str:
    """Render a minimal full HTML page for report artifacts."""

    style_block = ""
    if style:
        style_block = f"<style>\n{style}\n</style>"

    body_parts = [
        f"<h1>{escape(heading)}</h1>",
        f'<p><a href="{escape(back_href)}">{escape(back_label)}</a></p>',
    ]
    body_parts.extend(sections)
    body_html = "".join(body_parts)
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8" />'
        f"<title>{escape(title)}</title>{style_block}</head><body>{body_html}</body></html>"
    )


__all__ = ["render_report_page"]
