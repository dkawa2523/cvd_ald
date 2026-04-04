"""Shared lightweight HTML report helpers."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

from deposim_report.html_page import render_report_page


def write_artifact_list_report(
    *,
    run_dir: Path,
    run_id: str,
    title: str,
    artifact_links: Sequence[str],
    warnings: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
) -> Path:
    warning_html = "".join(f"<p><strong>Warning:</strong> {escape(str(msg))}</p>" for msg in (warnings or []))
    notes_html = "".join(f"<p>{escape(str(msg))}</p>" for msg in (notes or []))
    items = "".join(f"<li><a href='{escape(str(path))}'>{escape(str(path))}</a></li>" for path in artifact_links) or "<li>None</li>"
    body = f"{warning_html}{notes_html}<ul>{items}</ul>"

    page = render_report_page(
        title=title,
        heading=f"{title}: {run_id}",
        style="body { font-family: sans-serif; margin: 1.2rem 1.8rem; }",
        sections=[body],
    )
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


__all__ = ["write_artifact_list_report"]
