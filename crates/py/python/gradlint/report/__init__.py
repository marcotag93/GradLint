"""Report rendering for gradlint: consume ``report.json``, emit Markdown/HTML."""

from __future__ import annotations

from .html import render_html
from .markdown import render_markdown
from .model import Report, load_report

__all__ = ["Report", "load_report", "render_markdown", "render_html"]
