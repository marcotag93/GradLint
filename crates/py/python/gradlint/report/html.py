"""Render a canonical report to a self-contained HTML page (CSS + figures inline)."""

from __future__ import annotations

import jinja2

from .model import Report, correction_presentation

_STATUS_COLORS = {"PASS": "#1a8a3a", "WARN": "#c77800", "FLAG": "#c0271b"}

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gradlint report</title>
<style>
  :root { --accent: {{ status_color }}; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; color: #1a1a1a; background: #f4f5f7; }
  main { max-width: 900px; margin: 0 auto; padding: 1.5rem; }
  .banner { background: var(--accent); color: #fff; padding: 1rem 1.5rem;
            border-radius: 8px; display: flex; align-items: baseline;
            justify-content: space-between; gap: 1rem; }
  .banner h1 { margin: 0; font-size: 1.5rem; letter-spacing: 0.04em; }
  .banner .meta { font-size: 0.85rem; opacity: 0.9; text-align: right; }
  section { background: #fff; border-radius: 8px; padding: 1rem 1.25rem;
            margin-top: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }
  h2 { margin-top: 0; font-size: 1.1rem; border-bottom: 1px solid #eee;
       padding-bottom: 0.4rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  th, td { text-align: left; padding: 0.35rem 0.6rem;
           border-bottom: 1px solid #eee; }
  th { background: #fafafa; }
  code { background: #f0f0f0; padding: 0.1rem 0.3rem; border-radius: 3px; }
  .figures { display: grid; grid-template-columns: repeat(auto-fit,
             minmax(280px, 1fr)); gap: 1rem; }
  .figures img { width: 100%; height: auto; border: 1px solid #eee;
                 border-radius: 4px; }
  .figures img.glyphs { grid-column: 1 / -1; }
  ul { margin: 0.4rem 0; padding-left: 1.2rem; }
  .muted { color: #666; font-size: 0.85rem; }
</style>
</head>
<body>
<main>
  <div class="banner">
    <h1>gradlint · {{ report.status }}</h1>
    <div class="meta">{{ report.tool }} {{ report.tool_version }}<br>
      {{ report.timestamp }}</div>
  </div>

  {% if figures %}
  <section>
    <h2>Figures</h2>
    <div class="figures">
      {% if figures.shells %}<img alt="shell histogram"
        src="{{ figures.shells }}">{% endif %}
      {% if figures.sphere %}<img alt="direction sphere"
        src="{{ figures.sphere }}">{% endif %}
      {% if figures.drift %}<img alt="b0 drift"
        src="{{ figures.drift }}">{% endif %}
      {% if figures.coherence %}<img alt="candidate coherence"
        src="{{ figures.coherence }}">{% endif %}
      {% if figures.margin %}<img alt="repair margin gauge"
        src="{{ figures.margin }}">{% endif %}
      {% if figures.glyphs %}<img class="glyphs" alt="tensor glyph orientation"
        src="{{ figures.glyphs }}">{% endif %}
    </div>
  </section>
  {% endif %}

  {% if report.flip %}
  {% set flip = report.flip %}
  <section>
    <h2>Flip / permutation detection</h2>
    <ul>
      <li><strong>Decision:</strong> {{ flip.decision }}</li>
      <li><strong>Working shell:</strong>
        b&approx;{{ flip.working_b|g }} ({{ flip.n_wm_voxels }} WM voxels)</li>
      <li><strong>Best:</strong> <code>{{ flip.best.label }}</code>
        (coherence {{ flip.best.coherence|f4 }})</li>
      <li><strong>Runner-up:</strong> <code>{{ flip.runner_up.label }}</code>
        (coherence {{ flip.runner_up.coherence|f4 }})</li>
      <li><strong>Margin:</strong> {{ flip.margin|f4 }}
        (relative {{ flip.relative_margin|pct }}%)</li>
    </ul>
  </section>
  {% endif %}

  {% if correction %}
  <section>
    <h2>Correction status</h2>
    <p><strong>{{ correction.summary }}</strong></p>
    {% if report.repair %}
    <ul>
      <li><strong>Transform:</strong> <code>{{ report.repair.label }}</code></li>
      <li><strong>Outputs:</strong>
        {% for o in report.repair.outputs %}<code>{{ o }}</code>
        {% else %}none (dry run){% endfor %}</li>
    </ul>
    {% elif correction.kind in ["withheld", "recommended"] %}
    <p class="muted">The displayed best candidate is a preview only. The input
      gradient table was not modified.</p>
    {% endif %}
  </section>
  {% endif %}

  {% if report.shells %}
  <section>
    <h2>b-value / shell structure</h2>
    <table>
      <tr><th>Shell</th><th>b-value</th><th>Volumes</th></tr>
      {% for s in report.shells.shells %}
      <tr>
        <td>{% if s.is_b0 %}b0{% else %}b&approx;{{ s.nominal_b|g }}{% endif %}</td>
        <td>{{ s.min_b|g }}&ndash;{{ s.max_b|g }}</td>
        <td>{{ s.count }}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="muted">b0 volumes: {{ report.shells.b0.count }}
      {% if report.shells.non_integer_bvals %}
      · non-integer b-values at {{ report.shells.non_integer_bvals }}{% endif %}</p>
  </section>
  {% endif %}

  {% if report.angular and report.angular.shells %}
  <section>
    <h2>Angular scheme quality</h2>
    <table>
      <tr><th>b-value</th><th>Dirs</th><th>Energy</th><th>Cond. &#8470;</th>
        <th>DTI</th><th>CSD</th><th>Duplicates</th></tr>
      {% for s in report.angular.shells %}
      <tr>
        <td>b&approx;{{ s.nominal_b|g }}</td>
        <td>{{ s.count }}</td>
        <td>{{ s.electrostatic_energy|f1 }}</td>
        <td>{% if s.condition_number is none %}&mdash;
          {% else %}{{ s.condition_number|f1 }}{% endif %}</td>
        <td>{% if s.meets_dti_recommended %}&#10003;
          {% elif s.meets_dti_minimum %}min{% else %}&#10007;{% endif %}</td>
        <td>{% if s.meets_csd_minimum %}&#10003;{% else %}&#10007;{% endif %}</td>
        <td>{{ s.duplicates|length }}</td>
      </tr>
      {% endfor %}
    </table>
  </section>
  {% endif %}

  {% if report.b0_drift %}
  <section>
    <h2>b0 signal drift</h2>
    <ul>
      <li><strong>Slope:</strong> {{ report.b0_drift.slope|f4g }} per volume</li>
      <li><strong>Relative drift:</strong>
        {{ report.b0_drift.relative_drift|pct }}%</li>
    </ul>
  </section>
  {% endif %}

  {% if report.inputs %}
  <section>
    <h2>Inputs</h2>
    <table>
      <tr><th>Path</th><th>Bytes</th><th>sha256</th></tr>
      {% for f in report.inputs %}
      <tr><td><code>{{ f.path }}</code></td><td>{{ f.bytes }}</td>
        <td><code>{{ f.sha256[:16] }}&hellip;</code></td></tr>
      {% endfor %}
    </table>
  </section>
  {% endif %}

  {% if report.notes %}
  <section>
    <h2>Notes</h2>
    <ul>{% for note in report.notes %}<li>{{ note }}</li>{% endfor %}</ul>
  </section>
  {% endif %}
</main>
</body>
</html>
"""

_FILTERS = {
    "g": lambda x: f"{x:g}",
    "f1": lambda x: f"{x:.1f}",
    "f4": lambda x: f"{x:.4f}",
    "f4g": lambda x: f"{x:.4g}",
    "pct": lambda x: f"{x * 100:.2f}",
}


def render_html(
    report: Report,
    *,
    with_figures: bool = True,
    glyphs: dict[str, object] | None = None,
    figure_data: dict[str, str] | None = None,
) -> str:
    """Render a report as a self-contained HTML document.

    When ``with_figures`` is true, matplotlib figures are generated and embedded
    inline as base64 PNGs; otherwise the figure section is omitted.
    """
    figures: dict[str, str] = {}
    if with_figures:
        if figure_data is not None:
            figures = figure_data
        else:
            from .figures import build_figures

            figures = build_figures(report, glyphs=glyphs)

    env = jinja2.Environment(autoescape=True, trim_blocks=True)
    env.filters.update(_FILTERS)
    template = env.from_string(_TEMPLATE)
    return template.render(
        report=report,
        figures=figures,
        correction=correction_presentation(report),
        status_color=_STATUS_COLORS.get(report.status, "#444"),
    )
