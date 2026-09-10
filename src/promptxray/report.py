"""Build the single-file HTML report.

Everything is inlined - no CDN, no server, no build step. The file can be
opened from disk, mailed, or committed as a CI artefact.

The hero of the page is the prompt itself, laid out block by block with a bar
showing how much each block contributes to the score. That is the picture the
whole tool exists to produce.
"""

from __future__ import annotations

import html
from datetime import datetime

from .ablation import AblationReport
from .dataset import Example
from .metrics import UNPARSED, Score
from .runner import RunResult

CSS = """
:root {
  --paper: #f6f7f9;
  --card: #ffffff;
  --ink: #16202e;
  --muted: #6b7787;
  --rule: #dde2e9;
  --helps: #1b4d8f;
  --hurts: #b4690e;
  --flat: #98a2b0;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 40px 20px 80px;
  background: var(--paper); color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.55;
}
.wrap { max-width: 880px; margin: 0 auto; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em;
     font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
h2 { font-size: 15px; margin: 44px 0 12px; font-weight: 600;
     font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 28px; }
.note { color: var(--muted); font-size: 13px; max-width: 62ch; margin: 0 0 16px; }

.stats { display: flex; flex-wrap: wrap; gap: 1px; background: var(--rule);
         border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
.stat { background: var(--card); padding: 12px 16px; flex: 1 1 120px; }
.stat b { display: block; font-size: 20px; font-weight: 600;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.stat span { font-size: 12px; color: var(--muted); }

.legend { display: flex; flex-wrap: wrap; gap: 18px; margin: 0 0 18px;
  font-size: 12.5px; color: var(--muted); align-items: center; }
.swatch { display: inline-block; width: 22px; height: 11px; border-radius: 2px;
  vertical-align: -1px; margin-right: 7px; }
.sw-helps { background: rgba(27,77,143,.20); border-left: 3px solid var(--helps); }
.sw-hurts { background: rgba(180,105,14,.20); border-left: 3px solid var(--hurts); }
.sw-flat  { background: transparent; border-left: 3px solid var(--flat); }

.prompt { background: var(--card); border: 1px solid var(--rule); border-radius: 3px;
  padding: 20px 18px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 13.5px; line-height: 1.6; }

.blk { display: block; position: relative; padding: 8px 100px 8px 11px; margin: 0 0 12px;
  border-left: 3px solid transparent; white-space: pre-wrap; word-break: break-word;
  cursor: default; transition: background .12s ease; }
.blk:last-child { margin-bottom: 0; }
.blk.helps { background: rgba(27,77,143,.10); border-left-color: var(--helps); }
.blk.hurts { background: rgba(180,105,14,.12); border-left-color: var(--hurts); }
.blk.flat  { border-left-color: var(--flat); border-left-style: dotted; }
.blk.kept  { border-left-color: var(--rule); color: var(--muted); }
.blk:hover, .blk:focus-visible { background: rgba(22,32,46,.06); }
.blk:focus-visible { outline: 2px solid var(--helps); outline-offset: 2px; }
.blk .tag { position: absolute; right: 10px; top: 8px; font-size: 11px; color: var(--muted);
  letter-spacing: .02em; }
.blk .score { font-weight: 600; }
.blk.helps .score { color: var(--helps); }
.blk.hurts .score { color: var(--hurts); }

.tip { display: none; position: absolute; left: 0; top: calc(100% - 4px); z-index: 20;
  width: min(460px, 92vw); background: var(--ink); color: #eef2f7; border-radius: 4px;
  padding: 13px 15px; font-size: 12.5px; line-height: 1.5; white-space: normal;
  box-shadow: 0 8px 24px rgba(22,32,46,.28);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.blk:hover .tip, .blk:focus .tip, .blk:focus-within .tip { display: block; }
.tip h4 { margin: 0 0 6px; font-size: 13px; font-weight: 600; }
.tip p { margin: 0 0 8px; color: #c3cdd9; }
.tip p:last-child { margin-bottom: 0; }
.tip .fix { border-top: 1px solid rgba(255,255,255,.16); padding-top: 8px; color: #fff; }
.tip code { background: rgba(255,255,255,.12); padding: 1px 5px; border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; }
.tip .ex { color: #9fb2c7; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11.5px; display: block; margin: 3px 0 0; }
.hint { color: var(--muted); font-size: 12.5px; margin: 10px 0 0; }

table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--muted); font-size: 12px; }
td.num-cell, th.num-cell { text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.diag { background: #eef2f7; font-weight: 600; }
.pos { color: var(--hurts); }
.neg { color: var(--helps); }

details { border-top: 1px solid var(--rule); padding: 10px 0; }
summary { cursor: pointer; font-size: 13px; }
summary:focus-visible { outline: 2px solid var(--helps); outline-offset: 3px; }
.err { font-size: 13px; margin: 8px 0 0; padding-left: 14px; border-left: 2px solid var(--rule); }
.err b { font-weight: 600; }
footer { margin-top: 56px; color: var(--muted); font-size: 12px;
         border-top: 1px solid var(--rule); padding-top: 14px; }
@media (max-width: 640px) {
  .blk { padding-right: 12px; padding-top: 24px; }
  .blk .tag { top: 5px; right: 11px; }
}
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _fmt_delta(value: float) -> str:
    css = "pos" if value > 0.005 else "neg" if value < -0.005 else ""
    return f'<span class="{css}">{value:+.3f}</span>'


def _kind(effect) -> str:
    """helps / hurts / flat, from the measured contribution."""
    if effect.contribution > 0.005:
        return "helps"
    if effect.contribution < -0.005:
        return "hurts"
    return "flat"


def _explain(effect, by_id: dict[int, Example], tested: int) -> tuple[str, list[str], str]:
    """Turn one measurement into a headline, the evidence, and a suggested fix.

    Everything here comes from numbers the run actually produced. Nothing is
    guessed: if the tool cannot show which examples moved, it says so.
    """
    if effect.skipped_reason:
        return (
            "Not tested",
            [f"This block was left in every run because it is {_esc(effect.skipped_reason)}."],
            "Removing it would break the prompt instead of testing an idea, so its "
            "effect cannot be measured.",
        )

    moved = effect.delta_per_class or {}
    kind = _kind(effect)

    def samples(ids: list[int], arrow: str) -> list[str]:
        out = []
        for example_id in ids[:2]:
            example = by_id.get(example_id)
            if example:
                out.append(
                    f'<span class="ex">{arrow} [{_esc(example.label)}] '
                    f"{_esc(example.text[:110])}</span>"
                )
        return out

    if kind == "helps":
        worst = min(moved, key=lambda k: moved[k]) if moved else None
        why = [
            f"Taking this block out drops macro F1 by {abs(effect.delta_macro_f1):.3f} "
            f"on the {tested} examples it was tested on.",
            f"{len(effect.broken)} example(s) that the full prompt gets right become wrong "
            "without it."
            + (f" The class that suffers most is <code>{_esc(worst)}</code>." if worst else ""),
        ]
        why += samples(effect.broken, "breaks:")
        fix = ("Keep this block. If you want the prompt shorter, rewrite it rather than "
               "delete it, then confirm the rewrite with <code>promptxray diff</code>.")
        return "This block is doing work", why, fix

    if kind == "hurts":
        best = max(moved, key=lambda k: moved[k]) if moved else None
        why = [
            f"Removing this block raises macro F1 by {effect.delta_macro_f1:.3f}. "
            "It is costing you accuracy.",
            f"{len(effect.fixed)} example(s) that the full prompt gets wrong become correct "
            "once it is gone."
            + (f" The gain lands mostly on class <code>{_esc(best)}</code>." if best else ""),
        ]
        why += samples(effect.fixed, "fixes:")
        fix = ("Delete it, or narrow it so it only fires where you meant it to. The rule is "
               "probably right in spirit and too broad in wording. Verify the change with "
               "<code>promptxray diff --before old.txt --after new.txt</code>.")
        return "This block is costing you score", why, fix

    why = [
        f"Removing this block produced identical predictions on all {tested} tested "
        "examples. Macro F1 moved by less than 0.005.",
        "Instructions like this often read as useful to a human and are invisible to the "
        "model.",
    ]
    fix = ("Safe to delete: it spends tokens and latency for no measurable gain. If you are "
           "convinced it matters, re-run with a larger <code>--subset-size</code> before "
           "trusting this verdict.")
    return "This block changes nothing", why, fix


def render_report(
    *,
    prompt_path: str,
    data_path: str,
    model: str,
    provider_name: str,
    baseline: RunResult,
    baseline_score: Score,
    examples: list[Example],
    ablation: AblationReport | None = None,
    max_errors: int = 15,
) -> str:
    by_id = {e.id: e for e in examples}
    cost = baseline.cost_usd(model)
    parts: list[str] = []

    parts.append(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>promptxray - {_esc(prompt_path)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>promptxray</h1>
<p class="sub mono">{_esc(prompt_path)} &middot; {_esc(data_path)} &middot; {_esc(model)}
 via {_esc(provider_name)} &middot; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>""")

    # --- headline numbers -------------------------------------------------
    parts.append('<div class="stats">')
    parts.append(f'<div class="stat"><b>{baseline_score.accuracy:.1%}</b><span>accuracy</span></div>')
    parts.append(f'<div class="stat"><b>{baseline_score.macro_f1:.3f}</b><span>macro F1</span></div>')
    parts.append(f'<div class="stat"><b>{baseline_score.total}</b><span>examples</span></div>')
    parts.append(f'<div class="stat"><b>{baseline_score.unparsed}</b><span>unparsable answers</span></div>')
    calls = baseline.api_calls + (ablation.api_calls if ablation else 0)
    hits = baseline.cache_hits + (ablation.cache_hits if ablation else 0)
    parts.append(f'<div class="stat"><b>{calls}</b><span>API calls ({hits} cached)</span></div>')
    if cost is not None:
        parts.append(f'<div class="stat"><b>${cost:.4f}</b><span>estimated cost</span></div>')
    parts.append("</div>")

    # --- hero: the prompt, block by block --------------------------------
    if ablation:
        parts.append("<h2>What each block of the prompt does</h2>")
        parts.append(
            '<p class="note">This is your prompt. Each block was removed on its own and '
            f"the prompt re-run on {ablation.subset_size} examples. Hover a block - or tab "
            "to it - to see why it is coloured that way and what to do about it.</p>"
        )
        parts.append(
            '<div class="legend">'
            '<span><i class="swatch sw-helps"></i>removing it lowers the score</span>'
            '<span><i class="swatch sw-hurts"></i>removing it raises the score</span>'
            '<span><i class="swatch sw-flat"></i>no measurable effect</span>'
            "</div>"
        )

        parts.append('<div class="prompt">')
        for effect in ablation.effects:
            block = effect.block
            kind = "kept" if effect.skipped_reason else _kind(effect)
            headline, why, fix = _explain(effect, by_id, ablation.subset_size)

            if effect.skipped_reason:
                tag = "kept"
            else:
                tag = f'#{block.index + 1} <span class="score">' \
                      f"{effect.delta_macro_f1:+.3f}</span>"

            # Built as one string on purpose: the block is white-space: pre-wrap,
            # so a newline between these pieces would show up as a blank line.
            tooltip = (
                f'<span class="tip"><h4>{headline}</h4>'
                + "".join(f"<p>{line}</p>" for line in why)
                + f'<p class="fix">{fix}</p></span>'
            )
            parts.append(
                f'<span class="blk {kind}" tabindex="0">'
                f'<span class="tag">{tag}</span>'
                f"{_esc(block.text)}{tooltip}</span>"
            )
        parts.append("</div>")
        parts.append('<p class="hint">The number on each block is the change in macro F1 '
                     "when that block is removed. Negative means the block is holding the "
                     "score up.</p>")

        # --- ablation table ----------------------------------------------
        parts.append("<h2>Per-class effect of removing each block</h2>")
        parts.append("<table><thead><tr><th>block</th><th class='num-cell'>macro F1</th>")
        for label in baseline_score.labels:
            parts.append(f"<th class='num-cell'>F1 {_esc(label)}</th>")
        parts.append("<th class='num-cell'>fixed</th><th class='num-cell'>broken</th></tr></thead><tbody>")
        ordered = sorted(
            [e for e in ablation.effects if not e.skipped_reason],
            key=lambda e: e.contribution,
            reverse=True,
        )
        for effect in ordered:
            parts.append(f"<tr><td class='mono'>#{effect.block.index + 1} "
                         f"{_esc(effect.block.preview)}</td>")
            parts.append(f"<td class='num-cell'>{_fmt_delta(effect.delta_macro_f1)}</td>")
            for label in baseline_score.labels:
                parts.append(f"<td class='num-cell'>"
                             f"{_fmt_delta(effect.delta_per_class.get(label, 0.0))}</td>")
            parts.append(f"<td class='num-cell'>{len(effect.fixed)}</td>"
                         f"<td class='num-cell'>{len(effect.broken)}</td></tr>")
        parts.append("</tbody></table>")
        parts.append('<p class="note">Ordered by how much the block helps. Anything at the '
                     "bottom with a positive macro F1 delta is a candidate for deletion.</p>")

    # --- per class ---------------------------------------------------------
    parts.append("<h2>Scores by class</h2>")
    parts.append("<table><thead><tr><th>class</th><th class='num-cell'>precision</th>"
                 "<th class='num-cell'>recall</th><th class='num-cell'>F1</th>"
                 "<th class='num-cell'>support</th></tr></thead><tbody>")
    for label in baseline_score.labels:
        c = baseline_score.per_class[label]
        parts.append(f"<tr><td class='mono'>{_esc(label)}</td>"
                     f"<td class='num-cell'>{c.precision:.3f}</td>"
                     f"<td class='num-cell'>{c.recall:.3f}</td>"
                     f"<td class='num-cell'>{c.f1:.3f}</td>"
                     f"<td class='num-cell'>{c.support}</td></tr>")
    parts.append("</tbody></table>")

    # --- confusion matrix --------------------------------------------------
    parts.append("<h2>Confusion matrix</h2>")
    parts.append('<p class="note">Rows are the true label, columns are what the model '
                 "answered. The diagonal is correct.</p>")
    columns = baseline_score.labels + [UNPARSED]
    parts.append("<table><thead><tr><th>true \\ predicted</th>")
    for column in columns:
        parts.append(f"<th class='num-cell'>{_esc(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row_label in baseline_score.labels:
        parts.append(f"<tr><td class='mono'>{_esc(row_label)}</td>")
        for column in columns:
            count = baseline_score.confusion.get(row_label, {}).get(column, 0)
            cell_class = "num-cell diag" if row_label == column else "num-cell"
            parts.append(f"<td class='{cell_class}'>{count}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # --- errors ------------------------------------------------------------
    wrong = sorted(baseline_score.wrong_ids)[:max_errors]
    if wrong:
        parts.append("<h2>Examples the prompt got wrong</h2>")
        parts.append("<details open><summary>"
                     f"{len(baseline_score.wrong_ids)} wrong, showing {len(wrong)}</summary>")
        for example_id in wrong:
            example = by_id[example_id]
            got = baseline.predictions.get(example_id, UNPARSED)
            raw = baseline.raw.get(example_id, "")
            parts.append(
                f'<p class="err">{_esc(example.text[:300])}<br>'
                f'<b>expected</b> <span class="mono">{_esc(example.label)}</span> &middot; '
                f'<b>got</b> <span class="mono">{_esc(got)}</span>'
                + (f' &middot; raw: <span class="mono">{_esc(raw[:80])}</span>' if raw != got else "")
                + "</p>"
            )
        parts.append("</details>")

    parts.append('<footer>Generated by promptxray. Re-run with the same prompt to get the '
                 "same numbers - every answer is cached on disk.</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts)
