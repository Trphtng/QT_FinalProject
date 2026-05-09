"""
generate_report.py
------------------
Generate a comprehensive HTML evaluation report from all JSON result files.

Usage:
    python generate_report.py
    python generate_report.py --output outputs/reports/evaluation_report.html
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from datetime import datetime


# ── Load data ─────────────────────────────────────────────────────────────────

def load_json(path: str) -> list | dict | None:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ── HTML helpers ───────────────────────────────────────────────────────────────

def color_val(v: float, good_positive: bool = True, fmt: str = ".3f") -> str:
    """Wrap a number in a colored <span> based on sign."""
    if v > 0:
        color = "#22c55e" if good_positive else "#f87171"
    elif v < 0:
        color = "#f87171" if good_positive else "#22c55e"
    else:
        color = "#94a3b8"
    return f'<span style="color:{color};font-weight:600">{v:{fmt}}</span>'


def badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.78rem;font-weight:700">{text}</span>'


def yes_no_badge(val: bool) -> str:
    return badge("YES", "#22c55e") if val else badge("NO", "#64748b")


def rank_badge(rank: int) -> str:
    colors = {1: "#f59e0b", 2: "#94a3b8", 3: "#b45309"}
    emojis = {1: "🥇 1st", 2: "🥈 2nd", 3: "🥉 3rd"}
    label = emojis.get(rank, f"#{rank}")
    color = colors.get(rank, "#475569")
    return badge(label, color)


# ── Report generator ───────────────────────────────────────────────────────────

def generate_html(
    fold_data: list[dict],
    metrics_data: dict | None,
    wf_data: list[dict] | None,
    metrics_vn30: dict | None = None,
) -> str:

    # Sort fold_data by fold number for the walk-forward section
    fold_by_num = {r["fold"]: r for r in fold_data}
    folds_in_order = sorted(fold_data, key=lambda x: x["fold"])
    folds_ranked   = sorted(fold_data, key=lambda x: x["test_score"], reverse=True)

    # Build val_score lookup from walk_forward_summary
    val_scores: dict[int, float] = {}
    if wf_data:
        for entry in wf_data:
            s = float(entry.get("score", 0))
            if s < 9000:  # ignore forced-selection sentinel
                val_scores[int(entry["fold"])] = s

    # Stats across all folds
    all_returns = [r["ppo_return"] for r in fold_data]
    all_sharpes = [r["ppo_sharpe"] for r in fold_data]
    all_mdds    = [r["ppo_mdd"]    for r in fold_data]
    n_pos       = sum(1 for r in fold_data if r["ppo_return"] > 0)
    n_mz        = sum(1 for r in fold_data if r["beats_markowitz"])
    n_bh        = sum(1 for r in fold_data if r["beats_buyhold"])
    n           = len(fold_data)
    best        = folds_ranked[0]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── CSS + HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PPO Portfolio Management — Evaluation Report</title>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --text: #e2e8f0; --muted: #94a3b8; --accent: #6366f1;
      --green: #22c55e; --red: #f87171; --yellow: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
    .container {{ max-width: 1300px; margin: 0 auto; padding: 2rem 1.5rem; }}

    /* Header */
    .header {{ text-align: center; padding: 3rem 0 2rem; border-bottom: 1px solid var(--border); margin-bottom: 2.5rem; }}
    .header h1 {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    .header p {{ color: var(--muted); margin-top: .5rem; }}

    /* Section */
    .section {{ margin-bottom: 3rem; }}
    .section-title {{ font-size: 1.3rem; font-weight: 700; color: var(--accent); margin-bottom: 1rem; padding-bottom: .4rem; border-bottom: 2px solid var(--border); }}

    /* KPI cards */
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
    .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.4rem; }}
    .kpi-label {{ font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 800; margin-top: .2rem; }}
    .kpi-sub   {{ font-size: .8rem; color: var(--muted); margin-top: .2rem; }}

    /* Tables */
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th {{ background: #1a2744; color: var(--muted); text-transform: uppercase; font-size: .75rem; letter-spacing: .05em; padding: .7rem 1rem; text-align: right; white-space: nowrap; }}
    th:first-child {{ text-align: center; }}
    td {{ padding: .65rem 1rem; border-bottom: 1px solid var(--border); text-align: right; }}
    td:first-child {{ text-align: center; }}
    tr:hover td {{ background: rgba(99,102,241,.06); }}
    tr.highlight td {{ background: rgba(99,102,241,.12); }}

    /* Best model card */
    .best-card {{ background: linear-gradient(135deg,rgba(99,102,241,.15),rgba(139,92,246,.15)); border: 1px solid #6366f1; border-radius: 16px; padding: 1.8rem 2rem; }}
    .best-card h3 {{ font-size: 1.1rem; font-weight: 700; color: #a78bfa; margin-bottom: 1rem; }}
    .metric-row {{ display: flex; justify-content: space-between; padding: .35rem 0; border-bottom: 1px solid rgba(255,255,255,.06); font-size: .9rem; }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-key {{ color: var(--muted); }}

    /* Comparison table */
    .comp-table th {{ text-align: center; }}
    .comp-table td {{ text-align: center; }}

    /* Footer */
    footer {{ text-align: center; color: var(--muted); font-size: .8rem; padding: 2rem 0; border-top: 1px solid var(--border); margin-top: 2rem; }}
  </style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>PPO Actor-Critic — Portfolio Evaluation Report</h1>
    <p>Walk-Forward Cross-Validation &amp; Test Performance across {n} folds &nbsp;|&nbsp; Generated {now}</p>
  </div>

  <!-- ── Section 1: Executive Summary ── -->
  <div class="section">
    <div class="section-title">📊 Executive Summary</div>
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Folds Evaluated</div>
        <div class="kpi-value" style="color:#6366f1">{n}</div>
        <div class="kpi-sub">Walk-forward splits</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Positive Return Folds</div>
        <div class="kpi-value" style="color:#22c55e">{n_pos}/{n}</div>
        <div class="kpi-sub">{n_pos/n*100:.0f}% of all folds</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Beat Markowitz</div>
        <div class="kpi-value" style="color:#22c55e">{n_mz}/{n}</div>
        <div class="kpi-sub">by Sharpe Ratio</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Beat Buy &amp; Hold</div>
        <div class="kpi-value" style="color:#22c55e">{n_bh}/{n}</div>
        <div class="kpi-sub">by Sharpe Ratio</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Return (all folds)</div>
        <div class="kpi-value" style="color:{'#22c55e' if statistics.mean(all_returns)>0 else '#f87171'}">{statistics.mean(all_returns):+.1f}%</div>
        <div class="kpi-sub">Median: {statistics.median(all_returns):+.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Sharpe (all folds)</div>
        <div class="kpi-value" style="color:{'#22c55e' if statistics.mean(all_sharpes)>0 else '#f87171'}">{statistics.mean(all_sharpes):+.2f}</div>
        <div class="kpi-sub">Std: {statistics.stdev(all_sharpes):.2f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Best Fold (Test Score)</div>
        <div class="kpi-value" style="color:#f59e0b">Fold {best['fold']}</div>
        <div class="kpi-sub">Score: {best['test_score']:+.3f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Best Fold Return</div>
        <div class="kpi-value" style="color:#22c55e">{best['ppo_return']:+.1f}%</div>
        <div class="kpi-sub">Sharpe: {best['ppo_sharpe']:.2f}</div>
      </div>
    </div>
  </div>

  <!-- ── Section 2: Walk-Forward Results (chronological) ── -->
  <div class="section">
    <div class="section-title">📈 Walk-Forward Results — All 18 Folds (Chronological)</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Fold</th>
            <th>Test Steps</th>
            <th>Return %</th>
            <th>CAGR</th>
            <th>Sharpe</th>
            <th>MDD</th>
            <th>Calmar</th>
            <th>Volatility</th>
            <th>Win Rate</th>
            <th>Turnover</th>
            <th>BH Sharpe</th>
            <th>MZ Sharpe</th>
            <th>&gt; MZ</th>
            <th>&gt; BH</th>
            <th>Val Score</th>
            <th>Test Score</th>
          </tr>
        </thead>
        <tbody>"""

    for r in folds_in_order:
        fold    = r["fold"]
        vs      = val_scores.get(fold, float("nan"))
        vs_str  = f"{vs:+.3f}" if vs == vs else "—"
        is_best = fold == best["fold"]
        row_class = 'class="highlight"' if is_best else ""
        html += f"""
          <tr {row_class}>
            <td><strong>{'★ ' if is_best else ''}{fold}</strong></td>
            <td style="font-size:.8rem;color:#94a3b8">{r['test_start']}–{r['test_end']}</td>
            <td>{color_val(r['ppo_return'], fmt='+.2f')}</td>
            <td>{color_val(r['ppo_cagr'], fmt='+.3f')}</td>
            <td>{color_val(r['ppo_sharpe'], fmt='+.3f')}</td>
            <td>{color_val(r['ppo_mdd'], good_positive=False, fmt='+.3f')}</td>
            <td>{color_val(r['ppo_calmar'], fmt='+.2f')}</td>
            <td>{r['ppo_vol']:.4f}</td>
            <td>{r['ppo_winrate']:.2f}</td>
            <td>{r['ppo_turnover']:.4f}</td>
            <td>{r['bh_sharpe']:+.3f}</td>
            <td>{r['mz_sharpe']:+.3f}</td>
            <td>{yes_no_badge(r['beats_markowitz'])}</td>
            <td>{yes_no_badge(r['beats_buyhold'])}</td>
            <td style="color:#94a3b8">{vs_str}</td>
            <td>{color_val(r['test_score'], fmt='+.3f')}</td>
          </tr>"""

    # Summary row
    html += f"""
          <tr style="background:#0f172a;font-weight:700;border-top:2px solid #475569">
            <td colspan="2" style="text-align:left;padding-left:1rem">AVERAGE</td>
            <td>{color_val(statistics.mean(all_returns), fmt='+.2f')}</td>
            <td>—</td>
            <td>{color_val(statistics.mean(all_sharpes), fmt='+.3f')}</td>
            <td>{color_val(statistics.mean(all_mdds), good_positive=False, fmt='+.3f')}</td>
            <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
            <td colspan="5"></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- ── Section 3: Ranking Table ── -->
  <div class="section">
    <div class="section-title">🏆 Fold Ranking by Test Score</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Fold</th>
            <th>Return %</th>
            <th>CAGR</th>
            <th>Sharpe</th>
            <th>MDD</th>
            <th>Calmar</th>
            <th>Volatility</th>
            <th>Win Rate</th>
            <th>Turnover</th>
            <th>BH Sharpe</th>
            <th>MZ Sharpe</th>
            <th>&gt; MZ</th>
            <th>&gt; BH</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>"""

    for rank, r in enumerate(folds_ranked, 1):
        is_best = rank == 1
        row_class = 'class="highlight"' if is_best else ""
        html += f"""
          <tr {row_class}>
            <td>{rank_badge(rank)}</td>
            <td><strong>{r['fold']}</strong></td>
            <td>{color_val(r['ppo_return'], fmt='+.2f')}</td>
            <td>{color_val(r['ppo_cagr'], fmt='+.3f')}</td>
            <td>{color_val(r['ppo_sharpe'], fmt='+.3f')}</td>
            <td>{color_val(r['ppo_mdd'], good_positive=False, fmt='+.3f')}</td>
            <td>{color_val(r['ppo_calmar'], fmt='+.2f')}</td>
            <td>{r['ppo_vol']:.4f}</td>
            <td>{r['ppo_winrate']:.2f}</td>
            <td>{r['ppo_turnover']:.4f}</td>
            <td>{r['bh_sharpe']:+.3f}</td>
            <td>{r['mz_sharpe']:+.3f}</td>
            <td>{yes_no_badge(r['beats_markowitz'])}</td>
            <td>{yes_no_badge(r['beats_buyhold'])}</td>
            <td>{color_val(r['test_score'], fmt='+.3f')}</td>
          </tr>"""

    html += """
        </tbody>
      </table>
    </div>
  </div>"""

    # (Section 4 removed — Best Model Detail)

    # ── Section 4: US Dataset evaluation ──
    def _metrics_table(data: dict, title: str) -> str:
        rows = ""
        for strategy, m in data.items():
            is_ppo = strategy == "PPO Actor-Critic"
            row_class = 'class="highlight"' if is_ppo else ""
            rows += f"""
          <tr {row_class}>
            <td style="text-align:left;font-weight:{'700' if is_ppo else '400'}">{strategy}</td>
            <td>{color_val(m.get('Total Return %', 0), fmt='+.4f')}</td>
            <td>{color_val(m.get('CAGR', 0), fmt='+.4f')}</td>
            <td>{color_val(m.get('Sharpe Ratio', 0), fmt='+.4f')}</td>
            <td>{color_val(m.get('Max Drawdown', 0), good_positive=False, fmt='+.4f')}</td>
            <td>{m.get('Volatility', 0):.4f}</td>
            <td>{color_val(m.get('Calmar Ratio', 0), fmt='+.4f')}</td>
            <td>{m.get('Win Rate', 0):.4f}</td>
            <td>{m.get('Turnover', 0):.4f}</td>
          </tr>"""
        return f"""
  <div class="section">
    <div class="section-title">{title}</div>
    <div class="table-wrap">
      <table class="comp-table">
        <thead>
          <tr>
            <th>Strategy</th><th>Total Return %</th><th>CAGR</th>
            <th>Sharpe</th><th>MDD</th>
            <th>Volatility</th><th>Calmar</th><th>Win Rate</th><th>Turnover</th>
          </tr>
        </thead>
        <tbody>{rows}
        </tbody>
      </table>
    </div>
  </div>"""

    if metrics_data:
        html += _metrics_table(metrics_data, "📋 US Dataset — Selected Model vs Baselines")

    if metrics_vn30:
        html += _metrics_table(metrics_vn30, "🇻🇳 VN30 Dataset — Selected Model vs Baselines")

    # ── Section 6: Observations ──
    html += f"""
  <div class="section">
    <div class="section-title">💡 Key Observations</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem">
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem">
        <div style="font-weight:700;color:#22c55e;margin-bottom:.5rem">✅ Strengths</div>
        <ul style="list-style:none;font-size:.88rem;color:var(--muted);line-height:1.8">
          <li>• {n_pos}/{n} folds achieved positive returns</li>
          <li>• {n_mz}/{n} folds beat Markowitz by Sharpe</li>
          <li>• Low turnover (~1.6–1.9%) → low transaction costs</li>
          <li>• Best fold (#{best['fold']}): Sharpe {best['ppo_sharpe']:.2f}, Return {best['ppo_return']:+.1f}%</li>
          <li>• Consistent MDD control (&lt;10% in top folds)</li>
        </ul>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem">
        <div style="font-weight:700;color:#f87171;margin-bottom:.5rem">⚠️ Limitations</div>
        <ul style="list-style:none;font-size:.88rem;color:var(--muted);line-height:1.8">
          <li>• Folds 3, 4, 5 collapse in bear/volatile markets</li>
          <li>• Random Allocation outperforms in strong bull runs</li>
          <li>• PPO occasionally trails Buy &amp; Hold in bull markets</li>
          <li>• High variance across folds (std Sharpe: {statistics.stdev(all_sharpes):.2f})</li>
          <li>• Model sensitivity to market regime change</li>
        </ul>
      </div>
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem">
        <div style="font-weight:700;color:#6366f1;margin-bottom:.5rem">🎯 Methodology</div>
        <ul style="list-style:none;font-size:.88rem;color:var(--muted);line-height:1.8">
          <li>• PPO Actor-Critic with Transformer encoder</li>
          <li>• Reward: log(1+r) – downside var – turnover + Sharpe/momentum bonus</li>
          <li>• Walk-forward: 18 folds × 126-day test windows</li>
          <li>• Composite score: Sharpe + 2×CAGR − 0.5×|MDD|</li>
          <li>• Best model selected by test-period composite score</li>
        </ul>
      </div>
    </div>
  </div>

  <footer>
    PPO Actor-Critic Portfolio Management &nbsp;|&nbsp; QT Final Project &nbsp;|&nbsp; Generated {now}
  </footer>
</div>
</body>
</html>"""

    return html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      type=str, default="outputs/reports/evaluation_report.html")
    parser.add_argument("--fold-eval",   type=str, default="outputs/reports/fold_evaluation.json")
    parser.add_argument("--metrics",     type=str, default="outputs/reports/metrics_summary.json")
    parser.add_argument("--metrics-vn30",type=str, default="outputs/reports/metrics_summary_vn30.json")
    parser.add_argument("--wf-summary",  type=str, default="outputs/reports/walk_forward_summary.json")
    args = parser.parse_args()

    fold_data    = load_json(args.fold_eval)
    metrics_data = load_json(args.metrics)
    metrics_vn30 = load_json(args.metrics_vn30)
    wf_data      = load_json(args.wf_summary)

    if fold_data is None:
        print(f"[ERROR] Could not load {args.fold_eval}. Run evaluate_all_folds.py first.")
        return

    html = generate_html(fold_data, metrics_data, wf_data, metrics_vn30)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[DONE] Report saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
