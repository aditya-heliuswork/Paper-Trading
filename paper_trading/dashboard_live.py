# ============================================================================
# PETROQUANT PAPER TRADING — LIVE DASHBOARD
# ============================================================================
# Generates a premium 10-panel Plotly HTML dashboard from the trade log.
# Saved as paper_dashboard.html — auto-refreshes every 60 seconds in browser.
#
# Panels:
#   1. Equity Curve — portfolio value vs Buy & Hold
#   2. WTI Price + BUY/SELL signal markers  ← Bug #1 FIXED (price line was blank)
#   3. Per-Trade P&L Bar Chart
#   4. Market Regime Timeline
#   5. Model Confidence (probability over time)
#   6. Rolling Win Rate
#   7. Win / Loss Distribution Histogram
#   8. Feature Importance
#   9. Realized P&L Over Time
#   10. Drawdown Chart
#
# Bug Fixes:
#   #1  — Price line invisible: X-axes of price and signals are now aligned by
#          converting both to tz-naive UTC strings before passing to Plotly.
#   #7  — Dashboard crashes if current_price is None on cold start.
#   #11 — pos_str crashes if open_position dict is missing expected keys.
# ============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging

from . import config as cfg

logger = logging.getLogger(__name__)

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg'           : '#0d1117',
    'bg_panel'     : '#161b22',
    'bg_card'      : '#1c2128',
    'accent_green' : '#2ea043',
    'accent_red'   : '#f85149',
    'accent_blue'  : '#388bfd',
    'accent_gold'  : '#d29922',
    'accent_purple': '#a5a5f5',
    'text_primary' : '#e6edf3',
    'text_muted'   : '#8b949e',
    'grid'         : '#21262d',
    'buy_marker'   : '#2ea043',
    'sell_marker'  : '#f85149',
}


class LiveDashboard:
    """
    Generates a premium dark-mode Plotly HTML dashboard from trade log data.
    """

    def __init__(self, trade_log, portfolio=None, price_feed_df=None,
                 feature_importance=None):
        self.trade_log          = trade_log
        self.portfolio          = portfolio
        self.price_feed_df      = price_feed_df
        self.feature_importance = feature_importance

    # ── Main Render ───────────────────────────────────────────────────────────
    def render(self, current_price: float = None, regime: str = 'UNKNOWN',
               model_status: dict = None) -> str:
        """
        Builds the full 10-panel dashboard and returns HTML string.

        Bug #7 fix: current_price=None is now handled gracefully throughout.
        """
        trades    = self.trade_log.get_trade_history(limit=500)
        snapshots = self.trade_log.get_snapshots(limit=2000)
        summary   = self.trade_log.get_summary()

        # Bug #7 fix: pass current_price only if it's a valid positive number
        safe_price = current_price if (current_price and current_price > 0) else None
        snap_port  = self.portfolio.get_snapshot(safe_price) if self.portfolio else {}

        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                'Portfolio Equity Curve',
                f'WTI {cfg.ACTIVE_TIMEFRAME} Price + Signals',
                'Per-Trade P&L',
                'Market Regime Timeline',
                'Model Confidence (Probability)',
                'Rolling Win Rate',
                'Win / Loss Distribution',
                'Feature Importance',
                'Realized P&L Over Time',
                'Drawdown Chart',
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
            specs=[
                [{"type": "scatter"},   {"type": "scatter"}],
                [{"type": "bar"},       {"type": "scatter"}],
                [{"type": "scatter"},   {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"},   {"type": "scatter"}],
            ]
        )

        self._add_equity_curve(fig, snapshots, safe_price, row=1, col=1)
        self._add_price_signals(fig, trades, row=1, col=2)          # Bug #1 fix inside
        self._add_pnl_bars(fig, trades, row=2, col=1)
        self._add_regime_timeline(fig, snapshots, row=2, col=2)
        self._add_confidence_chart(fig, snapshots, row=3, col=1)
        self._add_rolling_winrate(fig, trades, row=3, col=2)
        self._add_win_loss_hist(fig, trades, row=4, col=1)
        self._add_feature_importance(fig, row=4, col=2)
        self._add_cumulative_pnl(fig, snapshots, row=5, col=1)
        self._add_drawdown(fig, snapshots, row=5, col=2)

        table_html = self._build_trade_table_html(trades)

        # Bug #7 fix: safe price formatting in title
        price_str = f'${safe_price:.2f}' if safe_price else 'Waiting...'
        fig.update_layout(
            title=dict(
                text=(f'PetroQuant Paper Trader  |  WTI {price_str}  |  '
                      f'Regime: {regime}  |  TF: {cfg.ACTIVE_TIMEFRAME}  |  '
                      f'Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}'),
                font=dict(size=16, color=COLORS['text_primary']),
                x=0.01
            ),
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg_panel'],
            font=dict(family='Inter, system-ui, sans-serif',
                      size=11, color=COLORS['text_primary']),
            height=2200,
            showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        )

        fig.update_xaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
                         showgrid=True, linecolor=COLORS['grid'])
        fig.update_yaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
                         showgrid=True, linecolor=COLORS['grid'])

        html = self._wrap_html(fig, table_html, summary, snap_port,
                               model_status, regime, safe_price)
        return html

    # ── Save ─────────────────────────────────────────────────────────────────
    def save(self, html: str, path: str = cfg.DASHBOARD_PATH) -> str:
        """Saves the HTML dashboard to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"[Dashboard] Saved to {path}")
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    def _add_equity_curve(self, fig, snapshots, current_price, row, col):
        if snapshots.empty:
            return

        equities = snapshots['equity'].values
        times    = snapshots['timestamp']

        fig.add_trace(go.Scatter(
            x=times, y=equities,
            name='Portfolio Equity',
            line=dict(color=COLORS['accent_blue'], width=2),
            fill='tozeroy', fillcolor='rgba(56, 139, 253, 0.06)',
        ), row=row, col=col)

        if current_price and len(snapshots) > 1 and 'wti_price' in snapshots.columns:
            first_price = snapshots['wti_price'].dropna()
            if not first_price.empty and first_price.iloc[0] > 0:
                bnh_equity = cfg.INITIAL_CAPITAL * (snapshots['wti_price'] / first_price.iloc[0])
                fig.add_trace(go.Scatter(
                    x=times, y=bnh_equity,
                    name='Buy & Hold WTI',
                    line=dict(color=COLORS['text_muted'], width=1.5, dash='dash'),
                ), row=row, col=col)

        fig.add_hline(y=cfg.INITIAL_CAPITAL, line_dash='dot',
                      line_color=COLORS['text_muted'], opacity=0.4,
                      annotation_text='Start Capital', row=row, col=col)

    def _add_price_signals(self, fig, trades, row, col):
        """
        Bug #1 fix: Price line was invisible because:
          - price_df.index is a tz-naive DatetimeIndex (UTC)
          - trade timestamps from SQLite are ISO string timestamps (no tz)
          - Plotly treated them as separate axes when mixed with string-formatted
            trade marker times

        Fix: Normalize both to pandas Timestamp (tz-naive UTC) so Plotly
        renders them on the same continuous time axis.
        """
        price_df = self.price_feed_df

        if price_df is not None and not price_df.empty:
            # Use last 200 bars for performance; ensure index is tz-naive
            plot_df = price_df.iloc[-200:].copy()
            if plot_df.index.tz is not None:
                plot_df.index = plot_df.index.tz_localize(None)

            fig.add_trace(go.Scatter(
                x=plot_df.index,                # tz-naive DatetimeIndex
                y=plot_df['Close'],
                name=f'WTI {cfg.ACTIVE_TIMEFRAME}',
                line=dict(color='#58a6ff', width=1.5),
                hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=row, col=col)

        if trades.empty:
            return

        # Normalize trade timestamps to tz-naive pandas Timestamps
        # so they share the same axis as the price line above
        def _to_naive_ts(ts_series):
            """Convert any timestamp format to tz-naive UTC DatetimeIndex."""
            ts = pd.to_datetime(ts_series, utc=False, errors='coerce')
            if ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            return ts

        # BUY markers (OPEN_LONG entries)
        buys = trades[trades['action'] == 'OPEN_LONG'].copy()
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(buys['timestamp']),
                y=buys['price'],
                mode='markers',
                name='LONG Entry',
                marker=dict(symbol='triangle-up', size=10,
                            color=COLORS['buy_marker'],
                            line=dict(color='white', width=1)),
                hovertemplate='LONG @ $%{y:.2f}<br>%{x}<extra></extra>',
            ), row=row, col=col)

        # SELL markers (OPEN_SHORT entries)
        sells = trades[trades['action'] == 'OPEN_SHORT'].copy()
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(sells['timestamp']),
                y=sells['price'],
                mode='markers',
                name='SHORT Entry',
                marker=dict(symbol='triangle-down', size=10,
                            color=COLORS['sell_marker'],
                            line=dict(color='white', width=1)),
                hovertemplate='SHORT @ $%{y:.2f}<br>%{x}<extra></extra>',
            ), row=row, col=col)

        # CLOSE markers (exits)
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if not closes.empty:
            close_colors = [
                COLORS['accent_green'] if p > 0 else COLORS['accent_red']
                for p in closes['pnl_realized'].fillna(0)
            ]
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(closes['timestamp']),
                y=closes['price'],
                mode='markers',
                name='Exit',
                marker=dict(symbol='x', size=8,
                            color=close_colors,
                            line=dict(color='white', width=1)),
                hovertemplate='Exit @ $%{y:.2f}<br>%{x}<extra></extra>',
            ), row=row, col=col)

    def _add_pnl_bars(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if closes.empty:
            return
        closes = closes.sort_values('timestamp')
        colors = [COLORS['accent_green'] if p > 0 else COLORS['accent_red']
                  for p in closes['pnl_realized'].fillna(0)]
        fig.add_trace(go.Bar(
            x=pd.to_datetime(closes['timestamp']),
            y=closes['pnl_realized'],
            name='Trade P&L',
            marker_color=colors,
        ), row=row, col=col)
        fig.add_hline(y=0, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.5, row=row, col=col)

    def _add_regime_timeline(self, fig, snapshots, row, col):
        if snapshots.empty or 'regime' not in snapshots.columns:
            return
        regime_colors = {
            'BULL'  : COLORS['accent_green'],
            'CHOPPY': COLORS['accent_gold'],
            'PANIC' : COLORS['accent_red'],
        }
        for regime_name, color in regime_colors.items():
            mask = snapshots['regime'] == regime_name
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=snapshots.loc[mask, 'timestamp'],
                    y=[regime_name] * mask.sum(),
                    mode='markers',
                    name=regime_name,
                    marker=dict(color=color, size=4, symbol='square'),
                ), row=row, col=col)

    def _add_confidence_chart(self, fig, snapshots, row, col):
        if snapshots.empty or 'probability' not in snapshots.columns:
            return
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'],
            y=snapshots['probability'],
            name='Model Prob (UP)',
            line=dict(color=COLORS['accent_purple'], width=1),
        ), row=row, col=col)
        fig.add_hline(y=cfg.BUY_THRESHOLD, line_dash='dash',
                      line_color=COLORS['accent_green'], opacity=0.6,
                      annotation_text=f'BUY >{cfg.BUY_THRESHOLD}', row=row, col=col)
        fig.add_hline(y=cfg.SELL_THRESHOLD, line_dash='dash',
                      line_color=COLORS['accent_red'], opacity=0.6,
                      annotation_text=f'SELL <{cfg.SELL_THRESHOLD}', row=row, col=col)
        fig.add_hline(y=0.5, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.4, row=row, col=col)

    def _add_rolling_winrate(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if len(closes) < 5:
            return
        closes = closes.sort_values('timestamp')
        closes['win'] = (closes['pnl_realized'] > 0).astype(float)
        closes['rolling_wr'] = closes['win'].rolling(min(20, len(closes))).mean() * 100
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(closes['timestamp']),
            y=closes['rolling_wr'],
            name='Rolling Win Rate %',
            line=dict(color=COLORS['accent_gold'], width=2),
            fill='tozeroy',
            fillcolor='rgba(210, 153, 34, 0.08)',
        ), row=row, col=col)
        fig.add_hline(y=50, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.5, row=row, col=col)

    def _add_win_loss_hist(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)]
        if closes.empty or 'pnl_realized' not in closes.columns:
            return
        pnls = closes['pnl_realized'].dropna()
        if pnls.empty:
            return
        fig.add_trace(go.Histogram(
            x=pnls, nbinsx=30, name='P&L Distribution',
            marker_color=[COLORS['accent_green'] if v > 0 else COLORS['accent_red']
                          for v in pnls],
            opacity=0.8,
        ), row=row, col=col)
        fig.add_vline(x=0, line_dash='solid', line_color='white',
                      opacity=0.5, row=row, col=col)

    def _add_feature_importance(self, fig, row, col):
        fi = self.feature_importance
        if fi is None or (hasattr(fi, 'empty') and fi.empty):
            return
        top10 = fi.head(10).sort_values()
        fig.add_trace(go.Bar(
            y=top10.index.tolist(),
            x=top10.values.tolist(),
            orientation='h',
            name='Feature Importance',
            marker_color=COLORS['accent_blue'],
        ), row=row, col=col)

    def _add_cumulative_pnl(self, fig, snapshots, row, col):
        if snapshots.empty or 'realized_pnl' not in snapshots.columns:
            return
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'],
            y=snapshots['realized_pnl'],
            name='Realized P&L',
            line=dict(color=COLORS['accent_gold'], width=2),
            fill='tozeroy',
            fillcolor='rgba(210, 153, 34, 0.08)',
        ), row=row, col=col)
        fig.add_hline(y=0, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.4, row=row, col=col)

    def _add_drawdown(self, fig, snapshots, row, col):
        if snapshots.empty or 'equity' not in snapshots.columns:
            return
        equities = snapshots['equity'].values
        peak     = np.maximum.accumulate(equities)
        dd_pct   = (equities - peak) / np.where(peak > 0, peak, 1) * 100
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'],
            y=dd_pct,
            name='Drawdown %',
            line=dict(color=COLORS['accent_red'], width=1.5),
            fill='tozeroy',
            fillcolor='rgba(248, 81, 73, 0.08)',
        ), row=row, col=col)

    # ── Trade Table ───────────────────────────────────────────────────────────
    def _build_trade_table_html(self, trades) -> str:
        """Renders the recent trade log as a styled HTML table."""
        recent = trades[trades['action'].str.startswith('OPEN', na=False)].head(50)
        if recent.empty:
            return '<p style="color:#8b949e;padding:16px">No trades yet.</p>'
        recent = recent.sort_values('timestamp', ascending=False)

        def _fmt_pnl(v):
            try:
                val = float(v)
                color = '#2ea043' if val > 0 else ('#f85149' if val < 0 else '#8b949e')
                return f'<span style="color:{color}">${val:+.2f}</span>'
            except Exception:
                return str(v)

        rows_html = ''
        for _, r in recent.iterrows():
            ts     = str(r.get('timestamp', ''))[:16]
            action = r.get('action', '')
            action_color = '#2ea043' if 'LONG' in str(action) else '#f85149'
            price  = f"${float(r.get('price', 0)):.2f}"
            qty    = f"{float(r.get('quantity', 0)):.3f}"
            regime = r.get('regime', '-')
            prob   = f"{float(r.get('probability', 0.5)):.1%}"
            pnl    = _fmt_pnl(r.get('pnl_realized', 0))
            rows_html += f"""
            <tr>
              <td>{ts}</td>
              <td style="color:{action_color};font-weight:600">{action}</td>
              <td>{price}</td><td>{qty}</td>
              <td>{regime}</td><td>{prob}</td><td>{pnl}</td>
            </tr>"""

        return f"""
        <div style="margin:24px 0;background:#161b22;border-radius:12px;
                    border:1px solid #21262d;overflow:hidden">
          <div style="padding:14px 20px;background:#1c2128;border-bottom:1px solid #21262d">
            <span style="font-weight:600;font-size:14px">Trade Log (Last 50 Opens)</span>
          </div>
          <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr style="background:#1c2128;color:#8b949e">
                <th style="padding:8px 12px;text-align:left">Time</th>
                <th style="padding:8px 12px;text-align:left">Action</th>
                <th style="padding:8px 12px;text-align:left">Price</th>
                <th style="padding:8px 12px;text-align:left">Qty</th>
                <th style="padding:8px 12px;text-align:left">Regime</th>
                <th style="padding:8px 12px;text-align:left">Prob</th>
                <th style="padding:8px 12px;text-align:left">P&amp;L</th>
              </tr>
            </thead>
            <tbody style="color:#e6edf3">{rows_html}</tbody>
          </table>
          </div>
        </div>"""

    # ── HTML Wrapper ───────────────────────────────────────────────────────────
    def _wrap_html(self, fig, table_html: str, summary, snap_port,
                   model_status, regime, current_price) -> str:
        """
        Wraps Plotly figure in a full HTML page with:
          - Auto-refresh
          - Status cards
          - Multi-timeframe dropdown selector
          - Bug #11 fix: safe access to open_position dict keys
          - Bug #15 fix: handles current_price=None gracefully
        """
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        equity  = snap_port.get('equity', cfg.INITIAL_CAPITAL)
        ret_pct = snap_port.get('total_return_pct', 0.0)
        pos     = snap_port.get('open_position')

        # Bug #11 fix: use .get() with defaults to avoid KeyError
        if pos:
            side       = pos.get('side', '?')
            qty_val    = pos.get('qty', 0)
            entry_p    = pos.get('entry_price', 0)
            unreal     = pos.get('unrealized_pnl', 0)
            pos_str    = (f"{side} {qty_val:.4f} units @ ${entry_p:.2f} "
                          f"(unrealized: ${unreal:+.2f})")
        else:
            pos_str = "No open position"

        model_str = ''
        if model_status and model_status.get('trained'):
            model_str = (f"Acc: {model_status.get('val_accuracy', 0):.1%} | "
                         f"Retrain in: {model_status.get('retrain_due_in_mins', 0):.0f}m")

        ret_color = '#2ea043' if ret_pct >= 0 else '#f85149'
        regime_colors_map = {'BULL': '#2ea043', 'CHOPPY': '#d29922', 'PANIC': '#f85149'}
        regime_color = regime_colors_map.get(regime, '#8b949e')

        # Bug #15 fix: safe price display — never shows $0.00 on cold start
        price_display = f'${current_price:.2f}' if current_price else 'Connecting...'

        # Build timeframe dropdown options
        tf_options = ''
        for tf, info in cfg.TIMEFRAME_CONFIGS.items():
            selected = 'selected' if tf == cfg.ACTIVE_TIMEFRAME else ''
            tf_options += f'<option value="{tf}" {selected}>{info["label"]}</option>'

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="{cfg.DASHBOARD_REFRESH}">
<meta name="description" content="PetroQuant WTI Paper Trading Dashboard — Live performance monitoring">
<title>PetroQuant Paper Trading | {cfg.ACTIVE_TIMEFRAME}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: {COLORS['bg']};
    color: {COLORS['text_primary']};
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
  }}
  .header-title {{
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #2ea043);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .status-bar {{
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  }}
  .badge {{
    padding: 4px 10px; border-radius: 20px; font-size: 12px;
    font-weight: 500; border: 1px solid;
  }}

  /* ── Timeframe Selector ── */
  .tf-selector {{
    display: flex; align-items: center; gap: 8px;
    background: #1c2128; border: 1px solid #30363d;
    border-radius: 8px; padding: 6px 10px;
  }}
  .tf-selector label {{
    font-size: 11px; color: #8b949e; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;
  }}
  .tf-select {{
    background: transparent; border: none; color: #e6edf3;
    font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600;
    cursor: pointer; outline: none;
    -webkit-appearance: none; appearance: none;
    padding-right: 16px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%238b949e' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 0px center;
  }}
  .tf-select option {{ background: #1c2128; color: #e6edf3; }}

  /* ── Cards ── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; padding: 16px 24px;
  }}
  .card {{
    background: {COLORS['bg_card']};
    border: 1px solid #21262d;
    border-radius: 10px; padding: 16px;
    transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: #388bfd44; }}
  .card-label {{ font-size: 11px; color: {COLORS['text_muted']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
  .card-value {{ font-size: 22px; font-weight: 700; }}
  .card-sub   {{ font-size: 11px; color: {COLORS['text_muted']}; margin-top: 4px; }}
  .chart-container {{ padding: 0 24px 24px; }}
  .refresh-note {{
    text-align: center; padding: 12px;
    font-size: 11px; color: {COLORS['text_muted']};
  }}

  /* ── Reset button ── */
  .reset-btn {{
    background: transparent; border: 1px solid #f8514944; color: #f85149;
    border-radius: 6px; padding: 5px 12px; font-size: 12px; font-family: inherit;
    cursor: pointer; transition: background 0.2s;
  }}
  .reset-btn:hover {{ background: #f8514911; }}

  /* ── Toast notification ── */
  #toast {{
    position: fixed; bottom: 24px; right: 24px;
    background: #1c2128; border: 1px solid #388bfd44; border-radius: 8px;
    padding: 12px 20px; font-size: 13px; color: #e6edf3;
    display: none; z-index: 9999;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }}
</style>
</head>
<body>

<div id="toast"></div>

<div class="header">
  <div class="header-title">🛢️ PetroQuant Paper Trading</div>
  <div class="status-bar">

    <!-- Timeframe Dropdown -->
    <div class="tf-selector">
      <label>Timeframe</label>
      <select id="tfSelect" class="tf-select" onchange="setTimeframe(this.value)">
        {tf_options}
      </select>
    </div>

    <span class="badge" style="color:{regime_color}; border-color:{regime_color}44; background:{regime_color}11">
      📊 {regime}
    </span>
    <span class="badge" style="color:#58a6ff; border-color:#388bfd44; background:#388bfd11">
      💰 WTI {price_display}
    </span>
    <span class="badge" style="color:#8b949e; border-color:#30363d; background:#21262d">
      🕐 {datetime.utcnow().strftime('%H:%M UTC')}
    </span>
    {"<span class='badge' style='color:#d29922; border-color:#d2992244; background:#d2992211'>🤖 " + model_str + "</span>" if model_str else ""}

    <button class="reset-btn" onclick="confirmReset()">⚠ Reset Account</button>
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Total Equity</div>
    <div class="card-value">${equity:,.2f}</div>
    <div class="card-sub">Started: ${cfg.INITIAL_CAPITAL:,.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">Total Return</div>
    <div class="card-value" style="color:{ret_color}">{ret_pct:+.3f}%</div>
    <div class="card-sub">Realized + Unrealized</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate</div>
    <div class="card-value">{summary.get('win_rate_pct', 0):.1f}%</div>
    <div class="card-sub">{summary.get('winning_trades', 0)}W / {summary.get('losing_trades', 0)}L</div>
  </div>
  <div class="card">
    <div class="card-label">Total Trades</div>
    <div class="card-value">{summary.get('total_trades', 0)}</div>
    <div class="card-sub">Commission: ${summary.get('total_commission', 0):,.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Max Drawdown</div>
    <div class="card-value" style="color:#f85149">-{summary.get('max_drawdown_pct', 0):.2f}%</div>
    <div class="card-sub">Peak to trough</div>
  </div>
  <div class="card">
    <div class="card-label">Open Position</div>
    <div class="card-value" style="font-size:13px; font-weight:500">{pos_str}</div>
  </div>
</div>

<div class="chart-container">
  {plot_html}
</div>

<div style="padding: 0 24px 24px">
  {table_html}
</div>

<div class="refresh-note">
  Auto-refreshes every {cfg.DASHBOARD_REFRESH} seconds &nbsp;|&nbsp;
  <a href="/trades" style="color:#388bfd">Download Trade CSV</a> &nbsp;|&nbsp;
  <a href="/status" style="color:#388bfd">JSON Status</a> &nbsp;|&nbsp;
  Timeframe: <strong>{cfg.ACTIVE_TIMEFRAME}</strong>
</div>

<script>
function showToast(msg, color) {{
  var t = document.getElementById('toast');
  t.style.color = color || '#e6edf3';
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(function() {{ t.style.display = 'none'; }}, 4000);
}}

function setTimeframe(tf) {{
  showToast('⏳ Switching to ' + tf + ' timeframe...', '#58a6ff');
  fetch('/set-timeframe', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{timeframe: tf}})
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    if (d.status === 'ok') {{
      showToast('✓ Switched to ' + d.timeframe + ' — reloading...', '#2ea043');
      setTimeout(function() {{ window.location.reload(); }}, 1500);
    }} else {{
      showToast('✗ Error: ' + (d.error || 'unknown'), '#f85149');
      // Revert dropdown to current timeframe
      document.getElementById('tfSelect').value = '{cfg.ACTIVE_TIMEFRAME}';
    }}
  }})
  .catch(function(e) {{
    showToast('✗ Request failed: ' + e, '#f85149');
    document.getElementById('tfSelect').value = '{cfg.ACTIVE_TIMEFRAME}';
  }});
}}

function confirmReset() {{
  if (!confirm('⚠ Reset ALL paper trades and return to $' + 
               new Intl.NumberFormat().format({cfg.INITIAL_CAPITAL}) + '?')) return;
  fetch('/reset', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
    body: 'confirm=yes'
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    showToast('✓ ' + d.message + ' — reloading...', '#2ea043');
    setTimeout(function() {{ window.location.reload(); }}, 1500);
  }})
  .catch(function(e) {{ showToast('✗ Reset failed', '#f85149'); }});
}}
</script>

</body>
</html>"""
        return html
