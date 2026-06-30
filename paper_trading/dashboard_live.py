# ============================================================================
# PETROQUANT PAPER TRADING — LIVE DASHBOARD
# ============================================================================
# Generates a premium 10-panel Plotly HTML dashboard from the trade log.
# Saved as paper_dashboard.html — auto-refreshes every 60 seconds in browser.
#
# Panels:
#   1. Equity Curve — portfolio value vs Buy & Hold
#   2. WTI 1-Min Price + BUY/SELL signal markers
#   3. Current Position card (side, entry, unrealized P&L)
#   4. Daily Regime Timeline (BULL/CHOPPY/PANIC color bands)
#   5. Recent Trade Log Table (last 50 fills)
#   6. Per-Trade P&L Bar Chart
#   7. Rolling Model Confidence (probability over time)
#   8. Win/Loss Distribution Histogram
#   9. Key Performance Metrics (Sharpe, win rate, drawdown, etc.)
#   10. Feature Importance Bar Chart
# ============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import logging

from . import config as cfg

logger = logging.getLogger(__name__)

# ── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg'          : '#0d1117',
    'bg_panel'    : '#161b22',
    'bg_card'     : '#1c2128',
    'accent_green': '#2ea043',
    'accent_red'  : '#f85149',
    'accent_blue' : '#388bfd',
    'accent_gold' : '#d29922',
    'accent_purple': '#a5a5f5',
    'text_primary': '#e6edf3',
    'text_muted'  : '#8b949e',
    'grid'        : '#21262d',
    'regime_bull' : 'rgba(46, 160, 67,  0.12)',
    'regime_choppy':'rgba(210, 153, 34, 0.12)',
    'regime_panic': 'rgba(248, 81, 73,  0.12)',
    'buy_marker'  : '#2ea043',
    'sell_marker' : '#f85149',
}


class LiveDashboard:
    """
    Generates a premium dark-mode Plotly HTML dashboard from trade log data.
    """

    def __init__(self, trade_log, portfolio=None, price_feed_df=None,
                 feature_importance=None):
        """
        Parameters
        ----------
        trade_log         : TradeLog    — SQLite trade log instance
        portfolio         : Portfolio   — current portfolio state
        price_feed_df     : pd.DataFrame — latest 1-min OHLCV bars (for price chart)
        feature_importance: pd.Series  — XGBoost feature importance
        """
        self.trade_log         = trade_log
        self.portfolio         = portfolio
        self.price_feed_df     = price_feed_df
        self.feature_importance= feature_importance

    # ── Main Render ───────────────────────────────────────────────────────────
    def render(self, current_price: float = None, regime: str = 'UNKNOWN',
               model_status: dict = None) -> str:
        """
        Builds the full 10-panel dashboard and returns HTML string.

        Parameters
        ----------
        current_price : float — latest WTI price
        regime        : str   — current daily regime
        model_status  : dict  — XGBoost model status info

        Returns
        -------
        str — full HTML page content
        """
        # ── Load data ────────────────────────────────────────────────────────
        trades    = self.trade_log.get_trade_history(limit=500)
        snapshots = self.trade_log.get_snapshots(limit=2000)
        summary   = self.trade_log.get_summary()
        snap_port = self.portfolio.get_snapshot(current_price) if self.portfolio else {}

        # ── Build subplot grid (9 chart panels; table is separate) ─────────────
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                'Portfolio Equity Curve',
                'WTI 1-Min Price + Signals',
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

        # ── Panel 1: Equity Curve ─────────────────────────────────────────────
        self._add_equity_curve(fig, snapshots, current_price, row=1, col=1)

        # ── Panel 2: Price + Signals ──────────────────────────────────────────
        self._add_price_signals(fig, trades, row=1, col=2)

        # ── Panel 3: Per-Trade P&L ────────────────────────────────────────────
        self._add_pnl_bars(fig, trades, row=2, col=1)

        # ── Panel 4: Regime Timeline ──────────────────────────────────────────
        self._add_regime_timeline(fig, snapshots, row=2, col=2)

        # ── Panel 5: Model Confidence ─────────────────────────────────────────
        self._add_confidence_chart(fig, snapshots, row=3, col=1)

        # ── Panel 6: Rolling Win Rate ─────────────────────────────────────────
        self._add_rolling_winrate(fig, trades, row=3, col=2)

        # ── Panel 7: Win/Loss Histogram ───────────────────────────────────────
        self._add_win_loss_hist(fig, trades, row=4, col=1)

        # ── Panel 8: Feature Importance ───────────────────────────────────────
        self._add_feature_importance(fig, row=4, col=2)

        # ── Panel 9: Cumulative P&L ───────────────────────────────────────────
        self._add_cumulative_pnl(fig, snapshots, row=5, col=1)

        # ── Panel 10: Drawdown ────────────────────────────────────────────────
        self._add_drawdown(fig, snapshots, row=5, col=2)

        # ── Standalone Trade Table figure ─────────────────────────────────────
        table_html = self._build_trade_table_html(trades)

        # ── Layout ───────────────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=f'PetroQuant Paper Trader  |  WTI ${current_price:.2f}  |  '
                     f'Regime: {regime}  |  '
                     f'Updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}',
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

        # Apply dark grid styling to all xy-axes (skip domain/table types)
        fig.update_xaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
                         showgrid=True, linecolor=COLORS['grid'])
        fig.update_yaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'],
                         showgrid=True, linecolor=COLORS['grid'])

        # ── Generate HTML with auto-refresh ───────────────────────────────────
        html = self._wrap_html(fig, table_html, summary, snap_port, model_status, regime, current_price)
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

        # Strategy line
        fig.add_trace(go.Scatter(
            x=times, y=equities,
            name='Portfolio Equity',
            line=dict(color=COLORS['accent_blue'], width=2),
            fill='tozeroy', fillcolor='rgba(56, 139, 253, 0.06)',
        ), row=row, col=col)

        # Buy-and-hold baseline (if we had held WTI from start)
        if current_price and len(snapshots) > 1 and 'wti_price' in snapshots.columns:
            first_price = snapshots['wti_price'].dropna().iloc[0] if not snapshots['wti_price'].dropna().empty else None
            if first_price and first_price > 0:
                bnh_equity = cfg.INITIAL_CAPITAL * (snapshots['wti_price'] / first_price)
                fig.add_trace(go.Scatter(
                    x=times, y=bnh_equity,
                    name='Buy & Hold WTI',
                    line=dict(color=COLORS['text_muted'], width=1.5, dash='dash'),
                ), row=row, col=col)

        # Initial capital line
        fig.add_hline(y=cfg.INITIAL_CAPITAL, line_dash='dot',
                      line_color=COLORS['text_muted'], opacity=0.4,
                      annotation_text='Start Capital', row=row, col=col)

    def _add_price_signals(self, fig, trades, row, col):
        price_df = self.price_feed_df
        if price_df is not None and not price_df.empty:
            # Candlestick or line
            fig.add_trace(go.Scatter(
                x=price_df.index[-200:],        # last 200 bars (~3.3 hours)
                y=price_df['Close'].iloc[-200:],
                name='WTI 1-min',
                line=dict(color='#58a6ff', width=1.5),
            ), row=row, col=col)

        if trades.empty:
            return

        # BUY markers
        buys = trades[trades['action'] == 'OPEN_LONG']
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buys['timestamp']),
                y=buys['price'],
                mode='markers',
                name='LONG Entry',
                marker=dict(symbol='triangle-up', size=10,
                            color=COLORS['buy_marker'],
                            line=dict(color='white', width=1)),
            ), row=row, col=col)

        # SELL markers
        sells = trades[trades['action'] == 'OPEN_SHORT']
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sells['timestamp']),
                y=sells['price'],
                mode='markers',
                name='SHORT Entry',
                marker=dict(symbol='triangle-down', size=10,
                            color=COLORS['sell_marker'],
                            line=dict(color='white', width=1)),
            ), row=row, col=col)

    def _add_pnl_bars(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if closes.empty:
            return
        closes = closes.sort_values('timestamp')
        colors = [COLORS['accent_green'] if p > 0 else COLORS['accent_red']
                  for p in closes['pnl_realized']]
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
        regime_colors = {'BULL': COLORS['accent_green'],
                         'CHOPPY': COLORS['accent_gold'],
                         'PANIC': COLORS['accent_red']}
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
        fig.add_hline(y=cfg.BUY_THRESHOLD,  line_dash='dash',
                      line_color=COLORS['accent_green'], opacity=0.6,
                      annotation_text=f'BUY >{cfg.BUY_THRESHOLD}', row=row, col=col)
        fig.add_hline(y=cfg.SELL_THRESHOLD, line_dash='dash',
                      line_color=COLORS['accent_red'], opacity=0.6,
                      annotation_text=f'SELL <{cfg.SELL_THRESHOLD}', row=row, col=col)
        fig.add_hline(y=0.5, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.4, row=row, col=col)

    def _add_rolling_winrate(self, fig, trades, row, col):
        """Rolling 20-trade win rate line chart."""
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

    def _build_trade_table_html(self, trades) -> str:
        """Renders the recent trade log as a styled HTML table (no Plotly required)."""
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
            ts = str(r.get('timestamp', ''))[:16]
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
            <span style="font-weight:600;font-size:14px">Trade Log (Last 50)</span>
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

    def _add_win_loss_hist(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)]
        if closes.empty or 'pnl_realized' not in closes.columns:
            return
        pnls = closes['pnl_realized'].dropna()
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
        if fi is None or fi.empty:
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
        if snapshots.empty:
            return
        cum_pnl = snapshots['realized_pnl']
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'],
            y=cum_pnl,
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

    # ── HTML Wrapper ───────────────────────────────────────────────────────────
    def _wrap_html(self, fig, table_html: str, summary, snap_port, model_status, regime, current_price) -> str:
        """Wraps Plotly figure in a full HTML page with auto-refresh and status cards."""
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        equity  = snap_port.get('equity', cfg.INITIAL_CAPITAL)
        ret_pct = snap_port.get('total_return_pct', 0.0)
        pos     = snap_port.get('open_position')
        pos_str = (f"{pos['side']} {pos['qty']:.4f} units @ ${pos['entry_price']:.2f} "
                   f"(unrealized: ${pos['unrealized_pnl']:+.2f})"
                   if pos else "No open position")

        model_str = ''
        if model_status and model_status.get('trained'):
            model_str = (f"Acc: {model_status.get('val_accuracy', 0):.1%} | "
                         f"Retrain in: {model_status.get('retrain_due_in_mins', 0):.0f}m")

        ret_color = '#2ea043' if ret_pct >= 0 else '#f85149'
        regime_colors_map = {'BULL': '#2ea043', 'CHOPPY': '#d29922', 'PANIC': '#f85149'}
        regime_color = regime_colors_map.get(regime, '#8b949e')

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="{cfg.DASHBOARD_REFRESH}">
<title>PetroQuant Paper Trading</title>
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
    display: flex; gap: 8px; flex-wrap: wrap;
  }}
  .badge {{
    padding: 4px 10px; border-radius: 20px; font-size: 12px;
    font-weight: 500; border: 1px solid;
  }}
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
</style>
</head>
<body>

<div class="header">
  <div class="header-title">🛢️ PetroQuant Paper Trading</div>
  <div class="status-bar">
    <span class="badge" style="color:{regime_color}; border-color:{regime_color}44; background:{regime_color}11">
      📊 {regime}
    </span>
    <span class="badge" style="color:#58a6ff; border-color:#388bfd44; background:#388bfd11">
      💰 WTI ${current_price:.2f}
    </span>
    <span class="badge" style="color:#8b949e; border-color:#30363d; background:#21262d">
      🕐 {datetime.utcnow().strftime('%H:%M UTC')}
    </span>
    {"<span class='badge' style='color:#d29922; border-color:#d2992244; background:#d2992211'>🤖 " + model_str + "</span>" if model_str else ""}
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
  <a href="/status" style="color:#388bfd">JSON Status</a>
</div>

</body>
</html>"""
        return html

