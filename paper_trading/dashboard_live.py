# ============================================================================
# PETROQUANT PAPER TRADING — LIVE DASHBOARD
# ============================================================================
# Premium dark-mode Plotly HTML dashboard.
# 1-minute timeframe REMOVED. Supports: 5m, 15m, 1h, 1d
#
# Key changes from v1.0:
#   - Timeframe shown as large PILL BUTTONS (5 Min | 15 Min | 1 Hour | 1 Day)
#     — very clearly visible at top of page
#   - Price chart title updates dynamically with active timeframe
#   - Active TF button is highlighted in blue
#   - Bug #1 fixed: price line is now visible (tz-naive UTC alignment)
#   - Bug #7 fixed: cold-start None price handled gracefully
#   - Bug #11 fixed: open_position dict uses .get() safely
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
    Generates a premium dark-mode Plotly HTML dashboard.
    Supported timeframes: 5m, 15m, 1h, 1d (1m removed)
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
        """Builds the full dashboard and returns HTML string."""
        trades    = self.trade_log.get_trade_history(limit=500)
        snapshots = self.trade_log.get_snapshots(limit=2000)
        summary   = self.trade_log.get_summary()

        safe_price = current_price if (current_price and current_price > 0) else None
        snap_port  = self.portfolio.get_snapshot(safe_price) if self.portfolio else {}

        tf_label = cfg.get_timeframe_label()

        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                'Portfolio Equity Curve',
                f'WTI {tf_label} Price + Signals',
                'Per-Trade P&L',
                'Market Regime Timeline',
                f'Model Confidence — {tf_label} Bars',
                'Rolling Win Rate (last 20 trades)',
                'P&L Distribution',
                'Feature Importance',
                'Cumulative Realized P&L',
                'Drawdown %',
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
        self._add_price_signals(fig, trades, row=1, col=2)
        self._add_pnl_bars(fig, trades, row=2, col=1)
        self._add_regime_timeline(fig, snapshots, row=2, col=2)
        self._add_confidence_chart(fig, snapshots, row=3, col=1)
        self._add_rolling_winrate(fig, trades, row=3, col=2)
        self._add_win_loss_hist(fig, trades, row=4, col=1)
        self._add_feature_importance(fig, row=4, col=2)
        self._add_cumulative_pnl(fig, snapshots, row=5, col=1)
        self._add_drawdown(fig, snapshots, row=5, col=2)

        table_html = self._build_trade_table_html(trades)

        price_str = f'${safe_price:.2f}' if safe_price else 'Connecting...'
        fig.update_layout(
            title=dict(
                text=(f'PetroQuant WTI Paper Trading  |  {price_str}  |  '
                      f'Regime: {regime}  |  Timeframe: {tf_label}  |  '
                      f'{datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}'),
                font=dict(size=14, color=COLORS['text_primary']),
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
        fig.update_xaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])
        fig.update_yaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'])

        html = self._wrap_html(fig, table_html, summary, snap_port,
                               model_status, regime, safe_price)
        return html

    def save(self, html: str, path: str = cfg.DASHBOARD_PATH) -> str:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"[Dashboard] Saved to {path}")
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART PANEL BUILDERS
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
        if 'wti_price' in snapshots.columns:
            first_price = snapshots['wti_price'].dropna()
            if not first_price.empty and first_price.iloc[0] > 0:
                bnh = cfg.INITIAL_CAPITAL * (snapshots['wti_price'] / first_price.iloc[0])
                fig.add_trace(go.Scatter(
                    x=times, y=bnh,
                    name='Buy & Hold WTI',
                    line=dict(color=COLORS['text_muted'], width=1.5, dash='dash'),
                ), row=row, col=col)
        fig.add_hline(y=cfg.INITIAL_CAPITAL, line_dash='dot',
                      line_color=COLORS['text_muted'], opacity=0.4,
                      annotation_text='Start Capital', row=row, col=col)

    def _add_price_signals(self, fig, trades, row, col):
        """
        Bug #1 fix: normalize both price index and trade timestamps to
        tz-naive UTC so Plotly renders them on the same time axis.
        """
        price_df = self.price_feed_df
        tf_label = cfg.get_timeframe_label()

        if price_df is not None and not price_df.empty:
            plot_df = price_df.iloc[-200:].copy()
            if plot_df.index.tz is not None:
                plot_df.index = plot_df.index.tz_localize(None)
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df['Close'],
                name=f'WTI {tf_label}',
                line=dict(color='#58a6ff', width=1.5),
                hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=row, col=col)

        if trades.empty:
            return

        def _to_naive_ts(ts_series):
            ts = pd.to_datetime(ts_series, utc=False, errors='coerce')
            if ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            return ts

        buys = trades[trades['action'] == 'OPEN_LONG'].copy()
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(buys['timestamp']), y=buys['price'],
                mode='markers', name='LONG Entry',
                marker=dict(symbol='triangle-up', size=12,
                            color=COLORS['buy_marker'],
                            line=dict(color='white', width=1)),
                hovertemplate='LONG @ $%{y:.2f}<br>%{x}<extra></extra>',
            ), row=row, col=col)

        sells = trades[trades['action'] == 'OPEN_SHORT'].copy()
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(sells['timestamp']), y=sells['price'],
                mode='markers', name='SHORT Entry',
                marker=dict(symbol='triangle-down', size=12,
                            color=COLORS['sell_marker'],
                            line=dict(color='white', width=1)),
                hovertemplate='SHORT @ $%{y:.2f}<br>%{x}<extra></extra>',
            ), row=row, col=col)

        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if not closes.empty:
            close_colors = [
                COLORS['accent_green'] if p > 0 else COLORS['accent_red']
                for p in closes['pnl_realized'].fillna(0)
            ]
            fig.add_trace(go.Scatter(
                x=_to_naive_ts(closes['timestamp']), y=closes['price'],
                mode='markers', name='Exit',
                marker=dict(symbol='x', size=10, color=close_colors,
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
        for name, color in {'BULL': COLORS['accent_green'],
                            'CHOPPY': COLORS['accent_gold'],
                            'PANIC': COLORS['accent_red']}.items():
            mask = snapshots['regime'] == name
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=snapshots.loc[mask, 'timestamp'],
                    y=[name] * mask.sum(),
                    mode='markers', name=name,
                    marker=dict(color=color, size=5, symbol='square'),
                ), row=row, col=col)

    def _add_confidence_chart(self, fig, snapshots, row, col):
        if snapshots.empty or 'probability' not in snapshots.columns:
            return
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'], y=snapshots['probability'],
            name='Model Prob (UP)',
            line=dict(color=COLORS['accent_purple'], width=1),
        ), row=row, col=col)
        fig.add_hline(y=cfg.BUY_THRESHOLD, line_dash='dash',
                      line_color=COLORS['accent_green'], opacity=0.6,
                      annotation_text=f'BUY >{cfg.BUY_THRESHOLD}', row=row, col=col)
        fig.add_hline(y=cfg.SELL_THRESHOLD, line_dash='dash',
                      line_color=COLORS['accent_red'], opacity=0.6,
                      annotation_text=f'SELL <{cfg.SELL_THRESHOLD}', row=row, col=col)

    def _add_rolling_winrate(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)].copy()
        if len(closes) < 3:
            return
        closes = closes.sort_values('timestamp')
        closes['win'] = (closes['pnl_realized'] > 0).astype(float)
        closes['rolling_wr'] = closes['win'].rolling(min(20, len(closes))).mean() * 100
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(closes['timestamp']), y=closes['rolling_wr'],
            name='Rolling Win Rate %',
            line=dict(color=COLORS['accent_gold'], width=2),
            fill='tozeroy', fillcolor='rgba(210,153,34,0.08)',
        ), row=row, col=col)
        fig.add_hline(y=50, line_dash='dot', line_color=COLORS['text_muted'],
                      opacity=0.5, row=row, col=col)

    def _add_win_loss_hist(self, fig, trades, row, col):
        closes = trades[trades['action'].str.startswith('CLOSE', na=False)]
        if closes.empty:
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
        fig.add_vline(x=0, line_color='white', opacity=0.5, row=row, col=col)

    def _add_feature_importance(self, fig, row, col):
        fi = self.feature_importance
        if fi is None or (hasattr(fi, 'empty') and fi.empty):
            return
        top10 = fi.head(10).sort_values()
        fig.add_trace(go.Bar(
            y=top10.index.tolist(), x=top10.values.tolist(),
            orientation='h', name='Feature Importance',
            marker_color=COLORS['accent_blue'],
        ), row=row, col=col)

    def _add_cumulative_pnl(self, fig, snapshots, row, col):
        if snapshots.empty or 'realized_pnl' not in snapshots.columns:
            return
        fig.add_trace(go.Scatter(
            x=snapshots['timestamp'], y=snapshots['realized_pnl'],
            name='Realized P&L',
            line=dict(color=COLORS['accent_gold'], width=2),
            fill='tozeroy', fillcolor='rgba(210,153,34,0.08)',
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
            x=snapshots['timestamp'], y=dd_pct,
            name='Drawdown %',
            line=dict(color=COLORS['accent_red'], width=1.5),
            fill='tozeroy', fillcolor='rgba(248,81,73,0.08)',
        ), row=row, col=col)

    # ── Trade Table ───────────────────────────────────────────────────────────
    def _build_trade_table_html(self, trades) -> str:
        recent = trades[trades['action'].str.startswith('OPEN', na=False)].head(50)
        if recent.empty:
            return '<p style="color:#8b949e;padding:20px;text-align:center">No trades yet. Engine is waiting for a strong signal.</p>'
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
            ac     = '#2ea043' if 'LONG' in str(action) else '#f85149'
            price  = f"${float(r.get('price', 0)):.2f}"
            qty    = f"{float(r.get('quantity', 0)):.3f}"
            regime = r.get('regime', '-')
            prob   = f"{float(r.get('probability', 0.5)):.1%}"
            pnl    = _fmt_pnl(r.get('pnl_realized', 0))
            rows_html += f"""
            <tr style="border-bottom:1px solid #21262d">
              <td style="padding:8px 12px">{ts}</td>
              <td style="padding:8px 12px;color:{ac};font-weight:600">{action}</td>
              <td style="padding:8px 12px">{price}</td>
              <td style="padding:8px 12px">{qty}</td>
              <td style="padding:8px 12px">{regime}</td>
              <td style="padding:8px 12px">{prob}</td>
              <td style="padding:8px 12px">{pnl}</td>
            </tr>"""

        return f"""
        <div style="margin:0 0 24px 0;background:#161b22;border-radius:12px;border:1px solid #21262d;overflow:hidden">
          <div style="padding:14px 20px;background:#1c2128;border-bottom:1px solid #21262d;display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600;font-size:14px">Recent Trades</span>
            <span style="font-size:12px;color:#8b949e">Timeframe: {cfg.get_timeframe_label()}</span>
          </div>
          <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr style="background:#1c2128;color:#8b949e">
                <th style="padding:8px 12px;text-align:left">Time (UTC)</th>
                <th style="padding:8px 12px;text-align:left">Action</th>
                <th style="padding:8px 12px;text-align:left">Price</th>
                <th style="padding:8px 12px;text-align:left">Qty</th>
                <th style="padding:8px 12px;text-align:left">Regime</th>
                <th style="padding:8px 12px;text-align:left">Confidence</th>
                <th style="padding:8px 12px;text-align:left">P&amp;L</th>
              </tr>
            </thead>
            <tbody style="color:#e6edf3">{rows_html}</tbody>
          </table>
          </div>
        </div>"""

    # ── HTML Wrapper ──────────────────────────────────────────────────────────
    def _wrap_html(self, fig, table_html, summary, snap_port,
                   model_status, regime, current_price) -> str:
        """
        Full HTML page with:
          - BIG PILL BUTTONS for timeframe selection (5 Min | 15 Min | 1 Hour | 1 Day)
          - Status cards
          - Plotly charts
          - Trade table
          - Auto-refresh
        """
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        equity  = snap_port.get('equity', cfg.INITIAL_CAPITAL)
        ret_pct = snap_port.get('total_return_pct', 0.0)
        pos     = snap_port.get('open_position')

        if pos:
            side    = pos.get('side', '?')
            qty_val = pos.get('qty', 0)
            entry_p = pos.get('entry_price', 0)
            unreal  = pos.get('unrealized_pnl', 0)
            unreal_color = '#2ea043' if unreal >= 0 else '#f85149'
            pos_html = (f'<span style="font-weight:600;color:{"#2ea043" if side=="LONG" else "#f85149"}">'
                        f'{side}</span> {qty_val:.2f} units @ ${entry_p:.2f}'
                        f'<br><span style="color:{unreal_color};font-size:11px">'
                        f'Unrealized: ${unreal:+.2f}</span>')
        else:
            pos_html = '<span style="color:#8b949e">No open position</span>'

        ret_color    = '#2ea043' if ret_pct >= 0 else '#f85149'
        regime_map   = {'BULL': '#2ea043', 'CHOPPY': '#d29922', 'PANIC': '#f85149'}
        regime_color = regime_map.get(regime, '#8b949e')
        price_display = f'${current_price:.2f}' if current_price else 'Connecting...'

        model_badge = ''
        if model_status and model_status.get('trained'):
            acc = model_status.get('val_accuracy', 0)
            acc_color = '#2ea043' if acc > 0.55 else ('#d29922' if acc > 0.52 else '#f85149')
            model_badge = (f'<div class="badge" style="color:{acc_color};border-color:{acc_color}44;background:{acc_color}11">'
                           f'Model Acc: {acc:.1%} | Retrain in {model_status.get("retrain_due_in_mins", 0):.0f}m</div>')

        # ── Build PILL BUTTONS for each timeframe ──────────────────────────
        # These are big, clearly visible buttons at the top of the page
        tf_buttons = ''
        for tf, info in cfg.TIMEFRAME_CONFIGS.items():
            is_active = tf == cfg.ACTIVE_TIMEFRAME
            if is_active:
                btn_style = ('background:linear-gradient(135deg,#388bfd,#1a6fd4);'
                             'color:#fff;border:2px solid #388bfd;'
                             'box-shadow:0 0 12px rgba(56,139,253,0.4);')
            else:
                btn_style = ('background:#1c2128;color:#8b949e;border:2px solid #30363d;')

            desc = info.get('description', '')
            tf_buttons += f"""
            <button
              onclick="setTimeframe('{tf}')"
              id="tfbtn-{tf}"
              title="{desc}"
              style="{btn_style}
                     padding:10px 20px;border-radius:8px;font-family:Inter,sans-serif;
                     font-size:14px;font-weight:700;cursor:pointer;
                     transition:all 0.2s;min-width:80px;letter-spacing:0.03em">
              {info['label']}
            </button>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="{cfg.DASHBOARD_REFRESH}">
<meta name="description" content="PetroQuant WTI Paper Trading — {cfg.get_timeframe_label()} timeframe">
<title>PetroQuant | {cfg.get_timeframe_label()} | WTI Paper Trading</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing:border-box;margin:0;padding:0; }}
  body {{
    font-family:'Inter',system-ui,sans-serif;
    background:{COLORS['bg']};color:{COLORS['text_primary']};min-height:100vh;
  }}

  /* ── TOP HEADER ── */
  .header {{
    background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border-bottom:1px solid #21262d;
    padding:14px 24px;
    display:flex;align-items:center;justify-content:space-between;
    flex-wrap:wrap;gap:10px;
  }}
  .header-left {{ display:flex;align-items:center;gap:16px;flex-wrap:wrap; }}
  .header-title {{
    font-size:18px;font-weight:700;
    background:linear-gradient(135deg,#58a6ff,#2ea043);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    white-space:nowrap;
  }}
  .header-right {{ display:flex;gap:8px;align-items:center;flex-wrap:wrap; }}
  .badge {{
    padding:5px 12px;border-radius:20px;font-size:12px;
    font-weight:600;border:1px solid;white-space:nowrap;
  }}

  /* ── TIMEFRAME PILL SECTION — Big and clearly visible ── */
  .tf-section {{
    background:#0d1117;
    border-bottom:1px solid #21262d;
    padding:14px 24px;
    display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  }}
  .tf-label {{
    font-size:12px;color:#8b949e;font-weight:600;
    text-transform:uppercase;letter-spacing:0.08em;white-space:nowrap;
    margin-right:4px;
  }}
  .tf-buttons {{ display:flex;gap:10px;flex-wrap:wrap; }}

  /* ── STATUS CARDS ── */
  .cards {{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:12px;padding:16px 24px;
  }}
  .card {{
    background:{COLORS['bg_card']};border:1px solid #21262d;
    border-radius:10px;padding:16px;transition:border-color 0.2s;
  }}
  .card:hover {{ border-color:#388bfd44; }}
  .card-label {{ font-size:10px;color:{COLORS['text_muted']};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px; }}
  .card-value {{ font-size:20px;font-weight:700;line-height:1.2; }}
  .card-sub   {{ font-size:11px;color:{COLORS['text_muted']};margin-top:5px; }}

  .chart-wrap {{ padding:0 24px 24px; }}
  .table-wrap {{ padding:0 24px 24px; }}
  .footer     {{ text-align:center;padding:16px;font-size:11px;color:{COLORS['text_muted']}; }}

  /* ── Reset ── */
  .reset-btn {{
    background:transparent;border:1px solid #f8514944;color:#f85149;
    border-radius:6px;padding:6px 14px;font-size:12px;font-family:inherit;
    cursor:pointer;transition:background 0.2s;font-weight:500;
  }}
  .reset-btn:hover {{ background:#f8514911; }}

  /* ── Toast ── */
  #toast {{
    position:fixed;bottom:24px;right:24px;
    background:#1c2128;border:1px solid #388bfd44;border-radius:10px;
    padding:14px 20px;font-size:13px;color:#e6edf3;
    display:none;z-index:9999;box-shadow:0 4px 24px rgba(0,0,0,0.5);
    max-width:320px;
  }}

  /* ── Loading overlay on TF switch ── */
  #loading {{
    display:none;position:fixed;inset:0;
    background:rgba(13,17,23,0.85);z-index:9998;
    align-items:center;justify-content:center;flex-direction:column;gap:16px;
    font-size:16px;color:#e6edf3;font-weight:500;
  }}
  #loading.show {{ display:flex; }}
  .spinner {{
    width:40px;height:40px;border:3px solid #21262d;
    border-top-color:#388bfd;border-radius:50%;
    animation:spin 0.8s linear infinite;
  }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
</style>
</head>
<body>

<div id="toast"></div>
<div id="loading"><div class="spinner"></div><span id="loading-msg">Switching timeframe...</span></div>

<!-- ── TOP HEADER ── -->
<div class="header">
  <div class="header-left">
    <span class="header-title">WTI Paper Trading</span>
    <div class="badge" style="color:{regime_color};border-color:{regime_color}44;background:{regime_color}11">
      {regime}
    </div>
    <div class="badge" style="color:#58a6ff;border-color:#388bfd44;background:#388bfd11">
      WTI {price_display}
    </div>
    <div class="badge" style="color:#8b949e;border-color:#30363d;background:#21262d">
      {datetime.utcnow().strftime('%H:%M UTC')}
    </div>
    {model_badge}
  </div>
  <div class="header-right">
    <button class="reset-btn" onclick="confirmReset()">Reset Account</button>
  </div>
</div>

<!-- ── TIMEFRAME PILL BUTTONS — Large & clearly visible ── -->
<div class="tf-section">
  <span class="tf-label">Trading Timeframe</span>
  <div class="tf-buttons">
    {tf_buttons}
  </div>
  <span style="font-size:12px;color:#8b949e;margin-left:8px">
    Currently trading: <strong style="color:#e6edf3">{cfg.get_timeframe_description()}</strong>
  </span>
</div>

<!-- ── STATUS CARDS ── -->
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
    <div class="card-value" style="font-size:13px;font-weight:500;line-height:1.5">{pos_html}</div>
  </div>
</div>

<!-- ── CHARTS ── -->
<div class="chart-wrap">
  {plot_html}
</div>

<!-- ── TRADE TABLE ── -->
<div class="table-wrap">
  {table_html}
</div>

<div class="footer">
  Auto-refreshes every {cfg.DASHBOARD_REFRESH}s &nbsp;|&nbsp;
  Active timeframe: <strong>{cfg.get_timeframe_label()}</strong> &nbsp;|&nbsp;
  <a href="/trades" style="color:#388bfd">Download CSV</a> &nbsp;|&nbsp;
  <a href="/status" style="color:#388bfd">JSON Status</a>
</div>

<script>
function showToast(msg, color) {{
  var t = document.getElementById('toast');
  t.style.borderColor = (color || '#388bfd') + '44';
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(function() {{ t.style.display = 'none'; }}, 5000);
}}

function setTimeframe(tf) {{
  var loadEl = document.getElementById('loading');
  var msgEl  = document.getElementById('loading-msg');
  var labels = {{'5m':'5 Minutes','15m':'15 Minutes','1h':'1 Hour','1d':'1 Day'}};

  // Show loading overlay
  msgEl.textContent = 'Switching to ' + (labels[tf] || tf) + '...';
  loadEl.classList.add('show');

  fetch('/set-timeframe', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{timeframe: tf}})
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    if (d.status === 'ok') {{
      msgEl.textContent = 'Switched! Reloading dashboard...';
      setTimeout(function() {{ window.location.reload(); }}, 1000);
    }} else {{
      loadEl.classList.remove('show');
      showToast('Error: ' + (d.error || 'unknown'), '#f85149');
    }}
  }})
  .catch(function(e) {{
    loadEl.classList.remove('show');
    showToast('Request failed: ' + e, '#f85149');
  }});
}}

function confirmReset() {{
  if (!confirm('Reset ALL trades and return account to $1,000,000?')) return;
  fetch('/reset', {{
    method: 'POST',
    headers: {{'Content-Type':'application/x-www-form-urlencoded'}},
    body: 'confirm=yes'
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(d) {{
    showToast(d.message + ' - reloading...', '#2ea043');
    setTimeout(function() {{ window.location.reload(); }}, 1500);
  }})
  .catch(function() {{ showToast('Reset failed', '#f85149'); }});
}}
</script>

</body>
</html>"""
        return html
