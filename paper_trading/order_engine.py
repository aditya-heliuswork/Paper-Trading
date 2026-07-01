# ============================================================================
# PETROQUANT PAPER TRADING — ORDER ENGINE
# ============================================================================
# Translates model signals into portfolio actions.
# Handles: position closing, new entries, duplicate signal suppression, regime scaling.
#
# Logic:
#   BUY signal  + no position  → open LONG
#   SELL signal + no position  → open SHORT
#   BUY signal  + SHORT open   → close SHORT only, go FLAT (re-enter next bar)
#   SELL signal + LONG open    → close LONG only, go FLAT (re-enter next bar)
#   Same direction as current  → HOLD (don't churn)
#   HOLD signal                → do nothing
#
# Bug #9 fix: log_snapshot() is now called even when daily loss limit is hit,
#   ensuring the equity snapshot chain is never broken.
# ============================================================================

import logging
from datetime import datetime
from typing import Optional

from .portfolio import Portfolio
from .trade_log import TradeLog

logger = logging.getLogger(__name__)


class OrderEngine:
    """
    Translates model signals into Portfolio actions and logs everything to TradeLog.

    Parameters
    ----------
    portfolio  : Portfolio  — the paper account to trade on
    trade_log  : TradeLog   — SQLite logger for all fills and snapshots
    """

    def __init__(self, portfolio: Portfolio, trade_log: TradeLog):
        self.portfolio    = portfolio
        self.trade_log    = trade_log
        self._last_signal = 'HOLD'    # track last executed signal to avoid churning

    # ── Main Execute ──────────────────────────────────────────────────────────
    def execute(self,
                signal      : str,
                probability : float,
                price       : float,
                regime      : str,
                regime_mult : float,
                bar_time    : Optional[datetime] = None) -> dict:
        """
        Execute a trading decision based on model signal.

        Parameters
        ----------
        signal      : 'BUY' | 'SELL' | 'HOLD'
        probability : float — model's probability of price going UP
        price       : float — current market price for execution
        regime      : str   — 'BULL' / 'CHOPPY' / 'PANIC'
        regime_mult : float — position size multiplier from regime
        bar_time    : datetime — timestamp of current bar (UTC)

        Returns
        -------
        dict — action taken, fill details, position state
        """
        bar_time  = bar_time or datetime.utcnow()
        portfolio = self.portfolio
        log       = self.trade_log

        # ── Mark to market (updates equity curve) ────────────────────────────
        unrealized = portfolio.mark_to_market(price)
        snapshot   = portfolio.get_snapshot(price)

        # ── Circuit breaker check ─────────────────────────────────────────────
        if portfolio.is_daily_loss_limit_hit():
            # Bug #9 fix: log the snapshot even on daily loss limit hit
            log.log_snapshot(bar_time, snapshot, regime, signal, probability)
            _log_hold_trade(log, 'HOLD', signal, price, 0, probability, regime,
                            regime_mult, snapshot, 'DAILY_LOSS_LIMIT', bar_time)
            return {'action': 'HOLD', 'reason': 'daily_loss_limit'}

        # ── Determine action ──────────────────────────────────────────────────
        current_side = portfolio.position.side if portfolio.position else None
        action       = self._decide_action(signal, current_side)

        if action == 'HOLD':
            log.log_snapshot(bar_time, snapshot, regime, signal, probability)
            return {
                'action'      : 'HOLD',
                'signal'      : signal,
                'probability' : probability,
                'price'       : price,
                'regime'      : regime,
                'equity'      : snapshot['equity'],
                'unrealized'  : unrealized,
            }

        fills = []

        # ── Close existing position (opposite signal) — go FLAT, do NOT flip ─
        if action in ('CLOSE_LONG', 'CLOSE_SHORT'):
            close_result = portfolio.close_position(price)
            if close_result.get('status') == 'closed':
                fills.append(('CLOSE_' + close_result['side'], close_result))
                log.log_trade(
                    timestamp     = bar_time,
                    action        = f"CLOSE_{close_result['side']}",
                    signal        = signal,
                    price         = close_result['fill_price'],
                    quantity      = close_result['qty'],
                    commission    = close_result['commission'],
                    regime        = regime,
                    probability   = probability,
                    position_size = regime_mult,
                    pnl_realized  = close_result['net_pnl'],
                    pnl_unrealized= 0.0,
                    cash_balance  = portfolio.cash,
                    equity        = portfolio.get_equity(price),
                    notes         = f'Close on opposite signal ({signal}) — going FLAT'
                )
                self._last_signal = 'HOLD'  # reset so next bar can open fresh

            snapshot_after = portfolio.get_snapshot(price)
            log.log_snapshot(bar_time, snapshot_after, regime, signal, probability)

            logger.info(f"[OrderEngine] {action} | price=${price:.4f} | "
                        f"pnl=${close_result.get('net_pnl', 0):+.2f} | "
                        f"equity=${snapshot_after['equity']:,.2f} | regime={regime}")

            return {
                'action'      : action,
                'fills'       : fills,
                'signal'      : signal,
                'probability' : probability,
                'price'       : price,
                'regime'      : regime,
                'equity'      : snapshot_after['equity'],
                'total_return': snapshot_after['total_return_pct'],
            }

        # ── Open new position (only when FLAT) ────────────────────────────────
        qty = portfolio.position_size_units(price, regime_mult, probability)

        if qty <= 0:
            log.log_snapshot(bar_time, snapshot, regime, signal, probability)
            return {'action': 'HOLD', 'reason': 'zero_qty'}

        if action == 'OPEN_LONG':
            open_result = portfolio.open_long(qty, price)
        else:  # OPEN_SHORT
            open_result = portfolio.open_short(qty, price)

        if open_result.get('status') == 'filled':
            new_side = open_result['side']
            fills.append((f'OPEN_{new_side}', open_result))
            log.log_trade(
                timestamp     = bar_time,
                action        = f"OPEN_{new_side}",
                signal        = signal,
                price         = open_result['fill_price'],
                quantity      = open_result['qty'],
                commission    = open_result['commission'],
                regime        = regime,
                probability   = probability,
                position_size = regime_mult,
                pnl_realized  = 0.0,
                pnl_unrealized= 0.0,
                cash_balance  = portfolio.cash,
                equity        = portfolio.get_equity(price),
                notes         = f'regime={regime} mult={regime_mult}'
            )
            self._last_signal = signal
        else:
            logger.warning(f"[OrderEngine] Open {action} rejected: {open_result.get('reason')}")

        snapshot_after = portfolio.get_snapshot(price)
        log.log_snapshot(bar_time, snapshot_after, regime, signal, probability)

        logger.info(f"[OrderEngine] {action} | price=${price:.4f} | qty={qty:.4f} | "
                    f"equity=${snapshot_after['equity']:,.2f} | "
                    f"regime={regime} | prob={probability:.2%}")

        return {
            'action'      : action,
            'fills'       : fills,
            'signal'      : signal,
            'probability' : probability,
            'price'       : price,
            'qty'         : qty,
            'regime'      : regime,
            'regime_mult' : regime_mult,
            'equity'      : snapshot_after['equity'],
            'total_return': snapshot_after['total_return_pct'],
        }

    # ── Decision Logic ────────────────────────────────────────────────────────
    def _decide_action(self, signal: str, current_side: Optional[str]) -> str:
        """
        Maps (signal, current_position_side) → action string.

        Returns one of:
          'OPEN_LONG'   — no position, BUY signal  → enter long
          'OPEN_SHORT'  — no position, SELL signal → enter short
          'CLOSE_LONG'  — currently long,  SELL signal → close only, go FLAT
          'CLOSE_SHORT' — currently short, BUY  signal → close only, go FLAT
          'HOLD'        — same direction as open position, or HOLD signal
        """
        if signal == 'HOLD':
            return 'HOLD'

        if current_side is None:
            return 'OPEN_LONG' if signal == 'BUY' else 'OPEN_SHORT'

        if signal == 'BUY' and current_side == 'SHORT':
            return 'CLOSE_SHORT'
        elif signal == 'SELL' and current_side == 'LONG':
            return 'CLOSE_LONG'
        else:
            return 'HOLD'


# ─────────────────────────────────────────────────────────────────────────────
def _log_hold_trade(log, action, signal, price, qty, probability,
                    regime, regime_mult, snapshot, notes, bar_time):
    """Helper to log a HOLD/circuit-breaker event."""
    log.log_trade(
        timestamp     = bar_time,
        action        = action,
        signal        = signal,
        price         = price,
        quantity      = qty,
        commission    = 0.0,
        regime        = regime,
        probability   = probability,
        position_size = regime_mult,
        pnl_realized  = 0.0,
        pnl_unrealized= snapshot.get('unrealized_pnl', 0.0),
        cash_balance  = snapshot['cash'],
        equity        = snapshot['equity'],
        notes         = notes,
    )
