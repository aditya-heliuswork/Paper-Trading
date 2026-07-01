# ============================================================================
# PETROQUANT PAPER TRADING — PORTFOLIO MANAGER
# ============================================================================
# Tracks cash balance, open positions, and P&L for the paper trading account.
#
# Supports:
#   - Long and short positions (oil futures can go both ways)
#   - Mark-to-market unrealized P&L
#   - Commission + slippage on fills
#   - Daily loss limit (circuit breaker)
#   - Equity curve (list of snapshots over time)
#
# Portfolio:
#   open_long(qty, price)     — buy N units at fill price
#   open_short(qty, price)    — sell short N units at fill price
#   close_position(price)     — close current open position
#   mark_to_market(price)     — compute unrealized P&L
#   get_snapshot()            — dict of all current metrics
#   position_size_units(...)  — how many units to trade given capital rules
# ============================================================================

import threading
import numpy as np
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
import logging

from . import config as cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    """Represents a single open position."""
    side        : str        # 'LONG' or 'SHORT'
    qty         : float      # number of units
    entry_price : float      # average fill price (after slippage)
    entry_time  : datetime   = field(default_factory=datetime.utcnow)
    commission  : float      = 0.0

    @property
    def direction(self) -> int:
        return 1 if self.side == 'LONG' else -1

    def unrealized_pnl(self, current_price: float) -> float:
        return self.direction * self.qty * (current_price - self.entry_price)

    def unrealized_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.direction * (current_price - self.entry_price) / self.entry_price


# ─────────────────────────────────────────────────────────────────────────────
class Portfolio:
    """
    Paper trading portfolio — tracks all account state.

    Parameters
    ----------
    initial_capital  : float — starting cash (default from config)
    commission_pct   : float — commission per trade as decimal (e.g. 0.0002 = 2bps)
    slippage_pct     : float — slippage per trade as decimal (e.g. 0.0001 = 1bp)
    max_position_pct : float — max % of equity per trade (e.g. 0.15 = 15%)
    max_daily_loss   : float — stop trading if daily loss exceeds this % of equity
    """

    def __init__(self,
                 initial_capital  : float = cfg.INITIAL_CAPITAL,
                 commission_pct   : float = cfg.COMMISSION_PCT,
                 slippage_pct     : float = cfg.SLIPPAGE_PCT,
                 max_position_pct : float = cfg.MAX_POSITION_PCT,
                 max_daily_loss   : float = cfg.MAX_DAILY_LOSS_PCT):

        self.initial_capital  = initial_capital
        self.commission_pct   = commission_pct
        self.slippage_pct     = slippage_pct
        self.max_position_pct = max_position_pct
        self.max_daily_loss   = max_daily_loss

        # Thread safety — trading loop and web server both access portfolio
        self._lock            = threading.RLock()

        # Account state
        self.cash             = initial_capital
        self.position         : Optional[Position] = None
        self.equity_curve     = []           # list of (timestamp, equity) tuples
        self.realized_pnl     = 0.0          # total realized P&L
        self.total_commission = 0.0
        self.total_trades     = 0
        self.winning_trades   = 0
        self.losing_trades    = 0
        self.total_win_pnl    = 0.0
        self.total_loss_pnl   = 0.0

        # Daily tracking (for circuit breaker)
        self._today           = date.today()
        self._day_start_equity= initial_capital
        self._daily_pnl       = 0.0

        # Record starting point
        self._record_snapshot(initial_capital)
        logger.info(f"[Portfolio] Initialized with ${initial_capital:,.2f} paper capital")

    # ── State Restoration ─────────────────────────────────────────────────────
    def restore_from_db(self, trade_log) -> bool:
        """
        Restore portfolio state from SQLite after a process restart.
        Treats the account as FLAT on restart (any open position at crash time
        is logged as abandoned — its P&L is not captured).

        Returns True if restoration succeeded, False if no prior data.
        """
        with self._lock:
            try:
                state = trade_log.get_portfolio_state()
                if not state or state.get('equity') is None:
                    logger.info("[Portfolio] No prior DB state found — starting fresh")
                    return False

                self.cash             = state['equity']
                self.realized_pnl     = state['realized_pnl']
                self.total_trades     = state['total_trades']
                self.winning_trades   = state['winning_trades']
                self.losing_trades    = state['losing_trades']
                self.total_win_pnl    = state['total_win_pnl']
                self.total_loss_pnl   = state['total_loss_pnl']
                self.total_commission = state['total_commission']
                self._day_start_equity= state['equity']
                self.position         = None   # always restart flat

                if state.get('had_open_position'):
                    logger.warning(
                        "[Portfolio] Restart detected an unclosed position in DB. "
                        "The position has been abandoned — its unrealized P&L is not "
                        "captured. Consider reviewing the last open trade."
                    )

                logger.info(
                    f"[Portfolio] Restored from DB | "
                    f"equity=${state['equity']:,.2f} | "
                    f"realized_pnl=${state['realized_pnl']:+,.2f} | "
                    f"trades={state['total_trades']}"
                )
                return True

            except Exception as e:
                logger.error(f"[Portfolio] restore_from_db failed: {e}")
                return False

    # ── Position Sizing ───────────────────────────────────────────────────────
    def position_size_units(self, price: float, regime_mult: float = 1.0,
                             confidence: float = 0.5) -> float:
        """
        Calculate number of units to trade.

        Logic: equity × max_position_pct × regime_mult × confidence_factor / price

        Parameters
        ----------
        price       : float — current WTI price per unit
        regime_mult : float — from HMM regime (1.0/0.5/0.25)
        confidence  : float — model probability (0.5-1.0)

        Returns
        -------
        float — number of units to trade (min 0.01, capped by max_position_pct)
        """
        if price <= 0:
            return 0.0

        equity     = self.get_equity()
        # Confidence scaling: maps prob distance from 0.5 to a 0.5-1.0 multiplier
        conf_mult  = 0.5 + 0.5 * (abs(confidence - 0.5) * 2)

        trade_value = equity * self.max_position_pct * regime_mult * conf_mult
        units       = trade_value / price

        return round(max(units, 0.0), 4)

    # ── Trade Execution ───────────────────────────────────────────────────────
    def open_long(self, qty: float, market_price: float) -> dict:
        """Buy QTY units long. Applies slippage (buys at slightly higher price)."""
        with self._lock:
            if self.position is not None:
                logger.warning("[Portfolio] Cannot open long — position already open")
                return {'status': 'rejected', 'reason': 'position_already_open'}

            if self.is_daily_loss_limit_hit():
                return {'status': 'rejected', 'reason': 'daily_loss_limit'}

            fill_price = market_price * (1 + self.slippage_pct)
            cost       = fill_price * qty
            commission = cost * self.commission_pct

            if cost + commission > self.cash:
                qty = max((self.cash * 0.99) / (fill_price * (1 + self.commission_pct)), 0)
                if qty <= 0:
                    return {'status': 'rejected', 'reason': 'insufficient_cash'}
                cost       = fill_price * qty
                commission = cost * self.commission_pct

            self.cash -= (cost + commission)
            self.total_commission += commission
            self.position = Position(side='LONG', qty=qty, entry_price=fill_price,
                                     commission=commission)
            self.total_trades += 1

            logger.info(f"[Portfolio] OPEN LONG  {qty:.4f} units @ ${fill_price:.4f} "
                        f"| cost=${cost:.2f} | commission=${commission:.2f} | cash=${self.cash:,.2f}")

            return {
                'status'     : 'filled',
                'side'       : 'LONG',
                'qty'        : qty,
                'fill_price' : fill_price,
                'cost'       : cost,
                'commission' : commission,
            }

    def open_short(self, qty: float, market_price: float) -> dict:
        """
        Sell short QTY units. Applies slippage (fills at slightly lower price).
        For paper trading futures: we reserve the full notional as collateral
        (not just 10% margin) so cash accounting stays consistent — the notional
        is returned at close plus/minus P&L.
        """
        with self._lock:
            if self.position is not None:
                logger.warning("[Portfolio] Cannot open short — position already open")
                return {'status': 'rejected', 'reason': 'position_already_open'}

            if self.is_daily_loss_limit_hit():
                return {'status': 'rejected', 'reason': 'daily_loss_limit'}

            fill_price  = market_price * (1 - self.slippage_pct)
            notional    = fill_price * qty
            commission  = notional * self.commission_pct

            # Reserve full notional as collateral (returned at close ± P&L)
            if notional + commission > self.cash:
                return {'status': 'rejected', 'reason': 'insufficient_cash'}

            self.cash -= (notional + commission)   # lock notional + commission
            self.total_commission += commission
            self.position = Position(side='SHORT', qty=qty, entry_price=fill_price,
                                     commission=commission)
            self.total_trades += 1

            logger.info(f"[Portfolio] OPEN SHORT {qty:.4f} units @ ${fill_price:.4f} "
                        f"| notional=${notional:.2f} | commission=${commission:.2f} "
                        f"| cash=${self.cash:,.2f}")

            return {
                'status'     : 'filled',
                'side'       : 'SHORT',
                'qty'        : qty,
                'fill_price' : fill_price,
                'notional'   : notional,
                'commission' : commission,
            }

    def close_position(self, market_price: float) -> dict:
        """Close the current open position at market price (with slippage)."""
        with self._lock:
            if self.position is None:
                return {'status': 'no_position'}

            pos = self.position
            if pos.side == 'LONG':
                fill_price  = market_price * (1 - self.slippage_pct)   # exit lower
                pnl         = (fill_price - pos.entry_price) * pos.qty
                proceeds    = fill_price * pos.qty
                self.cash  += proceeds                                  # return notional + gain

            else:  # SHORT — return locked notional + capture P&L
                fill_price  = market_price * (1 + self.slippage_pct)   # exit higher (worse)
                pnl         = (pos.entry_price - fill_price) * pos.qty
                notional_at_entry = pos.entry_price * pos.qty
                self.cash  += notional_at_entry + pnl                  # return collateral ± P&L

            commission       = abs(fill_price * pos.qty * self.commission_pct)
            self.cash       -= commission
            self.total_commission += commission

            net_pnl           = pnl - commission - pos.commission
            self.realized_pnl += net_pnl
            self._daily_pnl  += net_pnl

            if net_pnl > 0:
                self.winning_trades += 1
                self.total_win_pnl  += net_pnl
            else:
                self.losing_trades  += 1
                self.total_loss_pnl += abs(net_pnl)

            closed_position = pos
            self.position   = None

            self._record_snapshot(self.get_equity())

            logger.info(f"[Portfolio] CLOSE {closed_position.side} {closed_position.qty:.4f} "
                        f"@ ${fill_price:.4f} | P&L=${net_pnl:+.2f} "
                        f"| total_realized=${self.realized_pnl:+.2f}")

            return {
                'status'      : 'closed',
                'side'        : closed_position.side,
                'qty'         : closed_position.qty,
                'entry_price' : closed_position.entry_price,
                'fill_price'  : fill_price,
                'pnl'         : pnl,
                'commission'  : commission,
                'net_pnl'     : net_pnl,
            }

    # ── Mark to Market ────────────────────────────────────────────────────────
    def mark_to_market(self, current_price: float) -> float:
        """
        Returns current unrealized P&L based on latest price.
        Also records equity snapshot.
        """
        with self._lock:
            unrealized = 0.0
            if self.position is not None:
                unrealized = self.position.unrealized_pnl(current_price)

            equity = self.cash + unrealized
            self._record_snapshot(equity)

            # Reset daily tracking if new day
            today = date.today()
            if today != self._today:
                self._today            = today
                self._day_start_equity = equity
                self._daily_pnl        = 0.0

            return unrealized

    # ── Account Metrics ───────────────────────────────────────────────────────
    def get_equity(self, current_price: float = None) -> float:
        """Total account value (cash + unrealized P&L)."""
        if self.position is None or current_price is None:
            return self.cash
        return self.cash + self.position.unrealized_pnl(current_price)

    def get_total_return_pct(self, current_price: float = None) -> float:
        equity = self.get_equity(current_price)
        return (equity / self.initial_capital - 1) * 100

    def get_max_drawdown(self) -> float:
        """Max peak-to-trough drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self.equity_curve]
        peak     = equities[0]
        max_dd   = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def get_win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return (self.winning_trades / total * 100) if total > 0 else 0.0

    def get_profit_factor(self) -> float:
        if self.total_loss_pnl == 0:
            return float('inf') if self.total_win_pnl > 0 else 0.0
        return self.total_win_pnl / self.total_loss_pnl

    def is_daily_loss_limit_hit(self) -> bool:
        """Circuit breaker: stop trading if daily loss > max_daily_loss% of equity."""
        if self._daily_pnl >= 0:
            return False
        loss_pct = abs(self._daily_pnl) / self._day_start_equity
        if loss_pct >= self.max_daily_loss:
            logger.warning(f"[Portfolio] Daily loss limit hit! "
                           f"Loss=${abs(self._daily_pnl):.2f} ({loss_pct:.1%})")
            return True
        return False

    def get_snapshot(self, current_price: float = None) -> dict:
        """Returns full current account state as a dict (thread-safe read)."""
        with self._lock:
            unrealized = 0.0
            pos_info   = None
            if self.position is not None and current_price:
                unrealized = self.position.unrealized_pnl(current_price)
                pos_info   = {
                    'side'          : self.position.side,
                    'qty'           : self.position.qty,
                    'entry_price'   : self.position.entry_price,
                    'current_price' : current_price,
                    'unrealized_pnl': round(unrealized, 2),
                    'unrealized_pct': round(self.position.unrealized_pct(current_price) * 100, 3),
                    'entry_time'    : self.position.entry_time.isoformat(),
                }

            equity = self.cash + unrealized

            return {
                'cash'             : round(self.cash, 2),
                'equity'           : round(equity, 2),
                'initial_capital'  : self.initial_capital,
                'total_return_pct' : round((equity / self.initial_capital - 1) * 100, 3),
                'realized_pnl'     : round(self.realized_pnl, 2),
                'unrealized_pnl'   : round(unrealized, 2),
                'total_commission' : round(self.total_commission, 2),
                'total_trades'     : self.total_trades,
                'winning_trades'   : self.winning_trades,
                'losing_trades'    : self.losing_trades,
                'win_rate_pct'     : round(self.get_win_rate(), 1),
                'profit_factor'    : round(self.get_profit_factor(), 2),
                'max_drawdown_pct' : round(self.get_max_drawdown(), 2),
                'daily_pnl'        : round(self._daily_pnl, 2),
                'open_position'    : pos_info,
                'snapshot_time'    : datetime.utcnow().isoformat(),
            }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _record_snapshot(self, equity: float):
        self.equity_curve.append((datetime.utcnow(), equity))
        # Keep only last 10,000 snapshots to prevent unbounded memory growth
        if len(self.equity_curve) > 10_000:
            self.equity_curve = self.equity_curve[-10_000:]
