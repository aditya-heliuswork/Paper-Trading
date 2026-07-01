# ============================================================================
# PETROQUANT PAPER TRADING — TRADE LOG (SQLite)
# ============================================================================
# Persistent SQLite database for all paper trades, equity snapshots,
# and session summaries. Survives server restarts.
#
# Tables:
#   trades    — every trade execution (BUY/SELL/HOLD) with full fill details
#   snapshots — periodic equity snapshots (logged every minute)
#
# TradeLog:
#   log_trade(...)     — insert a trade record
#   log_snapshot(...)  — insert an equity snapshot
#   get_trade_history()— pd.DataFrame of all trades
#   get_snapshots()    — pd.DataFrame of all snapshots
#   get_summary()      — dict of key stats
#   export_csv(path)   — export trades to CSV file
# ============================================================================

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
import logging
import os

from . import config as cfg

logger = logging.getLogger(__name__)


class TradeLog:
    """
    SQLite-backed persistent trade log for the paper trading engine.

    The database is stored at config.DB_PATH and persists across restarts.
    All timestamps are stored as UTC ISO-8601 strings.
    """

    def __init__(self, db_path: str = cfg.DB_PATH):
        self.db_path = db_path
        if db_path != ':memory:':
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._mem_conn = None
        else:
            # For in-memory DBs: keep one persistent connection (shared memory)
            self._mem_conn = sqlite3.connect(':memory:', check_same_thread=False)
        self._init_db()
        logger.info(f"[TradeLog] SQLite database: {db_path}")


    # ── Database Setup ────────────────────────────────────────────────────────
    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    action          TEXT NOT NULL,
                    signal          TEXT,
                    ticker          TEXT DEFAULT 'CL=F',
                    interval        TEXT DEFAULT '1m',
                    horizon         INTEGER DEFAULT 5,
                    price           REAL,
                    quantity        REAL,
                    value           REAL,
                    commission      REAL,
                    slippage_est    REAL,
                    regime          TEXT,
                    probability     REAL,
                    position_size   REAL,
                    pnl_realized    REAL DEFAULT 0.0,
                    pnl_unrealized  REAL DEFAULT 0.0,
                    cash_balance    REAL,
                    equity          REAL,
                    cumulative_pnl  REAL DEFAULT 0.0,
                    notes           TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    date            TEXT NOT NULL,
                    cash            REAL,
                    equity          REAL,
                    unrealized_pnl  REAL,
                    realized_pnl    REAL,
                    total_return_pct REAL,
                    max_drawdown_pct REAL,
                    total_trades    INTEGER,
                    win_rate_pct    REAL,
                    regime          TEXT,
                    signal          TEXT,
                    probability     REAL,
                    wti_price       REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp)")

    # ── Write ─────────────────────────────────────────────────────────────────
    def log_trade(self,
                  timestamp     : datetime,
                  action        : str,
                  signal        : str,
                  price         : float,
                  quantity      : float,
                  commission    : float,
                  regime        : str,
                  probability   : float,
                  position_size : float,
                  pnl_realized  : float,
                  pnl_unrealized: float,
                  cash_balance  : float,
                  equity        : float,
                  notes         : str = '') -> int:
        """Insert a trade record into the database. Returns the row id."""
        ts_str  = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        date_str= ts_str[:10]
        value   = price * quantity if price and quantity else 0.0
        slippage_est = value * cfg.SLIPPAGE_PCT

        # Running cumulative P&L
        cum_pnl = self._get_cumulative_pnl() + pnl_realized

        with self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO trades
                    (timestamp, date, action, signal, ticker, interval, horizon,
                     price, quantity, value, commission, slippage_est,
                     regime, probability, position_size,
                     pnl_realized, pnl_unrealized,
                     cash_balance, equity, cumulative_pnl, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?)
            """, (ts_str, date_str, action, signal, cfg.TICKER_INTRADAY,
                  cfg.CANDLE_INTERVAL, cfg.PREDICT_HORIZON,
                  price, quantity, value, commission, slippage_est,
                  regime, probability, position_size,
                  pnl_realized, pnl_unrealized,
                  cash_balance, equity, cum_pnl, notes))
            return cursor.lastrowid

    def log_snapshot(self,
                     timestamp  : datetime,
                     snapshot   : dict,
                     regime     : str,
                     signal     : str,
                     probability: float) -> None:
        """Insert an equity snapshot (called every minute)."""
        ts_str   = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        date_str = ts_str[:10]

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO snapshots
                    (timestamp, date, cash, equity, unrealized_pnl, realized_pnl,
                     total_return_pct, max_drawdown_pct, total_trades, win_rate_pct,
                     regime, signal, probability, wti_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts_str, date_str,
                  snapshot.get('cash', 0),
                  snapshot.get('equity', 0),
                  snapshot.get('unrealized_pnl', 0),
                  snapshot.get('realized_pnl', 0),
                  snapshot.get('total_return_pct', 0),
                  snapshot.get('max_drawdown_pct', 0),
                  snapshot.get('total_trades', 0),
                  snapshot.get('win_rate_pct', 0),
                  regime, signal, probability,
                  snapshot.get('open_position', {}).get('current_price') if snapshot.get('open_position') else None))

    # ── Read ──────────────────────────────────────────────────────────────────
    def get_trade_history(self, limit: int = None, actions_only: bool = False) -> pd.DataFrame:
        """
        Returns trade history as a DataFrame.

        Parameters
        ----------
        limit        : int  — max rows to return (most recent first), None = all
        actions_only : bool — if True, excludes HOLD rows
        """
        where = " WHERE action NOT IN ('HOLD')" if actions_only else ""
        query = f"SELECT * FROM trades{where} ORDER BY timestamp DESC"

        with self._get_conn() as conn:
            if limit:
                df = pd.read_sql_query(query + " LIMIT ?", conn,
                                       params=[int(limit)], parse_dates=['timestamp'])
            else:
                df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        return df

    def get_snapshots(self, limit: int = 1000) -> pd.DataFrame:
        """Returns equity snapshots as a DataFrame, ordered oldest-first."""
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM snapshots ORDER BY timestamp ASC LIMIT ?",
                conn, params=(limit,), parse_dates=['timestamp']
            )
        return df.reset_index(drop=True)

    def get_summary(self) -> dict:
        """Returns key statistics from the trade log."""
        with self._get_conn() as conn:
            # Trade counts
            counts = conn.execute("""
                SELECT
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN action LIKE 'OPEN%' THEN 1 ELSE 0 END) as total_trades,
                    SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl_realized) as total_realized_pnl,
                    SUM(commission) as total_commission,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trades
            """).fetchone()

            # Latest equity
            latest = conn.execute("""
                SELECT equity, total_return_pct, max_drawdown_pct
                FROM snapshots ORDER BY timestamp DESC LIMIT 1
            """).fetchone()

        total_trades = counts[1] or 0
        wins         = counts[2] or 0
        losses       = counts[3] or 0
        win_rate     = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        return {
            'total_logged_rows'  : counts[0],
            'total_trades'       : total_trades,
            'winning_trades'     : wins,
            'losing_trades'      : losses,
            'win_rate_pct'       : round(win_rate, 1),
            'total_realized_pnl' : round(counts[4] or 0, 2),
            'total_commission'   : round(counts[5] or 0, 2),
            'first_trade'        : counts[6],
            'last_trade'         : counts[7],
            'latest_equity'      : round(latest[0], 2) if latest else cfg.INITIAL_CAPITAL,
            'total_return_pct'   : round(latest[1], 3) if latest else 0.0,
            'max_drawdown_pct'   : round(latest[2], 2) if latest else 0.0,
        }

    def export_csv(self, path: str = None) -> str:
        """Export full trade history to CSV. Returns file path."""
        if path is None:
            path = os.path.join(cfg.OUTPUT_DIR, 'paper_trades_export.csv')
        df = self.get_trade_history()
        df.to_csv(path, index=False)
        logger.info(f"[TradeLog] Exported {len(df)} rows to {path}")
        return path

    def get_today_trades(self) -> pd.DataFrame:
        """Returns trades executed today only."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        with self._get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE date = ? ORDER BY timestamp",
                conn, params=(today,), parse_dates=['timestamp']
            )
        return df

    # ── Internal ──────────────────────────────────────────────────────────────
    @contextmanager
    def _get_conn(self):
        """
        Context manager that yields a db connection and always closes it.
        For in-memory DBs, reuses the single persistent connection.
        For file DBs, opens a fresh connection, commits on success, closes on exit.
        This fixes the connection leak that existed with the old _connect() approach.
        """
        if self._mem_conn is not None:
            self._mem_conn.row_factory = sqlite3.Row
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _get_cumulative_pnl(self) -> float:
        """Get the last cumulative_pnl value for running total calculation."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT cumulative_pnl FROM trades ORDER BY id DESC LIMIT 1"
                ).fetchone()
                return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            return 0.0

    def get_portfolio_state(self) -> dict:
        """
        Reconstruct portfolio state from DB for restoration after a restart.
        Returns empty dict if no prior data exists.
        """
        try:
            with self._get_conn() as conn:
                snap = conn.execute("""
                    SELECT equity, realized_pnl, total_trades
                    FROM snapshots ORDER BY timestamp DESC LIMIT 1
                """).fetchone()

                if not snap or snap[0] is None:
                    return {}

                stats = conn.execute("""
                    SELECT
                        SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized > 0
                                 THEN pnl_realized ELSE 0 END)          AS total_win_pnl,
                        SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized < 0
                                 THEN ABS(pnl_realized) ELSE 0 END)     AS total_loss_pnl,
                        SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized > 0
                                 THEN 1 ELSE 0 END)                     AS winning_trades,
                        SUM(CASE WHEN action LIKE 'CLOSE%' AND pnl_realized <= 0
                                 THEN 1 ELSE 0 END)                     AS losing_trades,
                        SUM(CASE WHEN action LIKE 'OPEN%'  THEN 1 ELSE 0 END) AS total_trades,
                        COALESCE(SUM(commission), 0)                    AS total_commission
                    FROM trades
                """).fetchone()

                last_action = conn.execute("""
                    SELECT action, price, quantity
                    FROM trades
                    WHERE action LIKE 'OPEN%' OR action LIKE 'CLOSE%'
                    ORDER BY id DESC LIMIT 1
                """).fetchone()

                had_open = bool(last_action and last_action[0].startswith('OPEN_'))

            return {
                'equity'          : float(snap[0]),
                'realized_pnl'    : float(snap[1] or 0.0),
                'total_trades'    : int(stats[4] or 0),
                'winning_trades'  : int(stats[2] or 0),
                'losing_trades'   : int(stats[3] or 0),
                'total_win_pnl'   : float(stats[0] or 0.0),
                'total_loss_pnl'  : float(stats[1] or 0.0),
                'total_commission': float(stats[5] or 0.0),
                'had_open_position': had_open,
            }
        except Exception as e:
            logger.error(f"[TradeLog] get_portfolio_state error: {e}")
            return {}
