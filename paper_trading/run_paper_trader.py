#!/usr/bin/env python3
# ============================================================================
# PETROQUANT PAPER TRADING — ENTRY POINT
# ============================================================================
# Run modes:
#   python run_paper_trader.py                       → full engine + web server
#   python run_paper_trader.py --timeframe 5m        → run on 5m candles
#   python run_paper_trader.py --once                → run one cycle and exit
#   python run_paper_trader.py --dashboard           → generate dashboard only
#   python run_paper_trader.py --reset               → reset paper account
#   python run_paper_trader.py --status              → print current status
#   python run_paper_trader.py --no-server           → engine without web server
#
# Valid timeframes: 1m (default), 5m, 15m, 1h, 1d
#
# Bug #13 fix: --timeframe CLI argument added.
# ============================================================================

import sys
import os
import argparse
import time
import signal
import logging

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trading import config as cfg
from paper_trading.engine     import PaperTradingEngine
from paper_trading.web_server import create_web_server, run_web_server

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='PetroQuant Paper Trading Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_trading/run_paper_trader.py                    # 1m candles (default)
  python paper_trading/run_paper_trader.py --timeframe 5m     # 5-minute candles
  python paper_trading/run_paper_trader.py --timeframe 15m    # 15-minute candles
  python paper_trading/run_paper_trader.py --timeframe 1h     # 1-hour candles
  python paper_trading/run_paper_trader.py --timeframe 1d     # Daily candles
  python paper_trading/run_paper_trader.py --once             # One cycle, then exit
  python paper_trading/run_paper_trader.py --dashboard        # Dashboard only
  python paper_trading/run_paper_trader.py --status           # Account status
  python paper_trading/run_paper_trader.py --reset            # Reset account
        """
    )
    # Bug #13 fix: timeframe argument
    parser.add_argument(
        '--timeframe', '-tf',
        default='1m',
        choices=list(cfg.TIMEFRAME_CONFIGS.keys()),
        help='Trading timeframe: 1m (default), 5m, 15m, 1h, 1d'
    )
    parser.add_argument('--once',      action='store_true', help='Run one cycle and exit')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard and exit')
    parser.add_argument('--reset',     action='store_true', help='Reset paper account')
    parser.add_argument('--status',    action='store_true', help='Print account status')
    parser.add_argument('--no-server', action='store_true', help='Run engine without web server')
    args = parser.parse_args()

    _print_banner(args.timeframe)

    # ── Initialize engine (applies timeframe config) ─────────────────────────
    engine = PaperTradingEngine(timeframe=args.timeframe)

    # ── Mode: reset ──────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("\n⚠  Reset paper account? This deletes ALL trades. Type 'yes' to confirm: ")
        if confirm.strip().lower() == 'yes':
            engine.reset_account()
            print("✓ Account reset to ${:,.0f}".format(cfg.INITIAL_CAPITAL))
        else:
            print("✗ Reset cancelled.")
        return

    # ── Mode: status ─────────────────────────────────────────────────────────
    if args.status:
        trade_log = engine.trade_log
        summary   = trade_log.get_summary()
        snap      = engine.portfolio.get_snapshot()
        print("\n" + "=" * 60)
        print("  PetroQuant Paper Trading — Account Status")
        print("=" * 60)
        print(f"  Timeframe      : {cfg.ACTIVE_TIMEFRAME} ({cfg.get_timeframe_label()})")
        print(f"  Capital        : ${cfg.INITIAL_CAPITAL:,.2f}")
        print(f"  Equity         : ${summary.get('latest_equity', cfg.INITIAL_CAPITAL):,.2f}")
        print(f"  Total Return   : {summary.get('total_return_pct', 0):+.3f}%")
        print(f"  Realized P&L   : ${summary.get('total_realized_pnl', 0):+,.2f}")
        print(f"  Total Trades   : {summary.get('total_trades', 0)}")
        print(f"  Win Rate       : {summary.get('win_rate_pct', 0):.1f}%")
        print(f"  Max Drawdown   : {summary.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Commission Paid: ${summary.get('total_commission', 0):,.2f}")
        print(f"  First Trade    : {summary.get('first_trade', 'N/A')}")
        print(f"  Last Trade     : {summary.get('last_trade', 'N/A')}")
        print("=" * 60)
        return

    # ── Mode: dashboard only ──────────────────────────────────────────────────
    if args.dashboard:
        from paper_trading.price_feed   import fetch_candles, fetch_latest_price
        from paper_trading.daily_regime import DailyRegimeDetector
        from paper_trading.dashboard_live import LiveDashboard

        print(f"\n[Dashboard] Fetching {cfg.ACTIVE_TIMEFRAME} candles...")
        price_df = fetch_candles(interval=cfg.CANDLE_INTERVAL, days=cfg.BARS_TO_FETCH)
        current  = fetch_latest_price()
        regime, _ = DailyRegimeDetector().get_regime()

        dash = LiveDashboard(engine.trade_log, engine.portfolio, price_df)
        html = dash.render(current_price=current, regime=regime)
        path = dash.save(html)
        print(f"[Dashboard] Saved → {path}")
        print(f"[Dashboard] Open in browser: file://{path}")
        return

    # ── Mode: once ────────────────────────────────────────────────────────────
    if args.once:
        print(f"\n[Engine] Running single cycle (timeframe: {cfg.ACTIVE_TIMEFRAME})...")
        result = engine.run_once()
        print(f"\n[Engine] Result: {result}")
        print(f"[Engine] Dashboard: {cfg.DASHBOARD_PATH}")
        return

    # ── Mode: full engine (default) ───────────────────────────────────────────
    engine_ref = {
        'engine'   : engine,
        'portfolio': engine.portfolio,
        'trade_log': engine.trade_log,
        'dashboard': engine.dashboard,
    }

    # Start web server (unless --no-server)
    if not args.no_server:
        app = create_web_server(engine_ref)
        run_web_server(app, port=cfg.WEB_PORT)
        print(f"\n[OK] Web server running at http://0.0.0.0:{cfg.WEB_PORT}")
        print(f"  Dashboard  : http://localhost:{cfg.WEB_PORT}/")
        print(f"  Status     : http://localhost:{cfg.WEB_PORT}/status")
        print(f"  Trades     : http://localhost:{cfg.WEB_PORT}/trades")
        print(f"  Timeframe  : http://localhost:{cfg.WEB_PORT}/set-timeframe (POST)")

    # Start trading loop in background
    engine.start()
    print(f"\n[OK] Paper trading engine started ({cfg.ACTIVE_TIMEFRAME} candles). "
          f"Press Ctrl+C to stop.\n")

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────────────────
    def _shutdown(sig, frame):
        print("\n\n[Engine] Shutting down gracefully...")
        engine.stop()
        snap = engine.portfolio.get_snapshot(engine.last_price)
        print(f"[Engine] Final equity: ${snap['equity']:,.2f} "
              f"({snap['total_return_pct']:+.3f}%)")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
    except SystemExit:
        pass


# ─────────────────────────────────────────────────────────────────────────────
def _print_banner(timeframe: str = '1m'):
    tf_info = cfg.TIMEFRAME_CONFIGS.get(timeframe, {})
    label   = tf_info.get('label', timeframe)
    horizon = tf_info.get('predict_horizon', 5)
    loop    = tf_info.get('loop_secs', 60)

    print(f"""
==========================================================
         PetroQuant - Paper Trading Engine v1.1           
                                                          
  Asset     : WTI Crude Oil Futures (CL=F)               
  Timeframe : {label:<40}
  Loop      : Every {loop}s                              
  Signal    : XGBoost -> {horizon}-bar direction prediction  
  Regime    : Daily HMM (BULL / CHOPPY / PANIC)           
  Capital   : $1,000,000 paper USD                        
==========================================================
""")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
