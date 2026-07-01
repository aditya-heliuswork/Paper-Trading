# ============================================================================
# PETROQUANT PAPER TRADING — FLASK WEB SERVER
# ============================================================================
# Serves the live dashboard and JSON API endpoints.
#
# Endpoints:
#   GET  /               → HTML dashboard (auto-refresh every 60s)
#   GET  /dashboard      → same as above
#   GET  /status         → JSON: current position, equity, last signal, timeframe
#   GET  /trades         → CSV download: full trade log
#   GET  /trades/json    → JSON: last 100 trades
#   GET  /health         → {"status":"ok"} for uptime monitoring
#   POST /set-timeframe  → JSON body {"timeframe":"5m"} — switch trading timeframe
#   GET  /reset          → confirmation page
#   POST /reset          → reset all trades (body: confirm=yes)
#
# Bug #15 fix: current_price is passed as None (not 0) when the engine hasn't
#   received a first tick yet — the dashboard handles None gracefully.
# ============================================================================

from flask import Flask, Response, request, jsonify, send_file
import threading
import logging
import os
import io

from . import config as cfg

logger = logging.getLogger(__name__)


def create_web_server(engine_ref: dict) -> Flask:
    """
    Creates the Flask app. engine_ref is a dict holding live references:
      engine_ref['engine']    → PaperTradingEngine instance
      engine_ref['portfolio'] → Portfolio instance
      engine_ref['trade_log'] → TradeLog instance
      engine_ref['dashboard'] → LiveDashboard instance
    """
    app = Flask(__name__, static_folder=None)
    app.logger.disabled = True
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)   # silence Flask request logs

    # ── Dashboard ─────────────────────────────────────────────────────────────
    @app.route('/')
    @app.route('/dashboard')
    def dashboard():
        try:
            dash      = engine_ref.get('dashboard')
            engine    = engine_ref.get('engine')

            # Bug #15 fix: use None (not 0) when no price yet — dashboard renders "Connecting..."
            current_price = engine.last_price if engine else None
            regime        = engine.current_regime if engine else 'UNKNOWN'
            model_status  = engine.model.get_model_status() if (engine and engine.model) else {}

            html = dash.render(
                current_price=current_price,   # can be None — handled in dashboard
                regime=regime,
                model_status=model_status,
            )
            return Response(html, mimetype='text/html')

        except Exception as e:
            logger.error(f"[WebServer] Dashboard error: {e}", exc_info=True)
            return Response(f"<pre>Dashboard error: {e}</pre>", status=500)

    # ── Status JSON ───────────────────────────────────────────────────────────
    @app.route('/status')
    def status():
        try:
            engine    = engine_ref.get('engine')
            portfolio = engine.portfolio if engine else engine_ref.get('portfolio')
            trade_log = engine_ref.get('trade_log')

            current_price = engine.last_price if engine else None
            snap    = portfolio.get_snapshot(current_price) if portfolio else {}
            summary = trade_log.get_summary() if trade_log else {}

            return jsonify({
                'status'        : 'running',
                'current_price' : current_price,
                'regime'        : engine.current_regime if engine else 'UNKNOWN',
                'last_signal'   : engine.last_signal    if engine else 'UNKNOWN',
                'last_prob'     : engine.last_prob       if engine else None,
                'timeframe'     : cfg.ACTIVE_TIMEFRAME,
                'loop_secs'     : cfg.LOOP_INTERVAL_SECS,
                'predict_horizon': cfg.PREDICT_HORIZON,
                'portfolio'     : snap,
                'summary'       : summary,
                'model'         : engine.model.get_model_status() if (engine and engine.model) else {},
                'market_open'   : engine.is_market_open() if engine else False,
                'uptime_secs'   : engine.uptime_seconds() if engine else 0,
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # ── Set Timeframe (NEW endpoint) ──────────────────────────────────────────
    @app.route('/set-timeframe', methods=['POST'])
    def set_timeframe():
        """
        Switch the paper trader to a different timeframe at runtime.

        Body: JSON {"timeframe": "5m"}
        Valid values: 1m, 5m, 15m, 1h, 1d

        The engine stops its loop, applies the new config, resets the model,
        then restarts. Existing trades in the DB are preserved.
        """
        try:
            data = request.get_json(silent=True) or {}
            tf   = str(data.get('timeframe', '')).strip()

            if not tf:
                return jsonify({'error': 'timeframe is required',
                                'valid': list(cfg.TIMEFRAME_CONFIGS.keys())}), 400

            if tf not in cfg.TIMEFRAME_CONFIGS:
                return jsonify({'error': f"Unknown timeframe '{tf}'",
                                'valid': list(cfg.TIMEFRAME_CONFIGS.keys())}), 400

            engine = engine_ref.get('engine')
            if engine is None:
                return jsonify({'error': 'Engine not available'}), 503

            result = engine.switch_timeframe(tf)
            logger.info(f"[WebServer] Timeframe switched to {tf} via API")

            return jsonify({
                'status'   : 'ok',
                'timeframe': result['timeframe'],
                'loop_secs': result['loop_secs'],
                'horizon'  : result['horizon'],
                'message'  : f"Switched to {cfg.get_timeframe_label(tf)} timeframe",
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"[WebServer] set-timeframe error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    # ── Trade Log CSV ─────────────────────────────────────────────────────────
    @app.route('/trades')
    def download_trades():
        try:
            trade_log = engine_ref.get('trade_log')
            df = trade_log.get_trade_history()
            csv_str = df.to_csv(index=False)
            return Response(
                csv_str,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=paper_trades.csv'}
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ── Trade Log JSON ────────────────────────────────────────────────────────
    @app.route('/trades/json')
    def trades_json():
        try:
            trade_log = engine_ref.get('trade_log')
            limit = int(request.args.get('limit', 100))
            df = trade_log.get_trade_history(limit=limit)
            return jsonify(df.to_dict(orient='records'))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.route('/health')
    def health():
        engine = engine_ref.get('engine')
        return jsonify({
            'status'    : 'ok',
            'version'   : '1.1.0',
            'running'   : engine._running if engine else False,
            'uptime_s'  : engine.uptime_seconds() if engine else 0,
            'timeframe' : cfg.ACTIVE_TIMEFRAME,
        })

    # ── Reset (protected) ─────────────────────────────────────────────────────
    @app.route('/reset', methods=['GET', 'POST'])
    def reset():
        engine = engine_ref.get('engine')

        if request.method == 'GET':
            capital_fmt = f"${cfg.INITIAL_CAPITAL:,.0f}"
            return Response(f"""
                <html><body style="font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:40px">
                <h2>&#9888; Reset Paper Account</h2>
                <p>This will permanently delete <strong>ALL trades</strong> and reset
                the portfolio to <strong>{capital_fmt}</strong>.</p>
                <form method="POST">
                    <input type="hidden" name="confirm" value="yes">
                    <button type="submit"
                        style="background:#f85149;color:#fff;border:none;padding:10px 24px;
                               font-size:16px;border-radius:6px;cursor:pointer">
                        Confirm Reset
                    </button>
                </form>
                <br><a href="/" style="color:#388bfd">&#8592; Back to dashboard</a>
                </body></html>
            """, mimetype='text/html')

        # POST: perform the reset
        confirm = request.form.get('confirm', '')
        if confirm != 'yes':
            return jsonify({'error': 'Must submit confirm=yes via POST'}), 400

        try:
            if engine:
                engine.reset_account()
                # Keep engine_ref references current after reset
                engine_ref['portfolio'] = engine.portfolio
            return jsonify({'status': 'reset',
                            'message': f'Account reset to ${cfg.INITIAL_CAPITAL:,.0f}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def run_web_server(app: Flask, port: int = cfg.WEB_PORT):
    """Runs Flask in a background daemon thread."""
    thread = threading.Thread(
        target=lambda: app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
        name='WebServer'
    )
    thread.start()
    logger.info(f"[WebServer] Started on http://0.0.0.0:{port}")
    return thread
