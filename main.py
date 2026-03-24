"""
TradingAgents-Pro -- minimal script entry point.

For the full CLI with modes (quick, compare, screen, backtest, portfolio,
demo, depth, preset), use:

    python run.py --help
"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can be set directly

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()

ta = TradingAgentsGraph(debug=True, config=config)
state, decision = ta.propagate("NVDA", "2026-03-24")
print(decision)
