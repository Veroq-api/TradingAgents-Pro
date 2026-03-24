"""
Polaris-exclusive tools for TradingAgents agents.

These tools expose Polaris Knowledge API capabilities that have no equivalent
in yfinance or other vendors: full technical analysis with signal summary,
composite trading signals, sector/peer intelligence, news-to-price impact
analysis, and SEC EDGAR filings.
"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_technicals(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
) -> str:
    """Get full technical analysis with 20 indicators and buy/sell/neutral signal for a stock.

    Returns all indicators at once: SMA, EMA, RSI, MACD, Bollinger, ATR,
    Stochastic, ADX, OBV, VWAP, Williams %R, CCI, MFI, ROC, and more.
    Includes a composite signal summary with buy/sell/neutral recommendation.
    """
    return route_to_vendor("get_technicals", symbol)


@tool
def get_sentiment_score(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
) -> str:
    """Get composite trading signal combining sentiment, momentum, volume, and events.

    Returns a multi-factor score combining:
    - Sentiment (40% weight)
    - Momentum (25% weight)
    - Coverage velocity (20% weight)
    - Event proximity (15% weight)
    """
    return route_to_vendor("get_sentiment_score", symbol)


@tool
def get_sector_analysis(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
) -> str:
    """Get competitor intelligence — same-sector peers with live price, RSI, sentiment.

    Returns a table of sector peers with their current price, change %,
    RSI(14), 7-day sentiment, brief count, and signal.
    """
    return route_to_vendor("get_sector_analysis", symbol)


@tool
def get_news_impact(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
) -> str:
    """Measure how news moved the stock price — brief-to-price causation analysis.

    Returns average 1-day and 3-day price impacts, plus the best and worst
    individual news events and their measured price effect.
    """
    return route_to_vendor("get_news_impact", symbol)


@tool
def get_sec_filings(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Current trading date, YYYY-mm-dd"],
) -> str:
    """Fetch SEC EDGAR earnings filings (8-K, 10-Q, 10-K).

    Returns recent filings with date, form type, description, and URL.
    """
    return route_to_vendor("get_sec_filings", symbol)
