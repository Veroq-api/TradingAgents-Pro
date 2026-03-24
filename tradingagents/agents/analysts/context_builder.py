"""
Context Builder — pre-fetches verified data from Polaris for all agents.

This is NOT an LLM agent. It runs FIRST in the graph, before any analyst,
and builds a shared context package that every subsequent agent can reference
via state["verified_context"]. Each Polaris call is independently wrapped
in try/except for graceful degradation.
"""

from datetime import datetime, timedelta


def create_context_builder(llm_client=None):
    """Build a verified context package from Polaris for all agents to share.

    Args:
        llm_client: Unused — included for API compatibility with other create_* functions.
                    The context builder does not use an LLM.

    Returns:
        A LangGraph node function that fetches data and writes to state["verified_context"].
    """

    def context_builder_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        # Import Polaris functions directly (not through route_to_vendor)
        # so we get the raw data without tool wrappers.
        from tradingagents.dataflows.polaris import (
            get_technicals,
            get_sentiment_score,
            get_sector_analysis,
            get_news_impact,
            get_news,
            get_global_news,
        )

        # Compute a reasonable look-back window for news
        try:
            trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")
            start_date = (trade_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            start_date = trade_date

        # Build context package — each call is wrapped in try/except
        context_parts = []

        try:
            technicals = get_technicals(ticker)
            context_parts.append(f"=== TECHNICAL ANALYSIS ===\n{technicals}")
        except Exception as e:
            context_parts.append(f"=== TECHNICAL ANALYSIS ===\nUnavailable: {e}")

        try:
            signal = get_sentiment_score(ticker)
            context_parts.append(f"=== COMPOSITE TRADING SIGNAL ===\n{signal}")
        except Exception as e:
            context_parts.append(f"=== COMPOSITE TRADING SIGNAL ===\nUnavailable: {e}")

        try:
            sector = get_sector_analysis(ticker)
            context_parts.append(f"=== SECTOR & PEER ANALYSIS ===\n{sector}")
        except Exception as e:
            context_parts.append(f"=== SECTOR & PEER ANALYSIS ===\nUnavailable: {e}")

        try:
            impact = get_news_impact(ticker)
            context_parts.append(f"=== NEWS IMPACT ANALYSIS ===\n{impact}")
        except Exception as e:
            context_parts.append(f"=== NEWS IMPACT ANALYSIS ===\nUnavailable: {e}")

        try:
            news = get_news(ticker, start_date, trade_date)
            context_parts.append(f"=== RECENT INTELLIGENCE BRIEFS ===\n{news}")
        except Exception as e:
            context_parts.append(f"=== RECENT INTELLIGENCE BRIEFS ===\nUnavailable: {e}")

        try:
            global_news = get_global_news(start_date, trade_date)
            context_parts.append(f"=== GLOBAL MARKET INTELLIGENCE ===\n{global_news}")
        except Exception as e:
            context_parts.append(f"=== GLOBAL MARKET INTELLIGENCE ===\nUnavailable: {e}")

        context = "\n\n".join(context_parts)

        return {"verified_context": context}

    return context_builder_node
