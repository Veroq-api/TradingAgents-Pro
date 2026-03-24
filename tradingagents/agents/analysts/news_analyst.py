from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_news,
)
from tradingagents.agents.utils.polaris_tools import get_news_impact
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            get_news_impact,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. You have access to verified intelligence briefs with confidence scores (0-1) and bias scores. Each source has been scored for reliability. When citing a brief, include its confidence score. Prefer briefs with confidence > 0.7. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, and get_news_impact(symbol, curr_date) to measure how news moved the stock price. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read.

# [TradingAgents-Pro Enhancement] Accuracy Safeguard (not present in original TradingAgents)
CRITICAL: Only reference data explicitly provided to you. NEVER fabricate numbers, prices, percentages, dates, or claims. If data is missing, state "Data unavailable" — do not guess. Accuracy over completeness. Attribute every number to its source."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        # Prepend verified context from the Context Builder if available
        messages = list(state["messages"])
        context = state.get("verified_context", "")
        if context:
            messages = [
                SystemMessage(content=f"VERIFIED CONTEXT (pre-fetched from Polaris Knowledge API):\n\n{context}\n\nUse this as your primary data source. You may also use tools for additional detail.")
            ] + messages

        result = chain.invoke(messages)

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
