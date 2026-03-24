from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import time
import json
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.agents.utils.polaris_tools import get_sentiment_score
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_sentiment_score,
        ]

        system_message = (
            "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You have access to numeric sentiment data (-1.0 to +1.0) with trend direction and 7-day averages, plus a composite trading signal that weights sentiment (40%), momentum (25%), volume (20%), and events (15%). Use specific numbers, not vague descriptions. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Use the get_news(query, start_date, end_date) tool to search for company-specific news and social media discussions, and get_sentiment_score(symbol, curr_date) for composite trading signals with numeric sentiment data. Try to look at all sources possible from social media to sentiment to news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
