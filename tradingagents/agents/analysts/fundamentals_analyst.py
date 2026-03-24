from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)
from tradingagents.agents.utils.polaris_tools import get_sec_filings, get_sector_analysis
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_sec_filings,
            get_sector_analysis,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. You have access to full company financials from Polaris including balance sheet, income statement, cash flow, and SEC filings. Also available: earnings data and sector peer comparison with live metrics. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, `get_sec_filings` for SEC EDGAR filings, and `get_sector_analysis` for sector peer comparison."
            + "\n\n[TradingAgents-Pro Enhancement] Accuracy Safeguard (not present in original TradingAgents): CRITICAL: Only reference data explicitly provided to you. NEVER fabricate numbers, prices, percentages, dates, or claims. If data is missing, state 'Data unavailable' — do not guess. Accuracy over completeness.",
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
