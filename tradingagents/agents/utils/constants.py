"""Shared constants for all agents."""

NO_HALLUCINATE = """
CRITICAL RULES:
- Only reference data explicitly provided to you. NEVER fabricate numbers, prices, percentages, dates, or claims.
- If data is missing or unavailable, state "Data unavailable" or "N/A" — do not guess or estimate.
- It is acceptable to have gaps in your analysis. Accuracy over completeness.
- If you are uncertain about a figure, say so. Do not present speculation as fact.
- Attribute every specific number to the data source it came from.
""".strip()
