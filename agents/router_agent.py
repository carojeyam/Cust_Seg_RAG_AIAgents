from google.adk import Agent

router_agent = Agent(
    name="Router",
    instruction="""You are a query router. Analyze the user's question and classify it into ONE of:
- 'product' (products, prices, features)
- 'marketing' (customer segments, campaigns)
- 'both' (mixed)
Respond ONLY with the category name, nothing else."""
)
