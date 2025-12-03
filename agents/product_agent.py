from google.adk import Agent

product_agent = Agent(
    name="ProductExpert",
    instruction="""You are a product expert. Answer questions about our products based on provided context.
Be detailed, helpful, and reference specific features and prices when relevant."""
)

