#!/usr/bin/env python3
import os
from agents.tools import product_search, marketing_search
from agents.llm_provider import enhance_with_llm, set_llm_provider, is_llm_enabled
from agents.router_agent import router_agent
from agents.product_agent import product_agent
from agents.marketing_agent import marketing_agent

# Enable LLM (Ollama by default)
try:
    set_llm_provider("ollama", model="mistral")
except Exception as e:
    print(f"⚠️ Ollama init failed: {e}")

def classify_query(query: str) -> str:
    """
    Classifies a query as 'product', 'marketing', or 'both' using enhanced keywords.
    Falls back to 'product' if no clear match.
    """
    query_lower = query.lower()

    # Product-related keywords
    product_keywords = [
        "product", "price", "buy", "available", "specifications", "feature", "features",
        "compare", "comparison", "lowest", "highest", "cheapest", "cost", "order",
        "stock", "quantity", "category", "categories", "type", "types",
        "fish", "fruits", "gold", "meat", "sweets", "wine"
    ]

    # Marketing-related keywords
    marketing_keywords = [
        "customer", "segment", "promotion", "campaign", "marketing", "loyalty",
        "offer", "discount", "bundle", "bundles", "subscription", "subscriptions",
        "online", "web", "first-purchase", "free delivery", "entry-price", "targeted",
        "time-limited", "activation", "digital", "engagement", "value shopper",
        "wine enthusiast", "price-conscious"
    ]

    # Try router agent first (if available)
    try:
        category = router_agent(query)
        if category in ["product", "marketing", "both"]:
            return category
    except Exception:
        pass

    # Count keyword hits
    product_score = sum(kw in query_lower for kw in product_keywords)
    marketing_score = sum(kw in query_lower for kw in marketing_keywords)

    # Decide category
    if product_score > 0 and marketing_score == 0:
        return "product"
    elif marketing_score > 0 and product_score == 0:
        return "marketing"
    elif product_score > 0 and marketing_score > 0:
        return "both"
    else:
        # Fallback for unknown queries → treat as product
        return "product"

def answer_query(query: str, role: str) -> str:
    qtype = classify_query(query)
    if role == "customer" and qtype in ["marketing", "both"]:
        return "❌ Access Denied: Customers can only view product information."
    if qtype == "product":
        results = product_search(query, top_k=10)
        answer = enhance_with_llm(results, query)
        return f"📦 Products:\n{answer}"
    elif qtype == "marketing":
        results = marketing_search(query, top_k=10)
        answer = enhance_with_llm(results, query)
        return f"🎯 Marketing & Segments:\n{answer}"
    else:  # both
        product_results = product_search(query, top_k=10)
        marketing_results = marketing_search(query, top_k=10)
        product_answer = enhance_with_llm(product_results, query)
        marketing_answer = enhance_with_llm(marketing_results, query)
        return f"📦 Products:\n{product_answer}\n\n🎯 Marketing & Segments:\n{marketing_answer}"

def get_user_role() -> str:
    while True:
        print("\n1. Customer\n2. Employee")
        choice = input("Choose role: ").strip()
        if choice == "1": return "customer"
        if choice == "2": return "employee"

def main():
    role = get_user_role()
    while True:
        query = input("❓ Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]: break
        if query.lower() == "role":
            role = get_user_role()
            continue
        if query.lower() == "ollama status":
            print("🤖 Ollama ACTIVE" if is_llm_enabled() else "⚠️ RAG-Only")
            continue
        if not query:
            print("⚠️ Enter a valid question."); continue
        try:
            answer = answer_query(query, role)
            print(f"\n✅ Answer:\n{answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
