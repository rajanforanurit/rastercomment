def generate_streaming_comment(row):
    """Generator for Copilot-like streaming"""
    yield "🔍 Analyzing row from Power BI selection...\n"
    yield f"GL Account: {row.get('gl_account', 'N/A')} | Amount: ${row.get('amount', 0):,.0f}\n"
    yield "📊 Feature engineering complete...\n"
    yield "⚖️ Severity classification in progress...\n"
    yield "✍️ Generating intelligent business comment...\n"
    # Final would come from main logic in real streaming endpoint
