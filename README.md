**FinPulse AI** is a sophisticated data science application designed to bridge the gap between unstructured financial news and quantitative market signals. By architecting a multi-layered NLP pipeline—utilizing specialized Transformer models for sentiment and Large Language Models (LLMs) for reasoning—the system provides a comprehensive "Market Intelligence" report for any publicly traded equity.

🚀**Core Engineering Features**  
Domain-Specific Sentiment Classification: Unlike general-purpose NLP, this system utilizes FinBERT, a model pre-trained on a massive financial corpus, ensuring accurate classification of finance-specific jargon (e.g., distinguishing between "interest rate hikes" and "favorable volatility").

AI-Driven Narrative Synthesis: Leverages Llama 3.3 (via Groq) to ingest high-volume news headlines and synthesize them into a concise "Market Outlook," identifying critical risk vectors and growth opportunities.

Dynamic Market Ingestion: Engineered with a robust integration of the yfinance API, enabling real-time extraction of global news metadata and historical pricing data.

Predictive Visualization: Developed a reactive frontend using Streamlit and Plotly to visualize the correlation between cumulative sentiment trends and stock price action.

🛡️ **Enterprise-Grade Infrastructure**  
Secure Credential Management: Implemented an industry-standard security layer using Streamlit Secrets and .toml configuration, ensuring zero-exposure of sensitive API credentials in public repositories.

High-Inference Optimization: Optimized model deployment by utilizing Groq's LPU architecture for sub-second LLM inference and Streamlit Caching (@st.cache_resource) to maintain a low memory footprint.

Defensive Programming: Architected the news parser with robust error handling (Safe Dictionary Access) to maintain system stability despite volatile third-party API data structures.

🛠️**Technical Stack**
Backend: Python, Pandas, yfinance

NLP/AI: FinBERT (Transformers), LangChain, Groq (Llama 3.3)

Frontend: Streamlit, Plotly Express

Security: Streamlit Secrets Management, .gitignore protocol
