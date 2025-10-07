import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
RESUME_DIR = BASE_DIR / "2025 DS Case Study"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOGO_PATH = BASE_DIR / "image.png"

# Try Streamlit secrets first, then fall back to environment variables
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
except (ImportError, FileNotFoundError, AttributeError):
    # Fallback for local development without Streamlit secrets
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384

MAX_TOKENS = 4000
TEMPERATURE = 0.0
PARSE_CONFIDENCE_THRESHOLD = 0.7

STRATEGIES = ["Fundamental", "Systematic", "Credit", "Macro"]
ASSET_CLASSES = ["Equities", "Credit", "Macro", "Commodities", "Fixed Income"]

SECTORS = [
    "Technology", "Semiconductors", "Software", "Hardware",
    "Healthcare", "Biotechnology", "Medical Devices", "Pharmaceuticals",
    "Financials", "Banks", "Insurance", "Asset Management",
    "Consumer", "Retail", "E-commerce", "Consumer Staples",
    "Industrials", "Aerospace", "Defense", "Transportation",
    "Energy", "Renewables", "Oil & Gas",
    "Materials", "Chemicals", "Metals & Mining",
    "Real Estate", "REITs",
    "Telecommunications", "Media",
    "Utilities"
]

SKILLS_TAXONOMY = {
    "Programming": ["Python", "C++", "R", "SQL", "Java", "C#", "JavaScript", "MATLAB", "Julia"],
    "ML_AI": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Time Series", "XGBoost", "PyTorch", "TensorFlow", "scikit-learn"],
    "Finance": ["DCF", "LBO", "Merger Models", "Valuation", "Financial Modeling", "Factor Modeling", "Risk Modeling"],
    "Fixed_Income": ["Credit Modeling", "Bond Pricing", "Duration", "Convexity", "Spread Analysis"],
    "Derivatives": ["Options", "Swaps", "Futures", "Greeks", "Vol Surface"],
    "Quant": ["Statistical Arbitrage", "Algorithmic Trading", "Backtesting", "Signal Research", "Portfolio Optimization"],
    "Data": ["Pandas", "NumPy", "Data Analysis", "Data Visualization", "ETL", "Feature Engineering"],
    "Tools": ["Bloomberg", "FactSet", "CapIQ", "Refinitiv", "Alphalens", "Quantopian", "Excel", "VBA"],
    "Statistics": ["Regression", "Hypothesis Testing", "Bayesian Methods", "Monte Carlo", "Econometrics"]
}

WORK_AUTH_OPTIONS = ["US Citizen", "US Green Card", "US Work Visa", "EU Work Auth", "UK Work Auth", "Canada Work Auth", "Singapore Work Auth", "Need Sponsorship"]
REGIONS = ["North America", "EMEA", "APAC", "Latin America"]
US_STATES = ["NY", "CA", "TX", "IL", "MA", "CT", "NJ", "PA", "FL", "WA", "DC"]

SCHOOL_TIERS = {
    "Tier 1": [
        "MIT", "Stanford", "Harvard", "Princeton", "Yale", "Columbia",
        "University of Chicago", "UC Berkeley", "Caltech", "Penn",
        "University of Pennsylvania", "Wharton", "Cornell", "Northwestern",
        "Carnegie Mellon", "Duke", "Dartmouth", "Brown", "NYU"
    ],
    "Tier 2": [
        "UCLA", "USC", "University of Michigan", "UT Austin",
        "Georgia Tech", "UIUC", "University of Illinois",
        "UW Madison", "UNC Chapel Hill", "Virginia", "Vanderbilt"
    ]
}

EXPERIENCE_BANDS = ["Intern", "New Grad", "1-2", "3-5", "6-8", "9-12", "13+"]

PII_FIELDS = ["email", "phone", "linkedin_url", "website"]

SEARCH_LATENCY_TARGET = 0.3
SEARCH_LATENCY_P95 = 0.8
