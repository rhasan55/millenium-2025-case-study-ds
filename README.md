# Millennium Candidate Search Platform

Enterprise-grade resume parsing and candidate search platform built with LLM-powered extraction, hybrid semantic search, and advanced filtering for hedge fund recruiting.

## Architecture

- **Extraction**: PDF/DOCX parsing with PyMuPDF and python-docx
- **Parsing**: Multi-tier OpenAI strategy (gpt-4o-mini → o3-mini) with structured JSON schema
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions) for semantic understanding
- **Search**: Hybrid BM25 + semantic embeddings with LLM reranking (gpt-4o-mini)
- **Storage**: In-memory Pydantic models with numpy vector arrays
- **UI**: Streamlit with faceted filters, career progression scoring, insights dashboards, PII-controlled exports

## Features

### Parsing Pipeline
- Multi-format extraction (PDF, DOCX, DOC)
- Multi-tier parsing strategy with automatic escalation
- PII detection with validation (email completeness tracking, phone, LinkedIn)
- LLM-based structured extraction with schema validation
- Ontology mapping for strategies, asset classes, sectors, skills
- Entity resolution and deduplication
- Parse confidence scoring with automatic quality escalation

### Search Engine
- Hybrid semantic + BM25 lexical search with LLM reranking
- OpenAI text-embedding-3-large for superior semantic understanding
- 15+ faceted filters (strategy, sector, experience, education, skills, location, work auth)
- Awards and school tier filtering
- Sub-second search latency
- Match highlighting and relevance ranking

### UI Components
- **Search Page**: Advanced filters, candidate cards with progression badges, shortlisting
- **Profile Page**: Full candidate timeline, career progression intelligence, skills, education, experience
- **Insights Dashboard**: Career progression distribution, strategy/sector/region/skills metrics, top performers leaderboard
- **Export**: CSV/JSON with PII masking controls

### Career Progression Intelligence
- 100-point scoring system: Pedigree (40) + Velocity (35) + Specialization (25)
- Automatic tier classification: Elite, High-Potential, Strong, Developing, Entry
- High-potential indicator tracking (Tier 1 education, prestigious firms, rapid advancement)
- Entry level detection and progression analysis

## Setup

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Run Application

```bash
streamlit run app.py
```

The application will:
1. Auto-load resumes from `2025 DS Case Study/` directory
2. Parse and extract structured data via OpenAI
3. Generate embeddings for semantic search
4. Launch web interface at `http://localhost:8501`

## Usage

### Search
1. Navigate to **Search** page
2. Apply filters in left sidebar (strategy, sector, experience, skills, etc.)
3. Enter semantic query (e.g., "GLP-1 coverage, DCF modeling, healthcare")
4. Click **Search** to see ranked results
5. Toggle **Show PII** to reveal contact information

### Profile View
- Click **View Full Profile** on any candidate card
- See complete education timeline, experience, skills breakdown
- Access contact information and metadata

### Insights
- Navigate to **Insights** page
- View pipeline health: strategy distribution, top sectors, regional coverage
- Analyze skill prevalence and school funnel

### Export
- Navigate to **Export** page
- Select export scope (All, Search Results, Shortlist)
- Toggle PII inclusion
- Download CSV or JSON

## Data Model

```json
{
  "candidate_id": "cand_abc123",
  "name": "Jane Doe",
  "contacts": {
    "email": "jane@example.com",
    "email_is_valid": true,
    "phone": "+1-555-0100",
    "linkedin": "linkedin.com/in/janedoe"
  },
  "location": "New York, NY",
  "work_auth": "US Citizen",
  "years_experience_band": "3-5",
  "strategy": ["Fundamental"],
  "asset_classes": ["Equities"],
  "sector_coverage": ["Healthcare", "Biotechnology"],
  "skills": [
    {"skill": "Python", "family": "Programming", "level": "Advanced"}
  ],
  "education": [
    {
      "institution": "MIT",
      "degree": "BS",
      "major": "Computer Science",
      "gpa": 3.9
    }
  ],
  "experience": [
    {
      "employer": "Goldman Sachs",
      "title": "Equity Research Analyst",
      "start_date": "2021-07",
      "end_date": null,
      "sector_coverage": ["Healthcare"]
    }
  ],
  "meta": {
    "source_file": "jane_doe_resume.pdf",
    "parse_confidence": 0.92,
    "processing_time_ms": 2345
  }
}
```

## Configuration

Edit `config.py` to modify:
- **Ontologies**: Strategies, asset classes, sectors, skills taxonomy
- **School Tiers**: University classifications
- **Experience Bands**: Seniority levels
- **LLM Settings**: Model, temperature, max tokens
- **Search Parameters**: Latency targets, semantic weights

## Evaluation

The platform includes built-in quality metrics:
- Parse confidence scoring (0-1)
- Field-level extraction accuracy
- Search latency monitoring (p50, p95)
- Deduplication detection

## Security & Privacy

- **PII Isolation**: Contact information stored separately, masked in logs
- **Export Controls**: PII redaction toggle on all exports
- **Field Encryption**: Contact fields use Pydantic validators
- **Audit Trail**: Parse metadata tracks source, timestamp, confidence

## Performance

- **Parse**: ~1-2s per resume (gpt-4o-mini), ~3-5s with escalation (o3-mini)
- **Embedding**: ~100ms per candidate (text-embedding-3-large)
- **Search**: <300ms median latency at 10 candidates
- **Reranking**: ~500ms for top-10 results (gpt-4o-mini)
- **Throughput**: Batch processing supported for 100+ resumes

## Ontologies

### Strategies
Fundamental, Systematic, Credit, Macro

### Asset Classes
Equities, Credit, Macro, Commodities, Fixed Income

### Sectors (GICS-aligned)
Technology, Healthcare, Financials, Consumer, Industrials, Energy, Materials, Real Estate, Telecom, Utilities

### Skills Taxonomy
- **Programming**: Python, C++, R, SQL, Java
- **ML/AI**: Machine Learning, NLP, Time Series, PyTorch
- **Finance**: DCF, LBO, Valuation, Factor Modeling, Risk
- **Fixed Income**: Credit Modeling, Bond Pricing, Spread Analysis
- **Derivatives**: Options, Swaps, Greeks, Vol Surface
- **Quant**: Statistical Arbitrage, Backtesting, Portfolio Optimization
- **Tools**: Bloomberg, FactSet, CapIQ, Excel

## Roadmap

- [ ] PostgreSQL persistence for large-scale deployment
- [ ] REST API endpoints for integration
- [ ] Active learning for parse quality improvement
- [ ] Interview scheduling integration
- [ ] Saved searches and alerts
- [ ] Candidate comparison (side-by-side)
- [ ] Multi-language support
- [ ] Diversity analytics

## Author

Raja Hasan
rajmhasan@gmail.com
