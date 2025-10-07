# Setup Instructions

## Step 1: Add Your OpenAI API Key

Open the `.env` file and replace `your_api_key_here` with your actual OpenAI API key:

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx
```

## Step 2: Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Run the Application

```bash
streamlit run app.py
```

Or use the convenience script:

```bash
./run.sh
```

## What Happens on First Run

1. App loads and reads `.env` file for your OpenAI API key
2. Scans `2025 DS Case Study/` folder for PDF/DOCX resumes
3. Extracts text from each resume (2-5 seconds per file)
4. Sends to OpenAI for structured parsing (2-4 seconds per resume)
5. Generates embeddings for semantic search
6. Builds in-memory candidate database
7. Opens browser at http://localhost:8501

**Expected first load time**: 30-90 seconds for 10 resumes

## Verify Setup

Once the app loads, you should see:
- Sidebar showing "Total Candidates: X"
- Search page with filters
- Navigation options (Search, Insights, Export)

## Troubleshooting

### "OPENAI_API_KEY not set in environment"
- Check that `.env` file exists in project root
- Verify your API key is correct (starts with `sk-`)
- Make sure there are no quotes around the key in `.env`

### "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Resumes not loading
- Verify files are in `2025 DS Case Study/` folder
- Check file extensions are .pdf or .docx
- Look at terminal logs for parsing errors

### Slow performance
- First load is slow (LLM parsing each resume)
- Subsequent page refreshes use cached data
- Consider using faster OpenAI model (gpt-3.5-turbo)

## Next Steps

1. Go to Search page
2. Try a search: "Python, healthcare, equity research"
3. Apply filters (Strategy, Sector, etc.)
4. View candidate profiles
5. Check Insights dashboard
6. Export results to CSV

## Environment Variables (Optional)

You can customize in `.env`:

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4-turbo-preview
```

Available models:
- `gpt-4-turbo-preview` (default, most accurate)
- `gpt-3.5-turbo` (faster, cheaper)
- `gpt-4` (highest quality)
