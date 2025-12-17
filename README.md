# AIED Bluesky Analysis

Collect, clean, and analyze Bluesky posts about AI in educational contexts.

## Requirements
- Python 3.11+ recommended
- A Bluesky account with an app password (Settings → App Passwords)

## Install
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Configure environment (`.env`)
Create a `.env` file in the project root with at least:

```bash
BLUESKY_HANDLE=your.handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

Optional configuration:
- `START_DATE` / `END_DATE` (ISO dates; default start is `2024-01-01`, default end is “now”)
- `AI_TERMS` / `EDU_TERMS` (comma-separated term lists; defaults are defined in `src/get_data.py`)

Example:

```bash
START_DATE=2024-01-01
END_DATE=2025-12-14
AI_TERMS=chatgpt,gpt,gemini,claude,llm,openai,anthropic,ai,"gen ai"
EDU_TERMS=student,teacher,faculty,professor,admin,principal,superintendent,school,classroom,syllabus,assignment,grading
```

## Get data
This project retrieves posts via the official Bluesky XRPC API (`app.bsky.feed.searchPosts`) and writes newline-delimited JSON plus a manifest.

```bash
python3 src/get_data.py
```

Outputs:
- Raw NDJSON: `data/raw/bluesky_posts.json`
- Manifest (run settings + counts): `data/raw/manifest.json`

## Clean data
Cleaning removes duplicates, filters to English, enforces a minimum length, removes duplicate text, filters likely ads/spam, produces a standardized `text_clean` field, and adds education-context tags.

```bash
python3 src/clean_data.py
```

Output:
- Cleaned CSV: `data/processed/cleaned_posts.csv`

## Run analysis
`src/run_analysis.py` prints baseline dataset checks, context bucket prevalence, and top tf-idf terms/phrases.

```bash
python3 src/run_analysis.py --top-n 15
```

## Utilities
Two small helper scripts support reproducibility and qualitative validation:

- Check configuration and expected files:
  ```bash
  python3 src/config_check.py
  ```
  Use `--mode env` or `--mode paths` to run a subset of checks.

- Sample posts for manual inspection (default: stratified by context bucket):
  ```bash
  python3 src/sample_posts.py
  ```
  This writes `results/samples.csv`. Use `--mode random` to sample uniformly.

## Visualizations
Use the notebooks to reproduce plots and exploratory analyses (context prevalence/co-occurrence, temporal trends, engagement, sentiment, and topic modeling).

```bash
jupyter lab
```
## Restoring large data files

Due to GitHub file size limits, large data files are stored in compressed (`.zip`) form.

After cloning the repository, please **unzip the following files and replace the extracted files in the same directory paths**:

```bash
unzip data/processed/cleaned_posts.csv.zip -d data/processed/
unzip data/raw/bluesky_posts.json.zip -d data/raw/

Then open and run:
- `results/visualize_results.ipynb`
