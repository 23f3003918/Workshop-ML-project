# Fake News Detector

A text classifier that reads a news article and judges whether it is written like
a real news story or like a fabricated one. It uses Fastapi with a webpage.

**Live:** https://workshop-ml-project.onrender.com
*(replace with your actual Render URL once deployed)*

## What the model predicts:-

Given the headline and body of a news article, it returns a binary
classification plus a confidence score:

| class | meaning |
|---|---|
| `1` | **fake** — reads like fabricated or fake content |
| `0` | **real** — reads like ordinary reported news |

Trained on the **WELFake** dataset: 72,134 labelled news articles, reduced to
63,676 after removing 8,456 duplicates. The pipeline is a **TF-IDF
vectoriser** (1–2 grams, 300,000 features) feeding a **calibrated LinearSVC**.

On a held-out test set of 9,552 articles never used during training or model
selection:

| metric | value |
|---|---|
| accuracy | 96.8% |
| F1 (fake class) | 0.965 |
| ROC-AUC | 0.995 |

> **Note on labels:** in the WELFake CSV, `1` is fake and `0` is real. This is
> the reverse of the dataset's widely-copied published description. It was
> verified against the data itself: label-0 headlines average 0.10 ALLCAPS words
> and 0.002 exclamation marks, label-1 headlines average 2.01 and 0.120.

**The model judges writing style, not factual accuracy. It keys on
capitalisation, punctuation and hyperpartisan phrasing in English-language US
political news from roughly 2016–2017.**

## API

| method | path | what it does |
|---|---|---|
| `GET` | `/health` | service status and whether the model loaded |
| `POST` | `/predict` | classify an article |
| `GET` | `/` | the web page |
| `GET` | `/docs` | interactive API documentation |

### Example request

```bash
curl -X POST https://workshop-ml-project.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "SHOCKING!! Doctors STUNNED by this ONE WEIRD TRICK the government has been HIDING from you!!!", "text": "Share this before they DELETE it!! What the mainstream media REFUSES to tell you about what is really going on. Experts are SPEECHLESS."}'
```

Request body:

```json
{
  "title": "SHOCKING!! Doctors STUNNED by this ONE WEIRD TRICK the government has been HIDING from you!!!",
  "text": "Share this before they DELETE it!! What the mainstream media REFUSES to tell you about what is really going on. Experts are SPEECHLESS."
}
```

Response:

```json
{
  "prediction": 1,
  "verdict": "fake",
  "p_real": 0.0002489506744316102,
  "p_fake": 0.9997510493255684
}
```

Both `title` and `text` are strings and both are optional, defaulting to `""` —
send only one if you like, since the model joins them into a
single string before classifying. Sending both empty returns `422`.

`GET /health` returns:

```json
{"status": "ok", "model_loaded": true}
```

## How to run it locally

Requires Python 3.14 (any 3.10+ works, but 3.14 matches the interpreter the
model was pickled with).

```bash
git clone https://github.com/23f3003918/Workshop-ML-project.git
cd Workshop-ML-project
```

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000> for the page, or
<http://127.0.0.1:8000/docs> for the interactive API docs.

Check it is alive:

```bash
curl http://127.0.0.1:8000/health
```

### With Docker

```bash
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api
```

## Project structure

```
main.py                        FastAPI app
requirements.txt               dependencies
Dockerfile                     container build
welfake_tfidf_linsvc.joblib    the trained model (13 MB)
static/index.html              the web page
```

## Deployment

Deployed on **Render** as a Docker web service, building from this repository's
`Dockerfile`. The container binds to Render's `$PORT` and is not hardcoded here.

If you don't get a response on first request push, wait a min I'm on render's free tier, it may have fell asleep.

## Limitations

- Trained on US political news from around 2016–2017; accuracy on other topics,
  regions or time periods will be lower it don't know anything beyond that.
- It detects style, not truth. A carefully written fake news can read as real to
  it, and a badly written true report can read as fake.
- It is most confident about obviously sensational writing. Plain, sober gentle writing
  sits closer to the decision boundary.
