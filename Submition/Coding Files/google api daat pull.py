# pip install requests pandas python-dateutil
import os, math, time, sys
import requests
import pandas as pd
from datetime import date, timedelta

# ========= CONFIG =========
API_KEY = os.getenv("GNEWS_API_KEY") or "af5045767ab93a8533d0ff02deeccefc"
OUT_CSV = "tesla_news_2024-09-30_to_2025-08-13.csv"

# Date window (inclusive)
START_DAY = date(2025, 6, 30)
END_DAY   = date(2025, 8, 13)

# Per-day fetch targets
TARGET_ARTICLES_PER_DAY = 100     # <= adjust to 75 if you want to stay under 1000/day for 327 days
MAX_PER_REQUEST         = 25      # Essential plan cap (Business=50, Enterprise=100)  # docs: https://gnews.io/pricing
SLEEP_BETWEEN_REQUESTS  = 0.25

# GNews search params
LANG = "en"
SORTBY = "publishedAt"            # or 'relevance'
COUNTRY = None                    # e.g., 'us'
IN_FIELDS = "title,description,content"  # docs: 'in' can be title,description,content
INCLUDE_CONTENT = True            # paid plans only: expand=content
BASE_URL = "https://gnews.io/api/v4/search"

# Query length guard — GNews throws 400 if q is too long; we keep a safety margin
MAX_Q_CHARS = 190

# ========= QUERY INGREDIENTS (will be automatically split to fit MAX_Q_CHARS) =========
BASE_OR = ['Tesla', 'TSLA', '"Elon Musk"', 'trsla']  # include common misspelling

PRODUCT_TERMS = ['Cybertruck', '"Model 3"', '"Model Y"', '"Model S"', '"Model X"', 'Roadster']
FACTORY_TERMS = ['Gigafactory', '"Giga Texas"', '"Giga Berlin"', '"Giga Shanghai"', '"Giga Nevada"']
TECH_TERMS    = ['Supercharger', 'FSD', '"Full Self-Driving"', 'Optimus']
INVEST_TERMS  = ['investment', 'invest', 'shares', 'stock', 'stake']
GOV_TERMS     = ['government', '"White House"', 'Congress', 'Senate', 'regulator', 'SEC', 'FTC', 'EU', 'summit', 'hearing', 'meeting']

# ========= HELPERS =========
def make_or(terms):
    return " OR ".join(terms)

def chunk_group(base_left, group_terms, max_len):
    """
    Build one or more queries of the form: (base_left) AND (t1 OR t2 OR ...),
    chunking group_terms so each final q <= max_len characters.
    """
    chunks, cur = [], []
    while group_terms:
        # try adding next term
        nxt = group_terms[0]
        trial_terms = cur + [nxt]
        q = f"({base_left}) AND ({make_or(trial_terms)})"
        if len(q) <= max_len:
            cur.append(group_terms.pop(0))
        else:
            if not cur:
                # single term too long (very unlikely) -> force add and move on
                chunks.append(f"({base_left}) AND ({nxt})")
                group_terms.pop(0)
            else:
                chunks.append(f"({base_left}) AND ({make_or(cur)})")
                cur = []
    if cur:
        chunks.append(f"({base_left}) AND ({make_or(cur)})")
    return chunks

def build_subqueries():
    subqs = []
    # Core broad query kept short
    core = make_or(BASE_OR)
    if len(core) > MAX_Q_CHARS:
        # fall back to minimal base if somehow too long
        core = 'Tesla OR "Elon Musk"'
    subqs.append(core)

    # Attach additional themed chunks with 'Tesla' on the left side
    base_left = "Tesla"
    for grp in (PRODUCT_TERMS[:], FACTORY_TERMS[:], TECH_TERMS[:], INVEST_TERMS[:], GOV_TERMS[:]):
        subqs.extend(chunk_group(base_left, grp, MAX_Q_CHARS))
    return subqs

def iso_day_bounds(d: date):
    return f"{d.isoformat()}T00:00:00Z", f"{d.isoformat()}T23:59:59Z"

def fetch_one(query, day, page, max_per_req):
    from_iso, to_iso = iso_day_bounds(day)
    params = {
        "q": query,
        "lang": LANG,
        "max": max_per_req,
        "from": from_iso,
        "to": to_iso,
        "sortby": SORTBY,
        "in": IN_FIELDS,
        "apikey": API_KEY,
        "page": page,
    }
    if COUNTRY:
        params["country"] = COUNTRY
    if INCLUDE_CONTENT:
        params["expand"] = "content"

    r = requests.get(BASE_URL, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(2.0)
        r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    arts = data.get("articles", []) or []
    rows = []
    for a in arts:
        src = a.get("source") or {}
        rows.append({
            "day": day.isoformat(),
            "id": a.get("id"),
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "image": a.get("image"),
            "publishedAt": a.get("publishedAt"),
            "source_name": src.get("name"),
            "source_url": src.get("url"),
            "q_used": query,
            "page": page,
        })
    return rows

def estimate_requests(num_days, target_per_day, max_per_req, num_subqueries):
    # We'll allocate requests (pages) across subqueries round-robin.
    # Total pages for the day is ceil(target/max_per_req).
    pages_per_day = math.ceil(target_per_day / max_per_req)
    return num_days * pages_per_day

# ========= MAIN =========
def main():
    if not API_KEY or API_KEY == "YOUR_GNEWS_API_KEY":
        print("❌ Please set GNEWS_API_KEY in your environment.")
        sys.exit(1)

    subqueries = build_subqueries()
    # Keep only the first N subqueries if you need to limit breadth
    # subqueries = subqueries[:5]

    # Show query lengths (helps diagnose 400/q-too-long)
    print("Subqueries and lengths:")
    for i, q in enumerate(subqueries, 1):
        print(f"  Q{i} ({len(q)}): {q}")

    total_days = (END_DAY - START_DAY).days + 1
    est_reqs = estimate_requests(total_days, TARGET_ARTICLES_PER_DAY, MAX_PER_REQUEST, len(subqueries))
    print(f"\nPlan: {total_days} days, target {TARGET_ARTICLES_PER_DAY}/day, max/request={MAX_PER_REQUEST} → ~{est_reqs} requests total.")
    if est_reqs > 1000:
        print("⚠️ This exceeds 1,000 requests/day if run in one calendar day. "
              "Lower TARGET_ARTICLES_PER_DAY (e.g., 75) or split across two days.")

    all_rows = []
    cur = START_DAY
    while cur <= END_DAY:
        try:
            # total pages we need for this day
            pages_needed = math.ceil(TARGET_ARTICLES_PER_DAY / MAX_PER_REQUEST)
            # round-robin across subqueries, incrementing page per subquery when reused
            page_counters = {q: 0 for q in subqueries}
            pulls = 0
            while pulls < pages_needed:
                for q in subqueries:
                    if pulls >= pages_needed:
                        break
                    page_counters[q] += 1
                    rows = fetch_one(q, cur, page_counters[q], MAX_PER_REQUEST)
                    print(f"{cur} | Q{list(page_counters).index(q)+1} page {page_counters[q]} → {len(rows)}")
                    all_rows.extend(rows)
                    pulls += 1
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
        except requests.HTTPError as e:
            # GNews returns structured error messages (docs show 4xx behavior)
            # Example you saw: {"errors":{"q":"The query is too long (maximum 200 characters)."}}
            # We'll just log and continue to the next day.
            print(f"{cur}: HTTPError {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            print(f"{cur}: error {e}")
        cur += timedelta(days=1)

    # Deduplicate and sort
    df = pd.DataFrame(all_rows)
    if not df.empty:
        if df["id"].notna().any():
            df = df.drop_duplicates(subset=["id"])
        else:
            df = df.drop_duplicates(subset=["url"])
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
        df = df.sort_values(["day", "publishedAt"], ascending=[True, False])

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
