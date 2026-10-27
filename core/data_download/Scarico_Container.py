import os
import requests
import pandas as pd
import io
import gzip
from typing import Dict, List, Optional

# =============================================================================
# CREDENTIALS
# =============================================================================
CLIENT_ID = os.environ.get("UNCTAD_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("UNCTAD_CLIENT_SECRET", "")

# =============================================================================
# CONFIGURATION
# =============================================================================
COUNTRY_CODES = sorted(set([
    "008", "012", "020", "024", "031", "032", "036", "040",
    "056", "070", "076", "100", "116", "120", "140", "156",
    "170", "174", "180", "191", "196", "203", "208", "214",
    "218", "233", "246", "250", "258", "270", "276", "288",
    "300", "304", "308", "320", "324", "328", "340", "344",
    "348", "352", "356", "360", "364", "372", "376", "380",
    "384", "388", "392", "398", "400", "404", "408", "410",
    "414", "417", "418", "422", "426", "428", "430", "434",
    "440", "442", "446", "450", "454", "458", "462", "466",
    "470", "478", "480", "484", "496", "498", "499", "500",
    "504", "508", "512", "516", "524", "528", "533", "554",
    "558", "562", "566", "579", "585", "586", "591", "600",
    "604", "608", "616", "620", "624", "626", "634", "642",
    "643", "646", "659", "662", "670", "678", "682", "686",
    "688", "690", "694", "702", "703", "704", "705", "716",
    "724", "729", "740", "748", "752", "756", "757", "762",
    "764", "768", "780", "784", "788", "792", "800", "804",
    "807", "818", "826", "834", "840", "854", "858", "860",
    "862", "882", "887", "894"
]))

YEARS = [2016,2017,2018,2019, 2020, 2021]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def call_api(dataset: str, form_data: Dict) -> pd.DataFrame:
    """Make API call and return DataFrame."""
    url     = f"https://unctadstat-user-api.unctad.org/{dataset}/cur/Facts?culture=en"
    headers = {"ClientId": CLIENT_ID, "ClientSecret": CLIENT_SECRET}
    form_data.update({"$format": "csv", "compress": "gz"})

    try:
        r = requests.post(url, data=form_data, headers=headers, timeout=60)
    except Exception as e:
        print(f"  [Request error] {e}")
        return pd.DataFrame()

    if r.status_code != 200:
        print(f"  [API {r.status_code}] {r.text[:300]}")
        return pd.DataFrame()

    try:
        with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as f:
            df = pd.read_csv(f, na_values=[""], encoding="utf-8")
            if not df.empty:
                print(f"    → Received {len(df):,} rows")
            return df
    except Exception as e:
        print(f"  [Parse error] {e}")
        return pd.DataFrame()


def probe_with_fallback(dataset: str, candidate_filters: List[str]):
    """Try a series of probe filters; return (columns, sample_df) from the first that works."""
    for f in candidate_filters:
        print(f"  [Probe attempt] filter: {f or '<none>'}")
        url     = f"https://unctadstat-user-api.unctad.org/{dataset}/cur/Facts?culture=en"
        headers = {"ClientId": CLIENT_ID, "ClientSecret": CLIENT_SECRET}
        data    = {"$top": "1", "$format": "csv", "compress": "gz"}
        if f:
            data["$filter"] = f
        r = requests.post(url, data=data, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"    [Probe error {r.status_code}] {r.text[:200]}")
            continue
        try:
            with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8") as fh:
                content = fh.read()
                if not content.strip():
                    print("    [Probe] empty response.")
                    continue
                df = pd.read_csv(io.StringIO(content), nrows=1)
            cols = df.columns.tolist()
            print(f"    [Probe OK] {dataset} columns: {cols}")
            return cols, df
        except Exception as e:
            print(f"    [Probe parse error] {e}")
            continue
    return [], None


def pick_col(cols: List[str], *keywords: str) -> Optional[str]:
    """Find column matching all keywords."""
    for c in cols:
        cl = c.lower()
        if all(k.lower() in cl for k in keywords):
            return c
    return None


def measure_code(flattened_col: str) -> str:
    """Extract measure code from flattened column name."""
    name = flattened_col.split("/")[-1]
    if name.endswith("_Value"):
        name = name[: -len("_Value")]
    return name


# =============================================================================
# CONTAINER THROUGHPUT DOWNLOADER
# =============================================================================

def download_container_throughput() -> pd.DataFrame:
    """Download container throughput data."""
    print("\n" + "="*60)
    print("Downloading Container Throughput...")
    print("="*60)

    cols, _ = probe_with_fallback("US.ContPortThroughput", [""])
    if not cols:
        print("  Cannot probe ContPortThroughput — aborting.")
        return pd.DataFrame()

    val_col = (pick_col(cols, "teu")
               or pick_col(cols, "throughput")
               or pick_col(cols, "value"))
    if not val_col:
        print(f"  No TEU/throughput column found in: {cols}")
        return pd.DataFrame()
    print(f"  Using TEU column: '{val_col}'")

    country_filter = "','".join(COUNTRY_CODES)
    year_filter    = ",".join(map(str, YEARS))

    print(f"  Querying {len(COUNTRY_CODES)} countries for years {YEARS}...")
    df = call_api("US.ContPortThroughput", {
        "$select":  f"Economy/Label, Year, {measure_code(val_col)}",
        "$filter":  (
            f"Economy/Code in ('{country_filter}') and "
            f"Year in ({year_filter})"
        ),
        "$orderby": "Economy/Order asc, Year asc",
    })
    if df.empty:
        return df

    cols        = df.columns.tolist()
    country_col = pick_col(cols, "economy", "label")
    year_col    = pick_col(cols, "year")
    found_val   = pick_col(cols, val_col.split("/")[-1].lower()[:5])
    if not found_val:
        found_val = next(
            (c for c in cols if c not in [country_col, year_col]
             and pd.api.types.is_numeric_dtype(df[c])),
            None
        )
    df = df.rename(columns={
        country_col: "Country",
        year_col:    "Year",
        found_val:   "TEU",
    })
    return df[["Country", "Year", "TEU"]]


def save_csv(df: pd.DataFrame, name: str):
    """Save DataFrame to CSV."""
    if df.empty:
        print(f"  [Skip] {name} — empty DataFrame.")
        return
    path = f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  [Saved] {path}  ({len(df):,} rows × {df.shape[1]} cols)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNCTAD Container Throughput Downloader")
    print(f"  Countries : {len(COUNTRY_CODES)}")
    print(f"  Years     : {YEARS}")
    print("=" * 60)

    df = download_container_throughput()
    
    print("\n" + "=" * 60)
    print("Saving file...")
    print("=" * 60)
    save_csv(df, "container_throughput_masses")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    if not df.empty:
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print("  No data retrieved")
    print("=" * 60)
