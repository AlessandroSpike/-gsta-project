import os
import requests
import pandas as pd
import io
import gzip
import time
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

YEARS      = [2016,2017,2018,2019, 2020, 2021]
CHUNK_SIZE = 20

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


def chunked_bilateral_loop(
    dataset:       str,
    dim1:          str,
    dim2:          str,
    select_cols:   str,
    extra_filters: str = "",
    time_filter:   str = "",
) -> pd.DataFrame:
    """Download bilateral data in chunks."""
    all_dfs     = []
    countries   = COUNTRY_CODES
    total       = len(countries)
    n_chunks    = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    first_chunk = True

    for i in range(0, total, CHUNK_SIZE):
        chunk1  = countries[i:i+CHUNK_SIZE]
        filter1 = "','".join(chunk1)
        ci      = i // CHUNK_SIZE + 1

        for j in range(0, total, CHUNK_SIZE):
            chunk2  = countries[j:j+CHUNK_SIZE]
            filter2 = "','".join(chunk2)
            cj      = j // CHUNK_SIZE + 1

            print(f"  {dim1} chunk {ci}/{n_chunks}  |  {dim2} chunk {cj}/{n_chunks}", end="")

            filter_str = (
                f"{dim1}/Code in ('{filter1}') and "
                f"{dim2}/Code in ('{filter2}')"
            )
            if time_filter:
                filter_str += f" and {time_filter}"
            if extra_filters:
                filter_str += f" and {extra_filters}"

            df_chunk = call_api(dataset, {
                "$select":  f"{dim1}/Label, {dim2}/Label, {select_cols}",
                "$filter":  filter_str,
                "$orderby": f"{dim1}/Order asc, {dim2}/Order asc",
            })

            if df_chunk.empty:
                print("  (empty)")
                continue
            else:
                print()

            if first_chunk:
                print(f"    First chunk columns: {df_chunk.columns.tolist()}")
                first_chunk = False

            all_dfs.append(df_chunk)
            time.sleep(0.5)

    if not all_dfs:
        print(f"  No data returned for {dataset}.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# TRANSPORT COSTS DOWNLOADER
# =============================================================================

def download_transport_costs() -> pd.DataFrame:
    """Download bilateral transport costs data."""
    print("\n" + "="*60)
    print("Downloading Transport Costs (bilateral)...")
    print("="*60)

    cols, _ = probe_with_fallback("US.TransportCosts", [
        "Year eq 2021 and Origin/Code eq '840' and Destination/Code eq '826'",
    ])
    if not cols:
        print("  Cannot probe TransportCosts — aborting.")
        return pd.DataFrame()

    rate_col = (pick_col(cols, "perunit")
                or pick_col(cols, "freight")
                or pick_col(cols, "m1980")
                or pick_col(cols, "value"))
    if not rate_col:
        print(f"  No freight-rate column found in: {cols}")
        return pd.DataFrame()
    print(f"  Using freight-rate column: '{rate_col}'")

    year_filter = ",".join(map(str, YEARS))
    df = chunked_bilateral_loop(
        dataset       = "US.TransportCosts",
        dim1          = "Origin",
        dim2          = "Destination",
        select_cols   = f"Year, {measure_code(rate_col)}",
        time_filter   = f"Year in ({year_filter})",
        extra_filters = "TransportMode/Code eq '00' and Product/Code eq 'TOTAL'",
    )
    if df.empty:
        return df

    cols       = df.columns.tolist()
    origin_col = pick_col(cols, "origin", "label")
    dest_col   = pick_col(cols, "destination", "label")
    year_col   = pick_col(cols, "year")
    found_rate = pick_col(cols, rate_col.split("/")[-1].lower()[:6])
    if not found_rate:
        found_rate = next(
            (c for c in cols
             if c not in [origin_col, dest_col, year_col]
             and pd.api.types.is_numeric_dtype(df[c])),
            None
        )
    df = df.rename(columns={
        origin_col: "Origin",
        dest_col:   "Destination",
        year_col:   "Year",
        found_rate: "FreightRate_USD_per_kg",
    })
    return df[["Origin", "Destination", "Year", "FreightRate_USD_per_kg"]]


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
    print("UNCTAD Transport Costs Downloader")
    print(f"  Countries : {len(COUNTRY_CODES)}")
    print(f"  Years     : {YEARS}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print("=" * 60)

    df = download_transport_costs()
    
    print("\n" + "=" * 60)
    print("Saving file...")
    print("=" * 60)
    save_csv(df, "transport_costs_bilateral")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    if not df.empty:
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print("  No data retrieved")
    print("=" * 60)
