import os
import requests
import pandas as pd
import io
import gzip
from typing import Dict, List

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

YEARS = [2018, 2019, 2020, 2021]

CHUNK_SIZE = 50

# Ship type mapping for cleaner column names
SHIP_TYPE_NAMES = {
    "All ships": "Total_Arrivals",
    "Passenger ships": "Passenger_Ships",
    "Liquid bulk carriers": "Liquid_Bulk",
    "Container ships": "Container_Ships",
    "Dry breakbulk carriers": "Dry_Breakbulk",
    "Dry bulk carriers": "Dry_Bulk",
    "Roll-on/ roll-off ships": "RoRo_Ships",
    "Liquefied petroleum gas carriers": "LPG_Carriers",
    "Liquefied natural gas carriers": "LNG_Carriers"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def chunk_list(lst, size):
    """Split list into chunks of specified size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def call_api(dataset: str, form_data: Dict) -> pd.DataFrame:
    """Make API call and return DataFrame."""
    url = f"https://unctadstat-user-api.unctad.org/{dataset}/cur/Facts?culture=en"

    headers = {
        "ClientId": CLIENT_ID,
        "ClientSecret": CLIENT_SECRET
    }

    form_data.update({
        "$format": "csv",
        "compress": "gz"
    })

    try:
        r = requests.post(
            url,
            data=form_data,
            headers=headers,
            timeout=60
        )
    except Exception as e:
        print(f"  [Request error] {e}")
        return pd.DataFrame()

    if r.status_code != 200:
        print(f"  [API {r.status_code}] {r.text[:300]}")
        return pd.DataFrame()

    try:
        with gzip.open(
            io.BytesIO(r.content),
            "rt",
            encoding="utf-8"
        ) as f:
            df = pd.read_csv(
                f,
                na_values=[""],
                encoding="utf-8"
            )
            if not df.empty:
                print(f"    → Received {len(df):,} rows")
            return df
    except Exception as e:
        print(f"  [Parse error] {e}")
        return pd.DataFrame()


def probe_dataset(dataset: str):
    """Probe dataset to discover column structure."""
    print("Probing dataset...")

    url = f"https://unctadstat-user-api.unctad.org/{dataset}/cur/Facts?culture=en"

    headers = {
        "ClientId": CLIENT_ID,
        "ClientSecret": CLIENT_SECRET
    }

    r = requests.post(
        url,
        data={
            "$top": "10",
            "$format": "csv",
            "compress": "gz"
        },
        headers=headers,
        timeout=30
    )

    if r.status_code != 200:
        print(f"Probe failed: {r.status_code}")
        return [], pd.DataFrame()

    with gzip.open(
        io.BytesIO(r.content),
        "rt",
        encoding="utf-8"
    ) as f:
        df = pd.read_csv(f)

    cols = df.columns.tolist()

    print("\nColumns found:")
    for c in cols:
        print("  ", c)
    
    print("\nSample data:")
    print(df.head())

    return cols, df


def save_csv(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV."""
    if df.empty:
        print("Nothing to save.")
        return

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Saved {filename} "
        f"({len(df):,} rows × {df.shape[1]} cols)"
    )

# =============================================================================
# MAIN DOWNLOADER
# =============================================================================

def download_port_calls():
    """Download port calls data with ship types as columns."""
    
    print("\n" + "=" * 60)
    print("Downloading Port Calls")
    print("=" * 60)

    cols, sample = probe_dataset("US.PortCallsArrivals_S")

    # Find the value column
    value_col = next(
        (c for c in cols if c.endswith("_Value")),
        None
    )

    if value_col is None:
        print("Could not find value column.")
        return pd.DataFrame()

    measure_code = value_col.replace("_Value", "")

    print(f"\nMeasure detected: {measure_code}")

    # Check if CommercialMarket dimension exists
    has_ship_type = any("CommercialMarket" in c for c in cols)
    
    if has_ship_type:
        print("✓ CommercialMarket dimension found - will download by ship type")
    else:
        print("⚠ CommercialMarket dimension not found - downloading aggregate only")

    all_dfs = []
    year_filter = ",".join(map(str, YEARS))
    chunks = list(chunk_list(COUNTRY_CODES, CHUNK_SIZE))

    print(
        f"\nDownloading {len(COUNTRY_CODES)} countries "
        f"in {len(chunks)} chunks..."
    )

    for i, countries in enumerate(chunks, start=1):
        print(
            f"\nChunk {i}/{len(chunks)} "
            f"({len(countries)} countries)"
        )

        country_filter = "','".join(countries)

        # Build select statement
        if has_ship_type:
            select_str = f"Economy/Label,Year,CommercialMarket/Label,{measure_code}"
        else:
            select_str = f"Economy/Label,Year,{measure_code}"

        df = call_api(
            "US.PortCallsArrivals_S",
            {
                "$select": select_str,
                "$filter":
                    f"Economy/Code in ('{country_filter}') "
                    f"and Year in ({year_filter})",
                "$orderby":
                    "Economy/Order asc, Year asc"
            }
        )

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n✓ Downloaded {len(df):,} total rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Rename columns
    rename_dict = {
        "Economy_Label": "Country",
        "Year": "Year",
        f"{measure_code}_Value": "Arrivals"
    }
    
    if has_ship_type:
        rename_dict["CommercialMarket_Label"] = "ShipType"
    
    df = df.rename(columns=rename_dict)

    # Check if we have ship type data
    if "ShipType" in df.columns:
        print("\n" + "=" * 60)
        print("Processing ship types...")
        print("=" * 60)
        
        # Show unique ship types
        ship_types = df["ShipType"].unique()
        print(f"\nFound {len(ship_types)} ship types:")
        for st in sorted(ship_types):
            count = len(df[df["ShipType"] == st])
            print(f"  • {st}: {count:,} records")
        
        # Aggregate to annual level (sum semi-annual data)
        print("\n→ Aggregating to annual totals...")
        df_agg = (
            df.groupby(["Country", "Year", "ShipType"], as_index=False)
            ["Arrivals"]
            .sum()
        )
        
        print(f"  After aggregation: {len(df_agg):,} rows")
        
        # Pivot to wide format
        print("\n→ Pivoting ship types to columns...")
        df_pivot = df_agg.pivot_table(
            index=["Country", "Year"],
            columns="ShipType",
            values="Arrivals",
            fill_value=0
        ).reset_index()
        
        # Flatten column names and rename using mapping
        df_pivot.columns.name = None
        
        # Rename ship type columns to cleaner names
        col_rename = {"Country": "Country", "Year": "Year"}
        for col in df_pivot.columns:
            if col not in ["Country", "Year"]:
                col_rename[col] = SHIP_TYPE_NAMES.get(col, col.replace(" ", "_"))
        
        df_pivot = df_pivot.rename(columns=col_rename)
        
        # Reorder columns: Country, Year, Total, then others alphabetically
        base_cols = ["Country", "Year"]
        ship_cols = [c for c in df_pivot.columns if c not in base_cols]
        
        # Put Total_Arrivals first if it exists
        if "Total_Arrivals" in ship_cols:
            ship_cols.remove("Total_Arrivals")
            ship_cols = ["Total_Arrivals"] + sorted(ship_cols)
        else:
            ship_cols = sorted(ship_cols)
        
        df_final = df_pivot[base_cols + ship_cols]
        
        print(f"\n✓ Final shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
        print(f"  Ship type columns: {ship_cols}")
        
    else:
        # No ship type dimension - just aggregate by country and year
        print("\n→ Aggregating to annual totals...")
        df_final = (
            df.groupby(["Country", "Year"], as_index=False)
            ["Arrivals"]
            .sum()
        )
    
    # Sort by country and year
    df_final = df_final.sort_values(
        ["Country", "Year"]
    ).reset_index(drop=True)

    return df_final

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNCTAD Port Calls Downloader")
    print(f"Countries: {len(COUNTRY_CODES)}")
    print(f"Years: {YEARS}")
    print("=" * 60)

    df = download_port_calls()

    print("\n" + "=" * 60)
    print("Saving...")
    print("=" * 60)

    save_csv(df, "port_calls_by_ship_type.csv")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    if not df.empty:
        print(f"\nRows: {len(df):,}")
        print(f"Columns: {list(df.columns)}\n")
        print("Sample data:")
        print(df.head(10))
        
        # Show statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Countries: {df['Country'].nunique()}")
        print(f"Years: {sorted(df['Year'].unique())}")
        
        if "Total_Arrivals" in df.columns:
            print(f"\nTop 5 countries by total arrivals (2021):")
            df_2021 = df[df["Year"] == 2021].nlargest(5, "Total_Arrivals")
            for _, row in df_2021.iterrows():
                print(f"  {row['Country']}: {row['Total_Arrivals']:,.0f}")
    else:
        print("No data downloaded.")

    print("=" * 60)
