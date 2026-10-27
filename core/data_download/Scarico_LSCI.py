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

YEARS = [2016,2017,2018,2019, 2020, 2021]

CHUNK_SIZE = 20

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
        print(f"  [API {r.status_code}] {r.text[:500]}")
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

def download_lsci():
    """Download quarterly LSCI and aggregate to annual."""
    
    print("\n" + "=" * 60)
    print("Downloading LSCI (Liner Shipping Connectivity Index)")
    print("=" * 60)

    all_dfs = []

    # Build quarters list
    quarters = []
    for year in YEARS:
        for q in [1, 2, 3, 4]:
            quarters.append(f"{year}Q{q:02d}")

    print(f"Quarters: {quarters}")

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
        quarter_filter = "','".join(quarters)

        # Build the exact filter as in R code
        filter_str = (
            f"Economy/Code in ('{country_filter}') and "
            f"Quarter/Code in ('{quarter_filter}')"
        )

        df = call_api(
            "US.LSCI",
            {
                "$select": 
                    "Economy/Label, Quarter/Label, Index_Average_Q1_2023__100_Value, "
                    "Index_Average_Q1_2023__100_Footnote, Index_Average_Q1_2023__100_MissingValue",
                
                "$filter": filter_str,
                
                "$orderby": "Economy/Order asc, Quarter/Order asc",
                
                "$compute": 
                    "round(M6047/Value div 1, 2) as Index_Average_Q1_2023__100_Value, "
                    "M6047/Footnote/Text as Index_Average_Q1_2023__100_Footnote, "
                    "M6047/MissingValue/Label as Index_Average_Q1_2023__100_MissingValue"
            }
        )

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data retrieved!")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n✓ Downloaded {len(df):,} total rows")
    print(f"  Columns: {df.columns.tolist()}")

    # Show sample
    print("\nSample data:")
    print(df.head(10))

    # Rename columns for clarity
    df = df.rename(columns={
        "Economy_Label": "Country",
        "Quarter_Label": "Quarter",
        "Index_Average_Q1_2023__100_Value": "LSCI"
    })

    # Extract year from Quarter (e.g., "2018 - Q1" → 2018)
    # The quarter format might be different, let's handle multiple formats
    if "Quarter" in df.columns:
        # Try to extract year (usually first 4 digits)
        df["Year"] = df["Quarter"].str.extract(r'(\d{4})')[0].astype(int)
    
    print("\n" + "=" * 60)
    print("Aggregating quarterly to annual...")
    print("=" * 60)
    
    # Aggregate to annual (average across quarters)
    df_annual = (
        df.groupby(["Country", "Year"], as_index=False)
        ["LSCI"]
        .mean()
    )
    
    print(f"  ✓ Aggregated to {len(df_annual):,} country-year observations")

    # Sort by country and year
    df_annual = df_annual.sort_values(
        ["Country", "Year"]
    ).reset_index(drop=True)

    # Round LSCI to 2 decimal places
    df_annual["LSCI"] = df_annual["LSCI"].round(2)

    return df_annual

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("UNCTAD LSCI Downloader")
    print(f"Countries: {len(COUNTRY_CODES)}")
    print(f"Years: {YEARS}")
    print("=" * 60)

    df = download_lsci()

    print("\n" + "=" * 60)
    print("Saving...")
    print("=" * 60)

    save_csv(df, "lsci_annual.csv")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    if not df.empty:
        print(f"\nRows: {len(df):,}")
        print(f"Columns: {list(df.columns)}\n")
        print("Sample data:")
        print(df.head(20))
        
        # Show statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Countries: {df['Country'].nunique()}")
        print(f"Years: {sorted(df['Year'].unique())}")
        
        print(f"\nTop 15 countries by LSCI (2021):")
        df_2021 = df[df["Year"] == 2021].nlargest(15, "LSCI")
        for idx, (_, row) in enumerate(df_2021.iterrows(), 1):
            print(f"  {idx:2d}. {row['Country']:<35} {row['LSCI']:>8.2f}")
            
        print(f"\nLSCI statistics for 2021:")
        lsci_2021 = df[df["Year"] == 2021]["LSCI"]
        print(f"  Mean:   {lsci_2021.mean():.2f}")
        print(f"  Median: {lsci_2021.median():.2f}")
        print(f"  Min:    {lsci_2021.min():.2f}")
        print(f"  Max:    {lsci_2021.max():.2f}")
    else:
        print("No data downloaded.")

    print("=" * 60)
