from sec_edgar_api import EdgarClient
import pandas as pd
from datetime import datetime

from sec_edgar_api import EdgarClient
import pandas as pd
from datetime import datetime 
from portfolio_utils import top_100 

def fetch_financial_data(cik_dict, variables, start_date=None, end_date=None, single_variable=False):
    """
    Fetch financial data from the SEC EDGAR API using a dictionary of {Ticker: CIK}.
    
    Parameters
    ----------
    cik_dict : dict
        Dictionary with keys as ticker names and values as 10-digit CIK strings.
    variables : list
        List of EDGAR tag names (e.g., ["CommonStockSharesOutstanding", ...]).
    start_date : str (YYYY-MM-DD), optional
        Filter data so that only entries on or after this date are returned.
    end_date : str (YYYY-MM-DD), optional
        Filter data so that only entries on or before this date are returned.
    single_variable : bool, default=False
        - If False, returns all variables separately.
        - If True, picks the first available variable from `variables` and pivots data.

    Returns
    -------
    pandas.DataFrame
        - If `single_variable=True`: returns a wide DataFrame with tickers as columns.
        - Otherwise, returns a standard DataFrame with all available variables.
    """

    # Initialize EdgarClient
    edgar = EdgarClient(user_agent="YourCompany your-email@example.com")

    # Prepare container for all data
    all_data = []

    for ticker, cik in cik_dict.items():
        # Fetch company facts for the given CIK
        try:
            company_facts = edgar.get_company_facts(cik=cik)
        except Exception as e:
            print(f"Could not retrieve data for {ticker} (CIK={cik}). Error: {e}")
            continue

        if "facts" not in company_facts:
            continue  # No financial data available

        # Iterate over requested variables
        for tag in variables:
            for namespace_key in company_facts['facts']:
                namespace = company_facts['facts'][namespace_key]
                if tag not in namespace:
                    continue  # This namespace doesn't have the requested tag

                metric_data = namespace[tag]

                # Process units
                for unit, entries in metric_data.get("units", {}).items():
                    for entry in entries:
                        entry_date = entry["end"]

                        # Apply date filters
                        if start_date and datetime.strptime(entry_date, "%Y-%m-%d") < datetime.strptime(start_date, "%Y-%m-%d"):
                            continue
                        if end_date and datetime.strptime(entry_date, "%Y-%m-%d") > datetime.strptime(end_date, "%Y-%m-%d"):
                            continue

                        all_data.append({
                            "ticker": ticker,  # Use ticker instead of entity name
                            "cik": cik,  
                            "metric": tag,
                            "date": entry_date,
                            "value": entry["val"],
                            "frame": entry.get("frame"),
                            "form": entry.get("form"),
                            "unit": unit
                        })

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data found for the given parameters.")
        return df

    # Convert date column to datetime and sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(by=["ticker", "metric", "date"], ignore_index=True)

    # Convert to quarterly periods
    df["quarter"] = df["date"].dt.to_period("Q")

    # Aggregate (last value per quarter per ticker)
    aggregated_df = (
        df.groupby(["ticker", "metric", "quarter"], as_index=False)
          .agg({
               "date": "max",     # Use last date of quarter
               "value": "last",   # Use last reported value
               "frame": "last",
               "form": "last",
               "unit": "last"
          })
    )

    # -------------------------------
    # 1) Normal retrieval (single_variable=False)
    # -------------------------------
    if not single_variable:
        # Convert quarter to datetime for long-form DataFrame
        aggregated_df["date"] = aggregated_df["quarter"].dt.start_time  
        aggregated_df = aggregated_df.set_index("date")  # Set datetime index
        return aggregated_df

    # -------------------------------
    # 2) Single-variable retrieval (single_variable=True)
    # -------------------------------
    else:
        pivoted = (
            aggregated_df
            .pivot_table(
                index=["ticker", "quarter"],
                columns="metric",
                values="value",
                aggfunc="first"
            )
            .reset_index()
        )

        def pick_first_non_null(row):
            for col in variables:  # Prioritize variables in given order
                if pd.notnull(row.get(col)):
                    return row[col]
            return pd.NA

        pivoted["combined_value"] = pivoted.apply(pick_first_non_null, axis=1)

        # Pivot to have tickers as columns
        pivoted_wide = pivoted.pivot(
            index="quarter",
            columns="ticker",
            values="combined_value"
        )
        pivoted_wide.index = pivoted_wide.index.to_timestamp()  # Convert PeriodIndex to DatetimeIndex
        
        return pivoted_wide

