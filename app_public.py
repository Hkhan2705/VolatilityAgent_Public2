import streamlit as st
import pandas as pd
import os
from datetime import datetime
import numpy as np

# --- THIS IS THE CRITICAL FIX FOR SERVER-SIDE PLOTTING ---
import matplotlib

matplotlib.use("Agg")  # Use the non-interactive 'Agg' backend
import matplotlib.pyplot as plt

# --- END OF FIX ---


DATA_DIR = "local_data_store"


@st.cache_data(show_spinner="Loading volatility data from cache...")
def build_df_from_local_cache():
    # This function is correct and remains unchanged
    results = []
    if not os.path.exists(DATA_DIR):
        return pd.DataFrame()
    parquet_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    for filename in parquet_files:
        ticker = os.path.splitext(filename)[0]
        file_path = os.path.join(DATA_DIR, filename)
        try:
            vol_data = pd.read_parquet(file_path)
            offset = pd.tseries.frequencies.to_offset("1YE")
            df_1y = vol_data[vol_data.index >= vol_data.index.max() - offset]
            if df_1y.empty or 'IV_30D' not in df_1y.columns or 'HV_30D' not in df_1y.columns or len(df_1y) < 20:
                continue
            iv_series = df_1y['IV_30D']
            current_iv = iv_series.iloc[-1]
            iv_low_52wk = iv_series.min()
            iv_high_52wk = iv_series.max()
            iv_rank = (current_iv - iv_low_52wk) / (iv_high_52wk - iv_low_52wk) if (
                                                                                               iv_high_52wk - iv_low_52wk) > 0 else np.nan
            current_hv = df_1y['HV_30D'].iloc[-1]
            iv_hv_ratio = current_iv / current_hv if not (np.isnan(current_hv) or current_hv == 0) else np.nan
            if pd.isna(iv_rank) or pd.isna(iv_hv_ratio):
                continue
            results.append({
                "Ticker": ticker, "Current IV": current_iv,
                "IV Rank (1Y)": iv_rank, "IV/HV Ratio": iv_hv_ratio
            })
        except Exception:
            continue
    return pd.DataFrame(results) if results else pd.DataFrame()


def format_df_for_display(df):
    # This function is correct and remains unchanged
    if df.empty: return df
    df_display = df.copy()
    if 'Current IV' in df_display.columns:
        df_display['Current IV'] = df_display['Current IV'].map('{:.2%}'.format)
    if 'IV Rank (1Y)' in df_display.columns:
        df_display['IV Rank (1Y)'] = (df_display['IV Rank (1Y)'] * 100).map('{:.1f}'.format)
    if 'IV/HV Ratio' in df_display.columns:
        df_display['IV/HV Ratio'] = df_display['IV/HV Ratio'].map('{:.2f}'.format)
    return df_display


def plot_volatility_analysis(ticker):
    """
    Reads a single stock's Parquet file and generates the plot
    using a server-safe Matplotlib backend.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        st.error(f"Data file not found for {ticker} in the local cache.")
        return None

    vol_data = pd.read_parquet(file_path)

    timeframes = {"5 Years": "5Y", "1 Year": "1Y", "6 Months": "6M", "YTD": "YTD", "1 Month": "1M"}
    fig, axes = plt.subplots(len(timeframes), 1, figsize=(12, 18))
    fig.suptitle(f'Historical vs. Implied Volatility for {ticker}', fontsize=16, y=0.99)

    for i, (name, period) in enumerate(timeframes.items()):
        plot_data = pd.DataFrame()

        try:
            if period == 'YTD':
                current_year = datetime.now().year
                mask = vol_data.index.year == current_year
                plot_data = vol_data[mask]
            else:
                offset = pd.tseries.frequencies.to_offset(period)
                start_date = vol_data.index.max() - offset
                plot_data = vol_data[vol_data.index >= start_date]
        except Exception:
            plot_data = pd.DataFrame()  # Ensure it's empty on any error

        ax = axes[i]

        if not plot_data.empty:
            ax.plot(plot_data.index, plot_data['HV_30D'], label='30-Day Historical Vol (HV)', color='royalblue')
            if 'IV_30D' in plot_data.columns and not plot_data['IV_30D'].isnull().all():
                ax.plot(plot_data.index, plot_data['IV_30D'], label='30-Day Implied Vol (IV)', color='red')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Data Available for this Timeframe', horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)

        ax.set_title(name)
        ax.set_ylabel("Annualized Volatility")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# --- Main Application UI ---
st.set_page_config(layout="wide")
st.title("📊 S&P 500 Volatility Screener (Snapshot)")

try:
    sample_file_path = os.path.join(DATA_DIR, "AAPL.parquet")
    if os.path.exists(sample_file_path):
        sample_df = pd.read_parquet(sample_file_path)
        last_data_date = sample_df.index.max()
        last_updated_str = last_data_date.strftime('%Y-%m-%d')
        st.write(f"This application displays a snapshot of volatility data. **Data as of: {last_updated_str}**")
    else:
        st.write("This application displays a snapshot of volatility data. (Data files not found).")
except Exception:
    st.write("This application displays a snapshot of volatility data.")

full_df = build_df_from_local_cache()

if full_df.empty:
    st.error("Volatility data is not available. The application's data files may be missing from the repository.")
else:
    final_df_sorted = full_df.sort_values(by="IV Rank (1Y)", ascending=False).dropna(subset=['IV Rank (1Y)'])

    st.subheader("Volatility Screener Results (Sorted by IV Rank)")
    st.dataframe(format_df_for_display(final_df_sorted), use_container_width=True)

    st.subheader("Detailed Volatility Analysis")

    tab1, tab2 = st.tabs(["Analyze from Scan Results", "Analyze Any Ticker"])

    with tab1:
        st.write("Select a stock from the high IV Rank list above.")
        selected_ticker_from_list = st.selectbox(
            "Choose a stock:", options=final_df_sorted['Ticker'].tolist(), key="selectbox_public"
        )
        if selected_ticker_from_list and st.button(f"Generate Volatility Plot for {selected_ticker_from_list}",
                                                   key="button_selectbox_public"):
            with st.spinner(f"Loading plot data for {selected_ticker_from_list}..."):
                fig = plot_volatility_analysis(selected_ticker_from_list)
                if fig:
                    st.pyplot(fig)

    with tab2:
        st.write("Enter any valid stock ticker for a custom volatility analysis.")
        custom_ticker = st.text_input("Custom Ticker (e.g., SPY, QQQ):", "", key="text_input_public").upper()
        if custom_ticker and st.button(f"Generate Volatility Plot for {custom_ticker}", key="button_custom_public"):
            with st.spinner(f"Loading plot data for {custom_ticker}..."):
                fig = plot_volatility_analysis(custom_ticker)
                if fig:
                    st.pyplot(fig)

st.markdown("---")
st.write("Built by a Volatility Agent.")