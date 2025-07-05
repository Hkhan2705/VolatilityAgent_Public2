def plot_volatility_analysis(ticker):
    """
    Reads a single stock's Parquet file from the local cache and generates
    the 5-panel volatility plot.
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
        plot_data = pd.DataFrame()  # Start with an empty DataFrame

        # --- THIS IS THE DEFINITIVE, ROBUST LOGIC FOR YTD ---
        if period == 'YTD':
            current_year = datetime.now().year
            # Create a boolean mask for the current year
            mask = vol_data.index.year == current_year
            plot_data = vol_data[mask]
        # --- END OF CORRECTED LOGIC ---
        else:
            # The logic for other timeframes was also flawed, let's fix it too.
            try:
                offset = pd.tseries.frequencies.to_offset(period)
                start_date = vol_data.index.max() - offset
                plot_data = vol_data[vol_data.index >= start_date]
            except Exception:
                # If there's an error (e.g., not enough data), plot_data remains empty
                pass

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