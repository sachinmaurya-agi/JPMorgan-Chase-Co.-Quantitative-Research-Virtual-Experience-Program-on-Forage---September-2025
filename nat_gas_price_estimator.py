#!/usr/bin/env python3
\"\"\"nat_gas_price_estimator.py

Standalone script to:
- load monthly natural gas CSV from /mnt/data/Nat_Gas.csv (expected columns like Date/Dates and Price/Prices)
- build a simple additive model (linear trend + month-of-year seasonal means)
- provide `estimate_price(date)` function
- export a 12-month forecast CSV and print estimates from CLI

Usage examples:
    python nat_gas_price_estimator.py --date 2024-10-15
    python nat_gas_price_estimator.py --forecast-csv   # saves /mnt/data/nat_gas_12mo_forecast.csv
    python nat_gas_price_estimator.py --show-plots     # display plots (requires GUI or Jupyter)

Requirements (install if needed):
    pip install pandas numpy scikit-learn matplotlib python-dateutil

Note: script expects the CSV at /mnt/data/Nat_Gas.csv with monthly month-end dates between 2020-10-31 and 2024-09-30.
\"\"\"

import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dateutil import parser

# Constants
CSV_PATH = '/mnt/data/Nat_Gas.csv'
ARTIFACTS_PATH = '/mnt/data/nat_gas_model_artifacts.npz'
FORECAST_CSV = '/mnt/data/nat_gas_12mo_forecast.csv'

def load_monthly_csv(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f\"CSV not found at {csv_path}. Place the monthly snapshot there.\")
    df = pd.read_csv(csv_path)
    # Try to auto-detect date & price columns
    date_col = None
    price_col = None
    for col in df.columns:
        cl = col.lower()
        if 'date' in cl or 'time' in cl:
            date_col = col
        if 'price' in cl or 'value' in cl or 'px' in cl:
            price_col = col
    if date_col is None:
        date_col = df.columns[0]
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        price_col = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns)>1 else df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, price_col]].rename(columns={date_col: 'date', price_col: 'price'})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df

def build_model(df):
    # Ensure month-end alignment
    start = df['date'].min()
    end = df['date'].max()
    full_idx = pd.date_range(start=start, end=end, freq='M')  # month ends
    ts = df.set_index('date').reindex(full_idx)['price'].rename_axis('date').to_frame()
    ts['price'] = ts['price'].interpolate(method='time')
    ts = ts.reset_index()
    ts['t_days'] = (ts['date'] - ts['date'].min()).dt.days.astype(float)
    X = ts[['t_days']].values
    y = ts['price'].values
    lr = LinearRegression()
    lr.fit(X, y)
    ts['trend'] = lr.predict(X)
    ts['month'] = ts['date'].dt.month
    monthly_means = ts.groupby('month')['price'].mean()
    seasonal = monthly_means - monthly_means.mean()
    ts['seasonal'] = ts['month'].map(seasonal)
    ts['residual'] = ts['price'] - ts['trend'] - ts['seasonal']
    ts['resid_smooth'] = ts['residual'].rolling(window=3, center=True, min_periods=1).mean()
    ts['model_fit'] = ts['trend'] + ts['seasonal'] + ts['resid_smooth']

    # Forecast next 12 months
    last_date = ts['date'].max()
    future_idx = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=12, freq='M')
    future_t_days = (future_idx - ts['date'].min()).days.astype(float).to_numpy().reshape(-1,1)
    future_trend = lr.predict(future_t_days)
    future_month = future_idx.month
    # seasonal is a Series indexed by month
    future_seasonal = np.array([seasonal.loc[m] for m in future_month])
    future_prices = future_trend.flatten() + future_seasonal

    forecast_df = pd.DataFrame({'date': future_idx, 'price': future_prices, 'trend': future_trend.flatten(), 'seasonal': future_seasonal})
    artifacts = {
        'ts': ts, 'lr': lr, 'seasonal': seasonal, 'monthly_means': monthly_means, 'forecast_df': forecast_df,
        'first_date': ts['date'].min().date(), 'last_date': ts['date'].max().date()
    }
    return artifacts

def estimate_price(date_input, artifacts):
    ts = artifacts['ts']
    lr = artifacts['lr']
    seasonal = artifacts['seasonal']

    # parse input
    if isinstance(date_input, str):
        dt = parser.parse(date_input).date()
    elif isinstance(date_input, datetime):
        dt = date_input.date()
    else:
        dt = date_input

    first_date = artifacts['first_date']
    last_date = artifacts['last_date']
    max_forecast_date = last_date + timedelta(days=365)

    if dt < first_date or dt > max_forecast_date:
        raise ValueError(f\"Date {dt} out of supported range: {first_date} through {max_forecast_date}\")

    if first_date <= dt <= last_date:
        monthly_series = ts.set_index('date')['price']
        daily_idx = pd.date_range(start=monthly_series.index.min(), end=monthly_series.index.max(), freq='D')
        daily = monthly_series.reindex(daily_idx).interpolate(method='time')
        return {'date': dt, 'estimate': float(daily.loc[pd.Timestamp(dt)]), 'method': 'interpolated'}
    else:
        t_days = (pd.Timestamp(dt) - ts['date'].min()).days
        trend_val = float(lr.predict(np.array([[t_days]])))
        month = dt.month
        seasonal_val = float(seasonal.loc[month])
        estimate = trend_val + seasonal_val
        return {'date': dt, 'estimate': float(estimate), 'method': 'model_forecast'}

def save_artifacts(artifacts, path=ARTIFACTS_PATH):
    ts = artifacts['ts']
    lr = artifacts['lr']
    seasonal = artifacts['seasonal']
    np.savez(path,
             monthly_dates=ts['date'].astype(str).values,
             monthly_prices=ts['price'].values,
             trend_coef_intercept=lr.intercept_,
             trend_coef_slope=lr.coef_[0],
             seasonal_months=np.array([seasonal.loc[m] for m in range(1,13)]),
             last_date=str(artifacts['last_date'])
            )
    return path

def export_forecast_csv(artifacts, outpath=FORECAST_CSV):
    forecast_df = artifacts['forecast_df']
    forecast_df.to_csv(outpath, index=False, float_format='%.6f')
    return outpath

def main(argv):
    parser_arg = argparse.ArgumentParser(description='Natural gas monthly price estimator.')
    parser_arg.add_argument('--date', type=str, help='Date to estimate (YYYY-MM-DD). If omitted, script prints supported range.')
    parser_arg.add_argument('--forecast-csv', action='store_true', help='Save 12-month forecast CSV to /mnt/data.')
    parser_arg.add_argument('--show-plots', action='store_true', help='Show diagnostic plots (requires display).')
    args = parser_arg.parse_args(argv)

    try:
        df = load_monthly_csv()
    except Exception as e:
        print(f'Error loading CSV: {e}', file=sys.stderr)
        sys.exit(2)

    artifacts = build_model(df)

    # Save artifacts
    art_path = save_artifacts(artifacts)
    print(f'Artifacts saved to {art_path}')

    if args.forecast_csv:
        path = export_forecast_csv(artifacts)
        print(f'12-month forecast CSV exported to {path}')

    if args.date:
        try:
            res = estimate_price(args.date, artifacts)
            print(f\"Estimate for {res['date']}: {res['estimate']:.6f}  (method: {res['method']})\")
        except Exception as e:
            print(f'Error estimating price: {e}', file=sys.stderr)
            sys.exit(3)
    else:
        print(f\"Supported date range: {artifacts['first_date']} through {artifacts['last_date'] + timedelta(days=365)}\")
        print('Provide --date YYYY-MM-DD to get an estimate, or --forecast-csv to save the 12-month forecast CSV.')

if __name__ == '__main__':
    main(sys.argv[1:])
