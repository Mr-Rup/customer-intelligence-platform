"""
Shared utility functions for the Customer Intelligence Platform.
"""

import time
import functools
import pandas as pd
from pathlib import Path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to CSV with logging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"[OK] Saved {len(df):,} rows x {len(df.columns)} cols -> {path.name}")


def timer(func):
    """Decorator that prints execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIME] {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_separator(title: str = "", char: str = "-", width: int = 60) -> None:
    """Print a formatted section separator."""
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{char * pad} {title} {char * pad}")
    else:
        print(f"\n{char * width}")


def df_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a concise summary of a DataFrame."""
    print(f"\n[INFO] {name}")
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"   Missing values: {n_missing:,}")
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"   Duplicate rows: {n_duplicates:,}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a proportion as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = "$") -> str:
    """Format a numeric value as currency."""
    return f"{symbol}{value:,.2f}"
