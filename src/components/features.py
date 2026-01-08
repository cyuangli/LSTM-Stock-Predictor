import pandas as pd
import numpy as np
import pandas_ta_classic as ta

# ============================================================================
# 1. MOMENTUM INDICATORS
# ============================================================================

def get_ema(data, periods=20):
    """Exponential Moving Average"""
    new_df = pd.DataFrame()
    ema = data["Adj Close"].ewm(span=periods, adjust=False).mean()
    new_df["EMA"] = (ema - ema.mean()) / ema.std()
    return new_df
    
def get_sma(data, periods=50):
    """Simple Moving Average"""
    new_df = pd.DataFrame()
    sma = data["Adj Close"].rolling(window=periods).mean()
    new_df["SMA"] = (sma - sma.mean()) / sma.std()
    return new_df

def get_rsis(data, periods=24):
    """Relative Strength Index"""
    rsi_range = list(range(2, periods + 1))
    rsis = pd.DataFrame()
    
    for p in rsi_range:
        # Calculate RSI manually
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsis["RSI_" + str(p)] = rsi
    return (rsis - rsis.mean()) / rsis.std()

def get_roc(data, window=10):
    """Rate of Change"""
    new_df = pd.DataFrame()
    roc = ((data["Adj Close"] - data["Adj Close"].shift(window)) / data["Adj Close"].shift(window)) * 100
    new_df["ROC"] = (roc - roc.mean()) / roc.std()
    return new_df

def get_macd(data, fast=12, slow=26, signal=9):
    """Moving Average Convergence/Divergence"""
    new_df = pd.DataFrame()
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    new_df["MACD"] = (macd - macd.mean()) / macd.std()
    return new_df


# ============================================================================
# 2. VOLATILITY INDICATORS
# ============================================================================

def get_gkvol(df):
    """Garman-Klass Volatility"""
    new_df = pd.DataFrame()
    gkvol = (((np.log(df["High"]) - np.log(df["Low"])) ** 2) / 2) - (2 * np.log(2) - 1) * (np.log(df["Adj Close"]) - np.log(df["Open"])) ** 2
    new_df["GKvol"] = (gkvol - gkvol.mean()) / gkvol.std()
    return new_df

def get_bb(df, length=20, num_std=2):
    """Bollinger Bands"""
    new_df = pd.DataFrame(index=df.index)
    
    # Calculate using pandas
    log_close = np.log1p(df["Adj Close"])
    sma = log_close.rolling(window=length).mean()
    std = log_close.rolling(window=length).std()
    
    new_df["BB Low"] = sma - (num_std * std)
    new_df["BB Mid"] = sma
    new_df["BB High"] = sma + (num_std * std)
    
    # Normalize each band
    for col in ["BB Low", "BB Mid", "BB High"]:
        new_df[col] = (new_df[col] - new_df[col].mean()) / new_df[col].std()
    
    return new_df

def get_atr(df, length=14):
    """Average True Range"""
    new_df = pd.DataFrame()
    
    # Calculate True Range
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR as moving average of True Range
    atr = true_range.rolling(window=length).mean()
    
    new_df["ATR"] = (atr - atr.mean()) / atr.std()
    return new_df

# ============================================================================
# 3. VOLUME INDICATORS
# ============================================================================

def get_dollarvolume(data):
    """Dollar Volume"""
    new_df = pd.DataFrame()
    dv = (data["Adj Close"] * data["Volume"]) / 1e6
    new_df["Dollar Volume"] = (dv - dv.mean()) / dv.std()
    return new_df

def get_vpt(data):
    """Volume Price Trend"""
    new_df = pd.DataFrame()
    vpt = [0]
    for i in range(1, len(data)):
        change = (data["Adj Close"].iloc[i] - data["Adj Close"].iloc[i-1]) / data["Adj Close"].iloc[i-1]
        vpt.append(vpt[-1] + data["Volume"].iloc[i] * change)
    vpt = pd.Series(vpt, index=data.index)
    new_df["VPT"] = (vpt - vpt.mean()) / vpt.std()  # Fixed: removed double assignment
    return new_df
    
def get_obv(data):
    """On-Balance Volume"""
    new_df = pd.DataFrame()
    obv = ta.obv(data["Adj Close"], data["Volume"])
    new_df["OBV"] = (obv - obv.mean()) / obv.std()
    return new_df

def get_vwap(data):
    """Volume Weighted Average Price"""
    new_df = pd.DataFrame()
    vwap = ta.vwap(data["High"], data["Low"], data["Close"], data["Volume"])
    new_df["VWAP"] = (vwap - vwap.mean()) / vwap.std()
    return new_df

def get_mfi(data, length=14):
    """Money Flow Index"""
    new_df = pd.DataFrame()
    mfi = ta.mfi(data["High"], data["Low"], data["Close"], data["Volume"], length=length)
    new_df["MFI"] = (mfi - mfi.mean()) / mfi.std()
    return new_df

# ============================================================================
# 4. PRICE ACTION
# ============================================================================

def get_price_position(data):
    """Position of close relative to high-low range"""
    new_df = pd.DataFrame()
    position = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
    new_df["price_position"] = (position - position.mean()) / position.std()
    return new_df

def get_hl_ratio(data):
    """High-Low ratio"""
    new_df = pd.DataFrame()
    hl_ratio = (data["High"] - data["Low"]) / data["Close"]
    new_df["HL_ratio"] = (hl_ratio - hl_ratio.mean()) / hl_ratio.std()
    return new_df

# ============================================================================
# 5. LAGGED FEATURES
# ============================================================================

def get_lagged_returns(data, lags=[1, 2, 3, 5, 10]):
    """Multiple lagged returns"""
    new_df = pd.DataFrame()
    returns = data["Adj Close"].pct_change()
    for lag in lags:
        lagged = returns.shift(lag)
        new_df[f"Return_Lag_{lag}"] = (lagged - lagged.mean()) / lagged.std()
    return new_df


FEATURE_REGISTRY = {
    # Momentum
    "ema": get_ema,
    "sma": get_sma,
    "rsi": get_rsis,
    "roc": get_roc,
    "macd": get_macd,

    # Volatility
    "gkvol": get_gkvol,
    "bollinger": get_bb,
    "atr": get_atr,

    # Volume
    "dollar_volume": get_dollarvolume,
    "vpt": get_vpt,
    "obv": get_obv,
    "vwap": get_vwap,
    "mfi": get_mfi,

    # Price action
    "price_position": get_price_position,
    "hl_ratio": get_hl_ratio,

    # Lagged
    "lagged_returns": get_lagged_returns,
}

def build_features(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    feature_dfs = []

    for name in feature_names:
        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature: {name}")

        feature_fn = FEATURE_REGISTRY[name]

        try:
            feature_df = feature_fn(data)
        except Exception as e:
            raise RuntimeError(f"Failed to compute feature '{name}': {e}")

        feature_dfs.append(feature_df)

    # Combine all features with original data
    features = pd.concat(feature_dfs, axis=1)

    return pd.concat([data, features], axis=1)
