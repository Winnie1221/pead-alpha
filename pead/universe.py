"""
Ticker universe: ~200 liquid US equities across the 11 GICS sectors.

Selection: S&P 500 / Russell 1000 names with a long 8-K filing history.
Note: the list is chosen as of today, so it carries survivorship bias
(names that were delisted or acquired before today are absent). See the
README's Limitations section.
"""

_RAW = [
    # Technology (40)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO", "ADBE", "CRM",
    "AMD", "INTC", "QCOM", "TXN", "AMAT", "KLAC", "LRCX", "MU", "MRVL", "ANET",
    "SNPS", "CDNS", "ANSS", "FTNT", "PANW", "CRWD", "ZBRA", "KEYS", "SWKS", "QRVO",
    "CRUS", "APH", "TEL", "GLW", "JNPR", "HPQ", "HPE", "WDC", "STX", "NTAP",
    # Communication Services (10)
    "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS", "FOX", "WBD", "PARA",
    # Consumer Discretionary (20)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "BKNG", "LOW", "GM",
    "F", "ROST", "ORLY", "AZO", "BBY", "DG", "DLTR", "YUM", "CMG", "HLT",
    # Consumer Staples (15)
    "WMT", "KO", "PEP", "COST", "PG", "PM", "MO", "CL", "KMB", "GIS",
    "CHD", "LW", "MKC", "SJM", "HRL",
    # Health Care (25)
    "UNH", "JNJ", "LLY", "ABT", "MRK", "ABBV", "TMO", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "CI", "HUM", "IQV", "WST", "CTLT", "IDXX", "HOLX", "TECH",
    "ISRG", "SYK", "BSX", "MDT", "ZBH",
    # Financials (20)
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "USB", "PNC",
    "ICE", "CME", "CBOE", "MKTX", "LPLA", "AFL", "MET", "PRU", "ALL", "TRV",
    # Industrials (25)
    "HON", "RTX", "GE", "LMT", "NOC", "GD", "BA", "EMR", "ETN", "PH",
    "ROK", "ITW", "CMI", "PCAR", "DE", "CAT", "MMM", "DOV", "XYL", "ROP",
    "FTV", "AME", "VRSK", "TDG", "HEI",
    # Energy (15)
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "MPC", "VLO", "PSX",
    "OXY", "PXD", "DVN", "FANG", "CLR",
    # Materials (10)
    "LIN", "APD", "ECL", "SHW", "PPG", "NEM", "FCX", "NUE", "ALB", "CF",
    # Real Estate (5)
    "AMT", "PLD", "EQIX", "CCI", "PSA",
    # Utilities (5)
    "NEE", "DUK", "SO", "AEP", "EXC",
    # Extra mid-caps
    "CLB", "JBL", "FLEX", "ON", "LHX",
]

# De-duplicate while preserving order
UNIVERSE = list(dict.fromkeys(_RAW))
