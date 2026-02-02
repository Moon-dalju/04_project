import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

BEST_CONTAM = 0.12
BEST_WIN = 7
TCP_WIN = 8
WARNING_WIN = 5
CAUTION_WIN = 3
TCP_VARIABLES = ['TCP Tuner', 'TCP Load']


def run_pdm_model(df, col):
    current_win = TCP_WIN if col in TCP_VARIABLES else BEST_WIN

    iso = IsolationForest(contamination=BEST_CONTAM, random_state=42)
    pred = iso.fit_predict(df[[col]])
    is_out = (pred == -1).astype(int)

    crit = pd.Series(is_out).rolling(current_win).sum() >= current_win
    warn = (
        pd.Series(is_out).rolling(WARNING_WIN).sum() >= WARNING_WIN
    ) & (~crit)
    caut = (
        pd.Series(is_out).rolling(CAUTION_WIN).sum() >= CAUTION_WIN
    ) & (~warn) & (~crit)

    return {
        "normal": ~(crit | warn | caut),
        "caution": caut,
        "warning": warn,
        "critical": crit
    }
