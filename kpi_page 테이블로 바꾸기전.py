import sys
import os
import pandas as pd
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import __main__
from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QLabel, QProgressBar, QComboBox, QApplication
from PySide6.QtCore import QTimer, Qt, QFile, QIODevice, QThread, Signal
from PySide6.QtUiTools import QUiLoader

# 머신러닝 라이브러리
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.impute import SimpleImputer
    from scipy.interpolate import interp1d
    from scipy.stats import spearmanr
    import torch
    import torch.nn as nn
except ImportError:
    print("필수 라이브러리가 설치되지 않았습니다. (scikit-learn, torch, scipy)")

# Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna가 설치되지 않아 기본 설정을 사용합니다.")

# =============================================================================
# [설정] 파라미터 및 상수
# =============================================================================
FIXED_PARAMS_BY_GROUP = {
    "Main": {"window": 6,  "z_th": 4.5, "min_run": 25},  
    "Over": {"window": 53, "z_th": 2.48, "min_run": 19}, 
    "Low":  {"window": 22, "z_th": 2.12, "min_run": 12}, 
    "Default": {"window": 10, "z_th": 3.0, "min_run": 15} 
}

BEST_CONTAM = 0.12
BEST_WIN = 7
TCP_WIN = 8
WARNING_WIN = 4
CAUTION_WIN = 3
TCP_VARIABLES = ['TCP Tuner', 'TCP Load']

TARGET_MONITOR_COLS = [
    'BCl3 Flow', 'Cl2 Flow', 'RF Btm Pwr', 'Endpt A', 'He Press', 
    'RF Tuner', 'RF Load', 'TCP Tuner', 'TCP Load', 'Vat Valve'
]

# =============================================================================
# [Helper] Utils
# =============================================================================
def load_ui_file(ui_file_path, base_instance):
    loader = QUiLoader()
    file = QFile(ui_file_path)
    if not file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_path}: {file.errorString()}")
        return
    ui_widget = loader.load(file, base_instance)
    file.close()
    for widget in ui_widget.findChildren(QWidget):
        if widget.objectName():
            setattr(base_instance, widget.objectName(), widget)
    if base_instance.layout() is None:
        layout = QVBoxLayout(base_instance)
        layout.setContentsMargins(0, 0, 0, 0)
    else:
        layout = base_instance.layout()
    layout.addWidget(ui_widget)

def extract_number(name):
    numbers = re.findall(r'\d+', str(name) if pd.notnull(name) else "")
    return int(numbers[0]) if numbers else 0

def assign_group(num):
    try: n = int(num)
    except: return 'Others'
    if 2901 <= n <= 2943: return 'Main'
    elif 3101 <= n <= 3143: return 'Over'
    elif 3301 <= n <= 3343: return 'Low'
    else: return 'Others'

# =============================================================================
# [1] Analyzers & Predictors
# =============================================================================

class ReliableOESRULPredictor:
    def __init__(self, n_components=2, k=2.5, smoothing_alpha=0.2):
        self.n = n_components
        self.k = k
        self.alpha = smoothing_alpha
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.window_candidates = [4, 6, 8, 10, 12, 15, 18, 20]
        self.T2_smooth = []
        self.threshold = 0
        self.trained = False
        self.valid_cols = []

    def _is_float(self, s):
        try: float(s); return True
        except: return False

    def fit(self, df):
        if df.empty: return
        self.valid_cols = [c for c in df.columns if self._is_float(c)]
        temp_df = df[self.valid_cols]
        self.valid_cols = [c for c in self.valid_cols if temp_df[c].std() > 1e-6]
        if not self.valid_cols: return
        
        if 'fault_name' in df.columns and df['fault_name'].str.contains('calibration').any():
            df_train = df[df['fault_name'].str.contains('calibration')].copy()
        else:
            df_train = df.copy()

        data_matrix = df[self.valid_cols].copy()
        data_matrix = data_matrix.interpolate(method='linear', limit_direction='both').ffill().bfill()
        X = data_matrix.values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        n_calib = max(5, len(df_train))
        X_calib = X[:n_calib]
        X_calib_scaled = self.scaler.fit_transform(X_calib)
        self.pca.fit(X_calib_scaled)

        X_all_scaled = self.scaler.transform(X)
        scores = self.pca.transform(X_all_scaled)
        explained_variance = self.pca.explained_variance_
        explained_variance[explained_variance < 1e-9] = 1e-9
        T2_raw = np.sum(scores**2 / explained_variance, axis=1)
        self.T2_smooth = pd.Series(T2_raw).ewm(alpha=self.alpha).mean().values
        T2_calib = self.T2_smooth[:n_calib]
        self.threshold = np.mean(T2_calib) + self.k * np.std(T2_calib)
        self.trained = True

    def predict(self, current_idx):
        if not self.trained or current_idx >= len(self.T2_smooth): return "Preparing...", 0, 0.0
        if current_idx < min(self.window_candidates): return "Collecting Data", 0, 0.0
        
        best_r2, best_slope, best_intercept = -999, 0, 0
        found_trend = False
        
        for w in self.window_candidates:
            if current_idx - w + 1 < 0: continue
            start_idx = current_idx - w + 1
            y_sub = self.T2_smooth[start_idx : current_idx + 1]
            X_sub = np.arange(start_idx, current_idx + 1).reshape(-1, 1)
            reg = LinearRegression().fit(X_sub, y_sub)
            pred = reg.predict(X_sub)
            r2 = r2_score(y_sub, pred)
            slope = reg.coef_[0]
            if slope > 0 and r2 > best_r2:
                best_r2 = r2
                best_slope = slope
                best_intercept = reg.intercept_
                found_trend = True
                
        if not found_trend: return "Stable (No Trend)", 9999, 0.0
        
        rul_val = 9999
        status = "Stable"
        current_val = self.T2_smooth[current_idx]
        
        if current_val >= self.threshold:
            status = "Failed"; rul_val = 0
        elif best_slope > 1e-9:
            pred_fail_idx = (self.threshold - best_intercept) / best_slope
            rul_val = max(0, pred_fail_idx - current_idx)
            if rul_val > 2000: rul_val = 9999
            if rul_val < 50: status = "Critical"
            elif rul_val < 150: status = "Warning"
            else: status = "Normal"
            
        return status, rul_val, best_r2

class ReliableRFMRULPredictor:
    def __init__(self, n_components=2, k=2.5, smoothing_alpha=0.2):
        self.n, self.k, self.alpha = n_components, k, smoothing_alpha
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.T2_smooth, self.threshold, self.trained = [], 0, False
    
    def fit(self, df):
        if df.empty: return
        sensor_cols = [c for c in df.columns if c.startswith('S') or (df[c].dtype in [float, int] and 'TIME' not in c and 'wafer' not in c)]
        temp_df = df[sensor_cols]
        valid_cols = [c for c in sensor_cols if temp_df[c].std() > 1e-6]
        if not valid_cols: return
        
        if 'fault_name' in df.columns and df['fault_name'].str.contains('calibration').any():
            df_train = df[df['fault_name'].str.contains('calibration')].copy()
        else:
            df_train = df.copy()

        data_matrix = df[valid_cols].interpolate(method='linear', limit_direction='both').ffill().bfill().values
        X = SimpleImputer(strategy='mean').fit_transform(data_matrix)

        n_calib = max(5, len(df_train))
        X_calib_scaled = self.scaler.fit_transform(X[:n_calib])
        self.pca.fit(X_calib_scaled)
        
        X_all_scaled = self.scaler.transform(X)
        scores = self.pca.transform(X_all_scaled)
        explained_variance = self.pca.explained_variance_
        explained_variance[explained_variance < 1e-9] = 1e-9
        T2_raw = np.sum(scores**2 / explained_variance, axis=1)
        
        self.T2_smooth = pd.Series(T2_raw).ewm(alpha=self.alpha).mean().values
        T2_calib = self.T2_smooth[:n_calib]
        self.threshold = np.mean(T2_calib) + self.k * np.std(T2_calib)
        self.trained = True

    def predict(self, current_idx):
        if not self.trained or current_idx >= len(self.T2_smooth): return "Preparing...", 0, 0.0
        if current_idx < 4: return "Collecting...", 0, 0.0
        if self.T2_smooth[current_idx] >= self.threshold: return "Failed", 0, 0.0
        
        best_window = min(15, current_idx)
        if OPTUNA_AVAILABLE:
            def objective(trial):
                max_w = min(20, current_idx + 1)
                if max_w < 3: return -1.0
                w = trial.suggest_int('window', 3, max_w)
                start_idx = max(0, current_idx - w + 1)
                y_sub = self.T2_smooth[start_idx : current_idx + 1]
                if len(y_sub) < 3: return -1.0
                corr, _ = spearmanr(np.arange(len(y_sub)), y_sub)
                return corr if not np.isnan(corr) else -1.0
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=5)
            best_window = study.best_params['window']
            
        start_idx = max(0, current_idx - best_window + 1)
        y_sub = self.T2_smooth[start_idx : current_idx + 1]
        X_sub = np.arange(start_idx, current_idx + 1).reshape(-1, 1)
        reg = LinearRegression().fit(X_sub, y_sub)
        
        slope = reg.coef_[0]
        rul_val = 9999
        status = "Normal"
        
        if slope > 1e-12:
            pred_fail_idx = (self.threshold - reg.intercept_) / slope
            rul_val = max(0, pred_fail_idx - current_idx)
            if rul_val > 5000: rul_val = 5000
            
            if rul_val < 50: status = "Critical"
            elif rul_val < 150: status = "Warning"
            else: status = "Normal"
        else:
            rul_val = 5000; status = "Healthy"
            
        return status, rul_val, r2_score(y_sub, reg.predict(X_sub))

class OESAnalyzer:
    def __init__(self, target_length=28):
        self.target_length = target_length
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.iso = IsolationForest(contamination=0.2, random_state=42)
        self.trained = False
        self.wafer_data_map = {}
        self.t2_th = 0; self.spe_th = 0; self.t2_ewma_th = 0
        self.history_t2 = []

    def _is_float(self, s):
        try: float(s); return True
        except: return False

    def fit(self, df):
        self.wavelength_cols = [c for c in df.columns if self._is_float(c)]
        if 'wafer_names' not in df.columns: return
        self.wafer_data_map = {w: df[df['wafer_names'] == w].reset_index(drop=True) for w in df['wafer_names'].unique()}

        df_calib = df.iloc[:50]
        if 'fault_name' in df.columns:
            calib = df[df['fault_name'].astype(str).str.contains('calibration', case=False, na=False)]
            if not calib.empty: df_calib = calib

        tensor_list = []
        for w in df_calib['wafer_names'].unique():
            sub = df_calib[df_calib['wafer_names'] == w].reset_index(drop=True)
            if len(sub) < 5: continue
            x_old = np.linspace(0, 1, len(sub))
            x_new = np.linspace(0, 1, self.target_length)
            mat = []
            for col in self.wavelength_cols:
                vals = pd.to_numeric(sub[col], errors='coerce').fillna(0).values
                mat.append(interp1d(x_old, vals, kind='linear')(x_new))
            tensor_list.append(np.array(mat).T)
        
        if not tensor_list: return
        X_tensor = np.array(tensor_list)
        self.I, self.J, self.K = X_tensor.shape
        X_flat = X_tensor.reshape(self.I, self.J * self.K)
        X_scaled = self.scaler.fit_transform(X_flat)
        scores = self.pca.fit_transform(X_scaled)
        
        t2 = np.sum((scores ** 2) / self.pca.explained_variance_, axis=1)
        diff = X_scaled - self.pca.inverse_transform(scores)
        spe = np.sum(diff.reshape(self.I, self.J, self.K) ** 2, axis=2).flatten()
        
        k = 1.5
        self.t2_th = t2.mean() + k * t2.std()
        self.spe_th = spe.mean() + k * spe.std()
        
        t2_exp = np.repeat(t2, self.J)
        t2_ewma = pd.Series(t2_exp).ewm(alpha=0.2).mean().values
        self.t2_ewma_th = t2_ewma.mean() + k * t2_ewma.std()
        
        self.iso.fit(X_scaled)
        self.trained = True

    def predict(self, wafer_name):
        if not self.trained or wafer_name not in self.wafer_data_map: return False
        sub = self.wafer_data_map[wafer_name]
        if len(sub) < 5: return False
        
        x_old = np.linspace(0, 1, len(sub))
        x_new = np.linspace(0, 1, self.target_length)
        mat = []
        for col in self.wavelength_cols:
            vals = pd.to_numeric(sub[col], errors='coerce').fillna(0).values
            mat.append(interp1d(x_old, vals, kind='linear')(x_new))
            
        X_sample = np.array(mat).T
        X_scaled = self.scaler.transform(X_sample.reshape(1, self.J * self.K))
        scores = self.pca.transform(X_scaled)
        
        t2 = np.sum((scores ** 2) / self.pca.explained_variance_, axis=1)[0]
        diff = X_scaled - self.pca.inverse_transform(scores)
        spe_vals = np.sum(diff.reshape(1, self.J, self.K) ** 2, axis=2).flatten()
        iso_pred = self.iso.predict(X_scaled)[0]
        
        self.history_t2.append(t2)
        if len(self.history_t2) > 10: self.history_t2.pop(0)
        
        score = 0
        if t2 > self.t2_th: score += 1
        if np.mean(spe_vals > self.spe_th) > 0.1: score += 1
        if iso_pred == -1: score += 1
        if np.mean(self.history_t2) > self.t2_ewma_th: score += 1
        
        return score >= 2

class RFMAnalyzer:
    def __init__(self, target_length=28):
        self.target_length = target_length
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.iso = IsolationForest(contamination=0.2, random_state=42)
        self.trained = False
        self.sensor_cols = []
        self.wafer_data_map = {}
        self.history_t2 = []
        self.t2_th = 0; self.spe_th = 0; self.t2_ewma_th = 0

    def fit(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.sensor_cols = [c for c in numeric_cols if c not in ['TIME', 'series', 'group', 'wafer_num']]
        if 'wafer_names' not in df.columns: return
        self.wafer_data_map = {w: df[df['wafer_names'] == w].sort_values('TIME') for w in df['wafer_names'].unique()}

        df_calib = df.iloc[:50]
        if 'fault_name' in df.columns:
            calib = df[df['fault_name'].astype(str).str.contains('calibration', case=False, na=False)]
            if not calib.empty: df_calib = calib

        tensor_list = []
        for w in df_calib['wafer_names'].unique():
            sub = df_calib[df_calib['wafer_names'] == w].sort_values('TIME')
            if len(sub) < 5: continue
            x_old = np.linspace(0, 1, len(sub))
            x_new = np.linspace(0, 1, self.target_length)
            mat = []
            for col in self.sensor_cols:
                vals = sub[col].values
                f = interp1d(x_old, vals, kind='linear')
                mat.append(f(x_new))
            tensor_list.append(np.array(mat).T)
        
        if not tensor_list: return
        X_tensor = np.array(tensor_list)
        self.I, self.J, self.K = X_tensor.shape
        X_flat = X_tensor.reshape(self.I, self.J * self.K)
        X_scaled = self.scaler.fit_transform(X_flat)
        scores = self.pca.fit_transform(X_scaled)
        
        t2 = np.sum((scores ** 2) / self.pca.explained_variance_, axis=1)
        diff = X_scaled - self.pca.inverse_transform(scores)
        spe = np.sum(diff.reshape(self.I, self.J, self.K) ** 2, axis=2).flatten()
        
        k = 1.5
        self.t2_th = t2.mean() + k * t2.std()
        self.spe_th = spe.mean() + k * spe.std()
        
        t2_exp = np.repeat(t2, self.J)
        t2_ewma = pd.Series(t2_exp).ewm(alpha=0.2).mean().values
        self.t2_ewma_th = t2_ewma.mean() + k * t2_ewma.std()
        
        self.iso.fit(X_scaled)
        self.trained = True

    def predict(self, wafer_name):
        if not self.trained or wafer_name not in self.wafer_data_map: return False
        sub = self.wafer_data_map[wafer_name]
        if len(sub) < 5: return False
        x_old = np.linspace(0, 1, len(sub))
        x_new = np.linspace(0, 1, self.target_length)
        mat = []
        for col in self.sensor_cols:
            vals = sub[col].values
            f = interp1d(x_old, vals, kind='linear')
            mat.append(f(x_new))
            
        X_sample = np.array(mat).T
        X_scaled = self.scaler.transform(X_sample.reshape(1, self.J * self.K))
        scores = self.pca.transform(X_scaled)
        t2 = np.sum((scores ** 2) / self.pca.explained_variance_, axis=1)[0]
        diff = X_scaled - self.pca.inverse_transform(scores)
        spe_vals = np.sum(diff.reshape(1, self.J, self.K) ** 2, axis=2).flatten()
        iso_pred = self.iso.predict(X_scaled)[0]
        
        self.history_t2.append(t2)
        if len(self.history_t2) > 10: self.history_t2.pop(0)
        
        score = 0
        if t2 > self.t2_th: score += 1
        if np.mean(spe_vals > self.spe_th) > 0.1: score += 1
        if iso_pred == -1: score += 1
        if np.mean(self.history_t2) > self.t2_ewma_th: score += 1
        
        return score >= 2

class RULAnalyzer:
    def __init__(self, n_components=0.95):
        self.scaler = RobustScaler()
        self.n_components = n_components
        self.model = None
        self.threshold = 1.0
        self.slope, self.intercept = 0, 0
        self.t2_norm = []
        self.r2 = 0

    def _resample(self, data, target_len):
        if len(data) == 0: return np.zeros((target_len, data.shape[1]))
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, data.shape[1]))
        for i in range(data.shape[1]):
            resampled[:, i] = np.interp(x_new, x_old, data[:, i])
        return resampled

    def fit_analysis(self, df, window=10):
        print("RUL 분석 시작...")
        exclude = ['wafer_names', 'wafer_num', 'Step Number', 'Time', 'fault_name', 'RUL', 'wafer_id', 'group']
        sensor_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [float, int]]
        median_len = int(df.groupby('wafer_names').size().median())
        
        X_list = []
        for w in df['wafer_names'].unique():
            w_vals = df[df['wafer_names'] == w][sensor_cols].values
            X_list.append(self._resample(w_vals, median_len).flatten())
        
        X_arr = np.array(X_list)
        n_calib = max(5, int(len(X_arr) * 0.2))
        
        self.scaler.fit(X_arr[:n_calib])
        X_scaled = self.scaler.transform(X_arr)
        
        pca = PCA(n_components=self.n_components)
        pca.fit(self.scaler.transform(X_arr[:n_calib]))
        scores = pca.transform(X_scaled)
        
        t2 = np.sum((scores ** 2) / pca.explained_variance_, axis=1)
        t2_smooth = pd.Series(t2).rolling(window=window, min_periods=1).mean().values
        
        if t2_smooth.max() == t2_smooth.min(): self.t2_norm = np.zeros_like(t2_smooth)
        else: self.t2_norm = (t2_smooth - t2_smooth.min()) / (t2_smooth.max() - t2_smooth.min())
        
        X_reg = np.arange(len(self.t2_norm)).reshape(-1, 1)
        self.model = LinearRegression().fit(X_reg, self.t2_norm)
        self.slope, self.intercept = self.model.coef_[0], self.model.intercept_
        self.r2 = r2_score(self.t2_norm, self.model.predict(X_reg))
        
        self.threshold = 1.0
        if OPTUNA_AVAILABLE and self.slope > 0:
            def objective(trial):
                th = trial.suggest_float('threshold', 0.5, 1.5)
                if self.slope <= 1e-6: return 9999
                pred_idx = (th - self.intercept) / self.slope
                return abs(pred_idx - (len(self.t2_norm) - 1))
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=30)
            self.threshold = study.best_params['threshold']

    def get_rul_status(self, current_idx):
        if len(self.t2_norm) == 0: return 0, "Unknown", 0
        idx = min(current_idx, len(self.t2_norm)-1)
        current_hi = self.t2_norm[idx]
        
        rul_val = 9999
        if self.slope > 1e-6:
            pred_end_idx = (self.threshold - self.intercept) / self.slope
            rul_val = pred_end_idx - current_idx
            
        status = "Normal"
        if current_hi >= self.threshold: status = "Critical"
        elif current_hi >= self.threshold * 0.8: status = "Warning"
        
        return rul_val, status, self.r2

class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features=1, hidden_dim=128, latent_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.enc = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.dec = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.out_fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        h_seq, _ = self.enc(x)
        h_last = h_seq[:, -1, :]
        z = torch.relu(self.enc_fc(h_last))
        h0 = torch.relu(self.dec_fc(z))
        dec_in = h0.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_seq, _ = self.dec(dec_in)
        out = self.out_fc(dec_seq)
        return out

class MultiSensorAlarmSystem:
    def __init__(self):
        self.models = {} 
        self.buffers = {}
        self.trained = False
        self.sensor_cols = []
        self.warn_win = WARNING_WIN
        self.caut_win = CAUTION_WIN
        
        # [추가] 스텝 안정화 가드를 위한 상태 변수
        self.last_step = None
        self.step_stable_count = 0
        self.last_wafer_num = None

    def fit(self, df):
        self.sensor_cols = [c for c in TARGET_MONITOR_COLS if c in df.columns]
        if not self.sensor_cols:
            print("경고: 지정된 핵심 센서가 데이터에 없습니다.")
            return
        
        if 'group' not in df.columns:
            if 'wafer_num' not in df.columns:
                df['wafer_num'] = df['wafer_names'].apply(extract_number)
            df['group'] = df['wafer_num'].apply(assign_group)
        
        groups = ['Main', 'Over', 'Low', 'Others']
        self.models = {col: {} for col in self.sensor_cols}
        self.buffers = {col: [] for col in self.sensor_cols}
        
        for col in self.sensor_cols:
            for g in groups:
                sub_df = df[df['group'] == g]
                if len(sub_df) > 5:
                    # contamination을 낮추면 민감도가 줄어들지만, 
                    # 스텝 가드를 적용하면 굳이 파라미터를 건드리지 않아도 해결됩니다.
                    iso = IsolationForest(contamination=BEST_CONTAM, random_state=42)
                    data = sub_df[[col]].fillna(0).values
                    iso.fit(data)
                    self.models[col][g] = iso
                else:
                    self.models[col][g] = None
        self.trained = True

    def check(self, row_series, wafer_num):
        # 학습되지 않았으면 정상 반환
        if not self.trained: return "Normal", [] 
        
        # -----------------------------------------------------
        # 1. [오탐지 방지 로직] 스텝 변경 감지 및 안정화 기간 설정
        # -----------------------------------------------------
        current_step = row_series.get('Step Number', 0)
        
        # 새로운 웨이퍼가 들어온 경우 -> 리셋
        if self.last_wafer_num != wafer_num:
            self.last_wafer_num = wafer_num
            self.last_step = current_step
            self.step_stable_count = 0
            for col in self.buffers: self.buffers[col] = [] # 버퍼 초기화
            
        # 같은 웨이퍼에서 스텝 번호가 바뀐 경우 -> 리셋
        elif self.last_step != current_step:
            self.last_step = current_step
            self.step_stable_count = 0
            
        # 스텝이 유지되는 경우 -> 카운트 증가
        else:
            self.step_stable_count += 1
            
        # [핵심] 카운트가 15 미만이면 '안정화 기간'으로 간주 (약 15초)
        is_stabilizing = self.step_stable_count < 15 
        # -----------------------------------------------------

        group = assign_group(wafer_num)
        
        # 알람이 발생한 센서들을 저장할 리스트
        critical_sensors = []
        warning_sensors = []
        
        for col in self.sensor_cols:
            if col not in row_series: continue
            val = row_series[col]
            model = self.models[col].get(group)
            
            # 모델 예측
            is_out = 0
            if model is not None:
                try:
                    pred = model.predict([[val]])[0]
                    is_out = 1 if pred == -1 else 0
                except: is_out = 0
            
            # -------------------------------------------------
            # 2. [오탐지 방지 적용] 안정화 기간이면 강제로 정상 처리
            # -------------------------------------------------
            if is_stabilizing: 
                is_out = 0
            
            # (참고: Calibration 제외 코드는 사용자 요청으로 삭제됨 -> 모든 웨이퍼 검사)

            # 버퍼링 (연속적인 이상 감지 확인)
            win_size = TCP_WIN if col in TCP_VARIABLES else BEST_WIN
            self.buffers[col].append(is_out)
            if len(self.buffers[col]) > win_size: self.buffers[col].pop(0)
            
            buf = self.buffers[col]
            buf_len = len(buf)
            buf_sum = sum(buf)
            
            # 상태 판정 (Critical / Warning)
            if buf_len >= win_size and buf_sum == win_size:
                critical_sensors.append(col)
            elif buf_len >= self.warn_win and sum(buf[-self.warn_win:]) >= self.warn_win:
                warning_sensors.append(col)
        
        # -----------------------------------------------------
        # 3. 최종 결과 반환 (상태, 센서리스트)
        # -----------------------------------------------------
        final_status = "Normal"
        problem_sensors = []
        
        if critical_sensors:
            final_status = "Critical"
            problem_sensors = list(set(critical_sensors)) # 중복 제거
        elif warning_sensors:
            final_status = "Warning"
            problem_sensors = list(set(warning_sensors))
        
        # (NameError를 유발하던 has_caution 부분은 삭제했습니다)

        return final_status, problem_sensors

# =============================================================================
# [2] Worker Threads
# =============================================================================
class RULWorker(QThread):
    finished = Signal()
    def __init__(self, analyzer, df):
        super().__init__()
        self.analyzer = analyzer; self.df = df
    def run(self):
        try: self.analyzer.fit_analysis(self.df)
        except: pass
        self.finished.emit()

class OESRULWorker(QThread):
    finished = Signal()
    def __init__(self, analyzer, df):
        super().__init__()
        self.analyzer = analyzer; self.df = df
    def run(self):
        try: self.analyzer.fit(self.df)
        except: pass
        self.finished.emit()

class RFMRULWorker(QThread):
    finished = Signal()
    def __init__(self, analyzer, df):
        super().__init__()
        self.analyzer = analyzer; self.df = df
    def run(self):
        try: self.analyzer.fit(self.df)
        except: pass
        self.finished.emit()

# =============================================================================
# [3] Main Class: KPIPage
# =============================================================================
class KPIPage(QWidget):
    def __init__(self):
        super().__init__()
        try: load_ui_file("kpi.ui", self)
        except: return

        self.target_col = 'Pressure'
        self.rf_col = 'RF Pwr'      

        self.oes_analyzer = OESAnalyzer()
        self.rfm_analyzer = RFMAnalyzer()
        self.rul_analyzer = RULAnalyzer()
        self.oes_rul_predictor = ReliableOESRULPredictor()
        self.rfm_rul_predictor = ReliableRFMRULPredictor()
        self.alarm_system = MultiSensorAlarmSystem()
        
        self.oes_detected_list = [] 
        self.rfm_detected_list = [] 
        self.current_wafer_name = None 
        self.rul_ready = False
        self.oes_rul_ready = False
        self.rfm_rul_ready = False 

        self.feature_cols = []  
        self.lstm_scaler = StandardScaler() 
        self.lstm_model = None

        self.window_buffer = []
        self.current_group_params = FIXED_PARAMS_BY_GROUP["Default"]

        self.lstm_detected_set = set() 
        self.lstm_defect_list = []
        self.lstm_time_list = []
        self.lstm_run_count = 0 
        self.lstm_mu = 0.0
        self.lstm_std = 1.0
        self.z_threshold = 2.5 
        self.min_run = 10 

        self.load_data()
        self.load_lstm_model()
        self.init_ui_elements()
        self.populate_combobox()
        
        self.current_wafer_index = 0
        self.processed_wafers = set()  
        self.defect_wafer_set = set()  
        
        self.cnt_normal = 0
        self.cnt_warning = 0
        self.cnt_critical = 0
        self.cnt_caution = 0
        
        self.timer = QTimer(self)
        self.timer.setInterval(200) 
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start()

    def load_data(self):
        if os.path.exists("ev_data.csv"):
            try:
                self.df_ev = pd.read_csv("ev_data.csv")
                self.df_ev['wafer_num'] = self.df_ev['wafer_names'].apply(extract_number)
                self.df_ev['group'] = self.df_ev['wafer_num'].apply(assign_group)
                self.df_ev = self.df_ev.sort_values(by=['wafer_num', 'Time']).reset_index(drop=True)
                
                exclude = ['wafer_names', 'wafer_num', 'Step Number', 'Time', 'fault_name', 'RUL', 'wafer_id', 'group', 'series']
                numeric_cols = self.df_ev.select_dtypes(include=[np.number]).columns.tolist()
                self.feature_cols = [c for c in numeric_cols if c not in exclude]
                
                if not self.df_ev.empty:
                    self.alarm_system.fit(self.df_ev)

                self.stats = {}
                for col in self.df_ev.columns:
                    if col in numeric_cols:
                        self.stats[col] = {'mean': self.df_ev[col].mean(), 'std': self.df_ev[col].std()}

                self.rul_worker = RULWorker(self.rul_analyzer, self.df_ev)
                self.rul_worker.finished.connect(self.on_rul_finished)
                self.rul_worker.start()
            except Exception as e:
                print(f"EV Load Error: {e}")
                self.df_ev = pd.DataFrame()
        else: self.df_ev = pd.DataFrame()

        if os.path.exists("oes_data.csv"):
            try:
                df_oes = pd.read_csv("oes_data.csv")
                self.oes_analyzer.fit(df_oes)
                self.oes_rul_worker = OESRULWorker(self.oes_rul_predictor, df_oes)
                self.oes_rul_worker.finished.connect(self.on_oes_rul_finished)
                self.oes_rul_worker.start()
            except: pass
        
        if os.path.exists("rfm_data.csv"):
            try:
                df_rfm = pd.read_csv("rfm_data.csv")
                self.rfm_analyzer.fit(df_rfm)
                self.rfm_rul_worker = RFMRULWorker(self.rfm_rul_predictor, df_rfm)
                self.rfm_rul_worker.finished.connect(self.on_rfm_rul_finished)
                self.rfm_rul_worker.start()
            except: pass

    def on_rul_finished(self): self.rul_ready = True
    def on_oes_rul_finished(self): self.oes_rul_ready = True
    def on_rfm_rul_finished(self): self.rfm_rul_ready = True

    def load_lstm_model(self):
        import pickle
        # 1. 기본 모델 초기화 (파일 로드 실패 대비)
        default_n = len(self.feature_cols) if self.feature_cols else 1
        self.lstm_model = LSTMAutoEncoder(n_features=default_n)

        # 2. Pickle 파일 존재 확인 및 로드
        if os.path.exists("pdm_lstm_ae_bundle.pkl"):
            try:
                with open("pdm_lstm_ae_bundle.pkl", "rb") as f: 
                    bundle = pickle.load(f)
                
                # 3. 그룹 정보 및 Scaler 로드
                groups = bundle.get("groups", {})
                # '29'번 그룹 모델을 우선 사용, 없으면 첫 번째 그룹 사용
                target = "29" if "29" in groups else list(groups.keys())[0]
                g_data = groups[target]
                
                if 'scaler' in g_data: 
                    self.lstm_scaler = g_data['scaler']
                
                # [수정 완료] LSTMAE_CONFIG 대신 FIXED_PARAMS_BY_GROUP 사용
                conf = FIXED_PARAMS_BY_GROUP.get(target, FIXED_PARAMS_BY_GROUP["Default"])
                self.z_threshold = conf["z_th"]
                self.min_run = conf["min_run"]
                
                # 4. 모델 가중치 형상을 보고 차원(Dimension) 역추적 및 모델 재생성
                sd = g_data['model_state_dict']
                n_feat = sd['enc.weight_ih_l0'].shape[1]
                n_hid = sd['enc.weight_ih_l0'].shape[0] // 4
                n_lat = sd['enc_fc.weight'].shape[0]
                
                self.lstm_model = LSTMAutoEncoder(n_features=n_feat, hidden_dim=n_hid, latent_dim=n_lat)
                self.lstm_model.load_state_dict(sd)
                self.lstm_model.eval()
                
                # 5. 통계값(mu, std) 초기화 (데이터가 있을 경우)
                # 모델의 기준점(정상 범위)을 잡기 위해 초반 데이터를 이용해 오차 평균/분산을 계산합니다.
                if not self.df_ev.empty:
                    c_df = self.df_ev.iloc[:200] # 초반 200개 사용
                    vecs = self.lstm_scaler.transform(c_df[self.feature_cols].values)
                    errors = []
                    
                    with torch.no_grad():
                        for i in range(min(len(vecs), 500)):
                            # (Batch=1, Window=1, Features) 형태로 임시 추론
                            t_in = torch.FloatTensor(vecs[i]).unsqueeze(0).unsqueeze(0)
                            t_out = self.lstm_model(t_in)
                            
                            # [중요] MSE(제곱오차) 기준으로 mu/std 계산 (업데이트된 로직 반영)
                            loss = np.mean((t_in.numpy() - t_out.numpy())**2)
                            errors.append(loss)
                    
                    self.lstm_mu = np.mean(errors)
                    self.lstm_std = np.std(errors) + 1e-9
                    print(f"LSTM Model Loaded. Target Group: {target}, Mu: {self.lstm_mu:.5f}, Std: {self.lstm_std:.5f}")
                    
            except Exception as e: 
                print(f"LSTM Load Error: {e}")
                import traceback
                traceback.print_exc()

    def init_ui_elements(self):
        # 1. 콤보박스 스타일 설정 (기존 코드)
        if hasattr(self, 'comboBox'):
            self.combo_sensor = self.comboBox
            self.combo_sensor.setStyleSheet("""
                QComboBox { 
                    color: black; 
                    background-color: white; 
                    border: 1px solid gray;
                    padding: 1px 18px 1px 3px;
                }
                QComboBox QAbstractItemView {
                    color: black; 
                    background-color: white;
                    selection-background-color: lightgray;
                }
            """)
            try: self.combo_sensor.currentTextChanged.disconnect()
            except: pass
            self.combo_sensor.currentTextChanged.connect(self.on_sensor_changed)
        
        # 2. 초기 텍스트 설정 (기존 코드)
        for name in ['label_31', 'label_32', 'label_33', 'label_34', 'label_35']:
            if hasattr(self, name): getattr(self, name).setText("...")
        if hasattr(self, 'label_29'): self.label_29.setText("No Anomaly")

        # =================================================================
        # [추가] 3. 폰트 크기 및 스타일 일괄 적용 (여기에 붙여넣기)
        # =================================================================
        
        # (1) 핵심 센서 수치 (가장 크게: 압력, RF 파워 등) -> label_24, label_25
        # 폰트: 28px, 굵게, 진한 남색
        big_number_style = "font-size: 28px; font-weight: bold; color: #2c3e50;"
        if hasattr(self, 'label_24'): self.label_24.setStyleSheet(big_number_style)
        if hasattr(self, 'label_25'): self.label_25.setStyleSheet(big_number_style)

        # (2) 수율, 생산량, 알람 카운트 (중간 크기) -> label_12, 22, 23, 26, 27, 28
        # 폰트: 20px, 굵게, 진한 회색
        stat_style = "font-size: 20px; font-weight: bold; color: #34495e;"
        stat_labels = ['label_12', 'label_22', 'label_23', 'label_26', 'label_27', 'label_28']
        for name in stat_labels:
            if hasattr(self, name):
                getattr(self, name).setStyleSheet(stat_style)

        # (3) RUL 및 상태 메시지 초기 스타일 -> label_33, 34, 35
        # 폰트: 15px, 굵게 (업데이트 시 색상은 바뀌지만 크기는 유지되도록 update_dashboard도 수정 필수)
        rul_style = "font-size: 15px; font-weight: bold;"
        rul_labels = ['label_33', 'label_34', 'label_35']
        for name in rul_labels:
            if hasattr(self, name):
                getattr(self, name).setStyleSheet(rul_style)

        # (4) 이상 감지 로그 리스트 -> label_29, 30, 31, 32
        # 폰트: 13px, 굵게, 중간 회색
        log_style = "font-size: 13px; font-weight: bold; color: #555555;"
        log_labels = ['label_29', 'label_30', 'label_31', 'label_32']
        for name in log_labels:
            if hasattr(self, name):
                getattr(self, name).setStyleSheet(log_style)

        # 4. 그래프 초기화 (기존 코드)
        self.init_graphs()

    def init_graphs(self):
        if hasattr(self, 'frame_30'):
            self.fig_sensor, self.ax_sensor, self.canvas_sensor = self.setup_figure(self.get_layout(self.frame_30), (5, 2.5))
        if hasattr(self, 'frame_48'):
            self.fig_fdc, self.ax_fdc, self.canvas_fdc = self.setup_figure(self.get_layout(self.frame_48), (5, 3))

    def get_layout(self, widget):
        if widget.layout() is None:
            layout = QVBoxLayout(widget); layout.setContentsMargins(0,0,0,0)
            return layout
        return widget.layout()

    def setup_figure(self, layout, figsize):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
        fig = Figure(figsize=figsize, dpi=100)
        fig.patch.set_facecolor('#FFFFFF')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        ax = fig.add_subplot(111)
        return fig, ax, canvas

    def populate_combobox(self):
        if self.df_ev.empty or not hasattr(self, 'combo_sensor'): return
        numeric_cols = self.df_ev.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric_cols if c not in ['wafer_num', 'wafer_id', 'group']]
        self.combo_sensor.clear()
        self.combo_sensor.addItems(cols)
        self.combo_sensor.setCurrentText('Pressure')

    def on_sensor_changed(self, text):
        if text: self.target_col = text

    def update_gauge_style(self, progress_bar, value, mean, std):
        min_v, max_v = mean - 4*std, mean + 4*std
        ratio = 0.5 if max_v == min_v else (value - min_v) / (max_v - min_v)
        progress_bar.setRange(0, 1000)
        progress_bar.setValue(max(0, min(1000, int(ratio * 1000))))
        z = abs(value - mean) / (std + 1e-6)
        color = "#2ecc71" if z < 1.0 else ("#f39c12" if z < 2.5 else "#c0392b")
        progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    def update_dashboard(self):
        if self.df_ev.empty: return
        
        current_data = self.df_ev.iloc[self.current_wafer_index]
        wafer_name = str(current_data.get('wafer_names', f'W_{self.current_wafer_index}'))
        current_time = current_data.get('Time', 'Unknown')
        wafer_num = current_data.get('wafer_num', 0)
        
        group_code = str(assign_group(wafer_num))
        
        params = FIXED_PARAMS_BY_GROUP.get(group_code, FIXED_PARAMS_BY_GROUP["Default"])
        
        target_window = params["window"]
        target_z_th = params["z_th"]
        target_min_run = params["min_run"]

        # OES / RFM Wafer-level Check
        if self.current_wafer_name != wafer_name:
            detect_oes = self.oes_analyzer.predict(wafer_name)
            if detect_oes:
                if wafer_name not in self.oes_detected_list: self.oes_detected_list.append(wafer_name)
                if hasattr(self, 'label_31'): 
                    self.label_31.setText("\n".join(self.oes_detected_list[-10:]))
                    self.label_31.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
            elif not self.oes_detected_list and hasattr(self, 'label_31'):
                self.label_31.setText("OES: 정상")
                self.label_31.setStyleSheet("color: #2ecc71; font-size: 11px;")

            detect_rfm = self.rfm_analyzer.predict(wafer_name)
            if detect_rfm:
                if wafer_name not in self.rfm_detected_list: self.rfm_detected_list.append(wafer_name)
                if hasattr(self, 'label_32'):
                    self.label_32.setText("\n".join(self.rfm_detected_list[-10:]))
                    self.label_32.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
            elif not self.rfm_detected_list and hasattr(self, 'label_32'):
                self.label_32.setText("RFM: 정상")
                self.label_32.setStyleSheet("color: #2ecc71; font-size: 11px;")

            self.current_wafer_name = wafer_name

        self.processed_wafers.add(wafer_name)
        total_prod = len(self.processed_wafers)
        
        # 1. RUL Updates (Sensor Data)
        if self.rul_ready and hasattr(self, 'label_33'):
            rul, st, r2 = self.rul_analyzer.get_rul_status(total_prod)
            c = "#2ecc71" if st=="Normal" else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else 'Inf'
            
            # [수정] HTML을 적용하여 RUL 숫자만 26px로 키움
            self.label_33.setText(f"""
                <html><head/><body><p align='center'>
                <span style="font-size:14px;">Status: {st}</span><br>
                <span style="font-size:26px; font-weight:bold;">RUL: {rul_val}</span><br>
                <span style="font-size:12px;">Reliability: {r2*100:.1f}%</span>
                </p></body></html>
            """)
            # 기본 색상은 여기서 설정 (글자 크기는 위 HTML이 덮어씀)
            self.label_33.setStyleSheet(f"color: {c}; font-weight: bold;")
            
        # 2. OES RUL Updates
        if self.oes_rul_ready and hasattr(self, 'label_34'):
            st, rul, r2 = self.oes_rul_predictor.predict(total_prod)
            c = "#2ecc71" if st in ["Normal", "Stable"] else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else 'Stable'
            
            # [수정] HTML 적용
            self.label_34.setText(f"""
                <html><head/><body><p align='center'>
                <span style="font-size:14px;">Status: {st}</span><br>
                <span style="font-size:26px; font-weight:bold;">RUL: {rul_val}</span><br>
                <span style="font-size:12px;">Reliability: {r2*100:.1f}%</span>
                </p></body></html>
            """)
            self.label_34.setStyleSheet(f"color: {c}; font-weight: bold;")

        # 3. RFM RUL Updates
        if self.rfm_rul_ready and hasattr(self, 'label_35'):
            st, rul, r2 = self.rfm_rul_predictor.predict(total_prod)
            c = "#2ecc71" if st in ["Normal", "Healthy"] else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else '>2000'
            
            # [수정] HTML 적용
            self.label_35.setText(f"""
                <html><head/><body><p align='center'>
                <span style="font-size:14px;">Status: {st}</span><br>
                <span style="font-size:26px; font-weight:bold;">RUL: {rul_val}</span><br>
                <span style="font-size:12px;">Reliability: {r2*100:.1f}%</span>
                </p></body></html>
            """)
            self.label_35.setStyleSheet(f"color: {c}; font-weight: bold;")

        
        
        alarm_status = self.alarm_system.check(current_data, wafer_num)
        
        
        
        z = 0.0 
        
        if self.lstm_model is not None and self.feature_cols:
            try:
                # 3-1. 현재 데이터를 버퍼에 추가 (Raw Data)
                raw_feat = current_data[self.feature_cols].values 
                self.window_buffer.append(raw_feat)

                # 3-2. 버퍼가 목표 윈도우 크기(target_window)보다 커지면 가장 오래된 데이터 제거
                if len(self.window_buffer) > target_window:
                    self.window_buffer.pop(0)
                
                # 3-3. 버퍼가 윈도우 크기만큼 꽉 찼을 때만 예측 수행
                if len(self.window_buffer) == target_window:
                    # (Window, Features) 형태로 변환
                    window_data = np.array(self.window_buffer) 
                    
                    # 스케일링 수행 (scaler는 2D 입력을 기대하므로 형태 유지)
                    window_scaled = self.lstm_scaler.transform(window_data)
                    
                    # Tensor 변환: (Batch=1, Window, Features) 형태여야 함
                    t_in = torch.FloatTensor(window_scaled).unsqueeze(0)
                    
                    with torch.no_grad():
                        t_out = self.lstm_model(t_in)
                    
                    # 3-4. [핵심 변경] MAE(절대오차) -> MSE(제곱오차)로 변경 (레퍼런스 스크립트와 동일)
                    loss = np.mean((t_in.numpy() - t_out.numpy())**2)
                    
                    # Z-score 계산 (기존 lstm_mu, lstm_std 사용)
                    z = (loss - self.lstm_mu) / (self.lstm_std + 1e-9)
                    
                    # 3-5. [핵심 변경] 고정 임계치(self.z_threshold) -> 그룹별 임계치(target_z_th) 사용
                    if z > target_z_th: 
                        self.lstm_run_count += 1
                    else: 
                        self.lstm_run_count = 0
                    
                    # 3-6. [핵심 변경] 고정 min_run -> 그룹별 min_run(target_min_run) 사용
                    if self.lstm_run_count >= target_min_run:
                        if not self.lstm_defect_list or self.lstm_defect_list[-1] != wafer_name:
                            self.lstm_defect_list.append(wafer_name)
                            self.lstm_time_list.append(str(current_time))
                            # 심각한 불량(1.5배 초과)인 경우 Set에 추가하여 수율 계산에 반영
                            if z > target_z_th * 1.5: 
                                self.lstm_detected_set.add(wafer_name)
            except Exception as e:
                # 에러 발생 시 콘솔에 출력 (디버깅용)
                print(f"LSTM Prediction Error: {e}")

        # 3. Critical Alarm Log Logic
        if alarm_status == 'Critical':
            if not self.lstm_defect_list or self.lstm_defect_list[-1] != wafer_name:
                self.lstm_defect_list.append(wafer_name)
                self.lstm_time_list.append(str(current_time))
                self.defect_wafer_set.add(wafer_name)

        # 4. Update Log Labels
        if hasattr(self, 'label_29'): self.label_29.setText("\n".join(self.lstm_defect_list[-10:][::-1]))
        if hasattr(self, 'label_30'): self.label_30.setText("\n".join(self.lstm_time_list[-10:][::-1]))

        # 5. Counters
        if alarm_status == 'Critical': self.cnt_critical += 1
        elif alarm_status == 'Warning': self.cnt_warning += 1
        elif alarm_status == 'Caution': self.cnt_caution += 1
        else: self.cnt_normal += 1
            
        combined_defects = self.defect_wafer_set.union(self.lstm_detected_set)
        yield_rate = 100.0 if total_prod == 0 else ((total_prod - len(combined_defects)) / total_prod) * 100
        
        if hasattr(self, 'label_12'): self.label_12.setText(f"{total_prod}")
        if hasattr(self, 'label_22'): self.label_22.setText(f"{yield_rate:.1f}%")
        if hasattr(self, 'label_23'): self.label_23.setText(f"{len(combined_defects)}")
        
        if hasattr(self, 'label_26'): self.label_26.setText(f"{self.cnt_critical}")
        if hasattr(self, 'label_27'): self.label_27.setText(f"{self.cnt_warning}")
        if hasattr(self, 'label_28'): self.label_28.setText(f"{self.cnt_caution}")

        # 6. Gauges & Graphs
        tgt = self.target_col
        val = current_data.get(tgt, 0)
        st = self.stats.get(tgt, {'mean':0, 'std':1})
        if hasattr(self, 'progressBar'): self.update_gauge_style(self.progressBar, val, st['mean'], st['std'])
        if hasattr(self, 'label_24'): self.label_24.setText(f"{val:.1f}")
        
        val_rf = current_data.get(self.rf_col, 0)
        st_rf = self.stats.get(self.rf_col, {'mean':0, 'std':1})
        if hasattr(self, 'progressBar_2'): self.update_gauge_style(self.progressBar_2, val_rf, st_rf['mean'], st_rf['std'])
        if hasattr(self, 'label_25'): self.label_25.setText(f"{val_rf:.1f}")

        if hasattr(self, 'ax_sensor'):
            self.ax_sensor.clear()
            s_idx = max(0, self.current_wafer_index - 30)
            recent = self.df_ev.iloc[s_idx:self.current_wafer_index+1]
            if not recent.empty:
                self.ax_sensor.plot(recent.index, recent[tgt], color='#8e44ad', linewidth=2)
                self.ax_sensor.set_title(f"Real-time {tgt}", fontsize=8)
            self.canvas_sensor.draw()
        
        if hasattr(self, 'ax_fdc') and self.lstm_model:
            self.ax_fdc.clear()
            s_idx = max(0, self.current_wafer_index - 50)
            recent_f = self.df_ev.iloc[s_idx:self.current_wafer_index+1][self.feature_cols].values
            if len(recent_f) > 0:
                try:
                    v = self.lstm_scaler.transform(recent_f)
                    t_in = torch.FloatTensor(v).unsqueeze(0)
                    with torch.no_grad(): t_out = self.lstm_model(t_in)
                    err = np.mean(np.abs(t_in.numpy()[0] - t_out.numpy()[0]), axis=1)
                    self.ax_fdc.plot(err, color='#e74c3c')
                    self.ax_fdc.set_title("Anomaly Score", fontsize=8)
                except: pass
            self.canvas_fdc.draw()
            
        self.current_wafer_index = (self.current_wafer_index + 1) % len(self.df_ev)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KPIPage()
    window.show()
    sys.exit(app.exec())