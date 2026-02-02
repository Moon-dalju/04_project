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
                    iso = IsolationForest(contamination=BEST_CONTAM, random_state=42)
                    data = sub_df[[col]].fillna(0).values
                    iso.fit(data)
                    self.models[col][g] = iso
                else:
                    self.models[col][g] = None
        self.trained = True

    def check(self, row_series, wafer_num):
        if not self.trained: return "Normal", [] 
        
        # --- 기존의 스텝/웨이퍼 변경 감지 로직 ---
        current_step = row_series.get('Step Number', 0)
        if self.last_wafer_num != wafer_num:
            self.last_wafer_num = wafer_num
            self.last_step = current_step
            self.step_stable_count = 0
            for col in self.buffers: self.buffers[col] = []
        elif self.last_step != current_step:
            self.last_step = current_step
            self.step_stable_count = 0
        else:
            self.step_stable_count += 1
            
        is_stabilizing = self.step_stable_count < 15 # 가드 타임
        # -----------------------------------------------------

        group = assign_group(wafer_num)
        
        critical_sensors = []
        warning_sensors = []
        
        for col in self.sensor_cols:
            if col not in row_series: continue
            val = row_series[col]
            model = self.models[col].get(group)
            
            is_out = 0
            if model is not None:
                try:
                    pred = model.predict([[val]])[0]
                    is_out = 1 if pred == -1 else 0
                except: is_out = 0
            
            # [수정됨] 안정화 기간만 예외 처리하고, Calibration 필터는 삭제함
            if is_stabilizing: is_out = 0
            
            # 삭제된 부분:
            # if 'fault_name' in row_series and 'calibration' in ...: is_out = 0 

            win_size = TCP_WIN if col in TCP_VARIABLES else BEST_WIN
            self.buffers[col].append(is_out)
            if len(self.buffers[col]) > win_size: self.buffers[col].pop(0)
            
            buf = self.buffers[col]
            buf_len = len(buf)
            buf_sum = sum(buf)
            
            if buf_len >= win_size and buf_sum == win_size:
                critical_sensors.append(col)
            elif buf_len >= self.warn_win and sum(buf[-self.warn_win:]) >= self.warn_win:
                warning_sensors.append(col)
        
        final_status = "Normal"
        problem_sensors = []
        
        if critical_sensors:
            final_status = "Critical"
            problem_sensors = list(set(critical_sensors))
        elif warning_sensors:
            final_status = "Warning"
            problem_sensors = list(set(warning_sensors))

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
        
        
        self.defect_wafer_set = set()   # Critical
        self.warning_wafer_set = set()  # Warning
        self.caution_wafer_set = set()  # Caution
        
        self.cnt_normal = 0
        # self.cnt_warning, self.cnt_critical 등은 이제 Set 길이로 대체하므로 필요 없지만
        # 에러 방지를 위해 변수 선언은 남겨둡니다.
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
        # ---------------------------------------------------------
        # 1. 콤보박스 스타일 설정 (기존 유지)
        # ---------------------------------------------------------
        if hasattr(self, 'comboBox'):
            self.combo_sensor = self.comboBox
            self.combo_sensor.setStyleSheet("""
                QComboBox { 
                    color: black; background-color: white; 
                    border: 1px solid gray; padding: 1px 18px 1px 3px; 
                }
                QComboBox QAbstractItemView { 
                    color: black; background-color: white; 
                    selection-background-color: lightgray; 
                }
            """)
            try: self.combo_sensor.currentTextChanged.disconnect()
            except: pass
            self.combo_sensor.currentTextChanged.connect(self.on_sensor_changed)
        
        # ---------------------------------------------------------
        # 2. 라벨 초기화 (기존 유지)
        # ---------------------------------------------------------
        for name in ['label_31', 'label_32', 'label_33', 'label_34', 'label_35']:
            if hasattr(self, name): getattr(self, name).setText("...")
        
        # ---------------------------------------------------------
        # 3. [핵심] 테이블 설정
        # ---------------------------------------------------------
        from PySide6.QtWidgets import QTableWidget, QHeaderView, QAbstractItemView

        if hasattr(self, 'tableWidget'):
            self.log_table = self.tableWidget
        else:
            self.log_table = QTableWidget()
            if hasattr(self, 'frame_50') and self.frame_50.layout():
                self.frame_50.layout().addWidget(self.log_table)
            elif self.layout():
                self.layout().addWidget(self.log_table)

        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["Time", "Wafer", "Status", "Issue Sensor"])
        
        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # [수정] 헤더 표시 및 행 높이 조절
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.horizontalHeader().setVisible(True) # 헤더 보이게
        self.log_table.verticalHeader().setDefaultSectionSize(35) # 행 높이

        # [수정] 스타일 개선 (글자 키움, 헤더 색상) + [[말풍선(ToolTip) 스타일 추가]]
        self.log_table.setStyleSheet("""
            /* 테이블 기본 스타일 */
            QTableWidget {
                font-size: 16px; 
                font-weight: bold; 
                gridline-color: #dcdcdc;
            }
            /* 헤더 스타일 */
            QHeaderView::section { 
                background-color: #2c3e50; 
                color: white;
                font-size: 16px; 
                font-weight: bold;
                border: 1px solid #dcdcdc; 
                padding: 4px;
            }
            /* [추가된 부분] 말풍선(Tooltip) 스타일: 흰 배경에 검은 글씨 */
            QToolTip {
                background-color: #ffffff; 
                color: #000000; 
                border: 1px solid #000000;
                font-size: 14px;
                font-weight: normal;
            }
        """)
        
        # [추가] 더블 클릭 시 Ack(확인) 처리
        self.log_table.cellDoubleClicked.connect(self.on_alarm_ack)

        # ---------------------------------------------------------
        # 4. 글자 크기 일괄 적용 (기존 유지)
        # ---------------------------------------------------------
        big_number_style = "font-size: 28px; font-weight: bold; color: #2c3e50;"
        if hasattr(self, 'label_24'): self.label_24.setStyleSheet(big_number_style)
        if hasattr(self, 'label_25'): self.label_25.setStyleSheet(big_number_style)

        stat_style = "font-size: 20px; font-weight: bold; color: #34495e;"
        for name in ['label_12', 'label_22', 'label_23', 'label_26', 'label_27', 'label_28']:
            if hasattr(self, name): getattr(self, name).setStyleSheet(stat_style)

        rul_style = "font-size: 15px; font-weight: bold;"
        for name in ['label_33', 'label_34', 'label_35']:
            if hasattr(self, name): getattr(self, name).setStyleSheet(rul_style)

        log_style = "font-size: 13px; font-weight: bold; color: #555555;"
        for name in ['label_31', 'label_32']:
            if hasattr(self, name): getattr(self, name).setStyleSheet(log_style)

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
        
        # 1. 데이터 로드
        current_data = self.df_ev.iloc[self.current_wafer_index]
        wafer_name = str(current_data.get('wafer_names', f'W_{self.current_wafer_index}'))
        current_time = current_data.get('Time', 'Unknown')
        wafer_num = current_data.get('wafer_num', 0)
        
        group_code = str(assign_group(wafer_num))
        params = FIXED_PARAMS_BY_GROUP.get(group_code, FIXED_PARAMS_BY_GROUP["Default"])
        target_window, target_z_th, target_min_run = params["window"], params["z_th"], params["min_run"]

        # 2. OES / RFM 체크
        if self.current_wafer_name != wafer_name:
            detect_oes = self.oes_analyzer.predict(wafer_name)
            if detect_oes:
                if wafer_name not in self.oes_detected_list: self.oes_detected_list.append(wafer_name)
                if hasattr(self, 'label_31'): 
                    self.label_31.setText("\n".join(self.oes_detected_list[-10:]))
                    self.label_31.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
            elif not self.oes_detected_list and hasattr(self, 'label_31'):
                self.label_31.setText("OES: 정상"); self.label_31.setStyleSheet("color: #2ecc71; font-size: 11px;")
            
            detect_rfm = self.rfm_analyzer.predict(wafer_name)
            if detect_rfm:
                if wafer_name not in self.rfm_detected_list: self.rfm_detected_list.append(wafer_name)
                if hasattr(self, 'label_32'):
                    self.label_32.setText("\n".join(self.rfm_detected_list[-10:]))
                    self.label_32.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
            elif not self.rfm_detected_list and hasattr(self, 'label_32'):
                self.label_32.setText("RFM: 정상"); self.label_32.setStyleSheet("color: #2ecc71; font-size: 11px;")

            self.current_wafer_name = wafer_name

        self.processed_wafers.add(wafer_name)
        total_prod = len(self.processed_wafers)
        
        # 3. RUL Updates
        if self.rul_ready and hasattr(self, 'label_33'):
            rul, st, r2 = self.rul_analyzer.get_rul_status(total_prod)
            c = "#2ecc71" if st=="Normal" else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else 'Inf'
            self.label_33.setText(f"<html><head/><body><p align='center'><span style='font-size:14px;'>Status: {st}</span><br><span style='font-size:26px; font-weight:bold;'>RUL: {rul_val}</span><br><span style='font-size:12px;'>Reliability: {r2*100:.1f}%</span></p></body></html>")
            self.label_33.setStyleSheet(f"color: {c}; font-weight: bold;")
            
        if self.oes_rul_ready and hasattr(self, 'label_34'):
            st, rul, r2 = self.oes_rul_predictor.predict(total_prod)
            c = "#2ecc71" if st in ["Normal", "Stable"] else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else 'Stable'
            self.label_34.setText(f"<html><head/><body><p align='center'><span style='font-size:14px;'>Status: {st}</span><br><span style='font-size:26px; font-weight:bold;'>RUL: {rul_val}</span><br><span style='font-size:12px;'>Reliability: {r2*100:.1f}%</span></p></body></html>")
            self.label_34.setStyleSheet(f"color: {c}; font-weight: bold;")

        if self.rfm_rul_ready and hasattr(self, 'label_35'):
            st, rul, r2 = self.rfm_rul_predictor.predict(total_prod)
            c = "#2ecc71" if st in ["Normal", "Healthy"] else ("#f39c12" if st=="Warning" else "#e74c3c")
            rul_val = int(rul) if rul < 2000 else '>2000'
            self.label_35.setText(f"<html><head/><body><p align='center'><span style='font-size:14px;'>Status: {st}</span><br><span style='font-size:26px; font-weight:bold;'>RUL: {rul_val}</span><br><span style='font-size:12px;'>Reliability: {r2*100:.1f}%</span></p></body></html>")
            self.label_35.setStyleSheet(f"color: {c}; font-weight: bold;")

        # 4. 알람 체크 및 LSTM
        alarm_status, failed_sensors = self.alarm_system.check(current_data, wafer_num)
        
        z = 0.0
        if self.lstm_model is not None and self.feature_cols:
            try:
                raw_feat = current_data[self.feature_cols].values 
                self.window_buffer.append(raw_feat)
                if len(self.window_buffer) > target_window: self.window_buffer.pop(0)
                
                if len(self.window_buffer) == target_window:
                    window_data = np.array(self.window_buffer) 
                    t_in = torch.FloatTensor(self.lstm_scaler.transform(window_data)).unsqueeze(0)
                    with torch.no_grad(): t_out = self.lstm_model(t_in)
                    loss = np.mean((t_in.numpy() - t_out.numpy())**2)
                    z = (loss - self.lstm_mu) / (self.lstm_std + 1e-9)
                    if z > target_z_th: self.lstm_run_count += 1
                    else: self.lstm_run_count = 0
                    if self.lstm_run_count >= target_min_run:
                        if not self.lstm_defect_list or self.lstm_defect_list[-1] != wafer_name:
                            self.lstm_defect_list.append(wafer_name)
                            self.lstm_time_list.append(str(current_time))
                            if z > target_z_th * 1.5: self.lstm_detected_set.add(wafer_name)
            except Exception as e: print(f"LSTM Prediction Error: {e}")

        # 5. [수정됨] 로그 테이블 스마트 업데이트 (상태 생애 주기 및 전체 툴팁 적용)
        if alarm_status in ["Critical", "Warning"] and hasattr(self, 'log_table'):
            from PySide6.QtGui import QColor, QBrush
            def get_colors(status):
                if status == "Critical": return QColor(255, 200, 200), QColor("red")
                return QColor(255, 230, 200), QColor(255, 140, 0)

            current_bg, current_fg = get_colors(alarm_status)
            is_same_wafer = False
            
            # [기존 줄 갱신 로직]
            if self.log_table.rowCount() > 0:
                last_wafer = self.log_table.item(0, 1).text()
                if last_wafer == wafer_name:
                    is_same_wafer = True
                    
                    # (1) Time 툴팁 업데이트 (Ongoing 상태 유지)
                    time_item = self.log_table.item(0, 0)
                    start_time = time_item.text() # 테이블에 적힌 건 시작 시간
                    
                    tooltip_text = (f"Start Time: {start_time}\n"
                                    f"Last Seen:  {current_time}\n"
                                    f"Status:     Ongoing... 🔥")
                    time_item.setToolTip(tooltip_text)
                    
                    # (2) 상태 업데이트 (악화 시)
                    old_status = self.log_table.item(0, 2).text()
                    if "Warning" in old_status and alarm_status == "Critical":
                        # Ack 상태가 아닐 때만 텍스트 변경
                        if "(Ack)" not in old_status:
                            self.log_table.item(0, 2).setText("Critical")
                            self.log_table.item(0, 2).setToolTip("Critical") # 툴팁 갱신
                            current_bg, current_fg = get_colors("Critical")
                    
                    # (3) 센서 정보 병합
                    old_sensors = self.log_table.item(0, 3).text().split(", ")
                    if old_sensors == ["-"]: old_sensors = []
                    merged = sorted(list(set(old_sensors + failed_sensors)))
                    new_str = ", ".join(merged) if merged else "-"
                    self.log_table.item(0, 3).setText(new_str)
                    self.log_table.item(0, 3).setToolTip(new_str) # 툴팁 갱신
                    
                    # (4) 색상 재적용 (Ack 안 된 경우만)
                    if self.log_table.item(0, 0).background().color().name() != "#eeeeee":
                        for c in range(4):
                            self.log_table.item(0, c).setBackground(QBrush(current_bg))
                            self.log_table.item(0, c).setForeground(QBrush(current_fg))

            # [새 줄 추가 로직 (이전 줄은 Finished 처리)]
            if not is_same_wafer:
                from PySide6.QtWidgets import QTableWidgetItem
                
                # (1) 이전 줄 상태 변경: Ongoing -> Finished
                if self.log_table.rowCount() > 0:
                    prev_time_item = self.log_table.item(0, 0)
                    prev_tooltip = prev_time_item.toolTip()
                    # 이미 Ack된 상태가 아니라면 Finished로 변경
                    if "Acknowledged" not in prev_tooltip:
                        new_tooltip = prev_tooltip.replace("Ongoing...", "Finished ✅")
                        new_tooltip = new_tooltip.replace("Status:     ", "Status:     ") 
                        prev_time_item.setToolTip(new_tooltip)

                # (2) 새 줄 추가
                self.log_table.insertRow(0)
                
                s_str = ", ".join(failed_sensors) if failed_sensors else "-"
                items = [QTableWidgetItem(str(current_time)), QTableWidgetItem(str(wafer_name)), 
                         QTableWidgetItem(alarm_status), QTableWidgetItem(s_str)]
                
                # (3) 초기 툴팁 설정
                items[0].setToolTip(f"Start Time: {current_time}\nLast Seen:  {current_time}\nStatus:     Ongoing... 🔥")
                items[2].setToolTip(alarm_status)
                items[3].setToolTip(s_str)
                
                for i, item in enumerate(items):
                    item.setBackground(QBrush(current_bg))
                    item.setForeground(QBrush(current_fg))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.log_table.setItem(0, i, item)
                
                if self.log_table.rowCount() > 100: self.log_table.removeRow(100)

        # 6. 카운터 & 수율 업데이트 (Set 기반)
        if alarm_status == 'Critical': self.defect_wafer_set.add(wafer_name)
        elif alarm_status == 'Warning': self.warning_wafer_set.add(wafer_name)
        elif alarm_status == 'Caution': self.caution_wafer_set.add(wafer_name)
        else: self.cnt_normal += 1
            
        combined_defects = self.defect_wafer_set.union(self.lstm_detected_set)
        yield_rate = 100.0 if total_prod == 0 else ((total_prod - len(combined_defects)) / total_prod) * 100
        
        if hasattr(self, 'label_12'): self.label_12.setText(f"{total_prod}")
        if hasattr(self, 'label_22'): self.label_22.setText(f"{yield_rate:.1f}%")
        if hasattr(self, 'label_23'): self.label_23.setText(f"{len(combined_defects)}")
        
        if hasattr(self, 'label_26'): self.label_26.setText(f"{len(self.defect_wafer_set)}") 
        if hasattr(self, 'label_27'): self.label_27.setText(f"{len(self.warning_wafer_set)}")
        if hasattr(self, 'label_28'): self.label_28.setText(f"{len(self.caution_wafer_set)}")

        # 7. 그래프 등
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
    
    def on_alarm_ack(self, row, col):
        """
        현업 기능: 알람 확인(Acknowledge)
        테이블 행을 더블 클릭하면 '확인됨(Checked)' 상태로 변경하고 색상을 회색으로 바꿈.
        또한, 툴팁의 상태도 'Acknowledged'로 변경함.
        """
        from PySide6.QtGui import QColor, QBrush
        
        # 이미 확인된 상태인지 체크 (배경색이 회색인지)
        current_bg = self.log_table.item(row, 0).background().color()
        ack_color = QColor("#eeeeee") # 연한 회색
        
        if current_bg == ack_color: return 
            
        # 해당 행의 모든 컬럼 색상을 회색으로 변경 (처리 완료 표시)
        ack_brush = QBrush(ack_color)
        text_brush = QBrush(QColor("#999999")) # 글자도 흐리게
        
        # 1. Status 텍스트 및 툴팁 변경
        status_item = self.log_table.item(row, 2)
        old_text = status_item.text()
        
        if "(Ack)" not in old_text:
            new_text = f"{old_text}(Ack)"
            status_item.setText(new_text)
            status_item.setToolTip(new_text) # 툴팁도 업데이트
            
        # 2. Time 툴팁 상태 변경 (Ongoing/Finished -> Acknowledged)
        time_item = self.log_table.item(row, 0)
        current_tooltip = time_item.toolTip()
        
        if "Status:" in current_tooltip:
            # 기존 시간 정보(Start/Last)는 유지하고 상태만 교체
            base_info = current_tooltip.split("Status:")[0]
            new_tooltip = f"{base_info}Status:     Acknowledged 👮"
            time_item.setToolTip(new_tooltip)
            
        # 3. 색상 적용
        for c in range(4):
            item = self.log_table.item(row, c)
            item.setBackground(ack_brush)
            item.setForeground(text_brush)
            
        print(f"Engineer Acknowledged Alarm at Row {row}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KPIPage()
    window.show()
    sys.exit(app.exec())