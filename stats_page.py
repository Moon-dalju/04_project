import pandas as pd
import numpy as np
import re

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidgetItem, QHeaderView
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QPropertyAnimation, QRect

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.ensemble import IsolationForest

from stats_iso import run_pdm_model


def calc_auto_cpk(data, low=0.01, high=0.99):
    """
    Percentile-based Auto CPK
    low=0.01, high=0.99  → P1 ~ P99
    """
    if len(data) < 10:
        return np.nan

    mean = data.mean()
    std = data.std()
    if std == 0:
        return np.nan

    LSL = data.quantile(low)
    USL = data.quantile(high)

    cpk = min(
        (USL - mean) / (3 * std),
        (mean - LSL) / (3 * std)
    )
    return round(cpk, 3)




class StatsPage(QWidget):
    def __init__(self, wafer_groups):
        super().__init__()

        self.wafer_groups = wafer_groups
        self.current_filter = "ALL"
        self.stats_open = False
        self.latest_trend_masks = {}

        self.current_mean = None
        self.current_std = None

        self.load_ui()
        self.load_data()
        self.init_charts()
        self.init_dropdown()
        self.init_math_button()

        self.ui.statssummary.hide()

        self.anim = QPropertyAnimation(self.ui.statssummary, b"geometry")
        self.anim.setDuration(400)

        # ✅ Top10은 한 번만 계산
        self.update_summary()

    # ======================================================
    # UI
    # ======================================================
    def load_ui(self):
        loader = QUiLoader()
        file = QFile("stats.ui")
        file.open(QFile.ReadOnly)
        self.ui = loader.load(file, self)
        file.close()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ui)

        self.ui.comboVariable.setStyleSheet("QComboBox { color: black; }")
        self.ui.listTop10.setStyleSheet("QListWidget { color: black; }")

    # ======================================================
    # DATA
    # ======================================================
    def load_data(self):
        self.df = pd.read_csv("ev_data.csv")
        self.df["wafer_num"] = self.df["wafer_names"].apply(
            lambda x: int(re.findall(r"\d+", str(x))[0])
        )

        def assign_group(num):
            for k, (s, e) in self.wafer_groups.items():
                if s <= num <= e:
                    return k
            return "Others"

        self.df["group"] = self.df["wafer_num"].apply(assign_group)

        exclude = []
        self.target_cols = [c for c in self.df.columns[2:21] if c not in exclude]

    # ======================================================
    # FILTER
    # ======================================================
    def set_wafer_filter(self, group):
        self.current_filter = group
        self.update_summary()  # ✅ Top10은 필터 변경 시만
        self.update_all(self.ui.comboVariable.currentText())

    def get_filtered_df(self):
        if self.current_filter == "ALL":
            return self.df
        return self.df[self.df["group"] == self.current_filter]

    # ======================================================
    # DROPDOWN
    # ======================================================
    def init_dropdown(self):
        self.ui.comboVariable.addItems(self.target_cols)
        self.ui.comboVariable.currentTextChanged.connect(self.update_all)

    # ======================================================
    # CHART INIT
    # ======================================================
    def init_charts(self):
        self.fig_hist = Figure()
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.canvas_hist = FigureCanvas(self.fig_hist)
        QVBoxLayout(self.ui.histChart).addWidget(self.canvas_hist)

        self.fig_trend = Figure()
        self.ax_trend = self.fig_trend.add_subplot(111)
        self.canvas_trend = FigureCanvas(self.fig_trend)
        QVBoxLayout(self.ui.trendChart).addWidget(self.canvas_trend)

    # ======================================================
    # HISTOGRAM
    # ======================================================
    def draw_hist(self, col):
        df = self.get_filtered_df()
        data = df[col].dropna()
        self.ax_hist.clear()

        self.ax_hist.hist(
            data, bins=30, density=True,
            alpha=0.6, edgecolor="black", label="Histogram"
        )

        mean, std = data.mean(), data.std()
        if std > 0:
            x = np.linspace(data.min(), data.max(), 200)
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x - mean) / std) ** 2)
            self.ax_hist.plot(x, pdf, linewidth=2, label="Normal PDF")

        self.ax_hist.set_title(f"{col} Hist + Normal Curve")
        self.ax_hist.legend()
        self.ax_hist.grid(alpha=0.3)
        self.canvas_hist.draw()

    # ======================================================
    # TREND
    # ======================================================
    def draw_trend(self, col):
        self.ax_trend.clear()
        df = self.get_filtered_df()

        COLOR = {"caution": "#d4ac0d", "warning": "#e67e22", "critical": "#c0392b"}
        MARK = {"caution": "^", "warning": "s", "critical": "o"}

        normal_values = []

        for group, masks in self.latest_trend_masks.items():
            gdf = df[df["group"] == group].reset_index(drop=True)

            # ✅ trend 기본선: 검은색
            self.ax_trend.plot(
                gdf.index, gdf[col],
                color="#5dade2", linewidth=1.2
            )

            normal_mask = ~(
                masks["caution"] |
                masks["warning"] |
                masks["critical"]
            )
            normal_values.append(gdf.loc[normal_mask, col])

            for k in ["caution", "warning", "critical"]:
                idx = masks[k]
                if idx.any():
                    self.ax_trend.scatter(
                        gdf.index[idx],
                        gdf.loc[idx, col],
                        color=COLOR[k],
                        marker=MARK[k],
                        s=70,
                        edgecolors="black",
                        zorder=5
                    )

        # ===============================
        # 시그마 영역 계산 (정상 데이터)
        # ===============================
        trend_data = pd.concat(normal_values).dropna()

        if len(trend_data) >= 2:
            mean = trend_data.mean()
            std = trend_data.std()

            s1_u, s1_l = mean + 1*std, mean - 1*std
            s2_u, s2_l = mean + 2*std, mean - 2*std
            s25_u, s25_l = mean + 2.5*std, mean - 2.5*std

            # ===== 형광 영역 =====
            self.ax_trend.axhspan(s1_l, s1_u, color="#2ecc71", alpha=0.25, zorder=0)   # 초록
            self.ax_trend.axhspan(s2_l, s1_l, color="#f1c40f", alpha=0.25, zorder=0)   # 노랑
            self.ax_trend.axhspan(s1_u, s2_u, color="#f1c40f", alpha=0.25, zorder=0)

            self.ax_trend.axhspan(s25_l, s2_l, color="#e74c3c", alpha=0.25, zorder=0)  # 빨강
            self.ax_trend.axhspan(s2_u, s25_u, color="#e74c3c", alpha=0.25, zorder=0)

            # ===== 기준선 =====
            self.ax_trend.axhline(mean, color="black", linewidth=1.5)
            self.ax_trend.axhline(s25_u, color="red", linestyle="--", linewidth=1.2)
            self.ax_trend.axhline(s25_l, color="blue", linestyle="--", linewidth=1.2)

        # === CPK / UCL / LCL 계산 (정상 데이터 기준) ===
        normal_values = []

        for group, masks in self.latest_trend_masks.items():
            gdf = df[df["group"] == group].reset_index(drop=True)

            # 정상 데이터 = 어떤 결함에도 걸리지 않은 포인트
            normal_mask = ~(
                masks["caution"] |
                masks["warning"] |
                masks["critical"]
            )

            normal_values.append(gdf.loc[normal_mask, col])

        trend_data = pd.concat(normal_values).dropna()


        if len(trend_data) >= 2:
            mean = trend_data.mean()
            std = trend_data.std()

            ucl = mean + 2.5 * std
            lcl = mean - 2.5 * std

            # UCL / LCL 선
            self.ax_trend.axhline(ucl, color="red", linestyle="--", linewidth=1.5)
            self.ax_trend.axhline(lcl, color="blue", linestyle="--", linewidth=1.5)
            

        self.ax_trend.grid(alpha=0.3)
        self.canvas_trend.draw()

    # ======================================================
    # TREND SUMMARY
    # ======================================================
    def update_trend_summary(self):
        total = {"caution": 0, "warning": 0, "critical": 0}
        for masks in self.latest_trend_masks.values():
            for k in total:
                total[k] += int(masks[k].sum())

        self.ui.lblCPK.setText(f"""
        <span style="color:black;">
            <span style="color:#c0392b; font-weight:bold; font-size:22px;">●</span> 심각 {total["critical"]}<br>
            <span style="color:#e67e22; font-weight:bold;">■</span> 경고 {total["warning"]}<br>                   
            <span style="color:#d4ac0d; font-weight:bold;">▲</span> 주의 {total["caution"]}
        </span>
        """)

    def update_auto_cpk(self, col):
        df = self.get_filtered_df()

        # ===============================
        # 1) Overall CPK (결함 포함)
        # ===============================
        overall_data = df[col].dropna()
        overall_cpk = calc_auto_cpk(overall_data)

        # ===============================
        # 2) Normal CPK (정상 데이터만)
        # ===============================
        normal_values = []

        for group, masks in self.latest_trend_masks.items():
            gdf = df[df["group"] == group].reset_index(drop=True)

            normal_mask = ~(
                masks["caution"] |
                masks["warning"] |
                masks["critical"]
            )

            normal_values.append(gdf.loc[normal_mask, col])

        if not normal_values:
            self.ui.cpk.setText("Auto-CPK : -")
            return

        normal_data = pd.concat(normal_values).dropna()
        normal_cpk = calc_auto_cpk(normal_data)

        # ===============================
        # 3) UI 표시 (HTML + Tooltip)
        # ===============================
        if np.isnan(normal_cpk):
            self.ui.cpk.setText("Auto-CPK : -")
            return

        html = f"""
        <div style="color:black; font-size:22px; font-weight:bold;">
            {normal_cpk}
        </div>
        """

        if not np.isnan(overall_cpk):
            html += f"""
            <div style="color:gray; font-size:12px;">
                Overall {overall_cpk}
            </div>
            """
            self.ui.cpk.setToolTip("결함 포함")
        else:
            self.ui.cpk.setToolTip("")

        self.ui.cpk.setText(html)





    # ======================================================
    # TOP10 (전역 기준)
    # ======================================================
    def update_summary(self):
        scores = []
        for c in self.target_cols:
            iso = IsolationForest(contamination=0.12)
            scores.append((c, (iso.fit_predict(self.df[[c]]) == -1).sum()))

        scores.sort(key=lambda x: x[1], reverse=True)

        self.ui.listTop10.clear()
        for c, _ in scores[:10]:
            self.ui.listTop10.addItem(c)

    # ======================================================
    # STAT TABLE
    # ======================================================
    def show_stat_summary(self):
        df = self.get_filtered_df()
        desc = df[self.target_cols].describe().T
        table = self.ui.txtStats
        table.clear()

        table.setRowCount(len(desc))
        table.setColumnCount(len(desc.columns))
        table.setHorizontalHeaderLabels(desc.columns.tolist())
        table.setVerticalHeaderLabels(desc.index.tolist())

        for r, (_, row) in enumerate(desc.iterrows()):
            for c, val in enumerate(row):
                item = QTableWidgetItem(f"{val:.4f}")
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                item.setForeground(Qt.black)
                table.setItem(r, c, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.resizeRowsToContents()

    # ======================================================
    # MATH BUTTON
    # ======================================================
    def init_math_button(self):
        self.ui.mathbtn.clicked.connect(self.on_math_clicked)

    def on_math_clicked(self):
        self.show_stat_summary()
        self.ui.statssummary.show()
        self.ui.statssummary.raise_()
        self.ui.mathbtn.raise_()

        self.anim.stop()
        self.anim.setStartValue(self.ui.statssummary.geometry())
        self.anim.setEndValue(
            self.panel_open if not self.stats_open else self.panel_closed
        )
        self.anim.start()

        self.stats_open = not self.stats_open

    # ======================================================
    # PANEL POSITION
    # ======================================================
    def showEvent(self, event):
        super().showEvent(event)
        frame = self.ui.mainf
        panel = self.ui.statssummary

        self.panel_closed = QRect(0, frame.height(), frame.width(), frame.height())
        self.panel_open = QRect(0, 0, frame.width(), frame.height())
        panel.setGeometry(self.panel_closed)
        panel.hide()

    # ======================================================
    # UPDATE ALL
    # ======================================================
    def update_all(self, col):
        if not col:
            return

        df = self.get_filtered_df()
        self.latest_trend_masks.clear()

        for group in ["Main", "Over", "Low"]:
            gdf = df[df["group"] == group].reset_index(drop=True)
            if len(gdf) < 10:
                continue
            self.latest_trend_masks[group] = run_pdm_model(gdf, col)

        self.draw_hist(col)
        self.draw_trend(col)
        self.update_trend_summary()
        self.update_auto_cpk(col)
