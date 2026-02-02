import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import PySide6.QtWidgets as QtW
from PySide6.QtCore import QFile, QIODevice, QPoint, Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QWidget, QMessageBox,
    QLabel, QLineEdit,
    QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QPlainTextEdit, QTextBrowser,
    QGroupBox, QVBoxLayout, QSizePolicy, QGridLayout, QHBoxLayout
)
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# =========================================================
# Settings
# =========================================================
UI_FILENAME = "ml.ui"
CSV_FILENAME = "ev_data.csv"
MODEL_DIRNAME = "saved_models"

# ✅ 여기서부터 핵심: "이름 매칭"으로만 채우기
ENABLE_GEOMETRY_FALLBACK_FOR_MEDIANS = False  # ← 무조건 False (대충 채우기 금지)

# 스킵할 라벨 텍스트(너 UI에 'TextLabel' 같은 게 섞여있어서)
SKIP_LABEL_TEXTS = {"TextLabel"}


# =========================
# wafer -> group
# =========================
def extract_wafer_num(text: str) -> int:
    nums = re.findall(r"\d+", str(text))
    return int(nums[0]) if nums else 0


def assign_group_by_wafer_num(num: int) -> str:
    if 2901 <= num <= 2943:
        return "Main"
    if 3101 <= num <= 3143:
        return "Over"
    if 3301 <= num <= 3343:
        return "Low"
    return "Main"


# =========================
# Robust artifact loader
# =========================
def _peek_file(path: Path, nbytes: int = 32) -> tuple[int, bytes]:
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(nbytes)
    return size, head


def load_artifact_auto(path: Path):
    """
    pkl 로드: pickle -> joblib -> torch 순으로 시도
    """
    size, head = _peek_file(path, 32)

    # 1) pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e_pickle:
        # 2) joblib
        try:
            import joblib  # type: ignore
            return joblib.load(path)
        except Exception as e_joblib:
            # 3) torch
            try:
                import torch  # type: ignore
                return torch.load(str(path), map_location="cpu")
            except Exception as e_torch:
                raise RuntimeError(
                    f"[{path.name}] 로드 실패\n"
                    f"- size={size} bytes\n"
                    f"- head={head!r}\n"
                    f"- pickle: {e_pickle}\n"
                    f"- joblib: {e_joblib}\n"
                    f"- torch : {e_torch}\n"
                    f"※ 저장 포맷 불일치(joblib/torch/pickle) 또는 파일 손상 가능"
                )


# =========================
# Model Hub
# =========================
class EVModelHub:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / MODEL_DIRNAME
        self.binary = {}
        self.multi = {}
        self.load_errors = []
        self._load_all()

    def _load_one(self, path: Path):
        return load_artifact_auto(path)

    def _load_all(self):
        print(f"\n[ML] model_dir={self.model_dir} | exists={self.model_dir.exists()}")
        if not self.model_dir.exists():
            self.load_errors.append(f"saved_models 폴더 없음: {self.model_dir}")
            return

        for g in ["Main", "Over", "Low"]:
            b_path = self.model_dir / f"binary_{g}.pkl"
            m_path = self.model_dir / f"multi_{g}.pkl"

            if b_path.exists():
                try:
                    self.binary[g] = self._load_one(b_path)
                    print(f"✅ loaded {b_path.name}")
                except Exception as e:
                    msg = f"❌ fail {b_path.name}\n{e}"
                    print(msg)
                    self.load_errors.append(msg)
            else:
                self.load_errors.append(f"missing: {b_path.name}")

            if m_path.exists():
                try:
                    self.multi[g] = self._load_one(m_path)
                    print(f"✅ loaded {m_path.name}")
                except Exception as e:
                    msg = f"❌ fail {m_path.name}\n{e}"
                    print(msg)
                    self.load_errors.append(msg)
            else:
                self.load_errors.append(f"missing: {m_path.name}")

        print(f"[ML] loaded binary={list(self.binary.keys())} | multi={list(self.multi.keys())}")

    def is_ready(self) -> bool:
        return bool(self.binary) and bool(self.multi)

    def _fallback_group(self, dct: dict, want: str) -> str:
        if want in dct:
            return want
        if dct:
            fb = list(dct.keys())[0]
            print(f"⚠️ {want} 모델이 없어 {fb}로 대체")
            return fb
        raise RuntimeError("로드된 모델이 없습니다. saved_models pkl을 확인하세요.")

    def _payload_to_model_feats(self, payload):
        if not isinstance(payload, dict):
            raise RuntimeError(f"payload가 dict가 아닙니다: {type(payload)}")

        model = payload.get("model", None)
        if model is None:
            raise RuntimeError(f"payload에 model 키 없음. keys={list(payload.keys())}")

        feats = payload.get("features", payload.get("feature_names", None))
        if feats is None:
            raise RuntimeError(f"payload에 features 키 없음. keys={list(payload.keys())}")

        return model, list(feats)

    def predict_binary(self, group: str, xdict: dict):
        group = self._fallback_group(self.binary, group)
        payload = self.binary[group]
        model, feats = self._payload_to_model_feats(payload)
        label_meaning = payload.get("label_meaning", {0: "calibration", 1: "others"})

        X = pd.DataFrame([[xdict.get(f, 0.0) for f in feats]], columns=feats)
        pred = int(model.predict(X)[0])

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0].tolist()
            except Exception:
                proba = None

        return pred, label_meaning.get(pred, str(pred)), proba

    def predict_multi(self, group: str, xdict: dict):
        group = self._fallback_group(self.multi, group)
        payload = self.multi[group]
        model, feats = self._payload_to_model_feats(payload)

        X = pd.DataFrame([[xdict.get(f, 0.0) for f in feats]], columns=feats)
        pred_enc = model.predict(X)[0]

        le = payload.get("label_encoder", None)
        if le is not None:
            try:
                pred_enc_i = int(pred_enc)
                pred_label = le.inverse_transform([pred_enc_i])[0]
                pred_enc = pred_enc_i
            except Exception:
                pred_label = str(pred_enc)
        else:
            pred_label = str(pred_enc)

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0].tolist()
            except Exception:
                proba = None

        try:
            pred_enc_i = int(pred_enc)
        except Exception:
            pred_enc_i = -1

        return pred_enc_i, str(pred_label), proba


# =========================
# MLPage
# =========================
class MLPage(QWidget):
    def __init__(self, get_group_cb=None):
        super().__init__()
        self.get_group_cb = get_group_cb

        self.base_dir = Path(__file__).resolve().parent
        os.chdir(self.base_dir)

        self.ui_path = self.base_dir / UI_FILENAME
        self.csv_path = self.base_dir / CSV_FILENAME

        print(f"[ML] base_dir={self.base_dir}")
        print(f"[ML] ui_path={self.ui_path} | exists={self.ui_path.exists()}")
        print(f"[ML] csv_path={self.csv_path} | exists={self.csv_path.exists()}")

        self._load_ui(str(self.ui_path))
        self._ensure_designer_pushbutton()
        self._connect_hint_button()   # ✅ pushButton 클릭 -> 방향성 힌트 출력


        # ✅ 핵심 요구사항: frame_3 : frame_4 = 1 : 1 (창 늘리면 정확히 반반)
        # - Designer가 안 먹어도 실행 시 강제
        QTimer.singleShot(0, self._force_frame2_equal_split)
        QTimer.singleShot(0, self._force_frame10_11_equal_split)

        # (기존) 출력칸 4개 강제 생성/확보
        self._force_output_boxes()
        self._style_output_groupboxes()
        self._init_action_box()

        self.label_to_input = self._build_label_input_map()
        print(f"[ML] label->input mapped: {len(self.label_to_input)}")
        if self.label_to_input:
            print("[ML] label keys preview:", list(self.label_to_input.keys())[:12])

        self.output_slots = [
            self.findChild(QTextBrowser, "out_truefalse"),
            self.findChild(QTextBrowser, "out_faulttype"),
            self.findChild(QTextBrowser, "out_trueprob"),
            self.findChild(QTextBrowser, "out_faultprob"),
        ]

        if any(w is None for w in self.output_slots):
            missing = []
            names = ["out_truefalse", "out_faulttype", "out_trueprob", "out_faultprob"]
            for n, w in zip(names, self.output_slots):
                if w is None:
                    missing.append(n)
            raise RuntimeError(
                f"❌ 출력 QTextBrowser를 찾지 못했습니다: {missing}\n"
                f"→ ml.ui에서 objectName 확인하세요."
            )

        print("[ML] output_slots fixed:",
              [(type(w).__name__, w.objectName()) for w in self.output_slots])

        self.hub = EVModelHub(self.base_dir)
        self._build_good_bad_profiles()
        


        # ✅ 중앙값 채우기(정확 이름 매칭)
        self.fill_inputs_with_csv_medians()

        self._connect_predict_button()

        if not self.hub.is_ready():
            self._write_outputs_4(
                "모델 미로드",
                "모델 미로드",
                "N/A",
                "saved_models 로드 실패. 터미널 로그 확인"
            )

        self.FEATURE_WIDGET_MAP = {
              "Time": "Time",
              "Step Number": "StepNumber",

              "BCl3 Flow": "BCl3Flow",
              "Cl2 Flow": "Cl2Flow",

              "RF Btm Pwr": "RFBtmPwr",
              "RF Btm Rfl Pwr": "RFBtmRflPwr",
              "Endpt A": "EndptA",
              "He Press": "HePress",
              "Pressure": "Pressure",

              "RF Tuner": "RFTuner",
              "RF Load": "RFLoad",
              "RF Phase Err": "RFPhaseErr",
              "RF Pwr": "RFPwr",
              "RF Impedance": "RFImpedance",

              "TCP Tuner": "TCPTuner",
              "TCP Phase Err": "TCPPhaseErr",
              "TCP Impedance": "TCPImpedance",
              "TCP Top Pwr": "TCPTopPwr",
              "TCP Rfl Pwr": "TCPRflPwr",
              "TCP Load": "TCPLoad",

              "Vat Valve": "VatValve"
              
        }
        self.debug_check_feature_widgets()
        

    
    def _install_group_combo_on_titlebar(self):
        # 상단 우측(타이틀바 쪽)에 콤보박스 만들기
        self.cb_group = QtW.QComboBox(self)
        self.cb_group.setObjectName("cb_group_runtime")
        self.cb_group.setMinimumWidth(180)

        # 표시 텍스트, 내부 data(실제 그룹 값)
        self.cb_group.addItem("전체", "ALL")
        self.cb_group.addItem("2900 (Main)", "Main")
        self.cb_group.addItem("3100 (Over)", "Over")
        self.cb_group.addItem("3300 (Low)", "Low")

        # 🔑 핵심: 최상단 Form의 레이아웃을 잡는다
        root_layout = self.layout()
        if root_layout is None:
            print("[UI] root layout not found")
            return

        # 최상단 frame(타이틀바 영역)
        header = self.findChild(QtW.QFrame, "frame")
        if header is None:
           print("[UI] header frame not found")
           return

        # header를 포함하고 있는 layout 찾기
        parent_layout = header.parentWidget().layout()
        if parent_layout is None:
           print("[UI] parent layout not found")
           return

        # header row를 HBox로 재구성
        hbox = QHBoxLayout()
        hbox.setContentsMargins(12, 6, 12, 6)

        # 기존 header 제거 후 다시 삽입
        parent_layout.removeWidget(header)
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        hbox.addWidget(header)
        hbox.addStretch(1)
        hbox.addWidget(self.cb_group)

        parent_layout.insertLayout(0, hbox)

        self.cb_group.currentIndexChanged.connect(
            lambda _: print(
                f"[UI] group changed -> {self.cb_group.currentText()} ({self.cb_group.currentData()})"
            )
        )

        print("[UI] ✅ group combo installed on top title bar")

        


    def _get_selected_group(self) -> str:
        if callable(getattr(self, "get_group_cb", None)):
            try:
                v = self.get_group_cb()
                return (v or "ALL")
            except Exception:
                return "ALL"
        return "ALL"
    
    def _resolve_group(self, raw_group: str) -> str:
        """
        ✅ 최종 group 결정 규칙
        1) raw_group이 Main/Over/Low면 그대로
        2) raw_group이 ALL/AUTO/전체면 wafer_names로 자동 판정
        3) 그래도 못하면 Main fallback
        """
        g = (raw_group or "").strip()

        if g in ("Main", "Over", "Low"):
            return g

        # ALL/AUTO 처리
        wafer_text = (self._get_wafer_text() or "").strip()
        num = extract_wafer_num(wafer_text)

        if num:
            return assign_group_by_wafer_num(num)

        return "Main"



    
    def _build_good_bad_profiles(self):
        """
        group(Main/Over/Low) 별로:
        - good_profile: calibration 중앙값
        - bad_profile : others 중앙값
        - fault_profiles: fault_name별 중앙값
        """
        self.good_profile = {}
        self.bad_profile = {}
        self.fault_profiles = {}

        if not self.csv_path.exists():
            print("[HINT] CSV not found -> profiles skipped")
            return

        df = pd.read_csv(self.csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        # numeric cols만
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            print("[HINT] no numeric cols -> profiles skipped")
            return

        # group 만들기
        if "wafer_names" in df.columns:
            df["wafer_num"] = df["wafer_names"].apply(extract_wafer_num)
            df["group"] = df["wafer_num"].apply(assign_group_by_wafer_num)
        else:
            df["group"] = "Main"

        # label 만들기
        if "fault_name" in df.columns:
            df["fault_clean"] = df["fault_name"].astype(str).str.strip().str.lower()
            df["target_binary"] = (df["fault_clean"] != "calibration").astype(int)
        else:
            print("[HINT] fault_name column missing -> profiles skipped")
            return

        for g in ["Main", "Over", "Low"]:
            sub = df[df["group"] == g].copy()
            if sub.empty:
                continue

            good = sub[sub["target_binary"] == 0]
            bad  = sub[sub["target_binary"] == 1]

            if not good.empty:
                self.good_profile[g] = good[num_cols].median(numeric_only=True).to_dict()
            if not bad.empty:
                self.bad_profile[g] = bad[num_cols].median(numeric_only=True).to_dict()

            # fault_name별 profile
            fp = {}
            for fname, sdf in bad.groupby("fault_clean"):
                if sdf.empty:
                    continue
                fp[fname] = sdf[num_cols].median(numeric_only=True).to_dict()
            self.fault_profiles[g] = fp

        print("[HINT] ✅ good/bad/fault profiles built:",
              "good=", list(self.good_profile.keys()),
              "bad=", list(self.bad_profile.keys()))




    def _connect_hint_button(self):
        btn = self.findChild(QPushButton, "pushButton")
        if btn is None:
            print("[UI] hint pushButton not found (objectName=pushButton)")
            return

        try:
           btn.clicked.disconnect()
        except Exception:
           pass

        btn.clicked.connect(self.show_direction_hints)
        print("[UI] ✅ hint pushButton connected -> show_direction_hints()")


    def show_direction_hints(self):
        if getattr(self, "action_box", None) is None:
            print("[UI] action_box not ready; cannot show hints")
            return

        html = self.generate_direction_hint_html()
        self.action_box.setHtml(html)


    def generate_direction_hint_html(self) -> str:
        if getattr(self, "action_box", None) is None:
            return "<div class='title'>조치 방향 힌트</div><div>action_box 없음</div>"

        raw_group = self._get_selected_group()      # ALL / Main / Over / Low (상단 필터 그대로)
        group = self._resolve_group(raw_group)      # Main / Over / Low (프로파일 계산용)

        # 화면 표시용 그룹
        display_group = "ALL" if raw_group in ("ALL", "AUTO", "전체") else group


        good = self.good_profile.get(group, None)
        bad_others = self.bad_profile.get(group, None)
        fault_dict = self.fault_profiles.get(group, {})

        if not good:
            return """
            <div class="title">조치 방향 힌트</div>
            <div style="color:#666;">
             양품(calibration) 프로파일이 없습니다.<br>
             (ev_data.csv의 fault_name / wafer_names / numeric cols 확인)
            </div>
            """

        # ✅ 현재 예측 결과에서 “Top-1 fault”를 읽어오고 싶으면:
        #   - 가장 최근 예측된 m_label을 저장해두는 변수를 run_prediction에서 set 해두면 깔끔함.
        # 여기서는 self.last_fault_label (있으면) 사용
        top_fault = (getattr(self, "last_fault_label", "") or "").strip().lower()

        # bad 기준 선택: fault_profile > others_profile
        bad = fault_dict.get(top_fault, bad_others)

        cols = list(getattr(self, "FEATURE_WIDGET_MAP", {}).keys())
        if not cols:
            return "<div class='title'>조치 방향 힌트</div><div>FEATURE_WIDGET_MAP 비어있음</div>"


        # 차이가 거의 없으면 ≈ 처리 (불량-양품 차이가 5% 이하면 영향 약함)
        TOL_PCT = 0.02
        TOL_ABS = 1e-9

        WEAK_PCT = 0.05

        def _to_float(v):
            try:
                return float(v)
            except Exception:
                return np.nan

        def fmt(v):
            v = _to_float(v)
            if np.isnan(v):
                return "N/A"
            # 큰 값은 정수처럼, 작은 값은 소수
            if abs(v) >= 100:
                return f"{v:.0f}"
            if abs(v) >= 1:
                return f"{v:.3f}"
            return f"{v:.6f}"

        def fmt_delta(d):
            d = _to_float(d)
            if np.isnan(d):
                return "N/A"
            sign = "+" if d >= 0 else ""
            if abs(d) >= 100:
                return f"{sign}{d:.0f}"
            if abs(d) >= 1:
                return f"{sign}{d:.3f}"
            return f"{sign}{d:.6f}"

        items_html = ""
        for col in cols:
            wname = self.FEATURE_WIDGET_MAP.get(col, col)
            w = (self.findChild(QDoubleSpinBox, wname)
                 or self.findChild(QSpinBox, wname)
                 or self.findChild(QLineEdit, wname))
            cur = _to_float(self._read_numeric_from_widget(w))
            gv = _to_float(good.get(col, np.nan))
            bv = _to_float(bad.get(col, np.nan)) if isinstance(bad, dict) else np.nan

            if np.isnan(gv) or np.isnan(cur):
                items_html += f"""
                <li><b>{col}</b> : <b>판단불가</b>
                  <div style='color:#666; margin-left:10px;'>• 양품 목표 또는 현재값이 없습니다.</div>
                </li>
                """
                continue

            # (1) 목표(good)로 얼마나 움직여야 하나
            delta = gv - cur  # +면 올려야(증가), -면 내려야(감소)

            # (2) 유지 판정(목표와 거의 같으면)
            denom = max(abs(gv), 1e-12)
            close_enough = (abs(delta) <= TOL_ABS) or (abs(delta) / denom <= TOL_PCT)

            # (3) 이 변수가 “의미 있게” 양품/불량이 갈리는지(약하면 그냥 유지 권장)
            weak = False
            if not np.isnan(bv):
                denom2 = max(abs(gv), abs(bv), 1e-12)
                if abs(bv - gv) / denom2 <= WEAK_PCT:
                    weak = True

            if close_enough:
                action = "유지 권장"
                action_color = "#555555"
                detail = f"현재가 양품 목표와 거의 동일 (조정 불필요)"
                move = "0"
            else:
                if delta > 0:
                    action = "올리는 거 권장"
                    action_color = "#D32F2F"   # 🔴 빨간색
                else:
                    action = "내리는 거 권장"
                    action_color = "#1976D2"   # 🔵 파란색
                move = fmt_delta(delta)
                detail = f"양품 목표({fmt(gv)})로 맞추려면 현재({fmt(cur)})에서 {move} 만큼 조정"

            # 약한 변수면 경고(“건드릴 우선순위 낮음”)
            weak_note = ""
            if weak:
                weak_note = "<div style='color:#888; margin-left:10px;'>• (참고) 양품/불량 대표값 차이가 작아 우선순위는 낮을 수 있음</div>"

            # “양품/불량/현재”는 ‘|’ 대신 문장으로
            items_html += f"""
            <li>
             <b>{col}</b> :
             <b style="color:{action_color};">{action}</b>
             <div style='color:#444; margin-top:4px; margin-left:10px;'>
              • 현재 {fmt(cur)} → 양품 목표 {fmt(gv)} (조정량 {move})
             </div>
             {weak_note}
            </li>
            """

        bad_title = "불량(others)"
        if top_fault and isinstance(fault_dict, dict) and top_fault in fault_dict:
            bad_title = f"불량({top_fault})"

        return f"""
        <div class="title">조치 방향 힌트</div>

        <div class="section">[현재 기준]</div>
        <ul>
         <li>Group: <b>{display_group}</b></li>
         <li>목표(양품): <b>calibration 대표값</b></li>
         <li>참고(불량): <b>{bad_title}</b></li>
        </ul>

        <div class="section">[규칙]</div>
        <ul>
         <li><b>올리는 거 권장</b> : 현재 &lt; 양품 목표 → 목표까지 +조정</li>
         <li><b>내리는 거 권장</b> : 현재 &gt; 양품 목표 → 목표까지 -조정</li>
         <li><b>유지 권장</b> : 현재 ≈ 양품 목표 (오차 허용 범위 내)</li>
        </ul>

        <div class="section">[변수별 권장 조치]</div>
        <ul>{items_html}</ul>
        """





    def _init_action_box(self):
        """
        frame_10 (하단 큰 흰 박스)에 조치라인 출력용 QTextBrowser를 확실히 생성/부착
        """
        f10 = self.findChild(QtW.QFrame, "frame_10")
        if f10 is None:
            print("[UI] frame_10 not found -> action box skip")
            self.action_box = None
            return

        tb = f10.findChild(QTextBrowser, "action_textbox")
        if tb is None:
            tb = QTextBrowser(f10)
            tb.setObjectName("action_textbox")
            tb.setOpenExternalLinks(False)
            tb.setReadOnly(True)

            lay = f10.layout()
            if lay is None:
                lay = QVBoxLayout(f10)
                lay.setContentsMargins(14, 14, 14, 14)
                lay.setSpacing(8)
            lay.addWidget(tb)

        # ✅ 여기만 조절하면 됨 (픽셀 기준)
        ACTION_FONT_PX = 20
        TITLE_FONT_PX  = 34   # "현업 조치라인" 같은 타이틀

       # 1) QTextBrowser 자체 폰트 (픽셀로 강제)
        font = tb.font()
        font.setPixelSize(ACTION_FONT_PX)
        font.setBold(False)
        tb.setFont(font)
        tb.document().setDefaultFont(font)

        # 2) HTML 렌더링에도 먹히도록 default stylesheet를 박아버림
        tb.document().setDefaultStyleSheet(f"""
            html, body, div, p, span, li, ul, ol {{
                font-size: {ACTION_FONT_PX}px;
                line-height: 1.7;
            }}
            ul, ol {{
                margin-left: 18px;
            }}
            li {{
                margin: 6px 0px;
            }}
            .title {{
                font-size: {TITLE_FONT_PX}px;
                font-weight: 900;
                margin-bottom: 10px;
            }}
            .section {{
                font-size: {ACTION_FONT_PX + 2}px;
                font-weight: 900;
                margin-top: 12px;
            }}
        """)

        # 3) 박스 스타일 (✅ f-string으로 치환되게)
        tb.setStyleSheet(f"""
            QTextBrowser {{
                background: white;
                border: 1px solid #E5E5E5;
                border-radius: 10px;
                padding: 14px;
            }}
        """)

        self.action_box = tb

        # 초기 안내 문구도 class로 관리 (font-size inline 제거해도 커짐)
        self.action_box.setHtml(f"""
            <div class="title">현업 조치라인</div>
            <div style="color:#666;">
             좌측 입력값을 확인한 뒤 <b>예측</b>을 누르면,<br>
             결함유형 확률 기반 점검 순서가 여기에 자동으로 출력됩니다.
            </div>
        """)

        print("[UI] ✅ action_textbox ready in frame_10")

        



    def _map_fault_to_system(self, fault_label: str) -> str:
        s = (fault_label or "").strip().lower()

        # calibration/정상
        if "calibration" in s or "정상" in s:
            return "Calibration"

        # RF 계통
        if s.startswith("rf") or "rf " in s or "rf_" in s:
            return "RF"

        # TCP 계통
        if s.startswith("tcp") or "tcp " in s or "tcp_" in s:
            return "TCP"

        # Gas/Pressure/He 계통 (키워드 기반)
        if "bcl3" in s or "cl2" in s or "gas" in s:
            return "Gas"
        if "pressure" in s or "press" in s:
            return "Pressure"
        if "he" in s:
            return "He"

        return "Etc"


    def generate_action_template(self, p_true: float, top_items: list[tuple[str, float]]) -> str:
        """
        p_true: 진성(others) 확률 (%)  e.g. 99.21
        top_items: [(fault_label, prob_percent), ...]  e.g. [("RF -12", 49.1), ("TCP +50", 45.2), ...]
        return: HTML 문자열
        """

        # 기본 가드
        if not top_items:
            return f"""
            <div style="font-size:1.35em; font-weight:900;">[상황 요약]</div>            <ul>
             <li>진성(이상) 확률: <b>{p_true:.2f}%</b></li>
             <li>결함 유형 확률 정보가 없습니다. (predict_proba 미지원/실패)</li>
            </ul>
            """

        # Top-1/2
        t1_lab, t1_p = top_items[0]
        t2_lab, t2_p = top_items[1] if len(top_items) >= 2 else ("-", 0.0)

        sys1 = self._map_fault_to_system(t1_lab)
        sys2 = self._map_fault_to_system(t2_lab) if t2_lab != "-" else None

        diff = abs(t1_p - t2_p)
        ambiguous = (len(top_items) >= 2 and diff <= 10.0)  # ✅ 애매 기준(10% 이내) - 너가 조정 가능

        # 점검 리스트(계통별 기본 템플릿)
        def steps_for(system: str) -> list[str]:
            if system == "RF":
                return [
                    "RF Load / RF Power 안정성 확인 (드리프트/튐 여부)",
                    "RF Phase Error 변동 여부 점검",
                    "RF Impedance 이상 여부 확인",
                    "필요 시 RF 계통 재튜닝 후 Recipe 재적용"
                ]
            if system == "TCP":
                return [
                    "TCP Tuner / TCP Load 편차 확인",
                    "TCP Top/Rfl Power 변동 폭 확인",
                    "TCP Phase Error 이상 여부 점검",
                    "필요 시 TCP 계통 재튜닝 또는 매칭 상태 점검"
                ]
            if system == "Gas":
                return [
                    "BCl3 / Cl2 Flow 설정값-실측값 괴리 확인",
                    "가스 공급/밸브 응답 지연 여부 점검",
                    "Recipe 가스 step 전환 구간에서 불안정 여부 확인"
                ]
            if system == "Pressure":
                return [
                    "Pressure 안정화 구간에서 overshoot/진동 여부 확인",
                    "Vat Valve 동작 범위/응답 지연 점검",
                    "챔버 누설/압력 제어 루프 상태 확인"
                ]
            if system == "He":
                return [
                    "He Press 안정성 확인 (급격한 하강/상승)",
                    "He 라인/레귤레이터 상태 점검",
                    "웨이퍼 백사이드 냉각 조건 이슈 여부 확인"
                ]
            if system == "Calibration":
                return [
                    "현재는 정상(calibration) 가능성이 높음",
                    "센서 입력값/레시피 선택/웨이퍼 번호(group) 확인",
                    "재측정 후 동일하면 정상 처리"
                ]
            return [
                "Top 결함 유형의 계통 분류가 불명확합니다.",
                "Top-3 결함 라벨을 확인하고 수동 점검 항목을 지정하세요."
            ]

        # 우선 점검 계통 순서
        priority = []
        priority.append(sys1)
        if ambiguous and sys2 and sys2 != sys1:
            priority.append(sys2)

        # 출력 문자열 구성
        top_lines = "".join([f"<li>{lab} : <b>{p:.1f}%</b></li>" for lab, p in top_items[:3]])

        # 조치 step 합치기
        step_html = ""
        for idx, sysname in enumerate(priority, 1):
            step_list = steps_for(sysname)
            step_html += f"<div style='margin-top:10px; font-size:1.10em; font-weight:900;'>[{idx}] {sysname} 계통 권장 조치</div><ol>"
            for s in step_list:
                step_html += f"<li>{s}</li>"
            step_html += "</ol>"

        caution = ""
        if ambiguous and sys2:
            caution = f"""
            <div style="margin-top:10px; color:#444;">
              <b>[참고]</b> Top-1({t1_p:.1f}%)과 Top-2({t2_p:.1f}%) 차이가 <b>{diff:.1f}%</b>로 작아
              단일 원인보다 <b>복합 영향</b> 가능성이 있습니다. (RF/TCP 병행 점검 권장)
            </div>
            """

        html = f"""
        <div style="font-size:1.35em; font-weight:900;">현업 조치라인 </div>

        <div style="margin-top:10px; font-size:1.10em; font-weight:900;">[상황 요약]</div>
        <ul>
         <li>진성(이상) 확률: <b>{p_true:.2f}%</b></li>
         <li>주요 결함 유형 Top-3</li>
         <ul>{top_lines}</ul>
        </ul>

        <div style="margin-top:10px; font-size:1.10em; font-weight:900;">[우선 점검 계통]</div>
        <ul>
          {''.join([f'<li><b>{i+1}. {p}</b></li>' for i, p in enumerate(priority)])}
        </ul>

        <div style="margin-top:10px; font-size:1.10em; font-weight:900;">[권장 조치 (Step-by-step)]</div>
        {step_html}
        {caution}
        """
        return html




    
    def _ensure_designer_pushbutton(self):
        """
        Designer에 있는 objectName='pushButton'이 실행에서 안 보일 때:
        - 숨김 해제
        - 최소 크기 부여
        - frame_9(없으면 frame_8)에 레이아웃 만들어서 버튼을 '확실히' 넣음
        - 맨 위로 올림
        """
        btn = self.findChild(QPushButton, "pushButton")
        ref = self.findChild(QPushButton, "btn_predict")
        if btn is None or ref is None:
            print("[UI] pushButton or btn_predict not found in ui (objectName=pushButton)")
            return

        # =========================
        # 1️⃣ 공통 호스트(frame_9 우선)
        # =========================
        host = self.findChild(QtW.QFrame, "frame_9") \
               or self.findChild(QtW.QFrame, "frame_8") \
               or btn.parentWidget()

        if host.layout() is None:
            lay = QVBoxLayout(host)
            lay.setContentsMargins(12, 12, 12, 12)
            lay.setSpacing(8)
        else:
            lay = host.layout()

        # 2) 크기 강제 (0으로 눌리는 케이스 방지)
        btn.setMinimumHeight(ref.minimumHeight())
        btn.setMinimumWidth(ref.maximumHeight())
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        
        # 가로 꽉 차게
        if lay.indexOf(btn) < 0:
            btn.setParent(host)
            lay.addWidget(btn)
        lay.setStretchFactor(btn, 1)

        # =========================
        # 3️⃣ 검은색 스타일 적용
        # =========================
        btn.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: white;
                font-size: 16px;
                font-weight: 700;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #222222;
            }
            QPushButton:pressed {
                background-color: #000000;
            }
        """)

        # =========================
        # 4️⃣ 최종 강제 표시
        # =========================
        btn.setVisible(True)
        btn.setEnabled(True)
        btn.raise_()

        print("[UI] ✅ pushButton styled & expanded like btn_predict")
    

        






    def _find_common_ancestor(self, a: QtW.QWidget, b: QtW.QWidget):
        """a,b의 공통 부모(가장 가까운 공통 조상)"""
        pa = set()
        cur = a
        while cur is not None:
            pa.add(cur)
            cur = cur.parentWidget()
        cur = b
        while cur is not None:
            if cur in pa:
                return cur
            cur = cur.parentWidget()
        return None


    def _find_parent_with_both(self, a: QtW.QWidget, b: QtW.QWidget):
        """
        더 강한 버전:
        a의 부모를 위로 타고 올라가면서, 그 부모가 b를 자식으로 포함하는지 검사
        (common ancestor가 None이거나 layout이 없는 경우를 대비)
        """
        cur = a.parentWidget()
        while cur is not None:
            if cur.isAncestorOf(b):
                return cur
            cur = cur.parentWidget()
        return None

    # ---------------------------------------------------------
    # ✅ 1) frame_2 안에서 frame_3 / frame_4를 정확히 1:1로 강제
    # ---------------------------------------------------------
    def _force_frame2_equal_split(self):
        """
        목표: frame_3 : frame_4 = 1 : 1 (창 리사이즈 시에도 정확히 반반)

        전제(너 스샷 기준):
        - frame_2 = QFrame
        - frame_2.layout() = gridLayout_2 (QGridLayout)
        - frame_3, frame_4가 grid의 서로 다른 column에 들어있음
        """
        frame2 = self.findChild(QtW.QFrame, "frame_2")
        left = self.findChild(QtW.QFrame, "frame_3")
        right = self.findChild(QtW.QFrame, "frame_4")

        if frame2 is None or left is None or right is None:
            print("[UI] _force_frame2_equal_split skip (frame_2/frame_3/frame_4 not found)")
            return

        lay = frame2.layout()
        if lay is None:
            print("[UI] _force_frame2_equal_split skip (frame_2 has no layout)")
            return

        # 둘 다 Expanding 강제
        for w in (left, right):
            sp = w.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Expanding)
            sp.setVerticalPolicy(QSizePolicy.Expanding)
            w.setSizePolicy(sp)
            w.setMinimumWidth(0)
            w.setMaximumWidth(16777215)

        # ✅ GridLayout이면: left/right가 들어간 column을 찾아서 1:1
        if isinstance(lay, QGridLayout):
            col_left = None
            col_right = None

            for i in range(lay.count()):
                item = lay.itemAt(i)
                ww = item.widget()
                if ww is None:
                    continue
                r, c, rs, cs = lay.getItemPosition(i)
                if ww is left:
                    col_left = c
                elif ww is right:
                    col_right = c

            # fallback: 보통 (0,0),(0,1)
            if col_left is None:
                col_left = 0
            if col_right is None:
                col_right = 1 if col_left == 0 else 0

            lay.setColumnStretch(col_left, 1)
            lay.setColumnStretch(col_right, 1)

            # margin/spacing이 커서 “체감상 반반 아닌 것처럼” 보이면 여기 조정
            # lay.setContentsMargins(0, 0, 0, 0)
            # lay.setHorizontalSpacing(12)

            print(f"[UI] ✅ frame_2 grid columnStretch forced: col{col_left}=1, col{col_right}=1")

        # HBox인 경우도 대비
        elif isinstance(lay, QHBoxLayout):
            lay.setStretchFactor(left, 1)
            lay.setStretchFactor(right, 1)
            print("[UI] ✅ frame_2 HBox stretchFactor forced: 1:1")

        else:
            # generic fallback
            iL = lay.indexOf(left)
            iR = lay.indexOf(right)
            if iL >= 0:
                lay.setStretch(iL, 1)
            if iR >= 0:
                lay.setStretch(iR, 1)
            print("[UI] ✅ frame_2 generic stretch forced")

        frame2.updateGeometry()
        self.updateGeometry()

    # 창 리사이즈 때도 절대 안 깨지게(끝장 모드)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._force_frame2_equal_split()
        self._force_frame10_11_equal_split()
    # ---------------------------------------------------------
    # 출력 박스 강제 생성/확보 (기존 유지)
    # ---------------------------------------------------------
    def _force_output_boxes(self):
        """
        ✅ groupBox_3~6 안에 QTextBrowser 4개를 '강제로' 생성/확보
        """
        mapping = [
            ("groupBox_3", "out_truefalse"),
            ("groupBox_4", "out_trueprob"),
            ("groupBox_5", "out_faulttype"),
            ("groupBox_6", "out_faultprob"),
        ]

        created = []
        for gb_name, out_name in mapping:
            gb = self.findChild(QtW.QGroupBox, gb_name)
            if gb is None:
                raise RuntimeError(f"❌ {gb_name} 를 UI에서 못 찾음 (objectName 확인)")

            tb = gb.findChild(QTextBrowser, out_name)
            if tb is None:
                tb = QTextBrowser(gb)
                tb.setObjectName(out_name)
                tb.setReadOnly(True)

                lay = gb.layout()
                if lay is None:
                    lay = QVBoxLayout(gb)
                    lay.setContentsMargins(10, 25, 10, 10)
                    lay.setSpacing(0)

                center = QWidget(gb)
                center.setObjectName(out_name + "_center")
                c_lay = QVBoxLayout(center)
                c_lay.setContentsMargins(0, 0, 0, 0)
                c_lay.setSpacing(0)

                # ✅ 위/아래/좌/우 가운데
                c_lay.addWidget(tb, alignment=Qt.AlignCenter)

                lay.addWidget(center)
                created.append(out_name)

            tb.setOpenExternalLinks(False)
            tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        print("[ML] force_output_boxes ok. created:", created)

        chk = [self.findChild(QTextBrowser, n) for _, n in mapping]
        if any(w is None for w in chk):
            missing = [name for w, (_, name) in zip(chk, mapping) if w is None]
            raise RuntimeError(f"❌ force_output_boxes 이후에도 QTextBrowser 없음: {missing}")
    
    def _style_output_groupboxes(self):
        """
        groupBox_3~6 + 그 안의 QTextBrowser(out_*) 글씨를
        크게 + 진하게 보이도록 강제
        """
        # ✅ 너가 원하는 폰트 크기/굵기 (여기만 바꾸면 됨)
        TITLE_PX = 18     # groupBox 제목 크기
        BODY_PX  = 30     # 결과 텍스트 크기

        gbs = ["groupBox_3", "groupBox_4", "groupBox_5", "groupBox_6"]
        outs = ["out_truefalse", "out_trueprob", "out_faulttype", "out_faultprob"]

        # 1) groupBox 제목(타이틀) 스타일
        for name in gbs:
            gb = self.findChild(QGroupBox, name)
            if not gb:
                continue

            # groupBox 타이틀만 굵고 크게
            gb.setStyleSheet(f"""
                QGroupBox {{
                   font-size: {TITLE_PX}px;
                   font-weight: 800;
                }}
                QGroupBox::title {{
                   subcontrol-origin: margin;
                   left: 12px;
                   padding: 0 6px 0 6px;
                }}
            """)

        # QTextBrowser
        for out_name in outs:
            tb = self.findChild(QTextBrowser, out_name)
            if not tb:
                continue

            # ✅ 폰트 자체를 강제로 크게/굵게 (스타일 안 먹는 환경 대비)
            f = tb.font()
            f.setPointSize(BODY_PX)
            f.setBold(True)
            tb.setFont(f)




            tb.setStyleSheet(f"""
                QTextBrowser {{
                    font-size: {BODY_PX}px;
                    font-weight: 800;
                    background: transparent;
                    margin: 0px;
                    padding: 0px;
                    border: none; 
                }}
            """)
            tb.setFrameShape(QtW.QFrame.NoFrame)
            tb.setAlignment(Qt.AlignCenter)
        print("[UI] ✅ output groupBox_3~6 + out_* font styled")



    # ---------- UI loader ----------
    def _load_ui(self, ui_file_path: str):
        if not Path(ui_file_path).exists():
            raise RuntimeError(f"UI 파일이 없습니다: {ui_file_path}")

        loader = QUiLoader()
        f = QFile(ui_file_path)
        if not f.open(QIODevice.ReadOnly):
            raise RuntimeError(f"Cannot open UI: {ui_file_path} | {f.errorString()}")
        ui_widget = loader.load(f, None)
        f.close()

        if ui_widget is None:
            raise RuntimeError("UI load returned None")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(ui_widget)
        self.setLayout(lay)

        for w in ui_widget.findChildren(QWidget):
            if w.objectName():
                setattr(self, w.objectName(), w)

    def _connect_predict_button(self):
        btn = self.findChild(QPushButton, "btn_predict")
        if btn is None:
            print("[ML] bts_predict NOT FOUND (ml.ui objectName 확인)")
            return

        try:
            btn.clicked.disconnect()
        except Exception:
            pass

        btn.clicked.connect(self.run_prediction)
        print(f"[ML] predict button connected: {btn.objectName()}")

    # ---------- helpers ----------
    def _norm(self, s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("_", "")

    def _set_text(self, w: QWidget, text: str):
        if isinstance(w, QLineEdit):
            w.setText(text)
        elif isinstance(w, QLabel):
            w.setText(text)
        elif isinstance(w, QTextBrowser):
            safe = ("" if text is None else str(text))
            safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = safe.replace("\n", "<br>")

            html = f"""
            <div style="
                width:100%;
                height:100%;
                display:flex;
                align-items:center;      /* 세로 가운데 */
                justify-content:center;  /* 가로 가운데 */
                text-align:center;
                padding: 12px;
                box-sizing:border-box;
                line-height:1.4;
            ">
                <div>{safe}</div>
            </div>
            """
            w.setHtml(html)
            return 
            
        elif isinstance(w, QPlainTextEdit):
            w.setPlainText(text)
        elif isinstance(w, QTextEdit):
            w.setPlainText(text)
        else:
            try:
                w.setProperty("text", text)
            except Exception:
                pass

    def _write_outputs_4(self, s1: str, s2: str, s3: str, s4: str):
        arr = [s1, s2, s3, s4]
        if len(self.output_slots) >= 4:
            for i in range(4):
                self._set_text(self.output_slots[i], arr[i])
        else:
            QMessageBox.information(self, "AI 예측 결과", "\n\n".join(arr))

    # ---------- input mapping ----------
    def _build_label_input_map(self) -> dict:
        """
        QLabel 텍스트 -> 가장 가까운 입력 위젯 매핑
        ✅ geometry() 대신 mapToGlobal() 사용
        ✅ 왼쪽 입력 패널만 대상으로 필터링
        """
        labels = [l for l in self.findChildren(QLabel) if (l.text() or "").strip()]
        labels = [l for l in labels if (l.text() or "").strip() not in SKIP_LABEL_TEXTS]

        inputs = []
        inputs += self.findChildren(QDoubleSpinBox)
        inputs += self.findChildren(QSpinBox)
        inputs += self.findChildren(QLineEdit)

        if not labels or not inputs:
            return {}

        def gpos(w):
            p = w.mapToGlobal(QPoint(0, 0))
            return p.x(), p.y()

        label_xs = [gpos(l)[0] for l in labels]
        input_xs = [gpos(i)[0] for i in inputs]
        x_mid = float(np.median(label_xs + input_xs))

        left_labels = [l for l in labels if gpos(l)[0] < x_mid]

        label_to_input = {}
        for lab in left_labels:
            txt = (lab.text() or "").strip()
            lx, ly = gpos(lab)

            cand = []
            for w in inputs:
                wx, wy = gpos(w)
                if wy <= ly:
                    continue
                dx = abs(wx - lx)
                if dx > 420:
                    continue
                dy = wy - ly
                cand.append((dy, dx, w))

            if not cand:
                continue

            cand.sort(key=lambda t: (t[0], t[1]))
            label_to_input[txt] = cand[0][2]

        print(f"[DEBUG] label_to_input size = {len(label_to_input)}")
        print(f"[DEBUG] label_to_input keys = {list(label_to_input.keys())[:15]}")
        return label_to_input

    # =========================================================
    # ✅ 중앙값 채우기: 라벨 텍스트 ↔ CSV 컬럼명 "정확 매칭" 중심
    # =========================================================
    def fill_inputs_with_csv_medians(self):
        if not self.csv_path.exists():
            print("[ML] CSV not found. skip median fill")
            return

        df = pd.read_csv(self.csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            print("[ML] CSV numeric columns not found. skip median fill")
            return

        med = df[num_cols].median(numeric_only=True)
        # ✅ 나중에 힌트 버튼에서 "중앙값 대비 ↑/↓" 계산하려고 저장
        self.csv_medians = med.to_dict()
        self.csv_numeric_cols = list(num_cols)


        # (A) 1순위: objectName ↔ 컬럼명 매칭
        inputs = []
        inputs += self.findChildren(QDoubleSpinBox)
        inputs += self.findChildren(QSpinBox)
        inputs += self.findChildren(QLineEdit)

        name_to_w = {w.objectName(): w for w in inputs if w.objectName()}
        normname_to_w = {self._norm(w.objectName()): w for w in inputs if w.objectName()}

        filled_by_obj = []
        for col in num_cols:
            val = med.get(col, np.nan)
            if pd.isna(val):
                continue

            w = None
            if col in name_to_w:
                w = name_to_w[col]
            else:
                w = normname_to_w.get(self._norm(col), None)

            if w is None:
                continue

            try:
                if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                    w.setValue(float(val))
                elif isinstance(w, QLineEdit):
                    w.setText(f"{float(val):.6f}")
                else:
                    continue
                filled_by_obj.append((col, float(val), w.objectName()))
            except Exception:
                pass

        print(f"[ML][MEDIAN FILL][objectName] filled: {len(filled_by_obj)}")
        if filled_by_obj:
            for t in filled_by_obj[:12]:
                print(f"  • col={t[0]} median={t[1]:.6f} -> widget={t[2]}")

        # (B) 2순위: 라벨 텍스트 ↔ 컬럼명 매칭
        exact_cols = {c: c for c in num_cols}
        norm_cols = {self._norm(c): c for c in num_cols}

        filled = []
        skipped = []
        not_found = []

        for lab_txt, w in self.label_to_input.items():
            if lab_txt in SKIP_LABEL_TEXTS:
                skipped.append((lab_txt, "skip_label"))
                continue

            col = exact_cols.get(lab_txt, None)
            if col is None:
                col = norm_cols.get(self._norm(lab_txt), None)

            if col is None:
                not_found.append(lab_txt)
                continue

            val = med.get(col, np.nan)
            if pd.isna(val):
                skipped.append((lab_txt, f"median_nan({col})"))
                continue

            try:
                if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                    w.setValue(float(val))
                elif isinstance(w, QLineEdit):
                    w.setText(f"{float(val):.6f}")
                else:
                    skipped.append((lab_txt, f"unsupported_widget({type(w)})"))
                    continue

                filled.append((lab_txt, col, float(val)))
            except Exception as e:
                skipped.append((lab_txt, f"set_failed({e})"))

        print("\n[ML][MEDIAN FILL RESULT]")
        print(f"- filled: {len(filled)}")
        print(f"- not_found(label->col 미매칭): {len(not_found)}")
        print(f"- skipped: {len(skipped)}")

        if filled:
            print("  filled preview (label -> col = value):")
            for t in filled[:12]:
                print(f"   • {t[0]} -> {t[1]} = {t[2]:.6f}")

        if not_found:
            print("  not_found labels preview:")
            for t in not_found[:20]:
                print(f"   • {t}")

        if skipped:
            print("  skipped preview:")
            for t in skipped[:20]:
                print(f"   • {t[0]} : {t[1]}")

        if ENABLE_GEOMETRY_FALLBACK_FOR_MEDIANS:
            print("[ML] (WARN) geometry fallback is enabled, but this build intends exact matching.")

    # ---------- reading inputs ----------
    def _read_numeric_from_widget(self, w) -> float:
        if w is None:
            return 0.0
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            return float(w.value())
        if isinstance(w, QLineEdit):
            try:
                return float(str(w.text()).strip())
            except Exception:
                return 0.0
        return 0.0

    def _get_wafer_text(self) -> str:
        wn = self.findChild(QLineEdit, "wafer_names")
        if wn is not None:
            return wn.text()

        w = self.label_to_input.get("wafer_names", None)
        if isinstance(w, QLineEdit):
            return w.text()

        return ""

    def _build_xdict_for_features(self, features) :
        xdict = {}

        for f in features:
            nf = self._norm(f)
            w = None

            if hasattr(self, "FEATURE_WIDGET_MAP") and f in self.FEATURE_WIDGET_MAP:
                wname = self.FEATURE_WIDGET_MAP[f]
                w = (self.findChild(QDoubleSpinBox, wname)
                     or self.findChild(QSpinBox, wname)
                     or self.findChild(QLineEdit, wname))
            else:
                # fallback (기존 방식)
                w = (self.findChild(QDoubleSpinBox, f)
                     or self.findChild(QSpinBox, f)
                     or self.findChild(QLineEdit, f))

            val = self._read_numeric_from_widget(w)
            xdict[f] = val

        # 🔍 디버그
        print("[DEBUG] xdict(final):")
        for k, v in xdict.items():
            print(f"  {k} = {v}")

        return xdict

    # ---------- prediction ----------
    def run_prediction(self):
        print("🔥 run_prediction CALLED by bts_predict")
        try:
            if not self.hub.is_ready():
                err_text = "\n".join(self.hub.load_errors[-8:]) if self.hub.load_errors else "unknown"
                self._write_outputs_4(
                    "모델 로드 실패",
                    "모델 로드 실패",
                    "N/A",
                    f"saved_models 확인 필요\n{err_text}"
                )
                return

            raw_group = self._get_selected_group()  # ALL/Main/Over/Low

            if raw_group in ("Main", "Over", "Low"):
                group = raw_group
            else:
                # ALL이면 wafer_names로 자동 판정
                wafer_text = (self._get_wafer_text() or "").strip()
                num = extract_wafer_num(wafer_text)
                group = assign_group_by_wafer_num(num) if num else "Main"


            b_payload = self.hub.binary[self.hub._fallback_group(self.hub.binary, group)]
            m_payload = self.hub.multi[self.hub._fallback_group(self.hub.multi, group)]

            _, b_feats = self.hub._payload_to_model_feats(b_payload)
            _, m_feats = self.hub._payload_to_model_feats(m_payload)

            all_feats = list(dict.fromkeys(b_feats + m_feats))
            xdict = self._build_xdict_for_features(all_feats)

            b_cls, b_label, b_proba = self.hub.predict_binary(group, xdict)
            m_cls, m_label, m_proba = self.hub.predict_multi(group, xdict)

            # =========================
            # ① 진성/가성 예측 (binary)
            # =========================
            if str(b_label).strip().lower() == "others":
                out1 = "불량품 [others]"
            else:
                out1 = "양품 [calibration]"

            # =========================
            # ② 결함 유형 (multi) - 정상일 땐 보정
            # =========================
            if str(b_label).strip().lower() == "calibration":
                out2 = "정상 (결함 없음)"
            else:
                out2 = str(m_label)

            # =========================
            # ③ 진성확률 = P(others)
            # =========================
            if isinstance(b_proba, list) and len(b_proba) >= 2:
                try:
                    true_prob = float(b_proba[1]) * 100.0
                    out3 = f"{true_prob:.2f}%"
                except Exception:
                    out3 = "N/A"
            else:
                out3 = "N/A"

            # =========================
            # ④ 결함 유형 확률 Top3 (multi)
            # =========================
            if str(b_label).strip().lower() == "calibration":
                out4 = "N/A"
            else:
                out4 = "N/A"
                if isinstance(m_proba, list) and len(m_proba) > 0:
                    try:
                        arr = np.array(m_proba, dtype=float)

                        le = m_payload.get("label_encoder", None)
                        if le is not None and hasattr(le, "classes_"):
                            classes = list(le.classes_)
                        else:
                            classes = [f"class_{i}" for i in range(len(arr))]

                        top3 = arr.argsort()[::-1][:3]
                        out4 = "\n".join([f"{classes[i]} : {arr[i]*100:.1f}%" for i in top3])
                    except Exception:
                        out4 = "N/A"

            self._write_outputs_4(out1, out2, out3, out4)
            
                        # =========================
            # ✅ frame_10 조치라인 템플릿 출력
            # =========================
            try:
                # p_true (%)
                p_true = 0.0
                if isinstance(b_proba, list) and len(b_proba) >= 2:
                    p_true = float(b_proba[1]) * 100.0

                top_items = []
                if str(b_label).strip().lower() == "calibration":
                    # 정상일 때
                    top_items = [("calibration", 100.0)]
                else:
                    if isinstance(m_proba, list) and len(m_proba) > 0:
                        arr = np.array(m_proba, dtype=float)

                        le = m_payload.get("label_encoder", None)
                        if le is not None and hasattr(le, "classes_"):
                            classes = list(le.classes_)
                        else:
                            classes = [f"class_{i}" for i in range(len(arr))]

                        topk = arr.argsort()[::-1][:3]
                        top_items = [
                            (classes[i], float(arr[i]) * 100.0)
                            for i in topk
                        ]

                action_html = self.generate_action_template(
                    p_true=p_true,
                    top_items=top_items
                )

                if getattr(self, "action_box", None) is not None:
                    self.action_box.setHtml(action_html)
                else:
                    print("[UI] action_box not ready; skip setHtml")

            except Exception as e:
                print("[UI] action template failed:", e)


        

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._write_outputs_4(
                "예측 실패",
                "예측 실패",
                "N/A",
                str(e)
            )
        print(f"[PRED] sel={raw_group} -> group={group}")



    def debug_check_feature_widgets(self):
        print("\n[CHECK] FEATURE_WIDGET_MAP widget existence")
        missing = []

        for col, obj in self.FEATURE_WIDGET_MAP.items():
            w = (
                self.findChild(QDoubleSpinBox, obj)
                or self.findChild(QSpinBox, obj)
                or self.findChild(QLineEdit, obj)
            )

            if w is None:
                missing.append((col, obj))
            else:
                typ = type(w).__name__
                try:
                    val = w.value() if hasattr(w, "value") else w.text()
                except Exception:
                    val = "<?>"
                print(f"  OK  col='{col}' -> obj='{obj}' ({typ}) current={val}")

        if missing:
            print("\n  ❌ MISSING widgets:")
            for col, obj in missing:
                print(f"    - col='{col}' expects obj='{obj}'")

        print("")

    
    def _force_frame10_11_equal_split(self):
        """
        목표: frame_10 : frame_11 = 1 : 1
        - frame_11 = 4개 결과박스(2x2) 모여있는 컨테이너
        - frame_10 = 그 외 큰 영역 컨테이너
        - 공통 부모(layout) 찾아서 stretch를 1:1로 강제
        """
        f10 = self.findChild(QtW.QFrame, "frame_10")
        f11 = self.findChild(QtW.QFrame, "frame_11")

        if f10 is None or f11 is None:
            print("[UI] _force_frame10_11_equal_split skip (frame_10/frame_11 not found)")
            return

        # 1) 공통 조상 찾기
        parent = self._find_common_ancestor(f10, f11)

        # 2) 공통조상은 찾았는데 layout이 없으면, 더 강한 탐색(부모를 타고 올라가며)로 보정
        if parent is None or parent.layout() is None:
            parent2 = self._find_parent_with_both(f10, f11)
            if parent2 is not None and parent2.layout() is not None:
                parent = parent2

        if parent is None or parent.layout() is None:
            print("[UI] _force_frame10_11_equal_split skip (no common parent layout)")
            # 디버그: 각자의 parent chain 확인
            print("   - f10 chain:", self._debug_parent_chain(f10, limit=8))
            print("   - f11 chain:", self._debug_parent_chain(f11, limit=8))
            return

        lay = parent.layout()
        print(f"[UI] frame10/11 parent={parent.objectName()} layout={type(lay).__name__}")

        # Expanding 강제
        for w in (f10, f11):
            sp = w.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Expanding)
            sp.setVerticalPolicy(QSizePolicy.Expanding)
            w.setSizePolicy(sp)
            w.setMinimumWidth(0)
            w.setMinimumHeight(0)
            w.setMaximumWidth(16777215)
            w.setMaximumHeight(16777215)

        # VBox면 top/bottom stretch 1:1
        if isinstance(lay, QtW.QVBoxLayout):
            lay.setStretchFactor(f10, 1)
            lay.setStretchFactor(f11, 1)
            print("[UI] ✅ frame_10:frame_11 VBox stretch = 1:1")
            parent.updateGeometry()
            parent.adjustSize()
            return

        # HBox면 left/right stretch 1:1
        if isinstance(lay, QHBoxLayout):
            lay.setStretchFactor(f10, 1)
            lay.setStretchFactor(f11, 1)
            print("[UI] ✅ frame_10:frame_11 HBox stretch = 1:1")
            parent.updateGeometry()
            parent.adjustSize()
            return

        # Grid면 row/col 찾아서 1:1
        if isinstance(lay, QGridLayout):
            # f10/f11이 layout에 직접 들어있는지 확인
            pos = {}
            for i in range(lay.count()):
                it = lay.itemAt(i)
                ww = it.widget()
                if ww is None:
                    continue
                r, c, rs, cs = lay.getItemPosition(i)
                pos[ww] = (r, c, rs, cs)

            # 직접 매칭이 안 되면, "f10/f11을 포함하는 레이아웃 아이템"을 찾아서 처리
            def find_layout_item_widget(target):
                for i in range(lay.count()):
                    ww = lay.itemAt(i).widget()
                    if ww is None:
                        continue
                    if ww is target or ww.isAncestorOf(target):
                        return ww
                return None

            w10 = find_layout_item_widget(f10) or f10
            w11 = find_layout_item_widget(f11) or f11

            # 다시 위치 매핑(이번엔 w10/w11 기준)
            pos = {}
            for i in range(lay.count()):
                it = lay.itemAt(i)
                ww = it.widget()
                if ww is None:
                    continue
                r, c, rs, cs = lay.getItemPosition(i)
                pos[ww] = (r, c, rs, cs)

            r10, c10, _, _ = pos.get(w10, (None, None, None, None))
            r11, c11, _, _ = pos.get(w11, (None, None, None, None))

            # 세로(같은 col, 다른 row)
            if c10 is not None and c11 is not None and c10 == c11 and r10 is not None and r11 is not None:
                lay.setRowStretch(r10, 1)
                lay.setRowStretch(r11, 1)
                print(f"[UI] ✅ frame_10:frame_11 Grid rowStretch r{r10}=1, r{r11}=1")
                parent.updateGeometry()
                parent.adjustSize()
                return

            # 가로(같은 row, 다른 col)
            if r10 is not None and r11 is not None and r10 == r11 and c10 is not None and c11 is not None:
                lay.setColumnStretch(c10, 1)
                lay.setColumnStretch(c11, 1)
                print(f"[UI] ✅ frame_10:frame_11 Grid colStretch c{c10}=1, c{c11}=1")
                parent.updateGeometry()
                parent.adjustSize()
                return

            # fallback
            i10 = lay.indexOf(w10)
            i11 = lay.indexOf(w11)
            if i10 >= 0:
                lay.setStretch(i10, 1)
            if i11 >= 0:
                lay.setStretch(i11, 1)
            print("[UI] ✅ frame_10:frame_11 Grid generic stretch = 1:1")
            parent.updateGeometry()
            parent.adjustSize()
            return

        # 기타 레이아웃: index stretch
        try:
            i10 = lay.indexOf(f10)
            i11 = lay.indexOf(f11)
            if i10 >= 0:
                lay.setStretch(i10, 1)
            if i11 >= 0:
                lay.setStretch(i11, 1)
            print("[UI] ✅ frame_10:frame_11 generic stretch = 1:1")
        except Exception as e:
            print("[UI] frame_10:frame_11 generic stretch failed:", e)

        parent.updateGeometry()
        parent.adjustSize()




if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    w = MLPage()
    w.show()
    sys.exit(app.exec())
