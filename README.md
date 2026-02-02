# 🚀 Semi-Conductor QA/QC Data Analysis Dashboard

반도체 제조 공정 데이터(FDC, OES, RFM)를 활용하여 설비의 상태를 모니터링하고 품질을 관리하는 PySide6 기반 대시보드 애플리케이션입니다.

## 🛠️ 주요 기능
- **통계 분석 (Statistics)**: Wafer 그룹별(Main/Over/Low) 센서 데이터 시각화 및 자동 CPK 계산
- **결함 탐지 (Fault Detection)**: Isolation Forest 알고리즘을 활용한 공정 이상치 감지
- **머신러닝 예측 (ML Prediction)**: 학습된 모델을 통한 데이터 분류 및 설비 건전성(Health Index) 분석
- **실시간 로그 (KPI Log)**: 공정 이슈 및 이상 감지 내역 실시간 리스트업 및 히스토리 관리

## 📦 설치 및 실행 방법
1. **저장소 클론 및 이동**
   ```bash
   git clone [https://github.com/Moon-dalju/04_project.git](https://github.com/Moon-dalju/04_project.git)
   cd 04_project
