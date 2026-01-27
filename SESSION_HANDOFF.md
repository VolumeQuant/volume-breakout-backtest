# Session Handoff - Volume Breakout Backtest Stage 3-1

**마지막 작업일**: 2026-01-27
**작업 환경**: Windows, Python 3.10 (volumequant conda 환경)
**Python 경로**: `C:/Users/jkw88/miniconda3/envs/volumequant/python.exe`

---

## 1. 프로젝트 개요

한국 주식시장에서 **거래량 폭발(Volume Breakout)** 시점을 포착하여 수익성을 검증하는 백테스팅 시스템.

### 분석 조건 (Stage 2 최적값)
- **Volume_Ratio** ≥ 6.5 (당일 거래량이 20일 평균의 6.5배 이상)
- **Z_Score** ≥ 3.0 (60일 평균에서 표준편차 3배 이상)
- **당일 수익률** ≥ 10% (시가 대비 종가 상승률)

---

## 2. 완료된 작업 (Stage 3-1)

### 2.1 데이터 인프라 정상화

**문제**: 기존 코드에서 `'연기금등'` 컬럼명 사용 → pykrx 실제 컬럼명은 `'연기금'`
**해결**: `data_loader.py`에 `FlowDataLoader` 클래스 추가

```python
# pykrx 실제 컬럼명 (data_loader.py:270-282)
INVESTOR_COLUMNS = {
    '금융투자': '금융투자',  # 증권사 자기매매
    '연기금': '연기금',      # 연기금 (연기금등 X) ← 핵심 수정
    '개인': '개인',          # 개인 투자자 (신규 추가)
    '외국인': '외국인',
    ...
}
```

### 2.2 수급 주체별 단독 성과 백테스팅 결과

| 필터 | 시그널 수 | 1일 수익률 | Alpha | 10일 수익률 |
|------|-----------|------------|-------|-------------|
| Baseline (필터 없음) | 373 | 0.872% | - | 2.580% |
| **금융투자 순매수** | 184 | **1.528%** | **+75.2%** | 2.490% |
| 연기금 순매수 | 186 | 0.508% | -41.7% | 2.267% |
| 개인 순매도 | 234 | 0.984% | +12.8% | 2.081% |

### 2.3 핵심 발견

1. **금융투자(증권사) 순매수**가 가장 효과적 - 익일 수익률 75% 개선
2. **연기금 순매수**는 오히려 역효과 - 수익률 하락
3. **개인 순매도**는 소폭 개선 (개미 털기 가설 부분 지지)

---

## 3. 수정/생성된 파일 목록

### 3.1 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `data_loader.py` | `FlowDataLoader` 클래스 추가 (+322 lines) |

### 3.2 신규 생성 파일

| 파일 | 설명 |
|------|------|
| `stage3_step1_main.py` | Stage 3-1 백테스트 메인 스크립트 |
| `stage3_step1_visualizer.py` | 시각화 모듈 (막대그래프, Boxplot) |
| `results/stage3_step1_baseline_comparison.csv` | 상세 결과 데이터 |
| `results/stage3_step1_return_comparison.png` | 수익률 비교 막대 그래프 |
| `results/stage3_step1_return_boxplot.png` | 10일 수익률 분포 Boxplot |
| `results/stage3_step1_alpha_comparison.png` | Alpha 비교 그래프 |
| `data/investor_flow_data_v2.csv` | 수급 데이터 (332개 종목, 231,883행) |
| `.claudeignore` | Claude Code 토큰 최적화 설정 |

---

## 4. 다음 작업 단계 (Stage 3-2 이후)

### 4.1 즉시 해야 할 작업

```bash
# 1. 커밋 완료 확인
cd volume-breakout-backtest
git status
git log --oneline -3

# 2. 원격 저장소에 푸시 (필요시)
git push origin main
```

### 4.2 권장 후속 분석 (Stage 3-2)

1. **금융투자 순매수 심화 분석**
   - 순매수 금액 임계치 최적화 (예: > 10억, > 50억)
   - 연속 순매수 효과 (2일, 3일 연속)

2. **복합 필터 테스트**
   - 금융투자(+) AND 개인(-) 조합
   - 금융투자(+) AND 연기금(-) 조합

3. **시가총액별 세분화**
   - 대형주 vs 중소형주 수급 효과 차이

### 4.3 코드 실행 방법

```bash
# Stage 3-1 백테스트 재실행
cd volume-breakout-backtest
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe stage3_step1_main.py

# 시각화만 재생성
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe -c "
from stage3_step1_visualizer import main
main()
"
```

---

## 5. 주요 클래스/함수 레퍼런스

### FlowDataLoader (data_loader.py)

```python
from data_loader import FlowDataLoader

# 초기화
loader = FlowDataLoader(start_date='20220101', end_date='20241231')

# 데이터 검증 (삼성전자)
loader.verify_data(ticker='005930', name='삼성전자', sample_date='20240115')

# 전체 수급 데이터 수집
flow_df = loader.collect_flow_data(stock_list_file='data/stock_list.csv')
loader.save_data(flow_df, 'investor_flow_data_v2.csv')
```

### Stage3Step1Backtest (stage3_step1_main.py)

```python
from stage3_step1_main import Stage3Step1Backtest

backtest = Stage3Step1Backtest(
    volume_ratio_threshold=6.5,
    z_score_threshold=3.0,
    price_threshold=10.0
)

# 필터 설정 (FILTER_CONFIGS 딕셔너리)
# - A0_Baseline: 수급 필터 없음
# - A1_FinancialInvestment: 금융투자 순매수 > 0
# - A2_Pension: 연기금 순매수 > 0
# - A3_IndividualSell: 개인 순매수 < 0
```

---

## 6. 데이터 파일 위치

| 파일 | 크기 | 설명 |
|------|------|------|
| `data/stock_list.csv` | 16KB | 350개 종목 리스트 |
| `data/stock_data.csv` | 16MB | 원본 OHLCV 데이터 |
| `data/stock_data_with_indicators.csv` | 38MB | 지표 포함 데이터 |
| `data/investor_flow_data_v2.csv` | 30MB | 수급 데이터 (금융투자/연기금/개인/외국인) |

---

## 7. 알려진 이슈

1. **한글 폰트 경고**: matplotlib에서 일부 한글 글자 누락 경고 발생 (결과에 영향 없음)
2. **DtypeWarning**: CSV 로드 시 mixed types 경고 (무시 가능)

---

**작성자**: Claude Opus 4.5
**마지막 업데이트**: 2026-01-27 20:30 KST
