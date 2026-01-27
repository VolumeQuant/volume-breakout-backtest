# Session Handoff - Volume Breakout Backtest Stage 4-1

**마지막 작업일**: 2026-01-28
**작업 환경**: Windows, Python 3.10 (volumequant conda 환경)
**Python 경로**: `C:/Users/jkw88/miniconda3/envs/volumequant/python.exe`

---

## 1. 프로젝트 개요

한국 주식시장에서 **거래량 폭발(Volume Breakout)** 시점을 포착하여 수익성을 검증하는 백테스팅 시스템.

### 현재 최적 전략 (OOS 검증 완료)

```
진입 조건:
  - Volume_Ratio ≥ 6.5 (당일 거래량이 20일 평균의 6.5배 이상)
  - Z_Score ≥ 3.0 (60일 평균에서 표준편차 3배 이상)
  - 당일 수익률 ≥ 10% (시가 대비 종가)
  - 개인 순매도 (< 0)
  - 연기금 순매도 (< 0)

보유 기간: 10일 (권장)
예상 성과: 월 2~3건, 10일 수익률 4~6%, 승률 40~45%
```

---

## 2. 완료된 작업 단계

### Stage 1~2: 거래량/가격 필터 최적화 ✅
- VR ≥ 6.5, ZS ≥ 3.0, 당일 ≥ 10% 최적값 도출

### Stage 3-1: 수급 단독 필터 ✅
- 금융투자 순매수 +75% Alpha (가장 효과적)
- 연기금 순매수 -42% Alpha (역효과)

### Stage 3-2: 조합 필터 + 보유 기간별 분석 ✅
- 최적 조합: 개인(-) + 연기금(-) → 1일 1.91%, 10일 3.85%

### Stage 3-3: 민감도 분석 ✅
| 분석 | 결과 |
|------|------|
| 임계치 민감도 | VR 4.0/ZS 2.0 → 시그널↑, 10일 승률 53.2% |
| 5일 매집 가설 | ❌ 기각 (당일 수급이 더 유효) |
| 연기금 시총별 | 대형주 매도 시 +6.12% (강력 시그널) |

### Stage 3-4: 수급 집중도(Inst_Ratio) 분석 ✅
- **Sweet Spot**: Inst_Ratio ≥ 3%
- 시그널 33개, 승률 60.6%, 10일 수익률 4.37%

### Stage 4-1: Out-of-Sample 검증 (2025년) ✅

| 지표 | In-Sample (22-24) | OOS (2025) | 판정 |
|------|------------------:|-----------:|------|
| 1일 수익률 | 1.91% | -0.61% | ❌ Alpha Decay |
| 10일 수익률 | 3.85% | 5.96% | ✅ 유효 |
| 10일 승률 | 43.3% | 41.7% | ✅ 유지 |

**결론**: 단기(1일) 비추천, **10일 보유 권장**

---

## 3. 파일 구조

```
volume-breakout-backtest/
├── data/
│   ├── stock_list.csv                 # 350개 종목 리스트
│   ├── stock_data.csv                 # 원본 OHLCV (gitignore)
│   ├── stock_data_with_indicators.csv # 지표 포함 (gitignore)
│   └── investor_flow_data_v2.csv      # 수급 데이터 (gitignore)
├── results/
│   ├── stage3_step1_*.csv/png         # 수급 단독 필터 결과
│   ├── stage3_step2_*.csv/png         # 조합 필터 결과
│   ├── stage3_step3_sensitivity.csv   # 민감도 분석
│   ├── stage3_step4_inst_ratio.*      # 수급 집중도 분석
│   ├── stage4_oos_validation.*        # OOS 검증 결과
│   └── stage4_yearly_stats.csv        # 연도별 통계
├── stage3_step1_main.py          # 수급 단독 필터
├── stage3_step2_combo_backtest.py    # 조합 필터
├── stage3_step3_sensitivity.py   # 민감도 분석
├── stage3_step4_inst_ratio.py    # 수급 집중도
├── stage4_oos_validation.py      # OOS 검증
├── RESULTS_SUMMARY.md            # 전체 결과 보고서
└── SESSION_HANDOFF.md            # 이 문서
```

---

## 4. 주요 발견 사항

### 4.1 Alpha Decay 추세 (1일 수익률)
```
2022: 3.83% → 2023: 2.20% → 2024: 0.89% → 2025: -0.61%
```
→ 단기 전략은 시장 효율화로 무효화됨

### 4.2 10일 보유는 여전히 유효
```
2022: 5.61% → 2023: 5.82% → 2024: 0.75% → 2025: 5.96%
```
→ 2024년 제외 안정적 수익

### 4.3 연기금 역설 (시가총액별)
| 구분 | 연기금 매수 | 연기금 매도 |
|------|-------------|-------------|
| 대형주 | -0.40% | **+6.12%** ✅ |
| 중소형주 | +2.74% | +1.53% |

### 4.4 Inst_Ratio Sweet Spot
- **≥ 3%**: 승률 60.6%, 10일 4.37% (최적)
- ≥ 5%: 시그널 감소, 성과 불안정

---

## 5. 코드 실행 방법

```bash
cd volume-breakout-backtest

# Stage 3-3: 민감도 분석
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe stage3_step3_sensitivity.py

# Stage 3-4: 수급 집중도 분석
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe stage3_step4_inst_ratio.py

# Stage 4-1: OOS 검증 (2025년 데이터 수집 포함, 약 5분 소요)
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe stage4_oos_validation.py
```

---

## 6. 다음 작업 제안

### 6.1 즉시 가능한 작업

1. **손절/익절 전략 추가**
   - 트레일링 스탑 (고점 대비 -5% 등)
   - 목표가 도달 시 익절 (+10%, +20%)

2. **실시간 시그널 감지 시스템**
   - 매일 장 마감 후 자동 스캔
   - 조건 충족 종목 알림

### 6.2 추가 분석

1. **섹터별 세분화**
   - IT, 바이오, 2차전지 등 테마별 성과 차이

2. **시장 상황별 분석**
   - 상승장 vs 하락장 성과 차이
   - 변동성(VIX) 레벨별 성과

3. **복합 전략 포트폴리오**
   - 여러 전략 조합 리밸런싱

---

## 7. 알려진 이슈

1. **한글 폰트 경고**: matplotlib에서 일부 한글 글자 누락 경고 (결과 영향 없음)
2. **FutureWarning**: pandas groupby observed 경고 (무시 가능)
3. **2025년 데이터**: stage4_oos_validation.py 실행 시 실시간 수집 (약 5분)

---

## 8. Git 상태

```bash
# 최근 커밋
git log --oneline -5

# 현재 상태
2b47751 docs: Update RESULTS_SUMMARY.md with Stage 3-3/3-4/4-1 results
390cec9 Stage 3-3/3-4: 민감도 분석 및 수급 집중도 분석
faa720a Step 4-1: 2025 OOS validation completed
c3047f4 docs: Update RESULTS_SUMMARY.md with Stage 1~3-2 results
626b901 Stage 3-2: 수급 필터 조합 백테스팅 및 보유 기간별 분석
```

---

**작성자**: Claude Opus 4.5
**마지막 업데이트**: 2026-01-28 KST
