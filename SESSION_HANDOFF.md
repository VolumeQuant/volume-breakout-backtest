# Session Handoff - Volume Breakout Backtest Stage 5

**마지막 작업일**: 2026-01-28
**작업 환경**: Windows, Python 3.10 (volumequant conda 환경)
**Python 경로**: `C:/Users/jkw88/miniconda3/envs/volumequant/python.exe`

---

## 1. 프로젝트 개요

한국 주식시장에서 **거래량 폭발(Volume Breakout)** 시점을 포착하여 수익성을 검증하는 백테스팅 시스템.

### 현재 최적 전략 (Stage 5 - 현실적 빈도 확보)

**빈도와 수익성의 균형을 위한 최종 권장 전략**:

```
진입 조건:
  - Volume_Ratio ≥ 4.0 (당일 거래량이 20일 평균의 4배 이상)
  - Z_Score ≥ 2.0 (60일 평균에서 표준편차 2배 이상)
  - 당일 수익률 ≥ 7% (시가 대비 종가)
  - 개인 순매도 (< 0)
  - 연기금 순매도 (< 0)

보유 기간: 10일 (권장)
예상 성과: 월 6.2건, 10일 수익률 3.91%, 승률 51.3%, 샤프 3.46
```

**전략 선택 가이드**:
- **보수적 (높은 수익률 우선)**: VR≥6.5, ZS≥3.0, Price≥10% → 월 2.6건, 10일 3.85%
- **균형적 (권장)**: VR≥4.0, ZS≥2.0, Price≥7% → 월 6.2건, 10일 3.91%
- **공격적 (높은 빈도 우선)**: VR≥4.0, ZS≥2.0, Price≥5% → 월 7.4건, 10일 2.89%

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

### Stage 5: 현실적 매매 빈도 전략 최적화 ✅

**문제 인식**: VR 6.5/ZS 3.0 기준이 너무 엄격하여 월 2.6건으로 실전 운용 어려움

**Stage 5-1: 가격 임계치 민감도 분석**

완화된 거래량 기준(VR 4.0, ZS 2.0)에서 가격 임계치별 성과 비교:

| 가격 임계치 | 시그널 수 | 월평균 | 1일 수익률 | 10일 수익률 | 10일 승률 | 샤프 | 손익비 |
|------------|----------|--------|-----------|------------|----------|------|--------|
| ≥ 5% | 243 | 7.4건 | 0.33% | 2.89% | 47.7% | 2.75 | 1.75 |
| ≥ 7% | 199 | 6.2건 | 0.47% | **3.91%** | **51.3%** | **3.46** | 1.71 |
| ≥ 10% | 130 | 4.3건 | 0.90% | **5.70%** | 51.5% | 4.50 | **2.01** |

**Stage 5-2: 전략 비교 분석**

| 전략 | 시그널 | 월평균 | 1일 | 10일 | 승률 | 샤프 |
|------|--------|--------|-----|------|------|------|
| Stage 3-3 (VR 6.5, ZS 3.0, Price 10%) | 62 | 2.6건 | 1.91% | 3.85% | 41.9% | 2.87 |
| **Stage 5 (VR 4.0, ZS 2.0, Price 7%)** | 199 | **6.2건** | 0.47% | **3.91%** | **51.3%** | **3.46** |

**핵심 발견**:
1. **Price 7% = Sweet Spot**: 빈도(6.2건/월)와 수익률(3.91%) 균형점
2. **승률 개선**: 51.3% (Stage 3-3: 41.9%)
3. **샤프 비율 향상**: 3.46 (Stage 3-3: 2.87)
4. **Trade-off**: 가격 임계치 ↓ → 빈도 ↑, 수익률 ↓

**전략 선정 이유 (VR 4.0, ZS 2.0, Price 7%)**:
- 월 6.2건으로 실전 운용 가능한 빈도 확보
- 10일 수익률 3.91%로 Stage 3-3와 동등한 성과 유지
- 승률 51.3%로 심리적 안정성 확보
- 샤프 비율 3.46으로 위험 대비 수익 우수

---

## 3. 파일 구조

```
volume-breakout-backtest/
├── data/
│   ├── stock_list.csv                 # 350개 종목 리스트
│   ├── stock_data.csv                 # 원본 OHLCV (gitignore)
│   ├── stock_data_with_indicators.csv # 지표 포함 (gitignore)
│   └── investor_flow_data.csv         # 수급 데이터 (gitignore)
├── results/
│   ├── stage3_step1_*.csv/png         # 수급 단독 필터 결과
│   ├── stage3_step2_*.csv/png         # 조합 필터 결과
│   ├── stage3_step3_sensitivity.csv   # 민감도 분석
│   ├── stage3_step4_inst_ratio.*      # 수급 집중도 분석
│   ├── stage4_oos_validation.*        # OOS 검증 결과
│   ├── stage4_yearly_stats.csv        # 연도별 통계
│   ├── stage5_realistic_sensitivity.* # Stage 5 가격 임계치 분석
│   └── stage5_realistic_comparison.*  # Stage 3-3 vs Stage 5 비교
├── stage3_step1_main.py               # 수급 단독 필터
├── stage3_step2_combo_backtest.py     # 조합 필터
├── stage3_step3_sensitivity.py        # 민감도 분석
├── stage3_step4_inst_ratio.py         # 수급 집중도
├── stage4_oos_validation.py           # OOS 검증
├── stage5_realistic_strategy.py       # 현실적 빈도 전략 (최종)
├── stage5_final_strategy.py           # 시장필터+Trailing Stop 시도 (참고용)
├── RESULTS_SUMMARY.md                 # 전체 결과 보고서
└── SESSION_HANDOFF.md                 # 이 문서
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

# Stage 5: 현실적 빈도 전략 (최종 권장)
C:/Users/jkw88/miniconda3/envs/volumequant/python.exe stage5_realistic_strategy.py
```

---

## 6. 다음 작업 제안

### 6.1 즉시 가능한 작업

1. **실시간 시그널 감지 시스템** ⭐ 최우선
   - 매일 장 마감 후 자동 스캔
   - Stage 5 조건(VR 4.0, ZS 2.0, Price 7%) 충족 종목 알림
   - Telegram/Discord 봇 연동

2. **손절/익절 로직 정교화**
   - Trailing Stop -3.5% 재시도 (단, 시장 필터 제외)
   - 고정 손절 vs Trailing Stop 성과 비교
   - 목표가 도달 시 익절 (+10%, +20%)

3. **2025년 OOS 재검증**
   - 2025년 실제 데이터 수집 시 Stage 5 전략 재검증
   - 연도별 Alpha Decay 추적

### 6.2 추가 분석

1. **시장 필터 재설계** (Stage 5에서 제외된 기능)
   - 현재: 지수 > 20일 이평 시 진입 → 시그널 0건 발생
   - 개선 방향:
     - 더 완화된 조건 테스트 (지수 > 60일 이평)
     - 또는 시장 필터를 선택적 적용 (필터링 vs 가중치)

2. **섹터/테마별 세분화**
   - IT, 바이오, 2차전지 등 테마별 성과 차이
   - 시가총액별 세분화 (대형주 vs 중소형주)

3. **복합 전략 포트폴리오**
   - 보수적(Stage 3-3) + 균형적(Stage 5) 조합
   - 동적 비중 조절

---

## 7. Stage 5 개발 과정 및 주요 교훈

### 7.1 시도했으나 실패한 기능들

**시장 필터 (Market Filter)**:
- **시도**: KOSPI 200/KOSDAQ 지수 > 20일 이평선 조건 추가
- **결과**: 시그널 0건 발생 (필터 통과 비율 49% × 기타 조건 = 과도한 제약)
- **교훈**: 필터 조합 시 각 필터의 통과율을 사전 검증 필요

**Inst_Ratio ≥ 3% 조건**:
- **시도**: 금융투자+연기금 비중 ≥ 3% 조건 추가 (Stage 3-4에서 효과적이었음)
- **결과**: 시장 필터와 함께 적용 시 시그널 0건
- **교훈**: 단독으로는 유효하나 다른 필터와 조합 시 과최적화 위험

**Trailing Stop -3.5%**:
- **시도**: 고점 대비 -3.5% 하락 시 매도, 최대 10일 보유
- **결과**: 구현 완료했으나 복잡도 증가로 단순 10일 보유로 회귀
- **교훈**: 백테스팅 단계에서는 단순 전략으로 기본 성과 확인 후 점진적 개선 필요

### 7.2 성공적인 단순화 전략

**최종 채택 조건**:
```python
signals = df[
    (df['Volume_Ratio'] >= 4.0) &     # VR 완화
    (df['Z_Score'] >= 2.0) &           # ZS 완화
    (df['Return_0D'] >= 7.0) &         # Price threshold
    (df['Return_0D'] < 30.0) &         # 상한가 제외
    (df['개인'] < 0) &                  # 개인 순매도
    (df['연기금'] < 0)                   # 연기금 순매도
].copy()
```

**핵심 원칙**:
1. **과최적화 방지**: 필터 3~4개로 제한
2. **검증 가능성**: 각 필터의 독립적 효과 확인
3. **실전 적용성**: 매일 실행 가능한 단순 로직

---

## 8. 알려진 이슈

1. **한글 폰트 경고**: matplotlib에서 일부 한글 글자 누락 경고 (결과 영향 없음)
2. **FutureWarning**: pandas groupby observed 경고 (무시 가능)
3. **2025년 데이터**: stage4_oos_validation.py 실행 시 실시간 수집 (약 5분)

---

## 9. Git 상태

**다음 커밋 예정**:
```bash
# Stage 5 최종 작업 커밋
git add stage5_realistic_strategy.py stage5_final_strategy.py
git add results/stage5_realistic_*.csv results/stage5_realistic_*.png
git add SESSION_HANDOFF.md
git commit -m "Stage 5: Final realistic strategy with VR 4.0/ZS 2.0 - price threshold optimization

- 문제 해결: VR 6.5/ZS 3.0 기준의 낮은 빈도(월 2.6건) → VR 4.0/ZS 2.0로 완화
- 가격 임계치 민감도 분석: 5%, 7%, 10% 비교
- 최종 권장: VR 4.0, ZS 2.0, Price 7% (월 6.2건, 10일 3.91%, 승률 51.3%)
- 시장 필터/Trailing Stop 시도 → 과최적화로 제외, 단순화 전략 채택

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**최근 커밋 히스토리**:
```
2b47751 docs: Update RESULTS_SUMMARY.md with Stage 3-3/3-4/4-1 results
390cec9 Stage 3-3/3-4: 민감도 분석 및 수급 집중도 분석
faa720a Step 4-1: 2025 OOS validation completed
c3047f4 docs: Update RESULTS_SUMMARY.md with Stage 1~3-2 results
626b901 Stage 3-2: 수급 필터 조합 백테스팅 및 보유 기간별 분석
```

---

**작성자**: Claude Sonnet 4.5
**마지막 업데이트**: 2026-01-28 KST (Stage 5 완료)
