"""
Phase 3-3: Unbiased Hybrid Grid Search
======================================

목표: 돌파 시그널 내에서 모든 수급 요인의 영향력을 평등하게 전수조사

방법:
1. Rough Signal (VR>=3.0, Price>=5%) 통과 종목 대상
2. 모든 수급 변수 전수조사:
   - 주체: 개인, 외국인, 금융투자, 연기금
   - 기간: 1, 3, 5, 10, 20, 30, 50, 60일
   - 임계치: -5% ~ +5% (0.5% 단위) - 구간별 분석
3. 금융투자 10D, 연기금 30D는 0.1% 단위 정밀 스캔
4. Top 5 조건 추출 및 2025 OOS 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("=" * 80)
    print("Phase 3-3: Unbiased Hybrid Grid Search")
    print("=" * 80)

    data_dir = project_root / 'data'

    # 주가 데이터
    print("\n[1/4] Loading data...")
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # 수급 데이터
    try:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='utf-8-sig', low_memory=False)
    except:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)

    # 수급 데이터 전처리
    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        '금융투자_비중': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        '연기금_비중': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        '외국인_비중': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })
    flow_data['개인_비중'] = -(flow_data['금융투자_비중'] + flow_data['연기금_비중'] + flow_data['외국인_비중'])

    # 병합
    merged = stock_data.merge(flow_data, on=['Code', 'Date'], how='inner')
    print(f"   Merged data: {len(merged):,} rows")

    return merged


def calculate_all_cumulative_flows(df):
    """모든 Duration에 대해 누적 수급 계산"""
    print("\n[2/4] Calculating cumulative flows...")

    investors = ['개인_비중', '외국인_비중', '금융투자_비중', '연기금_비중']
    durations = [1, 3, 5, 10, 20, 30, 50, 60]

    df = df.sort_values(['Code', 'Date'])

    for duration in durations:
        print(f"   Duration {duration}D...")
        for investor in investors:
            col_name = f'{investor}_{duration}D'
            df[col_name] = df.groupby('Code')[investor].transform(
                lambda x: x.rolling(window=duration, min_periods=duration).sum()
            )

    return df


def apply_rough_signal_filter(df):
    """Rough Signal 필터 적용 (VR>=3.0, Price>=5%)"""
    print("\n[3/4] Applying Rough Signal filter...")

    # 당일 수익률 계산
    df['Price_Change'] = df.groupby('Code')['Close'].transform(
        lambda x: (x / x.shift(1) - 1) * 100
    )

    # Rough Signal 조건
    rough_signals = df[
        (df['Volume_Ratio'] >= 3.0) &
        (df['Price_Change'] >= 5.0)
    ].copy()

    print(f"   Rough Signal passed: {len(rough_signals):,} signals")

    return rough_signals


def analyze_threshold_range(signals, investor, duration, threshold_low, threshold_high, period='IS'):
    """특정 임계치 구간에 대한 분석"""
    col_name = f'{investor}_{duration}D'

    if col_name not in signals.columns:
        return None

    # 구간 필터링
    if threshold_low < 0 and threshold_high <= 0:
        # SELL 구간: threshold_low <= value < threshold_high
        condition = (signals[col_name] >= threshold_low) & (signals[col_name] < threshold_high)
        direction = 'SELL'
        threshold_label = f"{threshold_high:.1f}"
    else:
        # BUY 구간: threshold_low < value <= threshold_high
        condition = (signals[col_name] > threshold_low) & (signals[col_name] <= threshold_high)
        direction = 'BUY'
        threshold_label = f"{threshold_low:.1f}"

    filtered = signals[condition].copy()

    if len(filtered) < 10:
        return None

    # 수익률 계산
    valid_10d = filtered.dropna(subset=['Return_10D'])
    valid_20d = filtered.dropna(subset=['Return_20D'])

    if len(valid_20d) < 10:
        return None

    result = {
        'Period': period,
        'Investor': investor.replace('_비중', ''),
        'Duration': duration,
        'Threshold_Low': threshold_low,
        'Threshold_High': threshold_high,
        'Threshold': float(threshold_label),
        'Direction': direction,
        'Signal_Count': len(valid_20d),
        'Avg_Return_10D': valid_10d['Return_10D'].mean() if len(valid_10d) >= 10 else np.nan,
        'Avg_Return_20D': valid_20d['Return_20D'].mean(),
        'Win_Rate_10D': (valid_10d['Return_10D'] > 0).mean() * 100 if len(valid_10d) >= 10 else np.nan,
        'Win_Rate_20D': (valid_20d['Return_20D'] > 0).mean() * 100,
        'Median_Return_20D': valid_20d['Return_20D'].median(),
        'Std_Return_20D': valid_20d['Return_20D'].std(),
    }

    # 복합 점수 계산 (수익률 x 승률 x log(시그널))
    if result['Signal_Count'] >= 30:
        result['Composite_Score'] = (
            result['Avg_Return_20D'] *
            (result['Win_Rate_20D'] / 100) *
            np.log1p(result['Signal_Count'])
        )
    else:
        result['Composite_Score'] = 0

    return result


def analyze_cumulative_threshold(signals, investor, duration, threshold, period='IS'):
    """누적 임계치 분석 (threshold 이상/이하 전체)"""
    col_name = f'{investor}_{duration}D'

    if col_name not in signals.columns:
        return None

    # 방향 결정 및 필터링
    if threshold < 0:
        condition = signals[col_name] <= threshold
        direction = 'SELL'
    else:
        condition = signals[col_name] >= threshold
        direction = 'BUY'

    filtered = signals[condition].copy()

    if len(filtered) < 10:
        return None

    valid_10d = filtered.dropna(subset=['Return_10D'])
    valid_20d = filtered.dropna(subset=['Return_20D'])

    if len(valid_20d) < 10:
        return None

    result = {
        'Period': period,
        'Investor': investor.replace('_비중', ''),
        'Duration': duration,
        'Threshold': threshold,
        'Direction': direction,
        'Signal_Count': len(valid_20d),
        'Avg_Return_10D': valid_10d['Return_10D'].mean() if len(valid_10d) >= 10 else np.nan,
        'Avg_Return_20D': valid_20d['Return_20D'].mean(),
        'Win_Rate_10D': (valid_10d['Return_10D'] > 0).mean() * 100 if len(valid_10d) >= 10 else np.nan,
        'Win_Rate_20D': (valid_20d['Return_20D'] > 0).mean() * 100,
        'Median_Return_20D': valid_20d['Return_20D'].median(),
        'Std_Return_20D': valid_20d['Return_20D'].std(),
    }

    if result['Signal_Count'] >= 30:
        result['Composite_Score'] = (
            result['Avg_Return_20D'] *
            (result['Win_Rate_20D'] / 100) *
            np.log1p(result['Signal_Count'])
        )
    else:
        result['Composite_Score'] = 0

    return result


def run_full_grid_search(signals_is, signals_oos):
    """전체 Grid Search 실행"""
    print("\n[4/4] Running Grid Search...")

    investors = ['개인_비중', '외국인_비중', '금융투자_비중', '연기금_비중']
    durations = [1, 3, 5, 10, 20, 30, 50, 60]

    # 기본 임계치: -5% ~ +5% (0.5% 단위)
    base_thresholds = np.arange(-5.0, 5.5, 0.5)

    # 정밀 스캔 임계치: 0.1% 단위
    fine_thresholds = np.arange(-5.0, 5.1, 0.1)

    all_results = []
    total_combos = len(investors) * len(durations)
    current = 0

    for investor in investors:
        for duration in durations:
            current += 1
            investor_name = investor.replace('_비중', '')

            # 정밀 스캔 대상 여부 확인
            is_fine_scan = (
                (investor == '금융투자_비중' and duration == 10) or
                (investor == '연기금_비중' and duration == 30)
            )

            thresholds = fine_thresholds if is_fine_scan else base_thresholds
            scan_type = "Fine" if is_fine_scan else "Base"

            print(f"   [{current}/{total_combos}] {investor_name} {duration}D ({scan_type}, {len(thresholds)} thresholds)...")

            # 누적 임계치 분석 (threshold 이상/이하 전체)
            for threshold in thresholds:
                # IS 기간 분석
                result_is = analyze_cumulative_threshold(signals_is, investor, duration, threshold, 'IS')
                if result_is:
                    all_results.append(result_is)

                # OOS 기간 분석
                result_oos = analyze_cumulative_threshold(signals_oos, investor, duration, threshold, 'OOS')
                if result_oos:
                    all_results.append(result_oos)

    return pd.DataFrame(all_results)


def find_optimal_thresholds(results_df):
    """각 투자자/Duration 조합에서 최적 임계치 찾기"""
    print("\n" + "=" * 80)
    print("Optimal Threshold Analysis")
    print("=" * 80)

    optimal_results = []

    for investor in ['개인', '외국인', '금융투자', '연기금']:
        for duration in [1, 3, 5, 10, 20, 30, 50, 60]:
            # IS 데이터에서 분석
            subset = results_df[
                (results_df['Investor'] == investor) &
                (results_df['Duration'] == duration) &
                (results_df['Period'] == 'IS') &
                (results_df['Signal_Count'] >= 30)
            ].copy()

            if len(subset) < 3:
                continue

            # SELL 방향 최적점
            sell_data = subset[subset['Direction'] == 'SELL']
            if len(sell_data) >= 2:
                best_sell = sell_data.loc[sell_data['Avg_Return_20D'].idxmax()]
                optimal_results.append({
                    'Investor': investor,
                    'Duration': duration,
                    'Direction': 'SELL',
                    'Optimal_Threshold': best_sell['Threshold'],
                    'Return_20D': best_sell['Avg_Return_20D'],
                    'Win_Rate': best_sell['Win_Rate_20D'],
                    'Signal_Count': best_sell['Signal_Count'],
                    'Composite_Score': best_sell['Composite_Score']
                })

            # BUY 방향 최적점
            buy_data = subset[subset['Direction'] == 'BUY']
            if len(buy_data) >= 2:
                best_buy = buy_data.loc[buy_data['Avg_Return_20D'].idxmax()]
                optimal_results.append({
                    'Investor': investor,
                    'Duration': duration,
                    'Direction': 'BUY',
                    'Optimal_Threshold': best_buy['Threshold'],
                    'Return_20D': best_buy['Avg_Return_20D'],
                    'Win_Rate': best_buy['Win_Rate_20D'],
                    'Signal_Count': best_buy['Signal_Count'],
                    'Composite_Score': best_buy['Composite_Score']
                })

    optimal_df = pd.DataFrame(optimal_results)

    # 상위 20개 출력
    print("\nTop 20 Optimal Conditions (by 20D Return):")
    print("-" * 90)
    print("Rank | Investor | Dur | Dir  | Threshold | Signals | 20D Return | WinRate | Score")
    print("-" * 90)

    top20 = optimal_df.nlargest(20, 'Return_20D')
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"{i:4d} | {row['Investor']:8s} | {row['Duration']:3.0f} | {row['Direction']:4s} | "
              f"{row['Optimal_Threshold']:+8.1f}% | {row['Signal_Count']:7.0f} | "
              f"{row['Return_20D']:9.2f}% | {row['Win_Rate']:6.1f}% | {row['Composite_Score']:.2f}")

    return optimal_df


def analyze_inflection_points(results_df):
    """변곡점 분석: 금융투자 10D, 연기금 30D 정밀 분석"""
    print("\n" + "=" * 80)
    print("Inflection Point Analysis (Fine-grained)")
    print("=" * 80)

    inflection_results = []

    for investor, duration in [('금융투자', 10), ('연기금', 30)]:
        subset = results_df[
            (results_df['Investor'] == investor) &
            (results_df['Duration'] == duration) &
            (results_df['Period'] == 'IS')
        ].sort_values('Threshold')

        if len(subset) < 5:
            continue

        print(f"\n{investor} {duration}D Inflection Analysis:")
        print("-" * 70)

        # SELL 방향 분석
        sell_data = subset[subset['Direction'] == 'SELL'].sort_values('Threshold', ascending=False)
        if len(sell_data) >= 3:
            # 수익률 피크 찾기
            peak_idx = sell_data['Avg_Return_20D'].idxmax()
            peak_row = sell_data.loc[peak_idx]

            print(f"SELL Direction:")
            print(f"   Peak: Threshold <= {peak_row['Threshold']:.1f}%")
            print(f"   20D Return: {peak_row['Avg_Return_20D']:.2f}%")
            print(f"   Win Rate: {peak_row['Win_Rate_20D']:.1f}%")
            print(f"   Signals: {peak_row['Signal_Count']:.0f}")

            # 변곡점: 수익률이 떨어지기 시작하는 지점
            inflection_results.append({
                'Investor': investor,
                'Duration': duration,
                'Direction': 'SELL',
                'Peak_Threshold': peak_row['Threshold'],
                'Peak_Return': peak_row['Avg_Return_20D'],
                'Peak_WinRate': peak_row['Win_Rate_20D'],
                'Peak_Signals': peak_row['Signal_Count']
            })

        # BUY 방향 분석
        buy_data = subset[subset['Direction'] == 'BUY'].sort_values('Threshold')
        if len(buy_data) >= 3:
            peak_idx = buy_data['Avg_Return_20D'].idxmax()
            peak_row = buy_data.loc[peak_idx]

            print(f"BUY Direction:")
            print(f"   Peak: Threshold >= {peak_row['Threshold']:.1f}%")
            print(f"   20D Return: {peak_row['Avg_Return_20D']:.2f}%")
            print(f"   Win Rate: {peak_row['Win_Rate_20D']:.1f}%")
            print(f"   Signals: {peak_row['Signal_Count']:.0f}")

            inflection_results.append({
                'Investor': investor,
                'Duration': duration,
                'Direction': 'BUY',
                'Peak_Threshold': peak_row['Threshold'],
                'Peak_Return': peak_row['Avg_Return_20D'],
                'Peak_WinRate': peak_row['Win_Rate_20D'],
                'Peak_Signals': peak_row['Signal_Count']
            })

    return pd.DataFrame(inflection_results)


def analyze_linearity(results_df):
    """수급 강도와 수익률의 선형성 분석"""
    print("\n" + "=" * 80)
    print("Linearity Analysis: Flow Ratio vs Return")
    print("=" * 80)

    linearity_results = []

    for investor in ['개인', '외국인', '금융투자', '연기금']:
        for duration in [10, 30, 50]:
            # IS 데이터
            subset = results_df[
                (results_df['Investor'] == investor) &
                (results_df['Duration'] == duration) &
                (results_df['Period'] == 'IS') &
                (results_df['Signal_Count'] >= 30)
            ].sort_values('Threshold')

            if len(subset) < 5:
                continue

            # SELL 방향 분석 (음수 threshold)
            sell_subset = subset[subset['Direction'] == 'SELL']
            if len(sell_subset) >= 3:
                # 상관관계 (더 작은 threshold = 더 강한 SELL)
                corr_sell = sell_subset['Threshold'].corr(sell_subset['Avg_Return_20D'])

                if corr_sell > 0.5:
                    pattern = "Linear(+): Stronger SELL = Lower Return"
                elif corr_sell < -0.5:
                    pattern = "Linear(-): Stronger SELL = Higher Return"
                else:
                    pattern = "Non-linear (inflection exists)"

                linearity_results.append({
                    'Investor': investor,
                    'Duration': duration,
                    'Direction': 'SELL',
                    'Correlation': corr_sell,
                    'Pattern': pattern,
                    'Data_Points': len(sell_subset)
                })
                print(f"{investor} {duration}D SELL: r={corr_sell:.3f} -> {pattern}")

            # BUY 방향 분석 (양수 threshold)
            buy_subset = subset[subset['Direction'] == 'BUY']
            if len(buy_subset) >= 3:
                corr_buy = buy_subset['Threshold'].corr(buy_subset['Avg_Return_20D'])

                if corr_buy > 0.5:
                    pattern = "Linear(+): Stronger BUY = Higher Return"
                elif corr_buy < -0.5:
                    pattern = "Linear(-): Stronger BUY = Lower Return"
                else:
                    pattern = "Non-linear (inflection exists)"

                linearity_results.append({
                    'Investor': investor,
                    'Duration': duration,
                    'Direction': 'BUY',
                    'Correlation': corr_buy,
                    'Pattern': pattern,
                    'Data_Points': len(buy_subset)
                })
                print(f"{investor} {duration}D BUY: r={corr_buy:.3f} -> {pattern}")

    return pd.DataFrame(linearity_results)


def find_top_conditions(results_df, top_n=5):
    """Top N 조건 추출 (Composite Score 기준)"""
    print("\n" + "=" * 80)
    print(f"Top {top_n} Conditions (IS Period, Composite Score)")
    print("=" * 80)

    # IS 기간만 필터링
    is_results = results_df[
        (results_df['Period'] == 'IS') &
        (results_df['Signal_Count'] >= 50)  # 최소 50개 시그널
    ].copy()

    # Composite Score 기준 정렬
    top_conditions = is_results.nlargest(top_n, 'Composite_Score')

    print("\nRank | Investor | Dur | Dir  | Threshold | Signals | 20D Return | WinRate | Score")
    print("-" * 90)

    for i, (_, row) in enumerate(top_conditions.iterrows(), 1):
        print(f"{i:4d} | {row['Investor']:8s} | {row['Duration']:3.0f} | {row['Direction']:4s} | "
              f"{row['Threshold']:+8.1f}% | {row['Signal_Count']:7.0f} | "
              f"{row['Avg_Return_20D']:9.2f}% | {row['Win_Rate_20D']:6.1f}% | {row['Composite_Score']:.2f}")

    return top_conditions


def validate_oos(results_df, top_conditions):
    """Top 조건들의 OOS 검증"""
    print("\n" + "=" * 80)
    print("OOS (2025) Validation")
    print("=" * 80)

    validation_results = []

    print("\nCondition                    | IS Return | OOS Return | IS WR | OOS WR | Status")
    print("-" * 85)

    for _, is_row in top_conditions.iterrows():
        # 동일 조건의 OOS 결과 찾기
        oos_match = results_df[
            (results_df['Period'] == 'OOS') &
            (results_df['Investor'] == is_row['Investor']) &
            (results_df['Duration'] == is_row['Duration']) &
            (results_df['Threshold'] == is_row['Threshold']) &
            (results_df['Direction'] == is_row['Direction'])
        ]

        condition_name = f"{is_row['Investor']} {is_row['Duration']:.0f}D {is_row['Direction']} {is_row['Threshold']:+.1f}%"

        if len(oos_match) > 0:
            oos_row = oos_match.iloc[0]

            # 성과 유지 여부 판단
            maintained = (oos_row['Avg_Return_20D'] > 0) and (oos_row['Win_Rate_20D'] > 45)
            status = "[OK]" if maintained else "[FAIL]"

            print(f"{condition_name:28s} | {is_row['Avg_Return_20D']:8.2f}% | "
                  f"{oos_row['Avg_Return_20D']:9.2f}% | {is_row['Win_Rate_20D']:5.1f}% | "
                  f"{oos_row['Win_Rate_20D']:6.1f}% | {status}")

            validation_results.append({
                'Condition': condition_name,
                'Investor': is_row['Investor'],
                'Duration': is_row['Duration'],
                'Direction': is_row['Direction'],
                'Threshold': is_row['Threshold'],
                'IS_Return': is_row['Avg_Return_20D'],
                'OOS_Return': oos_row['Avg_Return_20D'],
                'IS_WinRate': is_row['Win_Rate_20D'],
                'OOS_WinRate': oos_row['Win_Rate_20D'],
                'IS_Signals': is_row['Signal_Count'],
                'OOS_Signals': oos_row['Signal_Count'],
                'Maintained': maintained
            })
        else:
            print(f"{condition_name:28s} | {is_row['Avg_Return_20D']:8.2f}% | "
                  f"{'N/A':>9s} | {is_row['Win_Rate_20D']:5.1f}% | {'N/A':>6s} | [NO DATA]")

    return pd.DataFrame(validation_results)


def generate_final_report(results_df, optimal_df, validation_df, inflection_df, linearity_df):
    """최종 리포트 생성"""
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    # 1. 최우수 단일 수급 팩터
    print("\n### Q1: What is the statistically superior single supply factor in breakout situations?")
    print("-" * 70)

    if len(optimal_df) > 0:
        best = optimal_df.nlargest(1, 'Return_20D').iloc[0]
        print(f"   BEST: {best['Investor']} {best['Duration']:.0f}D {best['Direction']} (Threshold: {best['Optimal_Threshold']:+.1f}%)")
        print(f"   - 20D Return: {best['Return_20D']:.2f}%")
        print(f"   - Win Rate: {best['Win_Rate']:.1f}%")
        print(f"   - Signals: {best['Signal_Count']:.0f}")

    # 2. 선형성 분석 결과
    print("\n### Q2: Does return increase linearly with flow strength, or is there an inflection point?")
    print("-" * 70)

    if len(linearity_df) > 0:
        for _, row in linearity_df.iterrows():
            print(f"   {row['Investor']} {row['Duration']}D {row['Direction']}: {row['Pattern']}")

    # 3. 변곡점 분석
    if len(inflection_df) > 0:
        print("\n### Inflection Points (Peak Performance Thresholds):")
        print("-" * 70)
        for _, row in inflection_df.iterrows():
            print(f"   {row['Investor']} {row['Duration']}D {row['Direction']}: "
                  f"Peak at {row['Peak_Threshold']:+.1f}% (Return: {row['Peak_Return']:.2f}%)")

    # 4. OOS 검증 통과 조건
    if len(validation_df) > 0:
        maintained = validation_df[validation_df['Maintained'] == True]
        print(f"\n### OOS Validation: {len(maintained)}/{len(validation_df)} conditions maintained")
        print("-" * 70)
        for _, row in maintained.iterrows():
            print(f"   [OK] {row['Condition']}: IS {row['IS_Return']:.2f}% -> OOS {row['OOS_Return']:.2f}%")


def main():
    # 1. 데이터 로드
    merged = load_and_prepare_data()

    # 2. 누적 수급 계산
    merged = calculate_all_cumulative_flows(merged)

    # 3. Rough Signal 필터 적용
    rough_signals = apply_rough_signal_filter(merged)

    # 4. IS/OOS 분리
    rough_signals['Year'] = rough_signals['Date'].dt.year
    signals_is = rough_signals[rough_signals['Year'] <= 2024].copy()
    signals_oos = rough_signals[rough_signals['Year'] == 2025].copy()

    print(f"\n   IS (2021-2024): {len(signals_is):,} signals")
    print(f"   OOS (2025): {len(signals_oos):,} signals")

    # 5. Grid Search 실행
    results_df = run_full_grid_search(signals_is, signals_oos)

    # 6. 결과 저장
    output_dir = project_root / 'results' / 'p3'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'p3_03_total_grid_search_results.csv',
                      index=False, encoding='utf-8-sig')
    print(f"\nResults saved: {output_dir / 'p3_03_total_grid_search_results.csv'}")
    print(f"Total {len(results_df)} combinations analyzed")

    # 7. 최적 임계치 분석
    optimal_df = find_optimal_thresholds(results_df)

    # 8. 변곡점 분석
    inflection_df = analyze_inflection_points(results_df)

    # 9. 선형성 분석
    linearity_df = analyze_linearity(results_df)

    # 10. Top 5 조건 추출
    top_conditions = find_top_conditions(results_df, top_n=5)

    # 11. OOS 검증
    validation_df = validate_oos(results_df, top_conditions)

    # 12. 최종 리포트
    generate_final_report(results_df, optimal_df, validation_df, inflection_df, linearity_df)

    # 13. 추가 파일 저장
    optimal_df.to_csv(output_dir / 'p3_03_optimal_thresholds.csv',
                      index=False, encoding='utf-8-sig')
    top_conditions.to_csv(output_dir / 'p3_03_top_conditions.csv',
                          index=False, encoding='utf-8-sig')
    if len(validation_df) > 0:
        validation_df.to_csv(output_dir / 'p3_03_oos_validation.csv',
                            index=False, encoding='utf-8-sig')
    if len(inflection_df) > 0:
        inflection_df.to_csv(output_dir / 'p3_03_inflection_points.csv',
                            index=False, encoding='utf-8-sig')
    if len(linearity_df) > 0:
        linearity_df.to_csv(output_dir / 'p3_03_linearity_analysis.csv',
                            index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("Phase 3-3 Complete!")
    print("=" * 80)

    return results_df, optimal_df, validation_df


if __name__ == "__main__":
    results, optimal, validation = main()
