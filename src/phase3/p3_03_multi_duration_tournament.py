"""
Phase 3-3 Extended: Multi-Duration Tournament
==============================================

"20일 보유 전략에는 그에 맞는 누적 수급(Accumulated Supply) 분석이 필요하다"는 가설 검증

Duration List: [1, 3, 5, 7, 10, 20] days

Track A: Constrained Combinatorial (Anti-Overfitting)
- 4C_3 (3주체): 반드시 1개 이상의 SELL이 포함되어야 함
- 4C_4 (4주체): 반드시 2개 이상의 SELL이 포함되어야 함

Track B: Classic with D-Day Accumulated Supply
Track C: Z-Score on D-Day Accumulated Supply

Output: Heatmap - rows=strategies, cols=durations, values=OOS returns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent

DURATION_LIST = [1, 3, 5, 7, 10, 20]


def load_base_data():
    """기본 데이터 로드"""
    print("=" * 80)
    print("Phase 3-3 Extended: Multi-Duration Tournament")
    print("Hypothesis: 'Matching duration between supply analysis and holding period matters'")
    print("=" * 80)

    data_dir = project_root / 'data'

    print("\n[1/6] Loading base data...")
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    try:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='utf-8-sig', low_memory=False)
    except:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)

    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        'fininvest_raw': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        'pension_raw': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        'foreign_raw': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })
    flow_data['retail_raw'] = -(flow_data['fininvest_raw'] + flow_data['pension_raw'] + flow_data['foreign_raw'])

    merged = stock_data.merge(flow_data, on=['Code', 'Date'], how='inner')
    merged = merged.sort_values(['Code', 'Date']).reset_index(drop=True)
    print(f"   Merged data: {len(merged):,} rows")

    return merged


def calculate_rolling_sums(df, duration):
    """특정 Duration에 대한 Rolling Sum 계산"""
    investors = {
        'retail': 'retail_raw',
        'foreign': 'foreign_raw',
        'fininvest': 'fininvest_raw',
        'pension': 'pension_raw'
    }

    for name, col in investors.items():
        # D-day Rolling Sum (당일 포함 D일간 합계)
        df[f'{name}_{duration}D'] = df.groupby('Code')[col].transform(
            lambda x: x.rolling(window=duration, min_periods=duration).sum()
        )

    return df


def calculate_z_scores_for_duration(df, duration, window=60):
    """특정 Duration의 Rolling Sum에 대한 Z-Score 계산"""
    investors = ['retail', 'foreign', 'fininvest', 'pension']

    for inv in investors:
        col = f'{inv}_{duration}D'
        if col not in df.columns:
            continue

        # 60-day rolling mean/std on the D-day accumulated data
        mean_col = f'{inv}_{duration}D_mean'
        std_col = f'{inv}_{duration}D_std'
        z_col = f'{inv}_{duration}D_Z'

        df[mean_col] = df.groupby('Code')[col].transform(
            lambda x: x.rolling(window=window, min_periods=window).mean()
        )
        df[std_col] = df.groupby('Code')[col].transform(
            lambda x: x.rolling(window=window, min_periods=window).std()
        )
        df[z_col] = (df[col] - df[mean_col]) / df[std_col]
        df[z_col] = df[z_col].replace([np.inf, -np.inf], np.nan)

        # Clean up temp columns
        df.drop(columns=[mean_col, std_col], inplace=True, errors='ignore')

    return df


def apply_base_signal(df):
    """Base Signal 적용: VR >= 3.0 AND Price >= 5%"""
    df['Price_Change'] = df.groupby('Code')['Close'].transform(
        lambda x: (x / x.shift(1) - 1) * 100
    )

    base_signals = df[
        (df['Volume_Ratio'] >= 3.0) &
        (df['Price_Change'] >= 5.0)
    ].copy()

    return base_signals


def evaluate_strategy(signals, strategy_name):
    """전략 성과 평가 (20D Return 기준)"""
    if len(signals) < 10:
        return None

    valid = signals.dropna(subset=['Return_20D'])
    if len(valid) < 10:
        return None

    return {
        'Strategy': strategy_name,
        'Signals': len(valid),
        'Avg_Return_20D': valid['Return_20D'].mean(),
        'Win_Rate_20D': (valid['Return_20D'] > 0).mean() * 100,
        'Median_20D': valid['Return_20D'].median(),
        'Std_20D': valid['Return_20D'].std(),
    }


def check_sell_constraint(combo_directions, n_investors):
    """Anti-Overfitting Constraint: SELL 개수 체크

    - 3주체 (n=3): 반드시 1개 이상의 SELL
    - 4주체 (n=4): 반드시 2개 이상의 SELL
    """
    sell_count = sum(1 for d in combo_directions if d == 'SELL')

    if n_investors == 3:
        return sell_count >= 1
    elif n_investors == 4:
        return sell_count >= 2
    else:
        return True  # 1주체, 2주체는 제약 없음


def run_track_a_constrained(signals_is, signals_oos, duration):
    """Track A: Constrained Combinatorial (Anti-Overfitting)"""
    investors = ['retail', 'foreign', 'fininvest', 'pension']
    investor_labels = {'retail': 'Retail', 'foreign': 'Foreign', 'fininvest': 'FinInvest', 'pension': 'Pension'}
    directions = ['BUY', 'SELL']

    results_is = []
    results_oos = []

    col_suffix = f'_{duration}D'

    for n in range(1, 5):
        for investor_combo in combinations(investors, n):
            for dir_combo in product(directions, repeat=n):
                # Anti-Overfitting Constraint Check
                if not check_sell_constraint(dir_combo, n):
                    continue

                conditions = list(zip(investor_combo, dir_combo))
                strategy_name = " & ".join([f"{investor_labels[inv]}_{dir}" for inv, dir in conditions])

                # IS Filter
                mask_is = pd.Series([True] * len(signals_is), index=signals_is.index)
                for inv, direction in conditions:
                    col = f'{inv}{col_suffix}'
                    if col not in signals_is.columns:
                        mask_is = pd.Series([False] * len(signals_is), index=signals_is.index)
                        break
                    if direction == 'BUY':
                        mask_is &= (signals_is[col] > 0)
                    else:
                        mask_is &= (signals_is[col] <= 0)

                filtered_is = signals_is[mask_is]
                result_is = evaluate_strategy(filtered_is, strategy_name)
                if result_is:
                    result_is['Period'] = 'IS'
                    result_is['Track'] = 'A'
                    result_is['Duration'] = duration
                    result_is['N_Investors'] = n
                    results_is.append(result_is)

                # OOS Filter
                mask_oos = pd.Series([True] * len(signals_oos), index=signals_oos.index)
                for inv, direction in conditions:
                    col = f'{inv}{col_suffix}'
                    if col not in signals_oos.columns:
                        mask_oos = pd.Series([False] * len(signals_oos), index=signals_oos.index)
                        break
                    if direction == 'BUY':
                        mask_oos &= (signals_oos[col] > 0)
                    else:
                        mask_oos &= (signals_oos[col] <= 0)

                filtered_oos = signals_oos[mask_oos]
                result_oos = evaluate_strategy(filtered_oos, strategy_name)
                if result_oos:
                    result_oos['Period'] = 'OOS'
                    result_oos['Track'] = 'A'
                    result_oos['Duration'] = duration
                    result_oos['N_Investors'] = n
                    results_oos.append(result_oos)

    return pd.DataFrame(results_is), pd.DataFrame(results_oos)


def run_track_b_classic(signals_is, signals_oos, duration):
    """Track B: Classic Academic with D-Day Accumulated Supply"""
    col_suffix = f'_{duration}D'

    scenarios = {
        'Smart_Alignment': {
            'desc': 'Foreign BUY AND Pension BUY',
            'filter': lambda df: (df[f'foreign{col_suffix}'] > 0) & (df[f'pension{col_suffix}'] > 0)
        },
        'Retail_Exhaustion': {
            'desc': 'Retail SELL',
            'filter': lambda df: df[f'retail{col_suffix}'] <= 0
        },
        'Handover': {
            'desc': 'Retail SELL AND (Foreign OR Pension BUY)',
            'filter': lambda df: (df[f'retail{col_suffix}'] <= 0) &
                                  ((df[f'foreign{col_suffix}'] > 0) | (df[f'pension{col_suffix}'] > 0))
        },
        'Pension_SELL': {
            'desc': 'Pension SELL only',
            'filter': lambda df: df[f'pension{col_suffix}'] <= 0
        }
    }

    results_is = []
    results_oos = []

    for scenario_name, scenario in scenarios.items():
        try:
            # IS
            filtered_is = signals_is[scenario['filter'](signals_is)]
            result_is = evaluate_strategy(filtered_is, scenario_name)
            if result_is:
                result_is['Period'] = 'IS'
                result_is['Track'] = 'B'
                result_is['Duration'] = duration
                result_is['Description'] = scenario['desc']
                results_is.append(result_is)

            # OOS
            filtered_oos = signals_oos[scenario['filter'](signals_oos)]
            result_oos = evaluate_strategy(filtered_oos, scenario_name)
            if result_oos:
                result_oos['Period'] = 'OOS'
                result_oos['Track'] = 'B'
                result_oos['Duration'] = duration
                result_oos['Description'] = scenario['desc']
                results_oos.append(result_oos)
        except KeyError:
            continue

    return pd.DataFrame(results_is), pd.DataFrame(results_oos)


def run_track_c_zscore(signals_is, signals_oos, duration):
    """Track C: Z-Score on D-Day Accumulated Supply"""
    z_suffix = f'_{duration}D_Z'

    scenarios = {
        'Foreign_Z_gt2': {
            'desc': f'Foreign {duration}D Z > 2.0',
            'filter': lambda df: df[f'foreign{z_suffix}'] > 2.0
        },
        'Retail_Z_lt_neg2': {
            'desc': f'Retail {duration}D Z < -2.0',
            'filter': lambda df: df[f'retail{z_suffix}'] < -2.0
        },
        'Foreign_OR_Retail_Extreme': {
            'desc': f'Foreign {duration}D Z > 2 OR Retail {duration}D Z < -2',
            'filter': lambda df: (df[f'foreign{z_suffix}'] > 2.0) | (df[f'retail{z_suffix}'] < -2.0)
        },
        'Foreign_Z_gt1': {
            'desc': f'Foreign {duration}D Z > 1.0 (moderate)',
            'filter': lambda df: df[f'foreign{z_suffix}'] > 1.0
        },
        'Pension_Z_lt_neg1': {
            'desc': f'Pension {duration}D Z < -1.0',
            'filter': lambda df: df[f'pension{z_suffix}'] < -1.0
        }
    }

    results_is = []
    results_oos = []

    for scenario_name, scenario in scenarios.items():
        try:
            # IS - Drop NaN Z-scores
            valid_cols = [f'foreign{z_suffix}', f'retail{z_suffix}', f'pension{z_suffix}', f'fininvest{z_suffix}']
            valid_cols = [c for c in valid_cols if c in signals_is.columns]
            if not valid_cols:
                continue

            valid_is = signals_is.dropna(subset=valid_cols)
            if len(valid_is) > 0:
                filtered_is = valid_is[scenario['filter'](valid_is)]
                result_is = evaluate_strategy(filtered_is, scenario_name)
                if result_is:
                    result_is['Period'] = 'IS'
                    result_is['Track'] = 'C'
                    result_is['Duration'] = duration
                    result_is['Description'] = scenario['desc']
                    results_is.append(result_is)

            # OOS
            valid_oos = signals_oos.dropna(subset=valid_cols)
            if len(valid_oos) > 0:
                filtered_oos = valid_oos[scenario['filter'](valid_oos)]
                result_oos = evaluate_strategy(filtered_oos, scenario_name)
                if result_oos:
                    result_oos['Period'] = 'OOS'
                    result_oos['Track'] = 'C'
                    result_oos['Duration'] = duration
                    result_oos['Description'] = scenario['desc']
                    results_oos.append(result_oos)
        except KeyError:
            continue

    return pd.DataFrame(results_is), pd.DataFrame(results_oos)


def create_heatmap_data(all_results):
    """Heatmap 데이터 생성: rows=strategies, cols=durations, values=OOS returns"""
    oos_results = all_results[all_results['Period'] == 'OOS'].copy()

    if len(oos_results) == 0:
        return pd.DataFrame()

    # Pivot: Strategy x Duration
    heatmap = oos_results.pivot_table(
        index='Strategy',
        columns='Duration',
        values='Avg_Return_20D',
        aggfunc='mean'
    )

    # Sort by average OOS return
    heatmap['Mean'] = heatmap.mean(axis=1)
    heatmap = heatmap.sort_values('Mean', ascending=False)

    return heatmap


def analyze_duration_correlation(heatmap_df):
    """Duration과 수익률 간 상관관계 분석"""
    print("\n" + "=" * 80)
    print("DURATION CORRELATION ANALYSIS")
    print("=" * 80)

    if heatmap_df.empty:
        print("No data available for analysis")
        return

    # 각 Duration별 평균 수익률
    duration_cols = [c for c in heatmap_df.columns if isinstance(c, int)]

    print("\n### Average OOS Return by Duration")
    print("-" * 50)
    for d in sorted(duration_cols):
        avg = heatmap_df[d].mean()
        print(f"   Duration {d:2d}D: {avg:6.2f}%")

    # 최적 Duration 찾기
    duration_avgs = {d: heatmap_df[d].mean() for d in duration_cols}
    best_duration = max(duration_avgs, key=duration_avgs.get)

    print(f"\n   [BEST] Duration {best_duration}D with {duration_avgs[best_duration]:.2f}% avg return")

    # 20D 보유 vs 20D 분석 매칭 검증
    if 20 in duration_cols:
        d20_avg = heatmap_df[20].mean()
        other_avg = np.mean([duration_avgs[d] for d in duration_cols if d != 20])

        print("\n### Hypothesis Test: '20D holding needs 20D supply analysis'")
        print("-" * 50)
        print(f"   20D Supply Analysis avg: {d20_avg:.2f}%")
        print(f"   Other Durations avg:     {other_avg:.2f}%")

        if d20_avg > other_avg:
            print(f"   Result: CONFIRMED - 20D analysis outperforms by {d20_avg - other_avg:.2f}%p")
        else:
            print(f"   Result: REJECTED - Other durations outperform by {other_avg - d20_avg:.2f}%p")


def print_top_strategies(all_results):
    """Track별 Top 전략 출력"""
    print("\n" + "=" * 80)
    print("TOP STRATEGIES BY TRACK (OOS Performance)")
    print("=" * 80)

    oos_results = all_results[all_results['Period'] == 'OOS'].copy()

    for track in ['A', 'B', 'C']:
        track_data = oos_results[oos_results['Track'] == track]

        if len(track_data) == 0:
            continue

        track_name = {'A': 'Constrained Combinatorial', 'B': 'Classic Academic', 'C': 'Z-Score'}[track]
        print(f"\n### Track {track}: {track_name} (Top 5)")
        print("-" * 80)
        print(f"{'Rank':<5} {'Strategy':<40} {'Duration':>8} {'Return':>10} {'WinRate':>10}")
        print("-" * 80)

        top5 = track_data.nlargest(5, 'Avg_Return_20D')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"{i:<5} {row['Strategy'][:40]:<40} {row['Duration']:>6}D {row['Avg_Return_20D']:>9.2f}% {row['Win_Rate_20D']:>9.1f}%")


def main():
    # 1. 데이터 로드
    merged = load_base_data()

    # 2. 모든 Duration에 대한 Rolling Sum 및 Z-Score 계산
    print("\n[2/6] Calculating Rolling Sums for all durations...")
    for duration in DURATION_LIST:
        print(f"   Processing {duration}D rolling sum...")
        merged = calculate_rolling_sums(merged, duration)

    print("\n[3/6] Calculating Z-Scores for all durations...")
    for duration in DURATION_LIST:
        print(f"   Processing {duration}D Z-scores...")
        merged = calculate_z_scores_for_duration(merged, duration, window=60)

    # 3. Base Signal 적용
    print("\n[4/6] Applying Base Signal (VR >= 3.0, Price >= 5%)...")
    base_signals = apply_base_signal(merged)
    print(f"   Base signals: {len(base_signals):,}")

    # 4. IS/OOS 분리
    base_signals['Year'] = base_signals['Date'].dt.year
    signals_is = base_signals[base_signals['Year'] <= 2024].copy()
    signals_oos = base_signals[base_signals['Year'] == 2025].copy()

    print(f"   IS (2021-2024): {len(signals_is):,}")
    print(f"   OOS (2025): {len(signals_oos):,}")

    # 5. 각 Duration에 대해 Tournament 실행
    print("\n[5/6] Running Multi-Duration Tournament...")

    all_results = []

    for duration in DURATION_LIST:
        print(f"\n   === Duration {duration}D ===")

        # Track A
        track_a_is, track_a_oos = run_track_a_constrained(signals_is, signals_oos, duration)
        print(f"   Track A: IS={len(track_a_is)}, OOS={len(track_a_oos)} strategies")

        # Track B
        track_b_is, track_b_oos = run_track_b_classic(signals_is, signals_oos, duration)
        print(f"   Track B: IS={len(track_b_is)}, OOS={len(track_b_oos)} strategies")

        # Track C
        track_c_is, track_c_oos = run_track_c_zscore(signals_is, signals_oos, duration)
        print(f"   Track C: IS={len(track_c_is)}, OOS={len(track_c_oos)} strategies")

        # Combine
        for df in [track_a_is, track_a_oos, track_b_is, track_b_oos, track_c_is, track_c_oos]:
            if len(df) > 0:
                all_results.append(df)

    all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    # 6. 분석 및 리포트
    print("\n[6/6] Generating Analysis...")

    # Heatmap 데이터 생성
    heatmap_df = create_heatmap_data(all_results_df)

    # Top 전략 출력
    print_top_strategies(all_results_df)

    # Duration 상관관계 분석
    analyze_duration_correlation(heatmap_df)

    # 결과 저장
    output_dir = project_root / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 결과
    all_results_df.to_csv(output_dir / 'p3_03_multi_duration_results.csv',
                          index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] {output_dir / 'p3_03_multi_duration_results.csv'}")

    # Heatmap 데이터
    if not heatmap_df.empty:
        heatmap_df.to_csv(output_dir / 'p3_03_multi_duration_heatmap.csv',
                          encoding='utf-8-sig')
        print(f"[SAVED] {output_dir / 'p3_03_multi_duration_heatmap.csv'}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    oos_df = all_results_df[all_results_df['Period'] == 'OOS']
    if len(oos_df) > 0:
        best = oos_df.loc[oos_df['Avg_Return_20D'].idxmax()]
        print(f"\n[TOURNAMENT WINNER]")
        print(f"   Strategy: {best['Strategy']}")
        print(f"   Track: {best['Track']}")
        print(f"   Duration: {best['Duration']}D")
        print(f"   OOS Return: {best['Avg_Return_20D']:.2f}%")
        print(f"   OOS Win Rate: {best['Win_Rate_20D']:.1f}%")
        print(f"   OOS Signals: {best['Signals']:.0f}")

    print("\n" + "=" * 80)
    print("Phase 3-3 Extended: Multi-Duration Tournament Complete!")
    print("=" * 80)

    return all_results_df, heatmap_df


if __name__ == "__main__":
    results, heatmap = main()
