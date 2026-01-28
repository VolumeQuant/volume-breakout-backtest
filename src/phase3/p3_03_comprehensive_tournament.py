"""
Phase 3-3: Triple Tournament
============================

3개의 서로 다른 철학을 가진 전략을 동시에 경쟁시키는 트리플 토너먼트

Track A: 4Cn 전수조사 (The Bruteforce)
- 80개 조합 테스트 (개인/외국인/금투/연기금 x Buy/Sell)

Track B: Classic 논문 기반 (The Logic)
- Smart Alignment: 외국인 BUY AND 연기금 BUY
- Retail Exhaustion: 개인 SELL
- Handover: 개인 SELL AND (외국인 OR 연기금 BUY)

Track C: Modern SOTA (The Z-Score)
- 60일 Rolling Z-Score 기반 이상치 탐지
- Foreigner_Z > 2.0 OR Retail_Z < -2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("=" * 80)
    print("Phase 3-3: Triple Tournament")
    print("Combinatorial vs Classic vs Modern SOTA")
    print("=" * 80)

    data_dir = project_root / 'data'

    print("\n[1/5] Loading data...")
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    try:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='utf-8-sig', low_memory=False)
    except:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)

    # 수급 데이터 전처리
    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        '금융투자': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        '연기금': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        '외국인': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })
    flow_data['개인'] = -(flow_data['금융투자'] + flow_data['연기금'] + flow_data['외국인'])

    # 병합
    merged = stock_data.merge(flow_data, on=['Code', 'Date'], how='inner')
    merged = merged.sort_values(['Code', 'Date'])
    print(f"   Merged data: {len(merged):,} rows")

    return merged


def calculate_price_change(df):
    """당일 수익률 계산"""
    df['Price_Change'] = df.groupby('Code')['Close'].transform(
        lambda x: (x / x.shift(1) - 1) * 100
    )
    return df


def calculate_z_scores(df, window=60):
    """Track C용 Z-Score 계산 (60일 Rolling)"""
    print("\n[2/5] Calculating Z-Scores (60-day rolling)...")

    investors = ['개인', '외국인', '금융투자', '연기금']

    for investor in investors:
        # Rolling mean and std
        df[f'{investor}_mean'] = df.groupby('Code')[investor].transform(
            lambda x: x.rolling(window=window, min_periods=window).mean()
        )
        df[f'{investor}_std'] = df.groupby('Code')[investor].transform(
            lambda x: x.rolling(window=window, min_periods=window).std()
        )
        # Z-Score
        df[f'{investor}_Z'] = (df[investor] - df[f'{investor}_mean']) / df[f'{investor}_std']
        df[f'{investor}_Z'] = df[f'{investor}_Z'].replace([np.inf, -np.inf], np.nan)

    return df


def apply_base_signal(df):
    """Base Signal 적용: VR >= 3.0 AND Price >= 5%"""
    print("\n[3/5] Applying Base Signal (VR >= 3.0, Price >= 5%)...")

    df = calculate_price_change(df)

    base_signals = df[
        (df['Volume_Ratio'] >= 3.0) &
        (df['Price_Change'] >= 5.0)
    ].copy()

    print(f"   Base signals: {len(base_signals):,}")
    return base_signals


def evaluate_strategy(signals, strategy_name):
    """전략 성과 평가"""
    if len(signals) < 10:
        return None

    valid_10d = signals.dropna(subset=['Return_10D'])
    valid_20d = signals.dropna(subset=['Return_20D'])

    if len(valid_20d) < 10:
        return None

    return {
        'Strategy': strategy_name,
        'Signals': len(valid_20d),
        'Avg_Return_10D': valid_10d['Return_10D'].mean() if len(valid_10d) >= 10 else np.nan,
        'Avg_Return_20D': valid_20d['Return_20D'].mean(),
        'Win_Rate_10D': (valid_10d['Return_10D'] > 0).mean() * 100 if len(valid_10d) >= 10 else np.nan,
        'Win_Rate_20D': (valid_20d['Return_20D'] > 0).mean() * 100,
        'Median_Return_20D': valid_20d['Return_20D'].median(),
        'Std_Return_20D': valid_20d['Return_20D'].std(),
        'Max_Gain_20D': valid_20d['Return_20D'].max(),
        'Max_Loss_20D': valid_20d['Return_20D'].min(),
    }


# ==============================================================================
# Track A: 4Cn 전수조사 (The Bruteforce)
# ==============================================================================

def run_track_a(signals_is, signals_oos):
    """Track A: 모든 조합 전수조사 (80개)"""
    print("\n" + "=" * 80)
    print("TRACK A: Combinatorial Bruteforce (4Cn)")
    print("=" * 80)

    investors = ['개인', '외국인', '금융투자', '연기금']
    directions = ['BUY', 'SELL']

    all_results_is = []
    all_results_oos = []

    # 모든 조합 생성 (1개~4개 선택)
    combo_count = 0
    for n in range(1, 5):  # 1C4 to 4C4
        for investor_combo in combinations(investors, n):
            # 각 투자자에 대해 BUY/SELL 조합
            direction_combos = list(product(directions, repeat=n))

            for dir_combo in direction_combos:
                combo_count += 1
                conditions = list(zip(investor_combo, dir_combo))
                strategy_name = " & ".join([f"{inv}_{dir}" for inv, dir in conditions])

                # IS 필터링
                mask_is = pd.Series([True] * len(signals_is), index=signals_is.index)
                for inv, direction in conditions:
                    if direction == 'BUY':
                        mask_is &= (signals_is[inv] > 0)
                    else:  # SELL
                        mask_is &= (signals_is[inv] <= 0)

                filtered_is = signals_is[mask_is]
                result_is = evaluate_strategy(filtered_is, strategy_name)
                if result_is:
                    result_is['Period'] = 'IS'
                    result_is['Track'] = 'A'
                    result_is['Combo_Size'] = n
                    all_results_is.append(result_is)

                # OOS 필터링
                mask_oos = pd.Series([True] * len(signals_oos), index=signals_oos.index)
                for inv, direction in conditions:
                    if direction == 'BUY':
                        mask_oos &= (signals_oos[inv] > 0)
                    else:
                        mask_oos &= (signals_oos[inv] <= 0)

                filtered_oos = signals_oos[mask_oos]
                result_oos = evaluate_strategy(filtered_oos, strategy_name)
                if result_oos:
                    result_oos['Period'] = 'OOS'
                    result_oos['Track'] = 'A'
                    result_oos['Combo_Size'] = n
                    all_results_oos.append(result_oos)

    print(f"   Total combinations tested: {combo_count}")
    print(f"   Valid IS results: {len(all_results_is)}")
    print(f"   Valid OOS results: {len(all_results_oos)}")

    return pd.DataFrame(all_results_is), pd.DataFrame(all_results_oos)


# ==============================================================================
# Track B: Classic 논문 기반 (The Logic)
# ==============================================================================

def run_track_b(signals_is, signals_oos):
    """Track B: Classic Academic Scenarios"""
    print("\n" + "=" * 80)
    print("TRACK B: Classic Academic Logic")
    print("=" * 80)

    scenarios = {
        'Smart_Alignment': {
            'desc': '외국인 BUY AND 연기금 BUY',
            'filter': lambda df: (df['외국인'] > 0) & (df['연기금'] > 0)
        },
        'Retail_Exhaustion': {
            'desc': '개인 SELL (투매)',
            'filter': lambda df: df['개인'] <= 0
        },
        'Handover': {
            'desc': '개인 SELL AND (외국인 OR 연기금 BUY)',
            'filter': lambda df: (df['개인'] <= 0) & ((df['외국인'] > 0) | (df['연기금'] > 0))
        }
    }

    all_results_is = []
    all_results_oos = []

    for scenario_name, scenario in scenarios.items():
        print(f"   Testing: {scenario_name} - {scenario['desc']}")

        # IS
        filtered_is = signals_is[scenario['filter'](signals_is)]
        result_is = evaluate_strategy(filtered_is, scenario_name)
        if result_is:
            result_is['Period'] = 'IS'
            result_is['Track'] = 'B'
            result_is['Description'] = scenario['desc']
            all_results_is.append(result_is)

        # OOS
        filtered_oos = signals_oos[scenario['filter'](signals_oos)]
        result_oos = evaluate_strategy(filtered_oos, scenario_name)
        if result_oos:
            result_oos['Period'] = 'OOS'
            result_oos['Track'] = 'B'
            result_oos['Description'] = scenario['desc']
            all_results_oos.append(result_oos)

    return pd.DataFrame(all_results_is), pd.DataFrame(all_results_oos)


# ==============================================================================
# Track C: Modern SOTA (The Z-Score)
# ==============================================================================

def run_track_c(signals_is, signals_oos):
    """Track C: Modern Z-Score Based Anomaly Detection"""
    print("\n" + "=" * 80)
    print("TRACK C: Modern SOTA (Z-Score Anomaly)")
    print("=" * 80)

    scenarios = {
        'Foreigner_Extreme_Buy': {
            'desc': '외국인 Z > 2.0 (2시그마 폭매수)',
            'filter': lambda df: df['외국인_Z'] > 2.0
        },
        'Retail_Extreme_Sell': {
            'desc': '개인 Z < -2.0 (2시그마 투매)',
            'filter': lambda df: df['개인_Z'] < -2.0
        },
        'Foreigner_OR_Retail_Extreme': {
            'desc': '외국인 Z > 2.0 OR 개인 Z < -2.0',
            'filter': lambda df: (df['외국인_Z'] > 2.0) | (df['개인_Z'] < -2.0)
        },
        'Institution_Extreme_Buy': {
            'desc': '금융투자 Z > 2.0 (기관 폭매수)',
            'filter': lambda df: df['금융투자_Z'] > 2.0
        },
        'Smart_Money_Extreme': {
            'desc': '(외국인 Z > 1.5 OR 연기금 Z > 1.5) AND 개인 Z < -1.0',
            'filter': lambda df: ((df['외국인_Z'] > 1.5) | (df['연기금_Z'] > 1.5)) & (df['개인_Z'] < -1.0)
        },
        'Foreigner_Moderate': {
            'desc': '외국인 Z > 1.0 (1시그마 이상)',
            'filter': lambda df: df['외국인_Z'] > 1.0
        },
        'Retail_Moderate_Sell': {
            'desc': '개인 Z < -1.0 (1시그마 이상 매도)',
            'filter': lambda df: df['개인_Z'] < -1.0
        }
    }

    all_results_is = []
    all_results_oos = []

    for scenario_name, scenario in scenarios.items():
        print(f"   Testing: {scenario_name}")

        # IS - Z-Score NaN 처리
        valid_is = signals_is.dropna(subset=['외국인_Z', '개인_Z', '금융투자_Z', '연기금_Z'])
        if len(valid_is) > 0:
            filtered_is = valid_is[scenario['filter'](valid_is)]
            result_is = evaluate_strategy(filtered_is, scenario_name)
            if result_is:
                result_is['Period'] = 'IS'
                result_is['Track'] = 'C'
                result_is['Description'] = scenario['desc']
                all_results_is.append(result_is)

        # OOS
        valid_oos = signals_oos.dropna(subset=['외국인_Z', '개인_Z', '금융투자_Z', '연기금_Z'])
        if len(valid_oos) > 0:
            filtered_oos = valid_oos[scenario['filter'](valid_oos)]
            result_oos = evaluate_strategy(filtered_oos, scenario_name)
            if result_oos:
                result_oos['Period'] = 'OOS'
                result_oos['Track'] = 'C'
                result_oos['Description'] = scenario['desc']
                all_results_oos.append(result_oos)

    return pd.DataFrame(all_results_is), pd.DataFrame(all_results_oos)


def select_champions(track_a_is, track_b_is, track_c_is):
    """각 트랙에서 IS 성과 최우수 Champion 선발"""
    print("\n" + "=" * 80)
    print("CHAMPION SELECTION (IS Top 1 per Track)")
    print("=" * 80)

    champions = {}

    # Track A Champion
    if len(track_a_is) > 0:
        track_a_sorted = track_a_is.sort_values('Avg_Return_20D', ascending=False)
        champions['A'] = track_a_sorted.iloc[0]
        print(f"\n[Track A Champion] {champions['A']['Strategy']}")
        print(f"   20D Return: {champions['A']['Avg_Return_20D']:.2f}%")
        print(f"   Win Rate: {champions['A']['Win_Rate_20D']:.1f}%")
        print(f"   Signals: {champions['A']['Signals']}")

    # Track B Champion
    if len(track_b_is) > 0:
        track_b_sorted = track_b_is.sort_values('Avg_Return_20D', ascending=False)
        champions['B'] = track_b_sorted.iloc[0]
        print(f"\n[Track B Champion] {champions['B']['Strategy']}")
        print(f"   20D Return: {champions['B']['Avg_Return_20D']:.2f}%")
        print(f"   Win Rate: {champions['B']['Win_Rate_20D']:.1f}%")
        print(f"   Signals: {champions['B']['Signals']}")

    # Track C Champion
    if len(track_c_is) > 0:
        track_c_sorted = track_c_is.sort_values('Avg_Return_20D', ascending=False)
        champions['C'] = track_c_sorted.iloc[0]
        print(f"\n[Track C Champion] {champions['C']['Strategy']}")
        print(f"   20D Return: {champions['C']['Avg_Return_20D']:.2f}%")
        print(f"   Win Rate: {champions['C']['Win_Rate_20D']:.1f}%")
        print(f"   Signals: {champions['C']['Signals']}")

    return champions


def validate_champions_oos(champions, track_a_oos, track_b_oos, track_c_oos):
    """Champion들의 OOS 성과 검증"""
    print("\n" + "=" * 80)
    print("OOS VALIDATION (2025)")
    print("=" * 80)

    validation_results = []

    for track, champion in champions.items():
        strategy_name = champion['Strategy']

        # 해당 트랙의 OOS 결과에서 동일 전략 찾기
        if track == 'A':
            oos_df = track_a_oos
        elif track == 'B':
            oos_df = track_b_oos
        else:
            oos_df = track_c_oos

        oos_match = oos_df[oos_df['Strategy'] == strategy_name]

        if len(oos_match) > 0:
            oos_result = oos_match.iloc[0]
            maintained = (oos_result['Avg_Return_20D'] > 0) and (oos_result['Win_Rate_20D'] > 45)
            status = "[OK]" if maintained else "[FAIL]"

            print(f"\n[Track {track}] {strategy_name}")
            print(f"   IS  -> 20D: {champion['Avg_Return_20D']:.2f}%, WR: {champion['Win_Rate_20D']:.1f}%, Signals: {champion['Signals']}")
            print(f"   OOS -> 20D: {oos_result['Avg_Return_20D']:.2f}%, WR: {oos_result['Win_Rate_20D']:.1f}%, Signals: {oos_result['Signals']} {status}")

            validation_results.append({
                'Track': track,
                'Strategy': strategy_name,
                'IS_Return': champion['Avg_Return_20D'],
                'OOS_Return': oos_result['Avg_Return_20D'],
                'IS_WinRate': champion['Win_Rate_20D'],
                'OOS_WinRate': oos_result['Win_Rate_20D'],
                'IS_Signals': champion['Signals'],
                'OOS_Signals': oos_result['Signals'],
                'Maintained': maintained
            })
        else:
            print(f"\n[Track {track}] {strategy_name}")
            print(f"   IS  -> 20D: {champion['Avg_Return_20D']:.2f}%")
            print(f"   OOS -> NO DATA")

    return pd.DataFrame(validation_results)


def generate_final_report(track_a_is, track_b_is, track_c_is,
                          track_a_oos, track_b_oos, track_c_oos,
                          champions, validation_df):
    """최종 리포트 생성"""
    print("\n" + "=" * 80)
    print("FINAL TOURNAMENT RESULTS")
    print("=" * 80)

    # Track A Top 5
    print("\n### Track A: Combinatorial (Top 5 IS)")
    print("-" * 70)
    if len(track_a_is) > 0:
        top5_a = track_a_is.nlargest(5, 'Avg_Return_20D')
        print("Rank | Strategy                              | 20D Return | WinRate | Signals")
        print("-" * 80)
        for i, (_, row) in enumerate(top5_a.iterrows(), 1):
            print(f"{i:4d} | {row['Strategy'][:40]:40s} | {row['Avg_Return_20D']:9.2f}% | {row['Win_Rate_20D']:6.1f}% | {row['Signals']:7.0f}")

    # Track B Results
    print("\n### Track B: Classic Academic (All)")
    print("-" * 70)
    if len(track_b_is) > 0:
        print("Strategy               | 20D Return | WinRate | Signals | Description")
        print("-" * 80)
        for _, row in track_b_is.iterrows():
            desc = row.get('Description', '')[:25]
            print(f"{row['Strategy']:22s} | {row['Avg_Return_20D']:9.2f}% | {row['Win_Rate_20D']:6.1f}% | {row['Signals']:7.0f} | {desc}")

    # Track C Top 5
    print("\n### Track C: Modern Z-Score (Top 5 IS)")
    print("-" * 70)
    if len(track_c_is) > 0:
        top5_c = track_c_is.nlargest(5, 'Avg_Return_20D')
        print("Rank | Strategy                         | 20D Return | WinRate | Signals")
        print("-" * 80)
        for i, (_, row) in enumerate(top5_c.iterrows(), 1):
            print(f"{i:4d} | {row['Strategy'][:35]:35s} | {row['Avg_Return_20D']:9.2f}% | {row['Win_Rate_20D']:6.1f}% | {row['Signals']:7.0f}")

    # Final Ranking
    print("\n" + "=" * 80)
    print("TOURNAMENT WINNER")
    print("=" * 80)

    if len(validation_df) > 0:
        # OOS 기준 최종 순위
        validation_df_sorted = validation_df.sort_values('OOS_Return', ascending=False)
        winner = validation_df_sorted.iloc[0]

        print(f"\n[WINNER] Track {winner['Track']}: {winner['Strategy']}")
        print(f"   OOS 20D Return: {winner['OOS_Return']:.2f}%")
        print(f"   OOS Win Rate: {winner['OOS_WinRate']:.1f}%")
        print(f"   OOS Signals: {winner['OOS_Signals']:.0f}")

        print("\n### Final Ranking (OOS Performance)")
        print("-" * 70)
        print("Rank | Track | Strategy                    | IS Return | OOS Return | Status")
        print("-" * 80)
        for i, (_, row) in enumerate(validation_df_sorted.iterrows(), 1):
            status = "[OK]" if row['Maintained'] else "[FAIL]"
            print(f"{i:4d} | {row['Track']:5s} | {row['Strategy'][:28]:28s} | {row['IS_Return']:8.2f}% | {row['OOS_Return']:9.2f}% | {status}")

    return validation_df


def main():
    # 1. 데이터 로드
    merged = load_and_prepare_data()

    # 2. Z-Score 계산 (Track C용)
    merged = calculate_z_scores(merged, window=60)

    # 3. Base Signal 적용
    base_signals = apply_base_signal(merged)

    # 4. IS/OOS 분리
    print("\n[4/5] Splitting IS/OOS...")
    base_signals['Year'] = base_signals['Date'].dt.year
    signals_is = base_signals[base_signals['Year'] <= 2024].copy()
    signals_oos = base_signals[base_signals['Year'] == 2025].copy()

    print(f"   IS (2021-2024): {len(signals_is):,} signals")
    print(f"   OOS (2025): {len(signals_oos):,} signals")

    # 5. 트랙별 실행
    print("\n[5/5] Running Tournament...")

    # Track A: Combinatorial
    track_a_is, track_a_oos = run_track_a(signals_is, signals_oos)

    # Track B: Classic
    track_b_is, track_b_oos = run_track_b(signals_is, signals_oos)

    # Track C: Z-Score
    track_c_is, track_c_oos = run_track_c(signals_is, signals_oos)

    # 6. Champion 선발
    champions = select_champions(track_a_is, track_b_is, track_c_is)

    # 7. OOS 검증
    validation_df = validate_champions_oos(champions, track_a_oos, track_b_oos, track_c_oos)

    # 8. 최종 리포트
    final_report = generate_final_report(
        track_a_is, track_b_is, track_c_is,
        track_a_oos, track_b_oos, track_c_oos,
        champions, validation_df
    )

    # 9. 결과 저장
    output_dir = project_root / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 통합 결과 저장
    all_is = pd.concat([track_a_is, track_b_is, track_c_is], ignore_index=True)
    all_oos = pd.concat([track_a_oos, track_b_oos, track_c_oos], ignore_index=True)
    all_results = pd.concat([all_is, all_oos], ignore_index=True)

    all_results.to_csv(output_dir / 'p3_03_tournament_results.csv',
                       index=False, encoding='utf-8-sig')
    print(f"\nResults saved: {output_dir / 'p3_03_tournament_results.csv'}")

    # Validation 결과 저장
    if len(validation_df) > 0:
        validation_df.to_csv(output_dir / 'p3_03_tournament_champions.csv',
                            index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("Phase 3-3 Triple Tournament Complete!")
    print("=" * 80)

    return all_results, validation_df


if __name__ == "__main__":
    results, validation = main()
