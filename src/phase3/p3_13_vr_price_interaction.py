"""
Phase 3-13: VR × PRICE Interaction Analysis
VR과 PRICE의 상호작용 분석

기존: VR≥X AND Price≥Y (독립 조건)
새로운 접근:
1. VR × Price (곱)
2. VR + Price (합)
3. 가중합: α×VR + β×Price
4. 조합 점수 = VR^a × Price^b

직관: 거래량이 많이 터지면 가격은 조금만 올라도 되고,
      가격이 크게 오르면 거래량은 적어도 된다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'p3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOLDING_PERIOD = 10

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")

    stock_df = pd.read_csv(
        DATA_DIR / 'stock_data_with_indicators.csv',
        parse_dates=['Date']
    )

    # VR 매핑
    if 'VR' not in stock_df.columns:
        if 'Volume_Ratio' in stock_df.columns:
            stock_df['VR'] = stock_df['Volume_Ratio']

    # Price_Change 계산
    if 'Price_Change' not in stock_df.columns:
        if 'Change' in stock_df.columns:
            stock_df['Price_Change'] = stock_df['Change']
        elif 'Close' in stock_df.columns and 'Open' in stock_df.columns:
            stock_df['Price_Change'] = ((stock_df['Close'] - stock_df['Open']) / stock_df['Open'] * 100)

    print(f"전체 데이터: {len(stock_df):,}건")
    return stock_df

def calculate_mdd(returns_series):
    """MDD 계산"""
    if len(returns_series) == 0:
        return 0
    cumulative = (1 + returns_series / 100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    mdd = drawdown.min()
    return mdd

def backtest_interaction(df, score_type, threshold, params=None):
    """상호작용 점수 기반 백테스팅

    score_type:
    - 'multiply': VR × Price
    - 'add': VR + Price
    - 'weighted': w1×VR + w2×Price
    - 'power': VR^a × Price^b
    """

    return_col = f'Return_{HOLDING_PERIOD}D'

    # 점수 계산
    if score_type == 'multiply':
        df['Score'] = df['VR'] * df['Price_Change']
    elif score_type == 'add':
        df['Score'] = df['VR'] + df['Price_Change']
    elif score_type == 'weighted':
        w1, w2 = params['w1'], params['w2']
        df['Score'] = w1 * df['VR'] + w2 * df['Price_Change']
    elif score_type == 'power':
        a, b = params['a'], params['b']
        df['Score'] = (df['VR'] ** a) * (df['Price_Change'] ** b)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    # 필터링
    signals = df[df['Score'] >= threshold].copy()
    signals = signals.sort_values('Date')

    # IS/OOS 분리
    signals['Year'] = signals['Date'].dt.year
    is_data = signals[signals['Year'].between(2021, 2024)]
    oos_data = signals[signals['Year'] == 2025]

    def calc_metrics(data, period_name, years):
        if len(data) == 0:
            return {
                'Score_Type': score_type,
                'Threshold': threshold,
                'Period': period_name,
                'Years': years,
                'Signals': 0,
                'Signals_Per_Week': 0,
                'Avg_Return': 0,
                'Median_Return': 0,
                'Win_Rate': 0,
                'Std': 0,
                'Sharpe': 0,
                'MDD': 0,
                'Profit_Factor': 0
            }

        returns = data[return_col]
        signals_per_week = len(data) / years / 52

        avg_return = returns.mean()
        median_return = returns.median()
        win_rate = (returns > 0).mean() * 100
        std = returns.std()
        sharpe = (avg_return / std) if std > 0 else 0
        mdd = calculate_mdd(returns)

        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        total_profit = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

        return {
            'Score_Type': score_type,
            'Threshold': threshold,
            'Period': period_name,
            'Years': years,
            'Signals': len(data),
            'Signals_Per_Week': round(signals_per_week, 1),
            'Avg_Return': round(avg_return, 2),
            'Median_Return': round(median_return, 2),
            'Win_Rate': round(win_rate, 1),
            'Std': round(std, 2),
            'Sharpe': round(sharpe, 3),
            'MDD': round(mdd, 2),
            'Profit_Factor': round(profit_factor, 2)
        }

    is_metrics = calc_metrics(is_data, 'IS', 4)
    oos_metrics = calc_metrics(oos_data, 'OOS', 1)

    # 파라미터 정보 추가
    if params:
        is_metrics['Params'] = str(params)
        oos_metrics['Params'] = str(params)
    else:
        is_metrics['Params'] = 'None'
        oos_metrics['Params'] = 'None'

    return is_metrics, oos_metrics

def main():
    print("="*80)
    print("Phase 3-13: VR × PRICE Interaction Analysis")
    print("="*80)

    df = load_data()

    results = []

    # 1. VR × Price (곱)
    print("\n[1/4] VR × Price (곱) 분석 중...")
    multiply_thresholds = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]
    for thresh in multiply_thresholds:
        is_m, oos_m = backtest_interaction(df.copy(), 'multiply', thresh)
        results.extend([is_m, oos_m])

    # 2. VR + Price (합)
    print("[2/4] VR + Price (합) 분석 중...")
    add_thresholds = [8, 10, 12, 14, 16, 18, 20, 22, 25]
    for thresh in add_thresholds:
        is_m, oos_m = backtest_interaction(df.copy(), 'add', thresh)
        results.extend([is_m, oos_m])

    # 3. 가중합: w1×VR + w2×Price
    print("[3/4] 가중합 분석 중...")
    weighted_configs = [
        {'w1': 1.0, 'w2': 1.0, 'thresholds': [12, 15, 18, 20, 22, 25]},
        {'w1': 1.0, 'w2': 2.0, 'thresholds': [15, 18, 20, 25, 30, 35]},
        {'w1': 2.0, 'w2': 1.0, 'thresholds': [10, 12, 15, 18, 20, 25]},
        {'w1': 1.5, 'w2': 1.0, 'thresholds': [12, 15, 18, 20, 22, 25]},
        {'w1': 1.0, 'w2': 1.5, 'thresholds': [12, 15, 18, 20, 25, 30]}
    ]

    for config in weighted_configs:
        for thresh in config['thresholds']:
            params = {'w1': config['w1'], 'w2': config['w2']}
            is_m, oos_m = backtest_interaction(df.copy(), 'weighted', thresh, params)
            results.extend([is_m, oos_m])

    # 4. Power: VR^a × Price^b
    print("[4/4] Power 조합 분석 중...")
    power_configs = [
        {'a': 1.0, 'b': 1.0, 'thresholds': [15, 20, 25, 30, 40, 50]},  # 곱과 동일
        {'a': 0.5, 'b': 1.0, 'thresholds': [10, 12, 15, 18, 20, 25]},  # VR에 제곱근
        {'a': 1.0, 'b': 0.5, 'thresholds': [8, 10, 12, 15, 18, 20]},   # Price에 제곱근
        {'a': 0.5, 'b': 0.5, 'thresholds': [6, 8, 10, 12, 15, 18]},    # 둘 다 제곱근
        {'a': 1.5, 'b': 1.0, 'thresholds': [20, 30, 40, 50, 60, 80]},  # VR 강조
        {'a': 1.0, 'b': 1.5, 'thresholds': [15, 20, 30, 40, 50, 70]}   # Price 강조
    ]

    for config in power_configs:
        for thresh in config['thresholds']:
            params = {'a': config['a'], 'b': config['b']}
            is_m, oos_m = backtest_interaction(df.copy(), 'power', thresh, params)
            results.extend([is_m, oos_m])

    print("✅ 분석 완료\n")

    # 결과 저장
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / 'p3_13_vr_price_interaction.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장: {output_path}\n")

    # IS 결과 필터링 (최소 신호)
    is_results = results_df[
        (results_df['Period'] == 'IS') &
        (results_df['Signals'] >= 50)
    ].copy()

    # 종합 점수
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series))
        return (series - min_val) / (max_val - min_val) * 100

    is_results['Return_Score'] = normalize(is_results['Avg_Return'])
    is_results['WinRate_Score'] = normalize(is_results['Win_Rate'])
    is_results['Sharpe_Score'] = normalize(is_results['Sharpe'])
    is_results['MDD_Score'] = normalize(-is_results['MDD'])

    is_results['Total_Score'] = (
        is_results['Return_Score'] * 0.4 +
        is_results['WinRate_Score'] * 0.2 +
        is_results['Sharpe_Score'] * 0.2 +
        is_results['MDD_Score'] * 0.2
    )

    # Top 20 출력
    print("="*150)
    print("Top 20 조합 (종합 점수 기준)")
    print("="*150)

    top_combinations = is_results.nlargest(20, 'Total_Score')

    for idx, row in top_combinations.iterrows():
        score_type = row['Score_Type']
        thresh = row['Threshold']
        params = row['Params']
        total_score = row['Total_Score']

        is_signals = row['Signals']
        is_per_week = row['Signals_Per_Week']
        is_ret = row['Avg_Return']
        is_win = row['Win_Rate']
        is_sharpe = row['Sharpe']
        is_mdd = row['MDD']
        is_pf = row['Profit_Factor']

        # OOS 찾기
        oos_row = results_df[
            (results_df['Period'] == 'OOS') &
            (results_df['Score_Type'] == score_type) &
            (results_df['Threshold'] == thresh) &
            (results_df['Params'] == params)
        ].iloc[0]

        oos_signals = oos_row['Signals']
        oos_ret = oos_row['Avg_Return']
        oos_win = oos_row['Win_Rate']
        oos_mdd = oos_row['MDD']

        ret_diff = abs(oos_ret - is_ret)
        stability = "✅" if ret_diff < 10 else "⚠️"

        # 파라미터 표시
        if params != 'None':
            param_str = f" | {params}"
        else:
            param_str = ""

        print(f"\n🏆 {score_type.upper()} ≥ {thresh}{param_str} | 종합점수: {total_score:.1f}")
        print(f"  IS  ({is_signals:3d}건, 주 {is_per_week:.1f}개): "
              f"수익 {is_ret:6.2f}% | 승률 {is_win:5.1f}% | Sharpe {is_sharpe:.3f} | "
              f"MDD {is_mdd:6.2f}% | PF {is_pf:.2f}")
        print(f"  OOS ({oos_signals:3d}건): "
              f"수익 {oos_ret:6.2f}% | 승률 {oos_win:5.1f}% | MDD {oos_mdd:6.2f}% | "
              f"차이 {oos_ret - is_ret:+.2f}%p {stability}")

    # 기존 AND 조건과 비교
    print("\n" + "="*150)
    print("기존 방식 비교 (VR≥3.0 AND Price≥10%)")
    print("="*150)

    baseline_df = df[(df['VR'] >= 3.0) & (df['Price_Change'] >= 10.0)].copy()
    baseline_df['Year'] = baseline_df['Date'].dt.year
    baseline_is = baseline_df[baseline_df['Year'].between(2021, 2024)]
    baseline_oos = baseline_df[baseline_df['Year'] == 2025]

    baseline_is_ret = baseline_is['Return_10D'].mean()
    baseline_is_win = (baseline_is['Return_10D'] > 0).mean() * 100
    baseline_is_mdd = calculate_mdd(baseline_is['Return_10D'])

    baseline_oos_ret = baseline_oos['Return_10D'].mean()
    baseline_oos_win = (baseline_oos['Return_10D'] > 0).mean() * 100

    print(f"IS  ({len(baseline_is):4d}건, 주 {len(baseline_is)/4/52:.1f}개): "
          f"수익 {baseline_is_ret:6.2f}% | 승률 {baseline_is_win:5.1f}% | MDD {baseline_is_mdd:6.2f}%")
    print(f"OOS ({len(baseline_oos):4d}건): "
          f"수익 {baseline_oos_ret:6.2f}% | 승률 {baseline_oos_win:5.1f}%")

    # 개선 효과
    best = top_combinations.iloc[0]
    print(f"\n최적 조합 개선 효과:")
    print(f"  수익률: {baseline_is_ret:.2f}% → {best['Avg_Return']:.2f}% ({best['Avg_Return'] - baseline_is_ret:+.2f}%p)")
    print(f"  승률: {baseline_is_win:.1f}% → {best['Win_Rate']:.1f}% ({best['Win_Rate'] - baseline_is_win:+.1f}%p)")
    print(f"  MDD: {baseline_is_mdd:.2f}% → {best['MDD']:.2f}% ({best['MDD'] - baseline_is_mdd:+.2f}%p)")

    print("\n" + "="*150)

if __name__ == '__main__':
    main()
