"""
Phase 3-15: Exit Parameter Scan
다양한 손절/익절 수준 테스트

Trailing Stop: -5%, -7%, -10%, -15%
Stop Loss Only: -3%, -5%, -7%, -10%
Trailing + Take Profit: -10% trailing + 15%, 20%, 25% 익절
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

MAX_HOLDING_DAYS = 10

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")
    stock_df = pd.read_csv(DATA_DIR / 'stock_data_with_indicators.csv', parse_dates=['Date'])

    if 'VR' not in stock_df.columns:
        if 'Volume_Ratio' in stock_df.columns:
            stock_df['VR'] = stock_df['Volume_Ratio']

    if 'Price_Change' not in stock_df.columns:
        if 'Change' in stock_df.columns:
            stock_df['Price_Change'] = stock_df['Change']
        elif 'Close' in stock_df.columns and 'Open' in stock_df.columns:
            stock_df['Price_Change'] = ((stock_df['Close'] - stock_df['Open']) / stock_df['Open'] * 100)

    print(f"전체 데이터: {len(stock_df):,}건")
    return stock_df

def simulate_trailing_stop(stock_df, entry_date, code, entry_price, stop_pct=5):
    """Trailing Stop"""
    stock_data = stock_df[
        (stock_df['Code'] == code) &
        (stock_df['Date'] > entry_date)
    ].sort_values('Date').head(MAX_HOLDING_DAYS)

    if len(stock_data) == 0:
        return 0, 0, 'NO_DATA'

    max_price = entry_price

    for idx, (_, row) in enumerate(stock_data.iterrows(), start=1):
        day_high = row['High']
        day_low = row['Low']
        day_close = row['Close']

        if day_high > max_price:
            max_price = day_high

        max_return = (max_price - entry_price) / entry_price * 100
        current_low_return = (day_low - entry_price) / entry_price * 100
        drawdown_from_peak = current_low_return - max_return

        if drawdown_from_peak <= -stop_pct:
            exit_price = max_price * (1 - stop_pct/100)
            realized_return = (exit_price - entry_price) / entry_price * 100
            return realized_return, idx, f'TRAIL_{stop_pct}%'

        if idx == MAX_HOLDING_DAYS or idx == len(stock_data):
            realized_return = (day_close - entry_price) / entry_price * 100
            return realized_return, idx, 'MAX_DAYS'

    last_return = (stock_data.iloc[-1]['Close'] - entry_price) / entry_price * 100
    return last_return, len(stock_data), 'EARLY_EXIT'

def simulate_stop_loss_only(stock_df, entry_date, code, entry_price, stop_loss_pct=3):
    """손절만"""
    stock_data = stock_df[
        (stock_df['Code'] == code) &
        (stock_df['Date'] > entry_date)
    ].sort_values('Date').head(MAX_HOLDING_DAYS)

    if len(stock_data) == 0:
        return 0, 0, 'NO_DATA'

    for idx, (_, row) in enumerate(stock_data.iterrows(), start=1):
        day_low = row['Low']
        day_close = row['Close']

        low_return = (day_low - entry_price) / entry_price * 100

        if low_return <= -stop_loss_pct:
            realized_return = -stop_loss_pct
            return realized_return, idx, f'SL_{stop_loss_pct}%'

        if idx == MAX_HOLDING_DAYS or idx == len(stock_data):
            realized_return = (day_close - entry_price) / entry_price * 100
            return realized_return, idx, 'MAX_DAYS'

    last_return = (stock_data.iloc[-1]['Close'] - entry_price) / entry_price * 100
    return last_return, len(stock_data), 'EARLY_EXIT'

def simulate_trailing_with_tp(stock_df, entry_date, code, entry_price, stop_pct=10, tp_pct=20):
    """Trailing Stop + Take Profit"""
    stock_data = stock_df[
        (stock_df['Code'] == code) &
        (stock_df['Date'] > entry_date)
    ].sort_values('Date').head(MAX_HOLDING_DAYS)

    if len(stock_data) == 0:
        return 0, 0, 'NO_DATA'

    max_price = entry_price

    for idx, (_, row) in enumerate(stock_data.iterrows(), start=1):
        day_high = row['High']
        day_low = row['Low']
        day_close = row['Close']

        # 익절 체크 (장중 고가 기준)
        high_return = (day_high - entry_price) / entry_price * 100
        if high_return >= tp_pct:
            realized_return = tp_pct
            return realized_return, idx, f'TP_{tp_pct}%'

        if day_high > max_price:
            max_price = day_high

        # Trailing Stop
        max_return = (max_price - entry_price) / entry_price * 100
        current_low_return = (day_low - entry_price) / entry_price * 100
        drawdown_from_peak = current_low_return - max_return

        if drawdown_from_peak <= -stop_pct:
            exit_price = max_price * (1 - stop_pct/100)
            realized_return = (exit_price - entry_price) / entry_price * 100
            return realized_return, idx, f'TRAIL_{stop_pct}%'

        if idx == MAX_HOLDING_DAYS or idx == len(stock_data):
            realized_return = (day_close - entry_price) / entry_price * 100
            return realized_return, idx, 'MAX_DAYS'

    last_return = (stock_data.iloc[-1]['Close'] - entry_price) / entry_price * 100
    return last_return, len(stock_data), 'EARLY_EXIT'

def calculate_mdd(returns_series):
    """MDD 계산"""
    if len(returns_series) == 0:
        return 0
    cumulative = (1 + returns_series / 100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    mdd = drawdown.min()
    return mdd

def backtest_strategy(stock_df, signals, strategy_type, **params):
    """전략 백테스팅"""
    results = []

    for idx, row in signals.iterrows():
        entry_date = row['Date']
        code = row['Code']
        entry_price = row['Close']

        if strategy_type == 'baseline':
            if 'Return_10D' in row:
                realized_return = row['Return_10D']
                holding_days = 10
                exit_reason = 'MAX_DAYS'
            else:
                continue

        elif strategy_type == 'trailing':
            realized_return, holding_days, exit_reason = simulate_trailing_stop(
                stock_df, entry_date, code, entry_price, stop_pct=params['stop_pct']
            )

        elif strategy_type == 'stop_loss':
            realized_return, holding_days, exit_reason = simulate_stop_loss_only(
                stock_df, entry_date, code, entry_price, stop_loss_pct=params['stop_loss_pct']
            )

        elif strategy_type == 'trailing_tp':
            realized_return, holding_days, exit_reason = simulate_trailing_with_tp(
                stock_df, entry_date, code, entry_price,
                stop_pct=params['stop_pct'], tp_pct=params['tp_pct']
            )

        results.append({
            'Date': entry_date,
            'Code': code,
            'Year': entry_date.year,
            'Return': realized_return,
            'Holding_Days': holding_days,
            'Exit_Reason': exit_reason
        })

    return pd.DataFrame(results)

def analyze_results(results_df):
    """결과 분석"""
    if len(results_df) == 0:
        return {}

    returns = results_df['Return']

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    total_profit = wins.sum() if len(wins) > 0 else 0
    total_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

    return {
        'Signals': len(results_df),
        'Avg_Return': round(returns.mean(), 2),
        'Win_Rate': round((returns > 0).mean() * 100, 1),
        'Avg_Holding_Days': round(results_df['Holding_Days'].mean(), 1),
        'MDD': round(calculate_mdd(returns), 2),
        'Sharpe': round((returns.mean() / returns.std()) if returns.std() > 0 else 0, 3),
        'Profit_Factor': round(profit_factor, 2),
        'Avg_Win': round(avg_win, 2),
        'Avg_Loss': round(avg_loss, 2)
    }

def main():
    print("="*80)
    print("Phase 3-15: Exit Parameter Scan")
    print("="*80)

    stock_df = load_data()

    # 신호 필터링
    print("\n신호 필터링 (VR≥3.0, Price≥10%)...")
    signals = stock_df[(stock_df['VR'] >= 3.0) & (stock_df['Price_Change'] >= 10.0)].copy()

    signals['Year'] = signals['Date'].dt.year
    is_signals = signals[signals['Year'].between(2021, 2024)]
    oos_signals = signals[signals['Year'] == 2025]

    print(f"IS: {len(is_signals):,}건, OOS: {len(oos_signals):,}건")

    all_results = []

    # 1. Baseline
    print("\n[Baseline] 백테스팅...")
    for period_name, period_signals in [('IS', is_signals), ('OOS', oos_signals)]:
        results = backtest_strategy(stock_df, period_signals, 'baseline')
        metrics = analyze_results(results)
        metrics['Strategy'] = 'Baseline'
        metrics['Period'] = period_name
        metrics['Params'] = '-'
        all_results.append(metrics)

    # 2. Trailing Stop
    print("\n[Trailing Stop] 백테스팅...")
    trailing_params = [5, 7, 10, 15]
    for stop_pct in trailing_params:
        print(f"  -{stop_pct}% Trailing...")
        for period_name, period_signals in [('IS', is_signals), ('OOS', oos_signals)]:
            results = backtest_strategy(stock_df, period_signals, 'trailing', stop_pct=stop_pct)
            metrics = analyze_results(results)
            metrics['Strategy'] = f'Trailing_Stop_{stop_pct}%'
            metrics['Period'] = period_name
            metrics['Params'] = f'-{stop_pct}%'
            all_results.append(metrics)

    # 3. Stop Loss Only
    print("\n[Stop Loss Only] 백테스팅...")
    sl_params = [3, 5, 7, 10]
    for sl_pct in sl_params:
        print(f"  -{sl_pct}% Stop Loss...")
        for period_name, period_signals in [('IS', is_signals), ('OOS', oos_signals)]:
            results = backtest_strategy(stock_df, period_signals, 'stop_loss', stop_loss_pct=sl_pct)
            metrics = analyze_results(results)
            metrics['Strategy'] = f'Stop_Loss_{sl_pct}%'
            metrics['Period'] = period_name
            metrics['Params'] = f'-{sl_pct}%'
            all_results.append(metrics)

    # 4. Trailing + Take Profit
    print("\n[Trailing + Take Profit] 백테스팅...")
    tp_configs = [
        (10, 15),
        (10, 20),
        (10, 25),
        (15, 20),
        (15, 25)
    ]
    for stop_pct, tp_pct in tp_configs:
        print(f"  -{stop_pct}% Trailing + {tp_pct}% TP...")
        for period_name, period_signals in [('IS', is_signals), ('OOS', oos_signals)]:
            results = backtest_strategy(stock_df, period_signals, 'trailing_tp',
                                       stop_pct=stop_pct, tp_pct=tp_pct)
            metrics = analyze_results(results)
            metrics['Strategy'] = f'Trail_{stop_pct}%_TP_{tp_pct}%'
            metrics['Period'] = period_name
            metrics['Params'] = f'-{stop_pct}%/+{tp_pct}%'
            all_results.append(metrics)

    # 결과 저장
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(RESULTS_DIR / 'p3_15_exit_parameter_scan.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료\n")

    # IS 결과만 추출
    is_results = summary_df[summary_df['Period'] == 'IS'].copy()

    # 종합 점수 (수익률 40%, 승률 20%, MDD 30%, Sharpe 10%)
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series))
        return (series - min_val) / (max_val - min_val) * 100

    is_results['Return_Score'] = normalize(is_results['Avg_Return'])
    is_results['WinRate_Score'] = normalize(is_results['Win_Rate'])
    is_results['MDD_Score'] = normalize(-is_results['MDD'])
    is_results['Sharpe_Score'] = normalize(is_results['Sharpe'])

    is_results['Total_Score'] = (
        is_results['Return_Score'] * 0.4 +
        is_results['WinRate_Score'] * 0.2 +
        is_results['MDD_Score'] * 0.3 +
        is_results['Sharpe_Score'] * 0.1
    )

    # Top 10 출력
    print("="*150)
    print("Top 10 전략 (종합 점수: 수익률 40% + 승률 20% + MDD 30% + Sharpe 10%)")
    print("="*150)

    top_strategies = is_results.nlargest(10, 'Total_Score')

    for idx, row in top_strategies.iterrows():
        strategy = row['Strategy']
        params = row['Params']
        score = row['Total_Score']

        is_ret = row['Avg_Return']
        is_win = row['Win_Rate']
        is_mdd = row['MDD']
        is_sharpe = row['Sharpe']
        is_pf = row['Profit_Factor']
        is_hold = row['Avg_Holding_Days']

        # OOS 찾기
        oos_row = summary_df[
            (summary_df['Period'] == 'OOS') &
            (summary_df['Strategy'] == strategy)
        ].iloc[0]

        oos_ret = oos_row['Avg_Return']
        oos_win = oos_row['Win_Rate']
        oos_mdd = oos_row['MDD']

        stability = "✅" if abs(oos_ret - is_ret) < 5 else "⚠️"

        print(f"\n🏆 {strategy} ({params}) | 점수: {score:.1f}")
        print(f"  IS:  수익 {is_ret:6.2f}% | 승률 {is_win:5.1f}% | MDD {is_mdd:7.2f}% | "
              f"Sharpe {is_sharpe:.3f} | PF {is_pf:.2f} | 보유 {is_hold:.1f}일")
        print(f"  OOS: 수익 {oos_ret:6.2f}% | 승률 {oos_win:5.1f}% | MDD {oos_mdd:7.2f}% | "
              f"차이 {oos_ret - is_ret:+.2f}%p {stability}")

    # Baseline과 비교
    print("\n" + "="*150)
    print("Baseline 대비 개선")
    print("="*150)

    baseline_is = is_results[is_results['Strategy'] == 'Baseline'].iloc[0]
    best_is = top_strategies.iloc[0]

    print(f"\nBaseline:")
    print(f"  수익률: {baseline_is['Avg_Return']:.2f}% | 승률: {baseline_is['Win_Rate']:.1f}% | "
          f"MDD: {baseline_is['MDD']:.2f}%")

    print(f"\n최적 전략 ({best_is['Strategy']}):")
    print(f"  수익률: {best_is['Avg_Return']:.2f}% ({best_is['Avg_Return'] - baseline_is['Avg_Return']:+.2f}%p)")
    print(f"  승률: {best_is['Win_Rate']:.1f}% ({best_is['Win_Rate'] - baseline_is['Win_Rate']:+.1f}%p)")
    print(f"  MDD: {best_is['MDD']:.2f}% ({best_is['MDD'] - baseline_is['MDD']:+.2f}%p)")

    print("\n" + "="*150)

if __name__ == '__main__':
    main()
