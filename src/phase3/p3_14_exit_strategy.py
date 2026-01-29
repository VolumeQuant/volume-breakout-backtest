"""
Phase 3-14: Exit Strategy Backtest
손절/익절 로직 적용

Strategy 2: Trailing Stop
  - 최고점 대비 -5% 하락 시 청산
  - 최대 10일 보유

Strategy 3: 손절만
  - 손절: -3% 도달 시 청산
  - 익절: 없음 (10일 만기까지)
  - 최대 10일 보유

비교:
  - Baseline: 10일 무조건 보유 (아무 관리 안 함)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings
from datetime import timedelta

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

def simulate_trailing_stop(stock_df, entry_date, code, entry_price, stop_pct=5):
    """Trailing Stop 시뮬레이션

    Args:
        stock_df: 전체 주가 데이터
        entry_date: 진입일
        code: 종목코드
        entry_price: 진입가
        stop_pct: 최고점 대비 하락 % (기본 5%)

    Returns:
        (실현수익률, 보유일수, 청산사유)
    """

    # 해당 종목의 진입일 이후 데이터
    stock_data = stock_df[
        (stock_df['Code'] == code) &
        (stock_df['Date'] > entry_date)
    ].sort_values('Date').head(MAX_HOLDING_DAYS)

    if len(stock_data) == 0:
        return 0, 0, 'NO_DATA'

    max_price = entry_price  # 최고가 추적

    for idx, (_, row) in enumerate(stock_data.iterrows(), start=1):
        day_high = row['High']
        day_low = row['Low']
        day_close = row['Close']

        # 장중 최고가 갱신
        if day_high > max_price:
            max_price = day_high

        # Trailing Stop 체크 (장중 최저가 기준)
        max_return = (max_price - entry_price) / entry_price * 100
        current_low_return = (day_low - entry_price) / entry_price * 100
        drawdown_from_peak = current_low_return - max_return

        if drawdown_from_peak <= -stop_pct:
            # Trailing Stop 발동
            exit_price = max_price * (1 - stop_pct/100)
            realized_return = (exit_price - entry_price) / entry_price * 100
            return realized_return, idx, f'TRAIL_STOP_{stop_pct}%'

        # 최대 보유일 도달
        if idx == MAX_HOLDING_DAYS or idx == len(stock_data):
            realized_return = (day_close - entry_price) / entry_price * 100
            return realized_return, idx, 'MAX_DAYS'

    # 데이터 부족
    last_return = (stock_data.iloc[-1]['Close'] - entry_price) / entry_price * 100
    return last_return, len(stock_data), 'EARLY_EXIT'

def simulate_stop_loss_only(stock_df, entry_date, code, entry_price, stop_loss_pct=3):
    """손절만 적용 시뮬레이션

    Args:
        stock_df: 전체 주가 데이터
        entry_date: 진입일
        code: 종목코드
        entry_price: 진입가
        stop_loss_pct: 손절 % (기본 3%)

    Returns:
        (실현수익률, 보유일수, 청산사유)
    """

    # 해당 종목의 진입일 이후 데이터
    stock_data = stock_df[
        (stock_df['Code'] == code) &
        (stock_df['Date'] > entry_date)
    ].sort_values('Date').head(MAX_HOLDING_DAYS)

    if len(stock_data) == 0:
        return 0, 0, 'NO_DATA'

    for idx, (_, row) in enumerate(stock_data.iterrows(), start=1):
        day_low = row['Low']
        day_close = row['Close']

        # 손절 체크 (장중 최저가 기준)
        low_return = (day_low - entry_price) / entry_price * 100

        if low_return <= -stop_loss_pct:
            # 손절 발동
            exit_price = entry_price * (1 - stop_loss_pct/100)
            realized_return = -stop_loss_pct
            return realized_return, idx, f'STOP_LOSS_{stop_loss_pct}%'

        # 최대 보유일 도달
        if idx == MAX_HOLDING_DAYS or idx == len(stock_data):
            realized_return = (day_close - entry_price) / entry_price * 100
            return realized_return, idx, 'MAX_DAYS'

    # 데이터 부족
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

def backtest_strategy(stock_df, signals, strategy_name, **kwargs):
    """전략별 백테스팅"""

    print(f"\n  {strategy_name} 백테스팅 중...")

    results = []

    for idx, row in signals.iterrows():
        entry_date = row['Date']
        code = row['Code']
        entry_price = row['Close']  # 종가 진입 가정

        if strategy_name == 'Baseline':
            # 10일 무조건 보유
            if 'Return_10D' in row:
                realized_return = row['Return_10D']
                holding_days = 10
                exit_reason = 'MAX_DAYS'
            else:
                continue

        elif strategy_name == 'Trailing_Stop':
            realized_return, holding_days, exit_reason = simulate_trailing_stop(
                stock_df, entry_date, code, entry_price,
                stop_pct=kwargs.get('stop_pct', 5)
            )

        elif strategy_name == 'Stop_Loss_Only':
            realized_return, holding_days, exit_reason = simulate_stop_loss_only(
                stock_df, entry_date, code, entry_price,
                stop_loss_pct=kwargs.get('stop_loss_pct', 3)
            )

        results.append({
            'Date': entry_date,
            'Code': code,
            'Year': entry_date.year,
            'Entry_Price': entry_price,
            'Return': realized_return,
            'Holding_Days': holding_days,
            'Exit_Reason': exit_reason
        })

    return pd.DataFrame(results)

def analyze_results(results_df, strategy_name, period_name, years):
    """결과 분석"""

    if len(results_df) == 0:
        return {
            'Strategy': strategy_name,
            'Period': period_name,
            'Signals': 0,
            'Avg_Return': 0,
            'Win_Rate': 0,
            'Avg_Holding_Days': 0,
            'MDD': 0,
            'Sharpe': 0,
            'Profit_Factor': 0,
            'Avg_Win': 0,
            'Avg_Loss': 0
        }

    returns = results_df['Return']

    avg_return = returns.mean()
    win_rate = (returns > 0).mean() * 100
    avg_holding = results_df['Holding_Days'].mean()
    std = returns.std()
    sharpe = (avg_return / std) if std > 0 else 0
    mdd = calculate_mdd(returns)

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    total_profit = wins.sum() if len(wins) > 0 else 0
    total_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

    return {
        'Strategy': strategy_name,
        'Period': period_name,
        'Signals': len(results_df),
        'Avg_Return': round(avg_return, 2),
        'Win_Rate': round(win_rate, 1),
        'Avg_Holding_Days': round(avg_holding, 1),
        'MDD': round(mdd, 2),
        'Sharpe': round(sharpe, 3),
        'Profit_Factor': round(profit_factor, 2),
        'Avg_Win': round(avg_win, 2),
        'Avg_Loss': round(avg_loss, 2)
    }

def main():
    print("="*80)
    print("Phase 3-14: Exit Strategy Backtest")
    print("="*80)

    # 데이터 로드
    stock_df = load_data()

    # 신호 필터링 (VR≥3.0, Price≥10%)
    print("\n신호 필터링 (VR≥3.0, Price≥10%)...")
    signals = stock_df[
        (stock_df['VR'] >= 3.0) &
        (stock_df['Price_Change'] >= 10.0)
    ].copy()

    print(f"총 신호: {len(signals):,}건")

    # IS/OOS 분리
    signals['Year'] = signals['Date'].dt.year
    is_signals = signals[signals['Year'].between(2021, 2024)]
    oos_signals = signals[signals['Year'] == 2025]

    print(f"  IS (2021-2024): {len(is_signals):,}건")
    print(f"  OOS (2025): {len(oos_signals):,}건")

    # 전략 백테스팅
    strategies = [
        ('Baseline', {}),
        ('Trailing_Stop', {'stop_pct': 5}),
        ('Stop_Loss_Only', {'stop_loss_pct': 3})
    ]

    all_results = []

    for strategy_name, params in strategies:
        print(f"\n{'='*80}")
        print(f"[{strategy_name}] 백테스팅")
        print(f"{'='*80}")

        # IS 백테스팅
        is_results = backtest_strategy(stock_df, is_signals, strategy_name, **params)
        is_metrics = analyze_results(is_results, strategy_name, 'IS', 4)
        all_results.append(is_metrics)

        # OOS 백테스팅
        oos_results = backtest_strategy(stock_df, oos_signals, strategy_name, **params)
        oos_metrics = analyze_results(oos_results, strategy_name, 'OOS', 1)
        all_results.append(oos_metrics)

        # 상세 결과 저장
        is_results.to_csv(
            RESULTS_DIR / f'p3_14_{strategy_name}_IS_detail.csv',
            index=False, encoding='utf-8-sig'
        )
        oos_results.to_csv(
            RESULTS_DIR / f'p3_14_{strategy_name}_OOS_detail.csv',
            index=False, encoding='utf-8-sig'
        )

    # 요약 결과
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        RESULTS_DIR / 'p3_14_exit_strategy_summary.csv',
        index=False, encoding='utf-8-sig'
    )

    print(f"\n✅ 결과 저장 완료\n")

    # 결과 출력
    print("="*150)
    print("전략별 성과 비교")
    print("="*150)

    for strategy_name in ['Baseline', 'Trailing_Stop', 'Stop_Loss_Only']:
        strategy_data = summary_df[summary_df['Strategy'] == strategy_name]

        is_row = strategy_data[strategy_data['Period'] == 'IS'].iloc[0]
        oos_row = strategy_data[strategy_data['Period'] == 'OOS'].iloc[0]

        print(f"\n【{strategy_name}】")
        print("-" * 150)

        print(f"IS (2021-2024):")
        print(f"  신호: {is_row['Signals']}건 | 평균 보유: {is_row['Avg_Holding_Days']:.1f}일")
        print(f"  수익률: {is_row['Avg_Return']:6.2f}% | 승률: {is_row['Win_Rate']:5.1f}% | "
              f"Sharpe: {is_row['Sharpe']:.3f} | MDD: {is_row['MDD']:7.2f}%")
        print(f"  평균 승: +{is_row['Avg_Win']:.2f}% | 평균 패: {is_row['Avg_Loss']:.2f}% | "
              f"PF: {is_row['Profit_Factor']:.2f}")

        print(f"\nOOS (2025):")
        print(f"  신호: {oos_row['Signals']}건 | 평균 보유: {oos_row['Avg_Holding_Days']:.1f}일")
        print(f"  수익률: {oos_row['Avg_Return']:6.2f}% | 승률: {oos_row['Win_Rate']:5.1f}% | "
              f"Sharpe: {oos_row['Sharpe']:.3f} | MDD: {oos_row['MDD']:7.2f}%")

    # 개선 효과
    print("\n" + "="*150)
    print("개선 효과 분석")
    print("="*150)

    baseline_is = summary_df[(summary_df['Strategy'] == 'Baseline') & (summary_df['Period'] == 'IS')].iloc[0]
    trailing_is = summary_df[(summary_df['Strategy'] == 'Trailing_Stop') & (summary_df['Period'] == 'IS')].iloc[0]
    stoploss_is = summary_df[(summary_df['Strategy'] == 'Stop_Loss_Only') & (summary_df['Period'] == 'IS')].iloc[0]

    print(f"\nBaseline → Trailing Stop (최고점 대비 -5% 청산):")
    print(f"  수익률: {baseline_is['Avg_Return']:.2f}% → {trailing_is['Avg_Return']:.2f}% "
          f"({trailing_is['Avg_Return'] - baseline_is['Avg_Return']:+.2f}%p)")
    print(f"  승률: {baseline_is['Win_Rate']:.1f}% → {trailing_is['Win_Rate']:.1f}% "
          f"({trailing_is['Win_Rate'] - baseline_is['Win_Rate']:+.1f}%p)")
    print(f"  MDD: {baseline_is['MDD']:.2f}% → {trailing_is['MDD']:.2f}% "
          f"({trailing_is['MDD'] - baseline_is['MDD']:+.2f}%p)")

    print(f"\nBaseline → Stop Loss Only (-3% 손절):")
    print(f"  수익률: {baseline_is['Avg_Return']:.2f}% → {stoploss_is['Avg_Return']:.2f}% "
          f"({stoploss_is['Avg_Return'] - baseline_is['Avg_Return']:+.2f}%p)")
    print(f"  승률: {baseline_is['Win_Rate']:.1f}% → {stoploss_is['Win_Rate']:.1f}% "
          f"({stoploss_is['Win_Rate'] - baseline_is['Win_Rate']:+.1f}%p)")
    print(f"  MDD: {baseline_is['MDD']:.2f}% → {stoploss_is['MDD']:.2f}% "
          f"({stoploss_is['MDD'] - baseline_is['MDD']:+.2f}%p)")

    print("\n" + "="*150)

if __name__ == '__main__':
    main()
