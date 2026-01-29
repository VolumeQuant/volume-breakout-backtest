"""
Phase 3-11: Win/Loss Distribution Analysis
VR≥3.0, Price≥10% 전략의 수익 구조 분석

승률이 50% 미만인데 평균 수익률이 플러스인 이유:
- 이익 거래의 평균 vs 손실 거래의 평균
- 손익비(Profit Factor)
- 최대 이익 vs 최대 손실
- 분포 분석
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

HOLDING_PERIODS = [1, 3, 5, 10, 20]

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

def analyze_win_loss(df, holding_period):
    """승패 분석"""

    return_col = f'Return_{holding_period}D'

    if return_col not in df.columns:
        return None

    # VR≥3.0, Price≥10% 필터
    signals = df[
        (df['VR'] >= 3.0) &
        (df['Price_Change'] >= 10.0)
    ].copy()

    signals['Year'] = signals['Date'].dt.year
    is_data = signals[signals['Year'].between(2021, 2024)]
    oos_data = signals[signals['Year'] == 2025]

    def calc_win_loss_metrics(data, period_name):
        if len(data) == 0:
            return None

        returns = data[return_col]

        # 승/패 분리
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        # 기본 통계
        total_trades = len(returns)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # 승패 수익
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        median_win = wins.median() if len(wins) > 0 else 0
        median_loss = losses.median() if len(losses) > 0 else 0

        # 최대/최소
        best_win = wins.max() if len(wins) > 0 else 0
        worst_loss = losses.min() if len(losses) > 0 else 0

        # Profit Factor (총 이익 / 총 손실)
        total_profit = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

        # Win/Loss Ratio (평균 이익 / 평균 손실)
        win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

        # 기댓값 (Expected Value)
        expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        return {
            'Holding_Period': f'{holding_period}D',
            'Period': period_name,
            'Total_Trades': total_trades,
            'Win_Count': win_count,
            'Loss_Count': loss_count,
            'Win_Rate': round(win_rate, 1),
            'Avg_Win': round(avg_win, 2),
            'Avg_Loss': round(avg_loss, 2),
            'Median_Win': round(median_win, 2),
            'Median_Loss': round(median_loss, 2),
            'Best_Win': round(best_win, 2),
            'Worst_Loss': round(worst_loss, 2),
            'Win_Loss_Ratio': round(win_loss_ratio, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Expected_Value': round(expected_value, 2),
            'Total_Profit': round(total_profit, 2),
            'Total_Loss': round(total_loss, 2)
        }

    is_metrics = calc_win_loss_metrics(is_data, 'IS')
    oos_metrics = calc_win_loss_metrics(oos_data, 'OOS')

    return is_metrics, oos_metrics

def main():
    print("="*80)
    print("Phase 3-11: Win/Loss Distribution Analysis")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # 보유기간별 승패 분석
    print(f"\n🔍 승패 구조 분석 (VR≥3.0, Price≥10%)")

    results = []

    for hp in HOLDING_PERIODS:
        print(f"\n   [{hp}D] 분석 중...")
        is_metrics, oos_metrics = analyze_win_loss(df, hp)

        if is_metrics:
            results.append(is_metrics)
            results.append(oos_metrics)

    # 결과 저장
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / 'p3_11_win_loss_analysis.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_path}")

    # 결과 출력
    print("\n" + "="*140)
    print("보유기간별 승패 구조 (VR≥3.0, Price≥10%)")
    print("="*140)

    for hp in HOLDING_PERIODS:
        hp_data = results_df[results_df['Holding_Period'] == f'{hp}D']

        if len(hp_data) == 0:
            continue

        is_row = hp_data[hp_data['Period'] == 'IS'].iloc[0]
        oos_row = hp_data[hp_data['Period'] == 'OOS'].iloc[0]

        print(f"\n【{hp}일 보유】")
        print("-" * 140)

        # IS 분석
        print(f"IS (2021-2024):")
        print(f"  거래: {is_row['Total_Trades']}건 (승 {is_row['Win_Count']}건 / 패 {is_row['Loss_Count']}건)")
        print(f"  승률: {is_row['Win_Rate']}%")
        print(f"  평균 승: +{is_row['Avg_Win']}% | 평균 패: {is_row['Avg_Loss']}% | Win/Loss Ratio: {is_row['Win_Loss_Ratio']:.2f}배")
        print(f"  중간값 승: +{is_row['Median_Win']}% | 중간값 패: {is_row['Median_Loss']}%")
        print(f"  최대 승: +{is_row['Best_Win']}% | 최대 패: {is_row['Worst_Loss']}%")
        print(f"  총 이익: +{is_row['Total_Profit']:.0f}% | 총 손실: -{is_row['Total_Loss']:.0f}% | Profit Factor: {is_row['Profit_Factor']:.2f}")
        print(f"  기댓값: {is_row['Expected_Value']}% (거래당 기대 수익)")

        # OOS 분석
        print(f"\nOOS (2025):")
        print(f"  거래: {oos_row['Total_Trades']}건 (승 {oos_row['Win_Count']}건 / 패 {oos_row['Loss_Count']}건)")
        print(f"  승률: {oos_row['Win_Rate']}%")
        print(f"  평균 승: +{oos_row['Avg_Win']}% | 평균 패: {oos_row['Avg_Loss']}% | Win/Loss Ratio: {oos_row['Win_Loss_Ratio']:.2f}배")
        print(f"  중간값 승: +{oos_row['Median_Win']}% | 중간값 패: {oos_row['Median_Loss']}%")
        print(f"  최대 승: +{oos_row['Best_Win']}% | 최대 패: {oos_row['Worst_Loss']}%")
        print(f"  기댓값: {oos_row['Expected_Value']}% (거래당 기대 수익)")

    # 핵심 인사이트
    print("\n" + "="*140)
    print("핵심 인사이트")
    print("="*140)

    # 10일 보유 상세 분석
    hp_10d_is = results_df[(results_df['Holding_Period'] == '10D') & (results_df['Period'] == 'IS')].iloc[0]
    hp_10d_oos = results_df[(results_df['Holding_Period'] == '10D') & (results_df['Period'] == 'OOS')].iloc[0]

    print(f"\n【10일 보유 상세】")
    print(f"\nIS (2021-2024):")
    print(f"  승률: {hp_10d_is['Win_Rate']}% (과반수는 손실)")
    print(f"  하지만 평균 승({hp_10d_is['Avg_Win']}%)이 평균 패({hp_10d_is['Avg_Loss']}%)보다 {hp_10d_is['Win_Loss_Ratio']:.2f}배 큼")
    print(f"  → 손실은 자주 나지만 작고, 이익은 적게 나지만 큼")
    print(f"  → Profit Factor {hp_10d_is['Profit_Factor']:.2f} (1.0 초과면 수익)")
    print(f"  → 거래당 기댓값: {hp_10d_is['Expected_Value']}%")

    # 심리적 부담
    print(f"\n심리적 부담:")
    print(f"  - 10번 거래하면 약 5.3번은 손실 (평균 {hp_10d_is['Avg_Loss']}%)")
    print(f"  - 최대 손실 가능: {hp_10d_is['Worst_Loss']}%")
    print(f"  - 연속 손실 가능성 높음 → 정신적 압박")

    # 보유기간별 Win/Loss Ratio 비교
    print(f"\n보유기간별 Win/Loss Ratio (평균 승리 / 평균 손실):")
    for hp in HOLDING_PERIODS:
        hp_is = results_df[(results_df['Holding_Period'] == f'{hp}D') & (results_df['Period'] == 'IS')].iloc[0]
        ratio = hp_is['Win_Loss_Ratio']
        win_rate = hp_is['Win_Rate']
        exp_val = hp_is['Expected_Value']

        # 필요 승률 계산 (손익분기점)
        # win_rate * avg_win + (1-win_rate) * avg_loss = 0
        # win_rate * avg_win = -(1-win_rate) * avg_loss
        # win_rate = -avg_loss / (avg_win - avg_loss)
        # win_rate = 1 / (1 + ratio)
        breakeven_winrate = 1 / (1 + ratio) * 100 if ratio > 0 else 50

        status = "✅" if win_rate >= breakeven_winrate else "⚠️"

        print(f"  {hp}일: Win/Loss Ratio {ratio:.2f}배 | 승률 {win_rate}% | 손익분기 승률 {breakeven_winrate:.1f}% {status} | 기댓값 {exp_val:.2f}%")

    print("\n" + "="*140)

if __name__ == '__main__':
    main()
