"""
Phase 3-10: Holding Period Sweet Spot Analysis
VR≥3.0, Price≥10% 조합에서 최적 보유기간 찾기

보유기간: 1D, 3D, 5D, 10D, 20D
목표: IS/OOS 수익률, 승률 최대화
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

# 테스트할 보유기간
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

def analyze_holding_period(df, holding_period):
    """보유기간별 분석"""

    return_col = f'Return_{holding_period}D'

    if return_col not in df.columns:
        print(f"⚠️ {return_col} 컬럼 없음")
        return None, None

    # VR≥3.0, Price≥10% 필터
    signals = df[
        (df['VR'] >= 3.0) &
        (df['Price_Change'] >= 10.0)
    ].copy()

    # IS/OOS 분리
    signals['Year'] = signals['Date'].dt.year
    is_data = signals[signals['Year'].between(2021, 2024)]
    oos_data = signals[signals['Year'] == 2025]

    def calc_metrics(data, period_name, years):
        if len(data) == 0:
            return {
                'Holding_Period': f'{holding_period}D',
                'Period': period_name,
                'Years': years,
                'Signals': 0,
                'Signals_Per_Week': 0,
                'Avg_Return': 0,
                'Median_Return': 0,
                'Win_Rate': 0,
                'Std': 0,
                'Sharpe': 0,
                'Best_Return': 0,
                'Worst_Return': 0
            }

        returns = data[return_col]
        signals_per_week = len(data) / years / 52

        return {
            'Holding_Period': f'{holding_period}D',
            'Period': period_name,
            'Years': years,
            'Signals': len(data),
            'Signals_Per_Week': round(signals_per_week, 1),
            'Avg_Return': round(returns.mean(), 2),
            'Median_Return': round(returns.median(), 2),
            'Win_Rate': round((returns > 0).mean() * 100, 1),
            'Std': round(returns.std(), 2),
            'Sharpe': round((returns.mean() / returns.std()) if returns.std() > 0 else 0, 3),
            'Best_Return': round(returns.max(), 2),
            'Worst_Return': round(returns.min(), 2)
        }

    is_metrics = calc_metrics(is_data, 'IS', 4)
    oos_metrics = calc_metrics(oos_data, 'OOS', 1)

    return is_metrics, oos_metrics

def analyze_yearly(df, holding_period):
    """연도별 분석"""

    return_col = f'Return_{holding_period}D'

    if return_col not in df.columns:
        return []

    # VR≥3.0, Price≥10% 필터
    signals = df[
        (df['VR'] >= 3.0) &
        (df['Price_Change'] >= 10.0)
    ].copy()

    signals['Year'] = signals['Date'].dt.year
    yearly_results = []

    for year in range(2021, 2026):
        year_data = signals[signals['Year'] == year]

        if len(year_data) == 0:
            yearly_results.append({
                'Holding_Period': f'{holding_period}D',
                'Year': year,
                'Signals': 0,
                'Avg_Return': 0,
                'Win_Rate': 0
            })
        else:
            returns = year_data[return_col]
            yearly_results.append({
                'Holding_Period': f'{holding_period}D',
                'Year': year,
                'Signals': len(year_data),
                'Avg_Return': round(returns.mean(), 2),
                'Win_Rate': round((returns > 0).mean() * 100, 1)
            })

    return yearly_results

def main():
    print("="*80)
    print("Phase 3-10: Holding Period Sweet Spot Analysis")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # 보유기간별 분석
    print(f"\n🔍 보유기간 분석 (VR≥3.0, Price≥10%)")
    print(f"   테스트 기간: {HOLDING_PERIODS}")

    summary_results = []
    yearly_results = []

    for hp in HOLDING_PERIODS:
        print(f"\n   [{hp}D] 분석 중...")

        # IS/OOS 분석
        is_metrics, oos_metrics = analyze_holding_period(df, hp)
        if is_metrics:
            summary_results.append(is_metrics)
            summary_results.append(oos_metrics)

        # 연도별 분석
        yearly = analyze_yearly(df, hp)
        yearly_results.extend(yearly)

    # 결과 저장
    summary_df = pd.DataFrame(summary_results)
    yearly_df = pd.DataFrame(yearly_results)

    summary_path = RESULTS_DIR / 'p3_10_holding_period_summary.csv'
    yearly_path = RESULTS_DIR / 'p3_10_holding_period_yearly.csv'

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    yearly_df.to_csv(yearly_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ 요약 결과 저장: {summary_path}")
    print(f"✅ 연도별 결과 저장: {yearly_path}")

    # 결과 출력
    print("\n" + "="*120)
    print("보유기간별 성과 비교 (VR≥3.0, Price≥10%)")
    print("="*120)

    for hp in HOLDING_PERIODS:
        hp_data = summary_df[summary_df['Holding_Period'] == f'{hp}D']

        if len(hp_data) == 0:
            continue

        is_row = hp_data[hp_data['Period'] == 'IS'].iloc[0]
        oos_row = hp_data[hp_data['Period'] == 'OOS'].iloc[0]

        print(f"\n【{hp}일 보유】")
        print("-" * 120)
        print(f"IS (2021-2024)  | "
              f"신호: {is_row['Signals']:4d}건 (주 {is_row['Signals_Per_Week']:4.1f}개) | "
              f"수익률: {is_row['Avg_Return']:6.2f}% | "
              f"중간값: {is_row['Median_Return']:6.2f}% | "
              f"승률: {is_row['Win_Rate']:5.1f}% | "
              f"Sharpe: {is_row['Sharpe']:5.3f}")
        print(f"OOS (2025)      | "
              f"신호: {oos_row['Signals']:4d}건 | "
              f"수익률: {oos_row['Avg_Return']:6.2f}% | "
              f"중간값: {oos_row['Median_Return']:6.2f}% | "
              f"승률: {oos_row['Win_Rate']:5.1f}% | "
              f"Sharpe: {oos_row['Sharpe']:5.3f}")

        # 안정성 체크
        diff = abs(oos_row['Avg_Return'] - is_row['Avg_Return'])
        stability = "✅ 안정" if diff < 5 else "⚠️ 불안정"
        print(f"IS-OOS 차이: {oos_row['Avg_Return'] - is_row['Avg_Return']:+.2f}%p {stability}")

    # 연도별 안정성
    print("\n" + "="*120)
    print("연도별 안정성 분석")
    print("="*120)

    for hp in HOLDING_PERIODS:
        hp_yearly = yearly_df[yearly_df['Holding_Period'] == f'{hp}D']

        if len(hp_yearly) == 0:
            continue

        positive_years = len(hp_yearly[hp_yearly['Avg_Return'] > 0])
        total_years = len(hp_yearly)

        print(f"\n【{hp}일 보유】 양수 수익률: {positive_years}/{total_years}년 ({positive_years/total_years*100:.1f}%)")
        print("  ", end="")
        for _, row in hp_yearly.iterrows():
            year = int(row['Year'])
            ret = row['Avg_Return']
            signals = row['Signals']
            status = "✅" if ret > 0 else "❌"
            print(f"{year}: {ret:+6.2f}% ({signals:3d}건) {status} | ", end="")
        print()

    # 최적 보유기간 추천
    print("\n" + "="*120)
    print("최적 보유기간 추천")
    print("="*120)

    is_summary = summary_df[summary_df['Period'] == 'IS'].copy()
    is_summary = is_summary.sort_values('Avg_Return', ascending=False)

    print("\n1️⃣ IS 수익률 기준 랭킹:")
    for idx, row in is_summary.iterrows():
        hp = row['Holding_Period']
        ret = row['Avg_Return']
        win = row['Win_Rate']
        sharpe = row['Sharpe']
        print(f"   {hp}: {ret:6.2f}% 수익률, {win:5.1f}% 승률, Sharpe {sharpe:.3f}")

    # Sharpe 기준
    is_summary_sharpe = is_summary.sort_values('Sharpe', ascending=False)
    print("\n2️⃣ Sharpe Ratio 기준 랭킹:")
    for idx, row in is_summary_sharpe.iterrows():
        hp = row['Holding_Period']
        ret = row['Avg_Return']
        win = row['Win_Rate']
        sharpe = row['Sharpe']
        print(f"   {hp}: Sharpe {sharpe:.3f}, {ret:6.2f}% 수익률, {win:5.1f}% 승률")

    # 승률 기준
    is_summary_win = is_summary.sort_values('Win_Rate', ascending=False)
    print("\n3️⃣ 승률 기준 랭킹:")
    for idx, row in is_summary_win.iterrows():
        hp = row['Holding_Period']
        ret = row['Avg_Return']
        win = row['Win_Rate']
        sharpe = row['Sharpe']
        print(f"   {hp}: {win:5.1f}% 승률, {ret:6.2f}% 수익률, Sharpe {sharpe:.3f}")

    print("\n" + "="*120)

if __name__ == '__main__':
    main()
