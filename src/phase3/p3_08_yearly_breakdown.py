"""
Phase 3-8: Baseline vs Flow Filters - Yearly Breakdown
연도별 안정성 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 처리
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results' / 'p3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")

    # 주가 데이터
    stock_df = pd.read_csv(
        DATA_DIR / 'stock_data_with_indicators.csv',
        parse_dates=['Date']
    )

    # 수급 데이터
    flow_df = pd.read_csv(
        DATA_DIR / 'investor_flow_data.csv',
        parse_dates=['Date']
    )

    # 수급 전처리 (3D rolling)
    flow_df = flow_df.sort_values(['Code', 'Date'])
    for investor in ['개인', '외국인', '금융투자', '연기금']:
        if investor in flow_df.columns:
            flow_df[f'{investor}_1D'] = flow_df[investor] / 100_000_000  # 원 → 억원
            flow_df[f'{investor}_3D'] = flow_df.groupby('Code')[f'{investor}_1D'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )

    # 병합
    merged = stock_df.merge(
        flow_df[['Date', 'Code', '개인_3D', '외국인_3D', '금융투자_3D', '연기금_3D']],
        on=['Date', 'Code'],
        how='left'
    )

    # VR 컬럼 통일
    if 'VR' not in merged.columns:
        if 'Volume_Ratio' in merged.columns:
            merged['VR'] = merged['Volume_Ratio']
        else:
            print("⚠️ VR/Volume_Ratio 컬럼 없음")

    # Price_Change 계산 (당일 변화율: (Close - Open) / Open * 100)
    if 'Price_Change' not in merged.columns:
        if 'Change' in merged.columns:
            merged['Price_Change'] = merged['Change']
        elif 'Close' in merged.columns and 'Open' in merged.columns:
            merged['Price_Change'] = ((merged['Close'] - merged['Open']) / merged['Open'] * 100)
        else:
            print("⚠️ Price_Change를 계산할 수 없음 (Close, Open 컬럼 없음)")

    # Return_20D → Fwd_Return_20D로 매핑 (20일 수익률 기준)
    if 'Fwd_Return_20D' not in merged.columns:
        if 'Return_20D' in merged.columns:
            merged['Fwd_Return_20D'] = merged['Return_20D']
        else:
            print("⚠️ Return_20D 컬럼 없음")

    print(f"전체 데이터: {len(merged):,}건")
    return merged

def classify_pension_sell(row, flow_threshold=0):
    """연기금 3D SELL 조건"""
    pension_3d = row.get('연기금_3D', 0)

    if pension_3d < -flow_threshold:
        return 'Pension-SELL'
    return None

def classify_ss_sniper(row, flow_threshold=0):
    """SS-Sniper: 개인+금투 BUY, 외인+연기금 SELL (3D)"""
    retail_3d = row.get('개인_3D', 0)
    foreign_3d = row.get('외국인_3D', 0)
    financial_3d = row.get('금융투자_3D', 0)
    pension_3d = row.get('연기금_3D', 0)

    # 4개 조건 모두 만족
    if (retail_3d > flow_threshold and
        foreign_3d < -flow_threshold and
        financial_3d > flow_threshold and
        pension_3d < -flow_threshold):
        return 'SS-Sniper'

    return None

def analyze_strategy_yearly(df, strategy_name, filter_func=None):
    """전략별 연도별 분석"""

    # VR=3.0, Price=5% 기본 조건
    base_condition = (df['VR'] >= 3.0) & (df['Price_Change'] >= 5.0)

    if filter_func is None:
        # Baseline: 수급 조건 없음
        signals = df[base_condition].copy()
    else:
        # 수급 필터 적용
        df['Strategy'] = df.apply(filter_func, axis=1)
        signals = df[base_condition & (df['Strategy'] == strategy_name)].copy()

    # 연도별 분석
    signals['Year'] = signals['Date'].dt.year
    yearly_results = []

    for year in range(2021, 2026):
        year_data = signals[signals['Year'] == year]

        if len(year_data) == 0:
            yearly_results.append({
                'Year': year,
                'Signals': 0,
                'Avg_Return': 0,
                'Median_Return': 0,
                'Win_Rate': 0,
                'Std': 0,
                'Sharpe': 0
            })
        else:
            returns = year_data['Fwd_Return_20D']
            yearly_results.append({
                'Year': year,
                'Signals': len(year_data),
                'Avg_Return': round(returns.mean(), 2),
                'Median_Return': round(returns.median(), 2),
                'Win_Rate': round((returns > 0).mean() * 100, 1),
                'Std': round(returns.std(), 2),
                'Sharpe': round((returns.mean() / returns.std()) if returns.std() > 0 else 0, 3)
            })

    return pd.DataFrame(yearly_results)

def main():
    print("="*80)
    print("Phase 3-8: Yearly Breakdown - Baseline vs Flow Filters")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # 필수 컬럼 확인
    required_cols = ['VR', 'Price_Change', 'Fwd_Return_20D', '개인_3D', '외국인_3D', '금융투자_3D', '연기금_3D']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 누락된 컬럼: {missing_cols}")
        return

    results = []

    # 1. Baseline (수급 조건 없음)
    print("\n[1/3] Baseline 연도별 분석 중...")
    baseline_result = analyze_strategy_yearly(df, 'Baseline', filter_func=None)
    baseline_result.insert(0, 'Strategy', 'Baseline')
    results.append(baseline_result)

    # 2. Pension-SELL
    print("[2/3] Pension-SELL 연도별 분석 중...")
    pension_result = analyze_strategy_yearly(df, 'Pension-SELL', filter_func=classify_pension_sell)
    pension_result.insert(0, 'Strategy', 'Pension-SELL')
    results.append(pension_result)

    # 3. SS-Sniper
    print("[3/3] SS-Sniper 연도별 분석 중...")
    sniper_result = analyze_strategy_yearly(df, 'SS-Sniper', filter_func=classify_ss_sniper)
    sniper_result.insert(0, 'Strategy', 'SS-Sniper')
    results.append(sniper_result)

    # 결과 통합
    comparison_df = pd.concat(results, ignore_index=True)

    # 저장
    output_path = RESULTS_DIR / 'p3_08_yearly_breakdown.csv'
    comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_path}")

    # 결과 출력
    print("\n" + "="*120)
    print("연도별 성과 비교 (VR=3.0, Price=5%, 20일 수익률 기준)")
    print("="*120)

    for strategy in ['Baseline', 'Pension-SELL', 'SS-Sniper']:
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy]

        print(f"\n【{strategy}】")
        print("-" * 120)

        for _, row in strategy_data.iterrows():
            year = row['Year']
            signals = row['Signals']
            avg_ret = row['Avg_Return']
            win_rate = row['Win_Rate']
            sharpe = row['Sharpe']

            # 수익률 상태
            ret_status = "✅" if avg_ret > 0 else "❌"

            print(f"{year} | "
                  f"신호: {signals:4d}건 | "
                  f"수익률: {avg_ret:7.2f}% {ret_status} | "
                  f"승률: {win_rate:5.1f}% | "
                  f"Sharpe: {sharpe:6.3f}")

    # 핵심 인사이트
    print("\n" + "="*120)
    print("핵심 인사이트")
    print("="*120)

    # 양수 수익률 연도 카운트
    for strategy in ['Baseline', 'Pension-SELL', 'SS-Sniper']:
        strategy_data = comparison_df[comparison_df['Strategy'] == strategy]
        positive_years = len(strategy_data[strategy_data['Avg_Return'] > 0])
        total_years = len(strategy_data)
        positive_pct = (positive_years / total_years) * 100

        print(f"\n{strategy}:")
        print(f"  양수 수익률 연도: {positive_years}/{total_years}년 ({positive_pct:.1f}%)")

        # 연도별 수익률 리스트
        yearly_returns = strategy_data[['Year', 'Avg_Return']].values
        print(f"  연도별: ", end="")
        for year, ret in yearly_returns:
            status = "✅" if ret > 0 else "❌"
            print(f"{int(year)}년 {ret:+.2f}% {status} | ", end="")
        print()

    print("\n" + "="*120)

if __name__ == '__main__':
    main()
