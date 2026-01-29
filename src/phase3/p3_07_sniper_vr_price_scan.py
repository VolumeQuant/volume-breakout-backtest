"""
Phase 3-7: SS-Sniper VR/Price 완화 분석
========================================

목적: SS-Sniper 전략의 VR/Price 임계치를 완화하여
     - 시그널 증가 (목표: 주 2개)
     - IS (2021-2024) 4년 평균 안정성 확인
     - OOS (2025) 검증

SS-Sniper 조건:
  - 개인 3D BUY + 금투 3D BUY
  - 외인 3D SELL + 연기금 3D SELL
  - OOS: 16.91%, 승률 63.5%, 52건
  - IS: ???

작성일: 2026-01-29
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'p3'

STOCK_DATA_PATH = DATA_DIR / 'stock_data_with_indicators.csv'
INVESTOR_FLOW_PATH = DATA_DIR / 'investor_flow_data.csv'

FLOW_THRESHOLD_KOSPI = 1.5
FLOW_THRESHOLD_KOSDAQ = 2.5

# Grid Search 범위
VR_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
PRICE_THRESHOLDS = [3.0, 5.0, 7.0, 10.0]


def load_data():
    """데이터 로딩"""
    print("📂 데이터 로딩 중...")
    stock_df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['Date'])
    flow_df = pd.read_csv(INVESTOR_FLOW_PATH, parse_dates=['Date'])
    print(f"   ✅ 주가: {len(stock_df):,}건 | 수급: {len(flow_df):,}건")
    return stock_df, flow_df


def prepare_flow_data(flow_df):
    """수급 전처리"""
    flow_df = flow_df.copy()
    flow_df = flow_df.sort_values(['Code', 'Date'])

    for investor in ['개인', '외국인', '금융투자', '연기금']:
        if investor in flow_df.columns:
            flow_df[f'{investor}_1D'] = flow_df[investor] / 100_000_000
            flow_df[f'{investor}_3D'] = flow_df.groupby('Code')[f'{investor}_1D'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )
    return flow_df


def apply_ss_sniper(row, flow_threshold):
    """SS-Sniper 전략"""
    return (
        row.get('개인_3D', 0) > flow_threshold and
        row.get('금융투자_3D', 0) > flow_threshold and
        row.get('외국인_3D', 0) < -flow_threshold and
        row.get('연기금_3D', 0) < -flow_threshold
    )


def backtest_with_thresholds(df, vr_thresh, price_thresh, holding_period=20):
    """VR/Price 임계치별 백테스팅"""

    # Stage 1 필터
    filtered = df[
        (df['VR'] >= vr_thresh) &
        (df['Change'] >= price_thresh)
    ].copy()

    results = []

    for idx, row in filtered.iterrows():
        code = str(row['Code']).zfill(6)
        is_kospi = code[0] in ['0', '1', '2', '3', '4', '5']
        flow_threshold = FLOW_THRESHOLD_KOSPI if is_kospi else FLOW_THRESHOLD_KOSDAQ

        if apply_ss_sniper(row, flow_threshold):
            return_col = f'Return_{holding_period}D'
            if return_col in row:
                results.append({
                    'Year': row['Date'].year,
                    'Date': row['Date'],
                    'Code': code,
                    'Name': row.get('Name', 'N/A'),
                    'VR': row['VR'],
                    'Change': row['Change'],
                    'Return': row[return_col],
                    'Win': row[return_col] > 0,
                })

    return pd.DataFrame(results)


def analyze_threshold(signals_df, vr_thresh, price_thresh):
    """임계치별 성과 분석"""

    if len(signals_df) == 0:
        return None

    results = []

    # 연도별 분석
    for year in sorted(signals_df['Year'].unique()):
        year_data = signals_df[signals_df['Year'] == year]

        if len(year_data) > 0:
            results.append({
                'VR_Thresh': vr_thresh,
                'Price_Thresh': price_thresh,
                'Year': year,
                'Signals': len(year_data),
                'Avg_Return': year_data['Return'].mean(),
                'Median_Return': year_data['Return'].median(),
                'Win_Rate': (year_data['Win'].sum() / len(year_data) * 100) if len(year_data) > 0 else 0,
                'Std': year_data['Return'].std(),
                'Max_Return': year_data['Return'].max(),
                'Min_Return': year_data['Return'].min(),
            })

    # IS/OOS 통합
    is_data = signals_df[signals_df['Year'] < 2025]
    oos_data = signals_df[signals_df['Year'] == 2025]

    summary = []

    if len(is_data) > 0:
        is_years = is_data['Year'].nunique()
        summary.append({
            'VR_Thresh': vr_thresh,
            'Price_Thresh': price_thresh,
            'Period': 'IS (2021-2024)',
            'Years': is_years,
            'Total_Signals': len(is_data),
            'Avg_Signals_Per_Year': len(is_data) / is_years,
            'Avg_Return': is_data['Return'].mean(),
            'Median_Return': is_data['Return'].median(),
            'Win_Rate': (is_data['Win'].sum() / len(is_data) * 100),
            'Std': is_data['Return'].std(),
            'Sharpe': is_data['Return'].mean() / is_data['Return'].std() if is_data['Return'].std() > 0 else 0,
        })

    if len(oos_data) > 0:
        summary.append({
            'VR_Thresh': vr_thresh,
            'Price_Thresh': price_thresh,
            'Period': 'OOS (2025)',
            'Years': 1,
            'Total_Signals': len(oos_data),
            'Avg_Signals_Per_Year': len(oos_data),
            'Avg_Return': oos_data['Return'].mean(),
            'Median_Return': oos_data['Return'].median(),
            'Win_Rate': (oos_data['Win'].sum() / len(oos_data) * 100),
            'Std': oos_data['Return'].std(),
            'Sharpe': oos_data['Return'].mean() / oos_data['Return'].std() if oos_data['Return'].std() > 0 else 0,
        })

    return pd.DataFrame(results), pd.DataFrame(summary)


def main():
    """메인"""
    print("\n" + "=" * 100)
    print("🎯 Phase 3-7: SS-Sniper VR/Price 완화 분석")
    print("   목표: 시그널 증가 + IS/OOS 안정성 확인")
    print("=" * 100)

    # 데이터 로딩
    stock_df, flow_df = load_data()

    # 수급 전처리
    print("\n📈 수급 데이터 전처리 중...")
    flow_df = prepare_flow_data(flow_df)

    # 병합
    merged = stock_df.merge(flow_df, on=['Date', 'Code'], how='left', suffixes=('', '_flow'))

    if 'Volume_Ratio' in merged.columns:
        merged['VR'] = merged['Volume_Ratio']
    if 'Change' not in merged.columns:
        merged['Change'] = ((merged['Close'] - merged['Open']) / merged['Open'] * 100)

    # Grid Search
    print(f"\n🔍 Grid Search 시작...")
    print(f"   VR: {VR_THRESHOLDS}")
    print(f"   Price: {PRICE_THRESHOLDS}")
    print(f"   총 조합: {len(VR_THRESHOLDS) * len(PRICE_THRESHOLDS)}개")

    all_yearly = []
    all_summary = []

    total = len(VR_THRESHOLDS) * len(PRICE_THRESHOLDS)
    current = 0

    for vr_thresh in VR_THRESHOLDS:
        for price_thresh in PRICE_THRESHOLDS:
            current += 1
            print(f"\r   진행: {current}/{total} ({current/total*100:.1f}%)", end='')

            signals = backtest_with_thresholds(merged, vr_thresh, price_thresh)

            if len(signals) > 0:
                yearly, summary = analyze_threshold(signals, vr_thresh, price_thresh)

                if yearly is not None:
                    all_yearly.append(yearly)
                if summary is not None:
                    all_summary.append(summary)

    print("\n   ✅ Grid Search 완료!")

    # 결과 통합
    yearly_df = pd.concat(all_yearly, ignore_index=True) if all_yearly else pd.DataFrame()
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    # 저장
    if not yearly_df.empty:
        output_path1 = RESULTS_DIR / 'p3_07_sniper_yearly.csv'
        yearly_df.to_csv(output_path1, index=False, encoding='utf-8-sig')
        print(f"\n💾 연도별 결과 저장: {output_path1}")

    if not summary_df.empty:
        output_path2 = RESULTS_DIR / 'p3_07_sniper_summary.csv'
        summary_df.to_csv(output_path2, index=False, encoding='utf-8-sig')
        print(f"💾 요약 결과 저장: {output_path2}")

    # ============================================================
    # 결과 분석
    # ============================================================

    if summary_df.empty:
        print("\n⚠️ 시그널이 없습니다.")
        return

    print("\n" + "=" * 100)
    print("📊 SS-Sniper IS/OOS 비교")
    print("=" * 100)

    # 기준선 (VR 3.0, Price 5%)
    baseline_is = summary_df[
        (summary_df['VR_Thresh'] == 3.0) &
        (summary_df['Price_Thresh'] == 5.0) &
        (summary_df['Period'] == 'IS (2021-2024)')
    ]

    baseline_oos = summary_df[
        (summary_df['VR_Thresh'] == 3.0) &
        (summary_df['Price_Thresh'] == 5.0) &
        (summary_df['Period'] == 'OOS (2025)')
    ]

    print("\n[기준선: VR=3.0, Price=5%]")

    if len(baseline_is) > 0:
        row = baseline_is.iloc[0]
        print(f"\n  IS (2021-2024):")
        print(f"    시그널: {row['Total_Signals']:.0f}건 (년 {row['Avg_Signals_Per_Year']:.1f}개, 주 {row['Avg_Signals_Per_Year']/52:.1f}개)")
        print(f"    수익률: {row['Avg_Return']:.2f}% (중앙값 {row['Median_Return']:.2f}%)")
        print(f"    승률:   {row['Win_Rate']:.1f}%")
        print(f"    Sharpe: {row['Sharpe']:.2f}")

    if len(baseline_oos) > 0:
        row = baseline_oos.iloc[0]
        print(f"\n  OOS (2025):")
        print(f"    시그널: {row['Total_Signals']:.0f}건 (주 {row['Avg_Signals_Per_Year']/52:.1f}개)")
        print(f"    수익률: {row['Avg_Return']:.2f}% (중앙값 {row['Median_Return']:.2f}%)")
        print(f"    승률:   {row['Win_Rate']:.1f}%")
        print(f"    Sharpe: {row['Sharpe']:.2f}")

    # Sweet Spot 찾기
    print("\n" + "=" * 100)
    print("🎯 Sweet Spot 후보 (IS 안정성 + OOS 시그널)")
    print("=" * 100)

    # IS 데이터만 필터링
    is_summary = summary_df[summary_df['Period'] == 'IS (2021-2024)'].copy()

    # 조건: IS 수익률 > 0, IS 승률 > 40%
    is_summary = is_summary[
        (is_summary['Avg_Return'] > 0) &
        (is_summary['Win_Rate'] > 40)
    ].copy()

    # OOS 데이터 병합
    oos_summary = summary_df[summary_df['Period'] == 'OOS (2025)'].copy()

    combined = is_summary.merge(
        oos_summary,
        on=['VR_Thresh', 'Price_Thresh'],
        suffixes=('_IS', '_OOS')
    )

    if len(combined) > 0:
        # 점수 계산: IS 안정성 × OOS 시그널
        combined['Score'] = (
            combined['Avg_Return_IS'] * 0.3 +
            combined['Win_Rate_IS'] * 0.2 +
            (combined['Avg_Signals_Per_Year_OOS'] / 104) * 5.0 +  # 시그널 보너스 (목표 104건)
            combined['Avg_Return_OOS'] * 0.2
        )

        # Top 10
        top10 = combined.sort_values('Score', ascending=False).head(10)

        print(f"\n{'순위':<4} {'VR':<5} {'Price':<6} {'IS시그널':<8} {'IS수익':<8} {'IS승률':<7} {'OOS시그널':<9} {'OOS수익':<9} {'주':<6} {'점수':<6}")
        print("-" * 100)

        for idx, (i, row) in enumerate(top10.iterrows(), 1):
            weekly = row['Avg_Signals_Per_Year_OOS'] / 52
            verdict = "✅" if weekly >= 2 else "⚠️"

            print(f"{idx:<4} {row['VR_Thresh']:<5.1f} {row['Price_Thresh']:<6.1f} "
                  f"{row['Avg_Signals_Per_Year_IS']:<8.1f} {row['Avg_Return_IS']:<8.2f} "
                  f"{row['Win_Rate_IS']:<7.1f} {row['Avg_Signals_Per_Year_OOS']:<9.1f} "
                  f"{row['Avg_Return_OOS']:<9.2f} {weekly:<6.1f} {row['Score']:<6.2f} {verdict}")

        # 최고 추천
        best = top10.iloc[0]

        print("\n" + "=" * 100)
        print("✨ 최종 추천 설정")
        print("=" * 100)

        print(f"""
SS-Sniper 전략:
  - VR >= {best['VR_Thresh']:.1f} (기존 3.0에서 {'완화' if best['VR_Thresh'] < 3.0 else '유지/강화'})
  - Price >= {best['Price_Thresh']:.1f}% (기존 5.0%에서 {'완화' if best['Price_Thresh'] < 5.0 else '유지/강화'})

성과 (IS 2021-2024):
  - 시그널: {best['Avg_Signals_Per_Year_IS']:.1f}건/년
  - 수익률: {best['Avg_Return_IS']:.2f}%
  - 승률: {best['Win_Rate_IS']:.1f}%
  - Sharpe: {best['Sharpe_IS']:.2f}

성과 (OOS 2025):
  - 시그널: {best['Avg_Signals_Per_Year_OOS']:.0f}건/년 (주 {best['Avg_Signals_Per_Year_OOS']/52:.1f}개)
  - 수익률: {best['Avg_Return_OOS']:.2f}%
  - 승률: {best['Win_Rate_OOS']:.1f}%
  - Sharpe: {best['Sharpe_OOS']:.2f}

판정:
  - {'✅ 목표 달성 (주 2개 이상)' if best['Avg_Signals_Per_Year_OOS']/52 >= 2 else f"⚠️ 목표 미달 (주 {best['Avg_Signals_Per_Year_OOS']/52:.1f}개)"}
  - {'✅ IS 안정적 (수익률 양수)' if best['Avg_Return_IS'] > 0 else '⚠️ IS 불안정'}
  - {'✅ OOS 유효' if best['Avg_Return_OOS'] > best['Avg_Return_IS'] else '⚠️ OOS 하락'}
""")


if __name__ == '__main__':
    main()
