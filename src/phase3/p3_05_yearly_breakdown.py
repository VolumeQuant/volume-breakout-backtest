"""
Phase 3-5 Extended: 연도별 뱃지 효과 분석
==========================================

목적: S-Tactical + 뱃지 전략이 매년 안정적인지, 2025년만 운 좋았던 건지 검증

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

VR_THRESHOLD = 3.0
PRICE_CHANGE_THRESHOLD = 5.0
ATR_MULTIPLIER = 1.5
ZSCORE_VOL_THRESHOLD = 2.0
FLOW_THRESHOLD_KOSPI = 1.5
FLOW_THRESHOLD_KOSDAQ = 2.5


def load_data():
    """데이터 로딩"""
    print("📂 데이터 로딩 중...")
    stock_df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['Date'])
    flow_df = pd.read_csv(INVESTOR_FLOW_PATH, parse_dates=['Date'])
    print(f"   ✅ 주가: {len(stock_df):,}건 | 수급: {len(flow_df):,}건")
    return stock_df, flow_df


def calculate_atr(df, period=20):
    """ATR 계산"""
    df = df.copy()
    df = df.sort_values(['Code', 'Date'])
    df['PrevClose'] = df.groupby('Code')['Close'].shift(1)
    df['TR'] = df.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['PrevClose']) if pd.notna(row['PrevClose']) else 0,
            abs(row['Low'] - row['PrevClose']) if pd.notna(row['PrevClose']) else 0
        ), axis=1
    )
    df['ATR'] = df.groupby('Code')['TR'].transform(
        lambda x: x.rolling(window=period, min_periods=period).mean()
    )
    df['DailyRange'] = abs(df['Close'] - df['Open'])
    return df


def calculate_volume_zscore(df, period=20):
    """거래량 Z-Score"""
    df = df.copy()
    df = df.sort_values(['Code', 'Date'])
    df['Vol_Mean'] = df.groupby('Code')['Volume'].transform(
        lambda x: x.rolling(window=period, min_periods=period).mean()
    )
    df['Vol_Std'] = df.groupby('Code')['Volume'].transform(
        lambda x: x.rolling(window=period, min_periods=period).std()
    )
    df['Vol_ZScore'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
    df['Vol_ZScore'] = df['Vol_ZScore'].replace([np.inf, -np.inf], np.nan)
    return df


def add_badges(df):
    """뱃지 추가"""
    df = df.copy()
    df['Badge_ATR'] = ((df['DailyRange'] > df['ATR'] * ATR_MULTIPLIER) & df['ATR'].notna())
    df['Badge_ZVOL'] = ((df['Vol_ZScore'] > ZSCORE_VOL_THRESHOLD) & df['Vol_ZScore'].notna())
    df['Badge_BOTH'] = df['Badge_ATR'] & df['Badge_ZVOL']
    return df


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


def apply_s_tactical(row, flow_threshold):
    """S-Tactical 전략"""
    return (
        row.get('외국인_1D', 0) < -flow_threshold and
        row.get('연기금_1D', 0) < -flow_threshold
    )


def backtest_yearly(df, holding_period=20):
    """연도별 백테스팅"""
    results = []

    for idx, row in df.iterrows():
        code = str(row['Code']).zfill(6)
        is_kospi = code[0] in ['0', '1', '2', '3', '4', '5']
        flow_threshold = FLOW_THRESHOLD_KOSPI if is_kospi else FLOW_THRESHOLD_KOSDAQ

        if apply_s_tactical(row, flow_threshold):
            return_col = f'Return_{holding_period}D'
            if return_col in row:
                results.append({
                    'Year': row['Date'].year,
                    'Date': row['Date'],
                    'Code': code,
                    'Name': row.get('Name', 'N/A'),
                    'VR': row.get('Volume_Ratio', row.get('VR', np.nan)),
                    'Change': row.get('Change', np.nan),
                    'Badge_ATR': row.get('Badge_ATR', False),
                    'Badge_ZVOL': row.get('Badge_ZVOL', False),
                    'Badge_BOTH': row.get('Badge_BOTH', False),
                    'Return': row[return_col],
                    'Win': row[return_col] > 0,
                })

    return pd.DataFrame(results)


def analyze_yearly(signals_df):
    """연도별 뱃지 효과 분석"""
    results = []

    for year in sorted(signals_df['Year'].unique()):
        year_data = signals_df[signals_df['Year'] == year]

        badge_groups = {
            '기본': year_data,
            '🔥ATR': year_data[year_data['Badge_ATR']],
            '⚡ZVOL': year_data[year_data['Badge_ZVOL']],
            '🔥⚡둘다': year_data[year_data['Badge_BOTH']],
        }

        for badge_name, df in badge_groups.items():
            if len(df) > 0:
                results.append({
                    'Year': year,
                    'Badge': badge_name,
                    'Signals': len(df),
                    'Avg_Return': df['Return'].mean(),
                    'Median_Return': df['Return'].median(),
                    'Win_Rate': (df['Win'].sum() / len(df) * 100) if len(df) > 0 else 0,
                    'Std': df['Return'].std(),
                    'Max_Return': df['Return'].max(),
                    'Min_Return': df['Return'].min(),
                })

    return pd.DataFrame(results)


def main():
    """메인"""
    print("\n" + "=" * 90)
    print("📅 Phase 3-5 Extended: S-Tactical 연도별 뱃지 효과 분석")
    print("   목적: 2025년만 운 좋았는지 vs 매년 안정적인지 검증")
    print("=" * 90)

    # 데이터 로딩
    stock_df, flow_df = load_data()

    # 지표 계산
    print("\n📊 지표 계산 중...")
    stock_df = calculate_atr(stock_df)
    stock_df = calculate_volume_zscore(stock_df)
    stock_df = add_badges(stock_df)

    flow_df = prepare_flow_data(flow_df)

    # 병합
    merged = stock_df.merge(flow_df, on=['Date', 'Code'], how='left', suffixes=('', '_flow'))

    if 'Volume_Ratio' in merged.columns:
        merged['VR'] = merged['Volume_Ratio']
    if 'Change' not in merged.columns:
        merged['Change'] = ((merged['Close'] - merged['Open']) / merged['Open'] * 100)

    # Stage 1 필터
    filtered = merged[
        (merged['VR'] >= VR_THRESHOLD) &
        (merged['Change'] >= PRICE_CHANGE_THRESHOLD)
    ].copy()

    print(f"   ✅ Stage 1 통과: {len(filtered):,}건")

    # 백테스팅
    print("\n🔍 S-Tactical 전략 백테스팅...")
    signals = backtest_yearly(filtered, holding_period=20)

    if len(signals) == 0:
        print("⚠️ 시그널 없음")
        return

    print(f"   ✅ 총 시그널: {len(signals)}건 ({signals['Year'].min()}~{signals['Year'].max()})")

    # 연도별 분석
    yearly_results = analyze_yearly(signals)

    # 출력
    print("\n" + "=" * 90)
    print("📊 연도별 뱃지 효과 분석 (S-Tactical)")
    print("=" * 90)

    for year in sorted(yearly_results['Year'].unique()):
        year_data = yearly_results[yearly_results['Year'] == year]

        print(f"\n📅 {year}년")
        print("-" * 90)

        base = year_data[year_data['Badge'] == '기본']
        if len(base) > 0:
            base_return = base['Avg_Return'].values[0]
            base_signals = base['Signals'].values[0]

            for _, row in year_data.iterrows():
                improvement = ""
                if row['Badge'] != '기본':
                    delta = row['Avg_Return'] - base_return
                    improvement = f"({delta:+6.2f}%p)"

                print(f"  {row['Badge']:8s}: 시그널={row['Signals']:3d}건 | "
                      f"수익률={row['Avg_Return']:7.2f}% | 승률={row['Win_Rate']:5.1f}% | "
                      f"중앙값={row['Median_Return']:7.2f}% {improvement}")

    # 요약 통계
    print("\n" + "=" * 90)
    print("📈 요약: 뱃지별 연도 안정성")
    print("=" * 90)

    for badge in ['기본', '🔥ATR', '⚡ZVOL', '🔥⚡둘다']:
        badge_data = yearly_results[yearly_results['Badge'] == badge]

        if len(badge_data) > 0:
            avg_signals = badge_data['Signals'].mean()
            avg_return = badge_data['Avg_Return'].mean()
            std_return = badge_data['Avg_Return'].std()
            min_return = badge_data['Avg_Return'].min()
            max_return = badge_data['Avg_Return'].max()

            # 안정성 점수 (표준편차가 낮을수록 좋음)
            stability = std_return / abs(avg_return) if avg_return != 0 else np.inf

            print(f"\n[{badge}]")
            print(f"  평균 시그널: {avg_signals:.1f}건/년")
            print(f"  평균 수익률: {avg_return:.2f}% (σ={std_return:.2f}%)")
            print(f"  수익률 범위: {min_return:.2f}% ~ {max_return:.2f}%")
            print(f"  안정성 지수: {stability:.2f} (낮을수록 안정)")

            # 양의 수익률 연도 수
            positive_years = (badge_data['Avg_Return'] > 0).sum()
            total_years = len(badge_data)
            print(f"  양의 수익: {positive_years}/{total_years}년 ({positive_years/total_years*100:.1f}%)")

    # 최고 전략 추천
    print("\n" + "=" * 90)
    print("🎯 최종 판정")
    print("=" * 90)

    # 각 뱃지의 안정성과 수익률 종합 평가
    badge_summary = []

    for badge in ['기본', '🔥ATR', '⚡ZVOL', '🔥⚡둘다']:
        badge_data = yearly_results[yearly_results['Badge'] == badge]

        if len(badge_data) >= 3:  # 최소 3년 데이터
            avg_return = badge_data['Avg_Return'].mean()
            std_return = badge_data['Avg_Return'].std()
            avg_signals = badge_data['Signals'].mean()
            positive_years = (badge_data['Avg_Return'] > 0).sum()
            total_years = len(badge_data)

            # 점수 계산 (수익률 * 안정성 * 양의 년도 비율)
            stability_factor = 1 / (1 + std_return / abs(avg_return)) if avg_return != 0 else 0
            positive_factor = positive_years / total_years
            score = avg_return * stability_factor * positive_factor

            badge_summary.append({
                'Badge': badge,
                'Avg_Return': avg_return,
                'Stability': stability_factor,
                'Positive_Rate': positive_factor,
                'Score': score,
                'Avg_Signals': avg_signals,
            })

    summary_df = pd.DataFrame(badge_summary).sort_values('Score', ascending=False)

    print("\n종합 점수 순위 (수익률 × 안정성 × 양의 년도 비율):")
    print("-" * 90)

    for idx, row in summary_df.iterrows():
        verdict = "✅ 추천" if idx == 0 else ""
        print(f"  {row['Badge']:8s}: 점수={row['Score']:6.2f} | "
              f"수익률={row['Avg_Return']:6.2f}% | 안정성={row['Stability']:.2f} | "
              f"양의 비율={row['Positive_Rate']*100:5.1f}% | 시그널={row['Avg_Signals']:.1f}건/년 {verdict}")

    # 과적합 경고
    print("\n⚠️ 과적합 경고 체크:")
    print("-" * 90)

    for badge in ['🔥ATR', '⚡ZVOL', '🔥⚡둘다']:
        badge_data = yearly_results[yearly_results['Badge'] == badge]
        base_data = yearly_results[yearly_results['Badge'] == '기본']

        if len(badge_data) > 0 and len(base_data) > 0:
            # 2025년 개선폭
            badge_2025 = badge_data[badge_data['Year'] == 2025]['Avg_Return'].values
            base_2025 = base_data[base_data['Year'] == 2025]['Avg_Return'].values

            # 2021-2024 평균 개선폭
            badge_is = badge_data[badge_data['Year'] < 2025]['Avg_Return'].mean()
            base_is = base_data[base_data['Year'] < 2025]['Avg_Return'].mean()

            if len(badge_2025) > 0 and len(base_2025) > 0:
                oos_improvement = badge_2025[0] - base_2025[0]
                is_improvement = badge_is - base_is

                warning = ""
                if oos_improvement > 15 and is_improvement < 5:
                    warning = "🚨 과적합 가능성 높음!"
                elif oos_improvement > 10 and is_improvement < 0:
                    warning = "⚠️ 과적합 의심"

                print(f"  {badge:8s}: OOS 개선={oos_improvement:+6.2f}%p | "
                      f"IS 평균 개선={is_improvement:+6.2f}%p {warning}")

    # 결과 저장
    output_path = RESULTS_DIR / 'p3_05_yearly_breakdown.csv'
    yearly_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_path}")


if __name__ == '__main__':
    main()
