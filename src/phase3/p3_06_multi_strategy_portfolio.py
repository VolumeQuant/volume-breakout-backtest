"""
Phase 3-6: Multi-Strategy Portfolio Analysis
============================================

목적: 여러 전략을 조합하여 주 2개 이상 시그널 확보
     - S-Tactical (ATR 1.3 완화): 고수익률, 저빈도
     - Pension-SELL (기본): 중수익률, 중빈도
     - A-Base (상위 필터링): 안정적, 고빈도

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


# ============================================================
# 전략별 시그널 생성
# ============================================================

def apply_s_tactical_enhanced(row, flow_threshold, atr_mult=1.3):
    """S-Tactical + ATR 1.3 완화"""
    # S-Tactical 기본 조건
    tactical = (
        row.get('외국인_1D', 0) < -flow_threshold and
        row.get('연기금_1D', 0) < -flow_threshold
    )

    # ATR 뱃지 (1.3배 완화)
    if pd.notna(row.get('ATR')) and pd.notna(row.get('DailyRange')):
        atr_badge = row['DailyRange'] > row['ATR'] * atr_mult
    else:
        atr_badge = False

    return tactical and atr_badge


def apply_pension_sell_3d(row, flow_threshold):
    """Pension-SELL 3D (기본)"""
    return row.get('연기금_3D', 0) < -flow_threshold


def apply_a_base(row, flow_threshold):
    """A-Base (Handover) 기본"""
    return (
        row.get('개인_3D', 0) < -flow_threshold and
        (row.get('외국인_3D', 0) > flow_threshold or
         row.get('연기금_3D', 0) > flow_threshold)
    )


def calculate_quality_score(row):
    """A-Base 품질 점수 계산"""
    score = 0

    # VR 점수
    vr = row.get('VR', 0)
    if vr >= 5.0:
        score += 3
    elif vr >= 4.0:
        score += 2
    elif vr >= 3.0:
        score += 1

    # 가격 변동 점수
    change = row.get('Change', 0)
    if change >= 10.0:
        score += 3
    elif change >= 7.0:
        score += 2
    elif change >= 5.0:
        score += 1

    # Z-Score 점수
    if pd.notna(row.get('Vol_ZScore')):
        zscore = row['Vol_ZScore']
        if zscore >= 2.0:
            score += 3
        elif zscore >= 1.5:
            score += 2
        elif zscore >= 1.0:
            score += 1

    # ATR 점수
    if pd.notna(row.get('ATR')) and pd.notna(row.get('DailyRange')):
        atr_ratio = row['DailyRange'] / row['ATR']
        if atr_ratio >= 1.5:
            score += 3
        elif atr_ratio >= 1.3:
            score += 2
        elif atr_ratio >= 1.0:
            score += 1

    return score


def backtest_multi_strategy(df, holding_period=20):
    """멀티 전략 백테스팅"""
    results = []

    for idx, row in df.iterrows():
        code = str(row['Code']).zfill(6)
        is_kospi = code[0] in ['0', '1', '2', '3', '4', '5']
        flow_threshold = FLOW_THRESHOLD_KOSPI if is_kospi else FLOW_THRESHOLD_KOSDAQ

        # 각 전략 체크
        strategies = []

        # S-Tactical Enhanced (최고 우선순위)
        if apply_s_tactical_enhanced(row, flow_threshold):
            strategies.append(('S-Tactical', 1, 31.40, 60.0))

        # Pension-SELL (중간 우선순위)
        if apply_pension_sell_3d(row, flow_threshold):
            strategies.append(('Pension-SELL', 2, 11.23, 53.0))

        # A-Base (낮은 우선순위, 품질 점수로 필터링)
        if apply_a_base(row, flow_threshold):
            quality_score = calculate_quality_score(row)
            # 상위 품질만 (점수 8 이상)
            if quality_score >= 8:
                strategies.append(('A-Base-Premium', 3, 10.0, 52.0))

        # 전략이 있으면 최고 우선순위 하나만 선택
        if strategies:
            strategies.sort(key=lambda x: x[1])  # 우선순위로 정렬
            strategy_name, priority, expected_return, expected_winrate = strategies[0]

            return_col = f'Return_{holding_period}D'
            if return_col in row:
                results.append({
                    'Year': row['Date'].year,
                    'Date': row['Date'],
                    'Code': code,
                    'Name': row.get('Name', 'N/A'),
                    'Strategy': strategy_name,
                    'Priority': priority,
                    'VR': row.get('VR', np.nan),
                    'Change': row.get('Change', np.nan),
                    'Quality_Score': calculate_quality_score(row),
                    'Expected_Return': expected_return,
                    'Expected_WinRate': expected_winrate,
                    'Return': row[return_col],
                    'Win': row[return_col] > 0,
                })

    return pd.DataFrame(results)


def analyze_portfolio(signals_df):
    """포트폴리오 분석"""
    results = []

    # 연도별 분석
    for year in sorted(signals_df['Year'].unique()):
        year_data = signals_df[signals_df['Year'] == year]

        # 전체
        total = {
            'Year': year,
            'Strategy': '전체',
            'Signals': len(year_data),
            'Avg_Return': year_data['Return'].mean(),
            'Median_Return': year_data['Return'].median(),
            'Win_Rate': (year_data['Win'].sum() / len(year_data) * 100) if len(year_data) > 0 else 0,
            'Std': year_data['Return'].std(),
            'Sharpe': year_data['Return'].mean() / year_data['Return'].std() if year_data['Return'].std() > 0 else 0,
        }
        results.append(total)

        # 전략별
        for strategy in ['S-Tactical', 'Pension-SELL', 'A-Base-Premium']:
            strategy_data = year_data[year_data['Strategy'] == strategy]

            if len(strategy_data) > 0:
                results.append({
                    'Year': year,
                    'Strategy': strategy,
                    'Signals': len(strategy_data),
                    'Avg_Return': strategy_data['Return'].mean(),
                    'Median_Return': strategy_data['Return'].median(),
                    'Win_Rate': (strategy_data['Win'].sum() / len(strategy_data) * 100) if len(strategy_data) > 0 else 0,
                    'Std': strategy_data['Return'].std(),
                    'Sharpe': strategy_data['Return'].mean() / strategy_data['Return'].std() if strategy_data['Return'].std() > 0 else 0,
                })

    return pd.DataFrame(results)


def main():
    """메인"""
    print("\n" + "=" * 100)
    print("🎯 Phase 3-6: Multi-Strategy Portfolio Analysis")
    print("   목표: 여러 전략 조합으로 주 2개 이상 시그널 확보")
    print("=" * 100)

    # 데이터 로딩
    stock_df, flow_df = load_data()

    # 지표 계산
    print("\n📊 지표 계산 중...")
    stock_df = calculate_atr(stock_df)
    stock_df = calculate_volume_zscore(stock_df)

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

    # 멀티 전략 백테스팅
    print("\n🔍 Multi-Strategy 백테스팅...")
    signals = backtest_multi_strategy(filtered)

    print(f"   ✅ 총 시그널: {len(signals)}건 ({signals['Year'].min()}~{signals['Year'].max()})")

    # 연도별 분석
    portfolio_results = analyze_portfolio(signals)

    # ============================================================
    # 결과 출력
    # ============================================================

    print("\n" + "=" * 100)
    print("📊 연도별 포트폴리오 성과")
    print("=" * 100)

    for year in sorted(signals['Year'].unique()):
        year_results = portfolio_results[portfolio_results['Year'] == year]

        print(f"\n📅 {year}년")
        print("-" * 100)

        for _, row in year_results.iterrows():
            if row['Strategy'] == '전체':
                print(f"\n  {'[전체 포트폴리오]':20s}: 시그널={row['Signals']:4d}건 (월 {row['Signals']/12:4.1f}개, 주 {row['Signals']/52:4.1f}개)")
                print(f"  {'':20s}  수익률={row['Avg_Return']:6.2f}% | 승률={row['Win_Rate']:5.1f}% | Sharpe={row['Sharpe']:5.2f}")
                print()
            else:
                print(f"  {row['Strategy']:20s}: {row['Signals']:3d}건 | "
                      f"{row['Avg_Return']:6.2f}% | 승률 {row['Win_Rate']:5.1f}%")

    # ============================================================
    # 요약 통계
    # ============================================================

    print("\n" + "=" * 100)
    print("📈 5년 통합 요약")
    print("=" * 100)

    # IS/OOS 분리
    is_signals = signals[signals['Year'] < 2025]
    oos_signals = signals[signals['Year'] == 2025]

    # 전체 포트폴리오
    print("\n[전체 포트폴리오]")

    for period_name, period_data in [('IS (2021-2024)', is_signals), ('OOS (2025)', oos_signals)]:
        if len(period_data) > 0:
            years = period_data['Year'].nunique()
            avg_signals = len(period_data) / years

            print(f"\n  {period_name}:")
            print(f"    시그널: {len(period_data):4d}건 (년 {avg_signals:5.1f}개, 월 {avg_signals/12:4.1f}개, 주 {avg_signals/52:4.1f}개)")
            print(f"    수익률: {period_data['Return'].mean():6.2f}% (중앙값 {period_data['Return'].median():6.2f}%)")
            print(f"    승률:   {(period_data['Win'].sum() / len(period_data) * 100):5.1f}%")
            print(f"    Sharpe: {period_data['Return'].mean() / period_data['Return'].std():.2f}")

    # 전략별
    print("\n" + "-" * 100)
    print("[전략별 기여도]")

    for strategy in ['S-Tactical', 'Pension-SELL', 'A-Base-Premium']:
        print(f"\n  {strategy}:")

        for period_name, period_data in [('IS', is_signals), ('OOS', oos_signals)]:
            strategy_data = period_data[period_data['Strategy'] == strategy]

            if len(strategy_data) > 0:
                years = strategy_data['Year'].nunique()
                avg_signals = len(strategy_data) / years

                print(f"    {period_name:6s}: {len(strategy_data):3d}건 (년 {avg_signals:5.1f}개) | "
                      f"{strategy_data['Return'].mean():6.2f}% | 승률 {(strategy_data['Win'].sum() / len(strategy_data) * 100):5.1f}%")

    # ============================================================
    # 최종 판정
    # ============================================================

    print("\n" + "=" * 100)
    print("🎯 최종 판정")
    print("=" * 100)

    if len(oos_signals) > 0:
        oos_weekly = len(oos_signals) / 52
        target_met = "✅" if oos_weekly >= 2 else "⚠️"

        print(f"""
목표 달성 여부:
  - 목표: 주 2개 이상
  - 실제: 주 {oos_weekly:.1f}개 {target_met}
  - OOS 시그널: {len(oos_signals)}건/년 (월 {len(oos_signals)/12:.1f}개)

성과 (OOS 2025):
  - 평균 수익률: {oos_signals['Return'].mean():.2f}%
  - 승률: {(oos_signals['Win'].sum() / len(oos_signals) * 100):.1f}%
  - Sharpe: {oos_signals['Return'].mean() / oos_signals['Return'].std():.2f}

전략 구성:
  - S-Tactical: {len(oos_signals[oos_signals['Strategy'] == 'S-Tactical'])}건 (고수익률 특화)
  - Pension-SELL: {len(oos_signals[oos_signals['Strategy'] == 'Pension-SELL'])}건 (중수익률 안정)
  - A-Base-Premium: {len(oos_signals[oos_signals['Strategy'] == 'A-Base-Premium'])}건 (보조 전략)

안정성 평가:
  - IS 수익률: {is_signals['Return'].mean():.2f}%
  - OOS 수익률: {oos_signals['Return'].mean():.2f}%
  - IS→OOS 개선: {oos_signals['Return'].mean() - is_signals['Return'].mean():+.2f}%p
""")

    # 결과 저장
    output_path = RESULTS_DIR / 'p3_06_multi_strategy_portfolio.csv'
    signals.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 전체 시그널 저장: {output_path}")

    output_path2 = RESULTS_DIR / 'p3_06_portfolio_summary.csv'
    portfolio_results.to_csv(output_path2, index=False, encoding='utf-8-sig')
    print(f"💾 포트폴리오 요약 저장: {output_path2}")


if __name__ == '__main__':
    main()
