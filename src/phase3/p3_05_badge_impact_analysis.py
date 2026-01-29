"""
Phase 3-5: 폭발 품질 뱃지 효과 검증
====================================

목적: Phase 3-3 우승 전략에 통계적 폭발 필터(ATR, Z-Score)를 추가했을 때
     수익률과 승률이 유의미하게 개선되는지 검증

작성일: 2026-01-29
작성자: 조 (HTS 개발 8년차)

분석 설계:
    1. Phase 3-3 Multi-Duration 우승 전략 재현
    2. ATR, Z-VOL 뱃지 계산
    3. 뱃지별 성과 비교:
       - 기본 (뱃지 없음)
       - 🔥ATR만
       - ⚡Z-VOL만
       - 🔥⚡둘 다
    4. IS/OOS 성과 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# ============================================================
# 설정
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'p3'

STOCK_DATA_PATH = DATA_DIR / 'stock_data_with_indicators.csv'
INVESTOR_FLOW_PATH = DATA_DIR / 'investor_flow_data.csv'

# 기본 필터
VR_THRESHOLD = 3.0
PRICE_CHANGE_THRESHOLD = 5.0

# 폭발 품질 필터
ATR_MULTIPLIER = 1.5
ZSCORE_VOL_THRESHOLD = 2.0

# 수급 임계치
FLOW_THRESHOLD_KOSPI = 1.5
FLOW_THRESHOLD_KOSDAQ = 2.5

# 분석 기간
IS_START = '2021-01-01'
IS_END = '2024-12-31'
OOS_START = '2025-01-01'
OOS_END = '2025-12-31'


# ============================================================
# 데이터 로딩
# ============================================================

def load_data():
    """데이터 로딩"""
    print("📂 데이터 로딩 중...")

    stock_df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['Date'])
    flow_df = pd.read_csv(INVESTOR_FLOW_PATH, parse_dates=['Date'])

    print(f"   ✅ 주가 데이터: {len(stock_df):,}건")
    print(f"   ✅ 수급 데이터: {len(flow_df):,}건")

    return stock_df, flow_df


# ============================================================
# ATR 및 Z-Score 계산
# ============================================================

def calculate_atr(df, period=20):
    """ATR (Average True Range) 계산"""
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
    """거래량 Z-Score 계산"""
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


def add_explosion_badges(df):
    """폭발 품질 뱃지 추가"""
    df = df.copy()

    # ATR 뱃지
    df['Badge_ATR'] = (
        (df['DailyRange'] > df['ATR'] * ATR_MULTIPLIER) &
        df['ATR'].notna()
    )

    # Z-VOL 뱃지
    df['Badge_ZVOL'] = (
        (df['Vol_ZScore'] > ZSCORE_VOL_THRESHOLD) &
        df['Vol_ZScore'].notna()
    )

    # 둘 다
    df['Badge_BOTH'] = df['Badge_ATR'] & df['Badge_ZVOL']

    return df


# ============================================================
# 수급 데이터 전처리
# ============================================================

def prepare_flow_data(flow_df):
    """수급 데이터 전처리"""
    flow_df = flow_df.copy()
    flow_df = flow_df.sort_values(['Code', 'Date'])

    # 투자자 유형별 컬럼
    investors = ['개인', '외국인', '금융투자', '연기금']

    for investor in investors:
        if investor in flow_df.columns:
            # 원 → 억 원
            flow_df[f'{investor}_1D'] = flow_df[investor] / 100_000_000

            # 3일 누적
            flow_df[f'{investor}_3D'] = flow_df.groupby('Code')[f'{investor}_1D'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )

    return flow_df


# ============================================================
# Phase 3-3 우승 전략 재현
# ============================================================

def apply_ss_sniper(row, flow_threshold):
    """SS-Sniper: 개인+금투 BUY vs 외인+연기금 SELL (3D)"""
    return (
        row.get('개인_3D', 0) > flow_threshold and
        row.get('금융투자_3D', 0) > flow_threshold and
        row.get('외국인_3D', 0) < -flow_threshold and
        row.get('연기금_3D', 0) < -flow_threshold
    )


def apply_s_tactical(row, flow_threshold):
    """S-Tactical: 외인+연기금 SELL (1D)"""
    return (
        row.get('외국인_1D', 0) < -flow_threshold and
        row.get('연기금_1D', 0) < -flow_threshold
    )


def apply_a_base(row, flow_threshold):
    """A-Base: 개인 SELL + (외인 OR 연기금 BUY) (3D)"""
    return (
        row.get('개인_3D', 0) < -flow_threshold and
        (row.get('외국인_3D', 0) > flow_threshold or
         row.get('연기금_3D', 0) > flow_threshold)
    )


def apply_pension_sell_3d(row, flow_threshold):
    """Pension SELL 3D (Track B Winner)"""
    return row.get('연기금_3D', 0) < -flow_threshold


# ============================================================
# 백테스팅
# ============================================================

def backtest_strategy(df, strategy_name, strategy_func, holding_period=20):
    """전략 백테스팅"""

    results = []

    for idx, row in df.iterrows():
        # 시장 구분
        code = str(row['Code']).zfill(6)
        is_kospi = code[0] in ['0', '1', '2', '3', '4', '5']
        flow_threshold = FLOW_THRESHOLD_KOSPI if is_kospi else FLOW_THRESHOLD_KOSDAQ

        # 전략 조건 확인
        if strategy_func(row, flow_threshold):
            # 수익률
            return_col = f'Return_{holding_period}D'
            if return_col in row:
                results.append({
                    'Date': row['Date'],
                    'Code': code,
                    'Name': row.get('Name', 'N/A'),
                    'Market': 'KOSPI' if is_kospi else 'KOSDAQ',
                    'VR': row.get('Volume_Ratio', row.get('VR', np.nan)),
                    'Change': row.get('Change', np.nan),
                    'Badge_ATR': row.get('Badge_ATR', False),
                    'Badge_ZVOL': row.get('Badge_ZVOL', False),
                    'Badge_BOTH': row.get('Badge_BOTH', False),
                    'Return': row[return_col],
                    'Win': row[return_col] > 0,
                })

    return pd.DataFrame(results)


def analyze_badge_impact(signals_df, strategy_name, period):
    """뱃지별 성과 분석"""

    results = []

    # 1. 기본 (뱃지 없음)
    base_signals = signals_df.copy()

    # 2. ATR만
    atr_signals = signals_df[signals_df['Badge_ATR']].copy()

    # 3. Z-VOL만
    zvol_signals = signals_df[signals_df['Badge_ZVOL']].copy()

    # 4. 둘 다
    both_signals = signals_df[signals_df['Badge_BOTH']].copy()

    badge_groups = {
        '기본 (뱃지 없음)': base_signals,
        '🔥 ATR만': atr_signals,
        '⚡ Z-VOL만': zvol_signals,
        '🔥⚡ 둘 다': both_signals,
    }

    for badge_name, df in badge_groups.items():
        if len(df) > 0:
            results.append({
                'Strategy': strategy_name,
                'Period': period,
                'Badge': badge_name,
                'Signals': len(df),
                'Avg_Return': df['Return'].mean(),
                'Median_Return': df['Return'].median(),
                'Win_Rate': (df['Win'].sum() / len(df) * 100) if len(df) > 0 else 0,
                'Std': df['Return'].std(),
                'Sharpe': df['Return'].mean() / df['Return'].std() if df['Return'].std() > 0 else 0,
                'Max_Return': df['Return'].max(),
                'Min_Return': df['Return'].min(),
            })

    return pd.DataFrame(results)


# ============================================================
# 메인 분석
# ============================================================

def main():
    """메인 분석 함수"""

    print("\n" + "=" * 80)
    print("🔬 Phase 3-5: 폭발 품질 뱃지 효과 검증")
    print("   Phase 3-3 우승 전략 + ATR/Z-VOL 필터")
    print("=" * 80)

    # 데이터 로딩
    stock_df, flow_df = load_data()

    # ATR, Z-Score 계산
    print("\n📊 통계 지표 계산 중...")
    stock_df = calculate_atr(stock_df)
    stock_df = calculate_volume_zscore(stock_df)
    stock_df = add_explosion_badges(stock_df)

    # 수급 데이터 전처리
    print("📈 수급 데이터 전처리 중...")
    flow_df = prepare_flow_data(flow_df)

    # 데이터 병합
    print("🔗 데이터 병합 중...")
    merged = stock_df.merge(flow_df, on=['Date', 'Code'], how='left', suffixes=('', '_flow'))

    # VR, Change 계산
    if 'Volume_Ratio' in merged.columns:
        merged['VR'] = merged['Volume_Ratio']

    if 'Change' not in merged.columns:
        merged['Change'] = ((merged['Close'] - merged['Open']) / merged['Open'] * 100)

    # Stage 1 필터
    print(f"\n🔍 Stage 1 필터 적용 (VR≥{VR_THRESHOLD}, Change≥{PRICE_CHANGE_THRESHOLD}%)")
    filtered = merged[
        (merged['VR'] >= VR_THRESHOLD) &
        (merged['Change'] >= PRICE_CHANGE_THRESHOLD)
    ].copy()

    print(f"   ✅ Stage 1 통과: {len(filtered):,}건")

    # IS/OOS 분리
    is_data = filtered[
        (filtered['Date'] >= IS_START) &
        (filtered['Date'] <= IS_END)
    ].copy()

    oos_data = filtered[
        (filtered['Date'] >= OOS_START) &
        (filtered['Date'] <= OOS_END)
    ].copy()

    print(f"\n   📅 IS (2021-2024): {len(is_data):,}건")
    print(f"   📅 OOS (2025): {len(oos_data):,}건")

    # ============================================================
    # Phase 3-3 우승 전략 백테스팅
    # ============================================================

    strategies = {
        'SS-Sniper (3D)': apply_ss_sniper,
        'S-Tactical (1D)': apply_s_tactical,
        'A-Base (3D)': apply_a_base,
        'Pension-SELL (3D)': apply_pension_sell_3d,
    }

    all_results = []

    print("\n" + "=" * 80)
    print("🏆 전략별 뱃지 효과 분석")
    print("=" * 80)

    for strategy_name, strategy_func in strategies.items():
        print(f"\n📌 {strategy_name}")
        print("-" * 80)

        # IS 백테스팅
        is_signals = backtest_strategy(is_data, strategy_name, strategy_func, holding_period=20)

        if len(is_signals) > 0:
            is_impact = analyze_badge_impact(is_signals, strategy_name, 'IS')

            # OOS 백테스팅
            oos_signals = backtest_strategy(oos_data, strategy_name, strategy_func, holding_period=20)

            if len(oos_signals) > 0:
                oos_impact = analyze_badge_impact(oos_signals, strategy_name, 'OOS')

                # IS/OOS 결합
                is_impact['Period'] = 'IS (2021-2024)'
                oos_impact['Period'] = 'OOS (2025)'

                combined = pd.concat([is_impact, oos_impact], ignore_index=True)
                all_results.append(combined)

                # 결과 출력
                print("\n[IS (2021-2024)]")
                for _, row in is_impact.iterrows():
                    print(f"  {row['Badge']:20s}: 시그널={row['Signals']:4d}건 | "
                          f"수익률={row['Avg_Return']:6.2f}% | 승률={row['Win_Rate']:5.1f}% | "
                          f"Sharpe={row['Sharpe']:5.2f}")

                print("\n[OOS (2025)]")
                for _, row in oos_impact.iterrows():
                    improvement = ''
                    base_return = oos_impact[oos_impact['Badge'] == '기본 (뱃지 없음)']['Avg_Return'].values
                    if len(base_return) > 0 and row['Badge'] != '기본 (뱃지 없음)':
                        delta = row['Avg_Return'] - base_return[0]
                        improvement = f"({delta:+.2f}%p)"

                    print(f"  {row['Badge']:20s}: 시그널={row['Signals']:4d}건 | "
                          f"수익률={row['Avg_Return']:6.2f}% | 승률={row['Win_Rate']:5.1f}% | "
                          f"Sharpe={row['Sharpe']:5.2f} {improvement}")

    # ============================================================
    # 결과 저장
    # ============================================================

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        output_path = RESULTS_DIR / 'p3_05_badge_impact_analysis.csv'
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 80)
        print(f"💾 결과 저장: {output_path}")
        print("=" * 80)

        # ============================================================
        # 요약 분석
        # ============================================================

        print("\n" + "=" * 80)
        print("📊 종합 분석: 뱃지 효과 요약")
        print("=" * 80)

        oos_only = final_df[final_df['Period'] == 'OOS (2025)'].copy()

        if len(oos_only) > 0:
            print("\n🎯 OOS (2025) 성과 개선 효과:")
            print("-" * 80)

            for strategy in oos_only['Strategy'].unique():
                strategy_data = oos_only[oos_only['Strategy'] == strategy]

                base = strategy_data[strategy_data['Badge'] == '기본 (뱃지 없음)']

                if len(base) > 0:
                    base_return = base['Avg_Return'].values[0]
                    base_signals = base['Signals'].values[0]

                    print(f"\n[{strategy}]")
                    print(f"  기준: {base_return:.2f}% (시그널 {base_signals}건)")

                    for badge in ['🔥 ATR만', '⚡ Z-VOL만', '🔥⚡  둘 다']:
                        badge_data = strategy_data[strategy_data['Badge'] == badge]

                        if len(badge_data) > 0:
                            badge_return = badge_data['Avg_Return'].values[0]
                            badge_signals = badge_data['Signals'].values[0]
                            delta = badge_return - base_return
                            signal_ratio = badge_signals / base_signals * 100

                            verdict = "✅ 개선" if delta > 0 else "⚠️ 하락"

                            print(f"  {badge:12s}: {badge_return:6.2f}% ({delta:+.2f}%p) | "
                                  f"시그널 {badge_signals:4d}건 ({signal_ratio:5.1f}%) {verdict}")

            # 최고 개선 전략
            print("\n" + "=" * 80)
            print("🏆 최고 개선 전략 (OOS 기준)")
            print("=" * 80)

            # 기본 대비 개선률 계산
            improvements = []

            for strategy in oos_only['Strategy'].unique():
                strategy_data = oos_only[oos_only['Strategy'] == strategy]
                base = strategy_data[strategy_data['Badge'] == '기본 (뱃지 없음)']

                if len(base) > 0:
                    base_return = base['Avg_Return'].values[0]

                    for badge in ['🔥 ATR만', '⚡ Z-VOL만', '🔥⚡ 둘 다']:
                        badge_data = strategy_data[strategy_data['Badge'] == badge]

                        if len(badge_data) > 0:
                            badge_return = badge_data['Avg_Return'].values[0]
                            badge_signals = badge_data['Signals'].values[0]
                            delta = badge_return - base_return

                            if badge_signals >= 10:  # 최소 시그널 수
                                improvements.append({
                                    'Strategy': strategy,
                                    'Badge': badge,
                                    'Return': badge_return,
                                    'Improvement': delta,
                                    'Signals': badge_signals,
                                    'WinRate': badge_data['Win_Rate'].values[0],
                                })

            if improvements:
                imp_df = pd.DataFrame(improvements).sort_values('Improvement', ascending=False)

                print("\nTop 5 개선 효과:")
                for idx, row in imp_df.head(5).iterrows():
                    print(f"  {row['Strategy']:25s} + {row['Badge']:12s}: "
                          f"{row['Return']:6.2f}% ({row['Improvement']:+.2f}%p) | "
                          f"승률 {row['WinRate']:5.1f}% | 시그널 {row['Signals']}건")

    print("\n✅ 분석 완료!")


if __name__ == '__main__':
    main()
