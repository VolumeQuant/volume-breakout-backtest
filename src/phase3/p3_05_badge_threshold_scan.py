"""
Phase 3-5 Badge Threshold Scan: Sweet Spot 찾기
==============================================

목적: 뱃지 임계치를 점진적으로 완화하면서
     - 충분한 시그널 (주 2개 = 월 8~9개 = 년 96~108개)
     - 안정적인 수익률 개선
     를 동시에 만족하는 Sweet Spot 찾기

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

# Grid Search 범위
ATR_MULTIPLIERS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
ZSCORE_THRESHOLDS = [1.0, 1.2, 1.5, 1.8, 2.0]

# 목표 시그널 수
TARGET_SIGNALS_PER_WEEK = 2
TARGET_SIGNALS_PER_YEAR = TARGET_SIGNALS_PER_WEEK * 52  # ~104건


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


def apply_s_tactical(row, flow_threshold):
    """S-Tactical 전략"""
    return (
        row.get('외국인_1D', 0) < -flow_threshold and
        row.get('연기금_1D', 0) < -flow_threshold
    )


def add_flexible_badges(df, atr_mult, zscore_thresh):
    """유연한 임계치로 뱃지 추가"""
    df = df.copy()

    df['Badge_ATR'] = (
        (df['DailyRange'] > df['ATR'] * atr_mult) &
        df['ATR'].notna()
    )

    df['Badge_ZVOL'] = (
        (df['Vol_ZScore'] > zscore_thresh) &
        df['Vol_ZScore'].notna()
    )

    df['Badge_BOTH'] = df['Badge_ATR'] & df['Badge_ZVOL']

    return df


def backtest_with_badges(df, atr_mult, zscore_thresh, holding_period=20):
    """뱃지 필터 백테스팅"""
    df = add_flexible_badges(df, atr_mult, zscore_thresh)

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
                    'Badge_ATR': row['Badge_ATR'],
                    'Badge_ZVOL': row['Badge_ZVOL'],
                    'Badge_BOTH': row['Badge_BOTH'],
                    'Return': row[return_col],
                    'Win': row[return_col] > 0,
                })

    return pd.DataFrame(results)


def analyze_threshold_combination(signals_df, atr_mult, zscore_thresh):
    """임계치 조합 분석"""
    results = []

    # IS/OOS 분리
    is_signals = signals_df[signals_df['Year'] < 2025]
    oos_signals = signals_df[signals_df['Year'] == 2025]

    badge_configs = {
        '기본': (False, False),  # 뱃지 없음
        'ATR만': (True, False),
        'ZVOL만': (False, True),
        '둘다': (True, True),
    }

    for badge_name, (use_atr, use_zvol) in badge_configs.items():
        # IS 분석
        if use_atr and use_zvol:
            is_filtered = is_signals[is_signals['Badge_BOTH']]
        elif use_atr:
            is_filtered = is_signals[is_signals['Badge_ATR']]
        elif use_zvol:
            is_filtered = is_signals[is_signals['Badge_ZVOL']]
        else:
            is_filtered = is_signals

        # OOS 분석
        if use_atr and use_zvol:
            oos_filtered = oos_signals[oos_signals['Badge_BOTH']]
        elif use_atr:
            oos_filtered = oos_signals[oos_signals['Badge_ATR']]
        elif use_zvol:
            oos_filtered = oos_signals[oos_signals['Badge_ZVOL']]
        else:
            oos_filtered = oos_signals

        # IS 성과
        if len(is_filtered) > 0:
            is_years = is_filtered['Year'].nunique()
            is_avg_signals = len(is_filtered) / is_years if is_years > 0 else 0
            is_return = is_filtered['Return'].mean()
            is_winrate = (is_filtered['Win'].sum() / len(is_filtered) * 100) if len(is_filtered) > 0 else 0
            is_std = is_filtered['Return'].std()
        else:
            is_avg_signals = 0
            is_return = np.nan
            is_winrate = 0
            is_std = np.nan

        # OOS 성과
        if len(oos_filtered) > 0:
            oos_signals_count = len(oos_filtered)
            oos_return = oos_filtered['Return'].mean()
            oos_winrate = (oos_filtered['Win'].sum() / len(oos_filtered) * 100) if len(oos_filtered) > 0 else 0
            oos_std = oos_filtered['Return'].std()
        else:
            oos_signals_count = 0
            oos_return = np.nan
            oos_winrate = 0
            oos_std = np.nan

        results.append({
            'ATR_Mult': atr_mult,
            'ZVOL_Thresh': zscore_thresh,
            'Badge': badge_name,
            'IS_Signals': len(is_filtered),
            'IS_Signals_Per_Year': is_avg_signals,
            'IS_Return': is_return,
            'IS_WinRate': is_winrate,
            'IS_Std': is_std,
            'OOS_Signals': oos_signals_count,
            'OOS_Return': oos_return,
            'OOS_WinRate': oos_winrate,
            'OOS_Std': oos_std,
        })

    return pd.DataFrame(results)


def main():
    """메인"""
    print("\n" + "=" * 100)
    print("🔍 Phase 3-5 Badge Threshold Scan: Sweet Spot 찾기")
    print(f"   목표: 주 {TARGET_SIGNALS_PER_WEEK}개 (년 ~{TARGET_SIGNALS_PER_YEAR}개) + 안정적 수익률 개선")
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

    # Grid Search
    print("\n🔍 Grid Search 시작...")
    print(f"   ATR 배수: {ATR_MULTIPLIERS}")
    print(f"   Z-Score 임계치: {ZSCORE_THRESHOLDS}")
    print(f"   총 조합: {len(ATR_MULTIPLIERS) * len(ZSCORE_THRESHOLDS)} × 4 뱃지 = {len(ATR_MULTIPLIERS) * len(ZSCORE_THRESHOLDS) * 4}개")

    all_results = []

    total_combinations = len(ATR_MULTIPLIERS) * len(ZSCORE_THRESHOLDS)
    current = 0

    for atr_mult in ATR_MULTIPLIERS:
        for zscore_thresh in ZSCORE_THRESHOLDS:
            current += 1
            print(f"\r   진행: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)", end='')

            # 백테스팅
            signals = backtest_with_badges(filtered, atr_mult, zscore_thresh)

            if len(signals) > 0:
                # 분석
                result = analyze_threshold_combination(signals, atr_mult, zscore_thresh)
                all_results.append(result)

    print("\n   ✅ Grid Search 완료!")

    # 결과 통합
    final_df = pd.concat(all_results, ignore_index=True)

    # 저장
    output_path = RESULTS_DIR / 'p3_05_badge_threshold_scan.csv'
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 결과 저장: {output_path}")

    # ============================================================
    # Sweet Spot 찾기
    # ============================================================

    print("\n" + "=" * 100)
    print("🎯 Sweet Spot 분석")
    print("=" * 100)

    # 조건: OOS 시그널 >= 20건 (월 1.7개 이상, 현실적 기준)
    MIN_OOS_SIGNALS = 20

    # OOS 시그널 조건만 (기본 포함)
    badge_filtered = final_df[
        (final_df['OOS_Signals'] >= MIN_OOS_SIGNALS)
    ].copy()

    print(f"\n필터링 조건:")
    print(f"  - OOS 시그널 >= {MIN_OOS_SIGNALS}건 (주 {MIN_OOS_SIGNALS/52:.1f}개, 월 {MIN_OOS_SIGNALS/12:.1f}개)")
    print(f"\n통과한 조합: {len(badge_filtered)}개")

    if len(badge_filtered) == 0:
        print("\n⚠️ 조건을 만족하는 조합이 없습니다.")
        return

    # 각 ATR/ZVOL 조합별로 기본 대비 개선폭 계산
    def calculate_improvements(row):
        # 같은 ATR/ZVOL 조합의 기본 전략 찾기
        base = final_df[
            (final_df['Badge'] == '기본') &
            (final_df['ATR_Mult'] == row['ATR_Mult']) &
            (final_df['ZVOL_Thresh'] == row['ZVOL_Thresh'])
        ]

        if len(base) > 0:
            return pd.Series({
                'IS_Improvement': row['IS_Return'] - base['IS_Return'].values[0],
                'OOS_Improvement': row['OOS_Return'] - base['OOS_Return'].values[0],
            })
        else:
            return pd.Series({'IS_Improvement': 0, 'OOS_Improvement': 0})

    badge_filtered[['IS_Improvement', 'OOS_Improvement']] = badge_filtered.apply(
        calculate_improvements, axis=1
    )

    # 안정성 점수 계산
    badge_filtered['Stability_Score'] = (
        badge_filtered['OOS_Return'] / badge_filtered['OOS_Std']
        if badge_filtered['OOS_Std'].notna().all()
        else 0
    )

    # 종합 점수: IS 개선 + OOS 개선 + 시그널 수 보너스
    badge_filtered['Total_Score'] = (
        badge_filtered['IS_Improvement'] * 0.3 +
        badge_filtered['OOS_Improvement'] * 0.5 +
        (badge_filtered['OOS_Signals'] / TARGET_SIGNALS_PER_YEAR) * 2.0  # 시그널 수 보너스
    )

    # Top 10 정렬
    top_configs = badge_filtered.sort_values('Total_Score', ascending=False).head(15)

    print("\n" + "=" * 100)
    print("🏆 Top 15 Sweet Spot 후보 (종합 점수순)")
    print("=" * 100)
    print(f"\n{'순위':<4} {'ATR':<5} {'ZVOL':<5} {'뱃지':<6} {'OOS':<8} {'주':<6} {'IS%':<7} {'OOS%':<8} {'IS개선':<7} {'OOS개선':<8} {'점수':<6}")
    print("-" * 100)

    for idx, (i, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{idx:<4} {row['ATR_Mult']:<5.1f} {row['ZVOL_Thresh']:<5.1f} {row['Badge']:<6} "
              f"{row['OOS_Signals']:<8.0f} {row['OOS_Signals']/52:<6.1f} {row['IS_Return']:<7.2f} {row['OOS_Return']:<8.2f} "
              f"{row['IS_Improvement']:<7.2f} {row['OOS_Improvement']:<8.2f} {row['Total_Score']:<6.2f}")

    # 가장 좋은 설정
    best = top_configs.iloc[0]

    print("\n" + "=" * 100)
    print("✨ 최종 추천 설정 (Sweet Spot)")
    print("=" * 100)

    print(f"""
전략: S-Tactical + {best['Badge']}

뱃지 임계치:
  - ATR 배수: {best['ATR_Mult']} (기존 1.5에서 완화)
  - Z-Score: {best['ZVOL_Thresh']} (기존 2.0에서 완화)

성과 (OOS 2025):
  - 시그널: {best['OOS_Signals']:.0f}건/년 (주 {best['OOS_Signals']/52:.1f}개, 월 {best['OOS_Signals']/12:.1f}개)
  - 수익률: {best['OOS_Return']:.2f}% (기본 대비 {best['OOS_Improvement']:+.2f}%p)
  - 승률: {best['OOS_WinRate']:.1f}%

성과 (IS 2021-2024):
  - 연평균 시그널: {best['IS_Signals_Per_Year']:.1f}건
  - 수익률: {best['IS_Return']:.2f}% (기본 대비 {best['IS_Improvement']:+.2f}%p)
  - 승률: {best['IS_WinRate']:.1f}%

종합 평가:
  - ✅ 시그널 충분 (주 {best['OOS_Signals']/52:.1f}개 {'✅' if best['OOS_Signals']/52 >= 2 else '⚠️'})
  - {'✅' if best['IS_Improvement'] > 0 else '⚠️'} IS 개선 ({best['IS_Improvement']:+.2f}%p)
  - {'✅' if best['OOS_Improvement'] > 0 else '⚠️'} OOS 개선 ({best['OOS_Improvement']:+.2f}%p)
""")

    # 상위 3개 비교
    print("\n" + "=" * 100)
    print("📊 Top 3 상세 비교")
    print("=" * 100)

    for idx, (i, row) in enumerate(top_configs.head(3).iterrows(), 1):
        print(f"\n[{idx}위] ATR {row['ATR_Mult']:.1f} × ZVOL {row['ZVOL_Thresh']:.1f} + {row['Badge']}")
        print(f"  OOS: {row['OOS_Signals']:.0f}건 (주 {row['OOS_Signals']/52:.1f}개) | "
              f"{row['OOS_Return']:.2f}% | 승률 {row['OOS_WinRate']:.1f}%")
        print(f"  IS:  {row['IS_Signals_Per_Year']:.1f}건/년 | "
              f"{row['IS_Return']:.2f}% | 승률 {row['IS_WinRate']:.1f}%")
        print(f"  개선: IS {row['IS_Improvement']:+.2f}%p | OOS {row['OOS_Improvement']:+.2f}%p")


if __name__ == '__main__':
    main()
