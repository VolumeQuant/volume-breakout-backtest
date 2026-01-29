"""
Phase 3-9: Quality Filter Grid Search
수급 조건 제거, 순수 거래량/가격 품질만으로 승률/수익률 개선

필터:
1. VR 임계치 강화 (3.0 → 3.5, 4.0, 4.5, 5.0)
2. Price 임계치 강화 (5% → 7%, 10%)
3. Z-Score 필터 (거래량 통계적 이상치)
4. ATR 필터 (변동성 대비 당일 등락폭)

목표:
- IS 수익률 2.38% → 5%+ 개선
- 승률 44.8% → 50%+ 개선
- 신호 빈도: 주 1~2개 허용 (품질 우선)
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

# Grid Search 범위
VR_THRESHOLDS = [3.0, 3.5, 4.0, 4.5, 5.0]
PRICE_THRESHOLDS = [5.0, 7.0, 10.0]
Z_SCORE_THRESHOLDS = [0, 1.5, 2.0, 2.5, 3.0]  # 0 = 필터 없음
ATR_MULTIPLIERS = [0, 1.3, 1.5, 2.0]  # 0 = 필터 없음

# ATR 계산 기간
ATR_PERIOD = 20

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

    # Return_20D 매핑
    if 'Fwd_Return_20D' not in stock_df.columns:
        if 'Return_20D' in stock_df.columns:
            stock_df['Fwd_Return_20D'] = stock_df['Return_20D']

    # Z_Score는 이미 있음

    print(f"전체 데이터: {len(stock_df):,}건")
    return stock_df

def calculate_atr(df):
    """ATR 계산"""
    print("ATR 계산 중...")

    df = df.sort_values(['Code', 'Date'])

    # True Range 계산
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df.groupby('Code')['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df.groupby('Code')['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # ATR (20일 평균)
    df['ATR'] = df.groupby('Code')['TR'].transform(
        lambda x: x.rolling(window=ATR_PERIOD, min_periods=1).mean()
    )

    # Daily Range
    df['Daily_Range'] = df['High'] - df['Low']

    # 정리
    df = df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'])

    print("✅ ATR 계산 완료")
    return df

def backtest_combination(df, vr_thresh, price_thresh, z_thresh, atr_mult):
    """조합별 백테스팅"""

    # Stage 1: VR + Price
    filtered = df[
        (df['VR'] >= vr_thresh) &
        (df['Price_Change'] >= price_thresh)
    ].copy()

    # Stage 2: Z-Score 필터
    if z_thresh > 0:
        filtered = filtered[filtered['Z_Score'] >= z_thresh]

    # Stage 3: ATR 필터
    if atr_mult > 0:
        filtered = filtered[filtered['Daily_Range'] >= filtered['ATR'] * atr_mult]

    # IS/OOS 분리
    filtered['Year'] = filtered['Date'].dt.year
    is_data = filtered[filtered['Year'].between(2021, 2024)]
    oos_data = filtered[filtered['Year'] == 2025]

    def calc_metrics(data, period_name, years):
        if len(data) == 0:
            return {
                'Period': period_name,
                'Years': years,
                'Signals': 0,
                'Signals_Per_Week': 0,
                'Avg_Return': 0,
                'Median_Return': 0,
                'Win_Rate': 0,
                'Std': 0,
                'Sharpe': 0
            }

        returns = data['Fwd_Return_20D']
        signals_per_week = len(data) / years / 52

        return {
            'Period': period_name,
            'Years': years,
            'Signals': len(data),
            'Signals_Per_Week': round(signals_per_week, 1),
            'Avg_Return': round(returns.mean(), 2),
            'Median_Return': round(returns.median(), 2),
            'Win_Rate': round((returns > 0).mean() * 100, 1),
            'Std': round(returns.std(), 2),
            'Sharpe': round((returns.mean() / returns.std()) if returns.std() > 0 else 0, 3)
        }

    is_metrics = calc_metrics(is_data, 'IS', 4)
    oos_metrics = calc_metrics(oos_data, 'OOS', 1)

    return is_metrics, oos_metrics

def main():
    print("="*80)
    print("Phase 3-9: Quality Filter Grid Search")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # ATR 계산
    df = calculate_atr(df)

    # 필수 컬럼 확인
    required_cols = ['VR', 'Price_Change', 'Z_Score', 'ATR', 'Daily_Range', 'Fwd_Return_20D']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 누락된 컬럼: {missing_cols}")
        return

    # Grid Search
    print(f"\n🔍 Grid Search 시작...")
    print(f"   VR: {VR_THRESHOLDS}")
    print(f"   Price: {PRICE_THRESHOLDS}")
    print(f"   Z-Score: {Z_SCORE_THRESHOLDS}")
    print(f"   ATR: {ATR_MULTIPLIERS}")

    total_combinations = len(VR_THRESHOLDS) * len(PRICE_THRESHOLDS) * len(Z_SCORE_THRESHOLDS) * len(ATR_MULTIPLIERS)
    print(f"   총 조합: {total_combinations}개\n")

    results = []
    count = 0

    for vr in VR_THRESHOLDS:
        for price in PRICE_THRESHOLDS:
            for z_score in Z_SCORE_THRESHOLDS:
                for atr in ATR_MULTIPLIERS:
                    count += 1

                    if count % 20 == 0:
                        print(f"   진행: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")

                    is_metrics, oos_metrics = backtest_combination(df, vr, price, z_score, atr)

                    # 조합 정보 추가
                    combo_info = {
                        'VR': vr,
                        'Price': price,
                        'Z_Score': z_score if z_score > 0 else 'None',
                        'ATR_Mult': atr if atr > 0 else 'None'
                    }

                    is_result = {**combo_info, **is_metrics}
                    oos_result = {**combo_info, **oos_metrics}

                    results.append(is_result)
                    results.append(oos_result)

    print(f"✅ Grid Search 완료\n")

    # 결과 정리
    results_df = pd.DataFrame(results)

    # 저장
    output_path = RESULTS_DIR / 'p3_09_quality_filter_scan.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 전체 결과 저장: {output_path}\n")

    # Top 조합 찾기
    print("="*120)
    print("Top 10 조합 (IS 수익률 기준)")
    print("="*120)

    is_results = results_df[results_df['Period'] == 'IS'].copy()

    # 최소 신호 개수 필터 (IS에서 주 0.5개 이상 = 연 26개)
    is_results = is_results[is_results['Signals'] >= 26]

    # IS 수익률 기준 정렬
    top_is = is_results.nlargest(10, 'Avg_Return')

    for idx, row in top_is.iterrows():
        vr = row['VR']
        price = row['Price']
        z_score = row['Z_Score']
        atr = row['ATR_Mult']
        signals = row['Signals']
        per_week = row['Signals_Per_Week']
        avg_ret = row['Avg_Return']
        win_rate = row['Win_Rate']
        sharpe = row['Sharpe']

        # 같은 조합의 OOS 찾기
        oos_row = results_df[
            (results_df['Period'] == 'OOS') &
            (results_df['VR'] == vr) &
            (results_df['Price'] == price) &
            (results_df['Z_Score'] == z_score) &
            (results_df['ATR_Mult'] == atr)
        ].iloc[0]

        oos_ret = oos_row['Avg_Return']
        oos_win = oos_row['Win_Rate']
        oos_signals = oos_row['Signals']

        # 안정성 체크
        stability = "✅" if abs(oos_ret - avg_ret) < 10 else "⚠️"

        print(f"\nVR≥{vr}, Price≥{price}%, Z≥{z_score}, ATR×{atr}")
        print(f"  IS:  {signals:3d}건 (주 {per_week:.1f}개) | {avg_ret:6.2f}% | 승률 {win_rate:5.1f}% | Sharpe {sharpe:.3f}")
        print(f"  OOS: {oos_signals:3d}건 | {oos_ret:6.2f}% | 승률 {oos_win:5.1f}% {stability}")

    # Baseline 비교
    print("\n" + "="*120)
    print("Baseline 참고 (VR≥3.0, Price≥5%, 필터 없음)")
    print("="*120)
    baseline_is = results_df[
        (results_df['Period'] == 'IS') &
        (results_df['VR'] == 3.0) &
        (results_df['Price'] == 5.0) &
        (results_df['Z_Score'] == 'None') &
        (results_df['ATR_Mult'] == 'None')
    ].iloc[0]

    baseline_oos = results_df[
        (results_df['Period'] == 'OOS') &
        (results_df['VR'] == 3.0) &
        (results_df['Price'] == 5.0) &
        (results_df['Z_Score'] == 'None') &
        (results_df['ATR_Mult'] == 'None')
    ].iloc[0]

    print(f"IS:  {baseline_is['Signals']:4d}건 (주 {baseline_is['Signals_Per_Week']:4.1f}개) | "
          f"{baseline_is['Avg_Return']:6.2f}% | 승률 {baseline_is['Win_Rate']:5.1f}% | "
          f"Sharpe {baseline_is['Sharpe']:.3f}")
    print(f"OOS: {baseline_oos['Signals']:4d}건 | "
          f"{baseline_oos['Avg_Return']:6.2f}% | 승률 {baseline_oos['Win_Rate']:5.1f}%")

    print("\n" + "="*120)

if __name__ == '__main__':
    main()
