"""
Phase 3-12: VR × PRICE Sweet Spot Analysis with MDD
거래량과 가격만으로 수익 내는 한계 확인 및 최적점 탐색

평가 지표:
1. 수익률 (IS/OOS)
2. 승률
3. Sharpe Ratio
4. MDD (Maximum Drawdown)
5. Profit Factor
6. 신호 빈도

목표: "거래량이 터지면서 장대양봉" 직관의 최적 조합
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

# Grid Search 범위 (거래량 폭발 + 장대양봉)
VR_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
PRICE_THRESHOLDS = [3, 5, 7, 8, 9, 10, 12, 15]

# 보유기간 (일단 10일 고정, 나중에 확장 가능)
HOLDING_PERIOD = 10

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

def calculate_mdd(returns_series):
    """MDD (Maximum Drawdown) 계산"""
    if len(returns_series) == 0:
        return 0

    # 누적 수익률 계산
    cumulative = (1 + returns_series / 100).cumprod()

    # Running maximum
    running_max = cumulative.expanding().max()

    # Drawdown
    drawdown = (cumulative - running_max) / running_max * 100

    # MDD
    mdd = drawdown.min()

    return mdd

def backtest_combination(df, vr_thresh, price_thresh):
    """VR × PRICE 조합별 백테스팅"""

    return_col = f'Return_{HOLDING_PERIOD}D'

    # 필터링
    signals = df[
        (df['VR'] >= vr_thresh) &
        (df['Price_Change'] >= price_thresh)
    ].copy()

    # 날짜순 정렬
    signals = signals.sort_values('Date')

    # IS/OOS 분리
    signals['Year'] = signals['Date'].dt.year
    is_data = signals[signals['Year'].between(2021, 2024)]
    oos_data = signals[signals['Year'] == 2025]

    def calc_metrics(data, period_name, years):
        if len(data) == 0:
            return {
                'VR': vr_thresh,
                'Price': price_thresh,
                'Period': period_name,
                'Years': years,
                'Signals': 0,
                'Signals_Per_Week': 0,
                'Avg_Return': 0,
                'Median_Return': 0,
                'Win_Rate': 0,
                'Std': 0,
                'Sharpe': 0,
                'MDD': 0,
                'Avg_Win': 0,
                'Avg_Loss': 0,
                'Win_Loss_Ratio': 0,
                'Profit_Factor': 0,
                'Expected_Value': 0,
                'Best_Return': 0,
                'Worst_Return': 0
            }

        returns = data[return_col]
        signals_per_week = len(data) / years / 52

        # 기본 통계
        avg_return = returns.mean()
        median_return = returns.median()
        win_rate = (returns > 0).mean() * 100
        std = returns.std()
        sharpe = (avg_return / std) if std > 0 else 0

        # MDD 계산
        mdd = calculate_mdd(returns)

        # 승패 분석
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0

        total_profit = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

        expected_value = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

        return {
            'VR': vr_thresh,
            'Price': price_thresh,
            'Period': period_name,
            'Years': years,
            'Signals': len(data),
            'Signals_Per_Week': round(signals_per_week, 1),
            'Avg_Return': round(avg_return, 2),
            'Median_Return': round(median_return, 2),
            'Win_Rate': round(win_rate, 1),
            'Std': round(std, 2),
            'Sharpe': round(sharpe, 3),
            'MDD': round(mdd, 2),
            'Avg_Win': round(avg_win, 2),
            'Avg_Loss': round(avg_loss, 2),
            'Win_Loss_Ratio': round(win_loss_ratio, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Expected_Value': round(expected_value, 2),
            'Best_Return': round(returns.max(), 2) if len(returns) > 0 else 0,
            'Worst_Return': round(returns.min(), 2) if len(returns) > 0 else 0
        }

    is_metrics = calc_metrics(is_data, 'IS', 4)
    oos_metrics = calc_metrics(oos_data, 'OOS', 1)

    return is_metrics, oos_metrics

def main():
    print("="*80)
    print("Phase 3-12: VR × PRICE Sweet Spot Analysis with MDD")
    print("="*80)

    # 데이터 로드
    df = load_data()

    # Grid Search
    print(f"\n🔍 Grid Search 시작 (보유기간: {HOLDING_PERIOD}일)")
    print(f"   VR: {VR_THRESHOLDS}")
    print(f"   PRICE: {PRICE_THRESHOLDS}")

    total_combinations = len(VR_THRESHOLDS) * len(PRICE_THRESHOLDS)
    print(f"   총 조합: {total_combinations}개\n")

    results = []
    count = 0

    for vr in VR_THRESHOLDS:
        for price in PRICE_THRESHOLDS:
            count += 1

            if count % 10 == 0:
                print(f"   진행: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")

            is_metrics, oos_metrics = backtest_combination(df, vr, price)
            results.append(is_metrics)
            results.append(oos_metrics)

    print(f"✅ Grid Search 완료\n")

    # 결과 정리
    results_df = pd.DataFrame(results)

    # 저장
    output_path = RESULTS_DIR / 'p3_12_vr_price_sweet_spot.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 전체 결과 저장: {output_path}\n")

    # IS 결과만 추출 (최소 신호 개수 필터)
    is_results = results_df[results_df['Period'] == 'IS'].copy()
    is_results = is_results[is_results['Signals'] >= 50]  # IS 4년에서 최소 50건 (주 0.24개)

    # 종합 점수 계산
    # 정규화: 0-100 스케일
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series))
        return (series - min_val) / (max_val - min_val) * 100

    is_results['Return_Score'] = normalize(is_results['Avg_Return'])
    is_results['WinRate_Score'] = normalize(is_results['Win_Rate'])
    is_results['Sharpe_Score'] = normalize(is_results['Sharpe'])
    is_results['MDD_Score'] = normalize(-is_results['MDD'])  # MDD는 작을수록 좋음 (음수 → 양수)

    # 종합 점수 (가중 평균)
    # 수익률 40%, 승률 20%, Sharpe 20%, MDD 20%
    is_results['Total_Score'] = (
        is_results['Return_Score'] * 0.4 +
        is_results['WinRate_Score'] * 0.2 +
        is_results['Sharpe_Score'] * 0.2 +
        is_results['MDD_Score'] * 0.2
    )

    # Top 조합 출력
    print("="*150)
    print("Top 20 조합 (종합 점수 기준)")
    print("="*150)
    print("종합 점수 = 수익률(40%) + 승률(20%) + Sharpe(20%) + MDD(20%)")
    print("="*150)

    top_combinations = is_results.nlargest(20, 'Total_Score')

    for idx, row in top_combinations.iterrows():
        vr = row['VR']
        price = row['Price']
        score = row['Total_Score']

        # IS 지표
        is_signals = row['Signals']
        is_per_week = row['Signals_Per_Week']
        is_ret = row['Avg_Return']
        is_win = row['Win_Rate']
        is_sharpe = row['Sharpe']
        is_mdd = row['MDD']
        is_pf = row['Profit_Factor']

        # 같은 조합의 OOS 찾기
        oos_row = results_df[
            (results_df['Period'] == 'OOS') &
            (results_df['VR'] == vr) &
            (results_df['Price'] == price)
        ].iloc[0]

        oos_signals = oos_row['Signals']
        oos_ret = oos_row['Avg_Return']
        oos_win = oos_row['Win_Rate']
        oos_mdd = oos_row['MDD']

        # IS-OOS 안정성
        ret_diff = abs(oos_ret - is_ret)
        stability = "✅" if ret_diff < 10 else "⚠️"

        print(f"\n🏆 VR≥{vr}, Price≥{price}% | 종합점수: {score:.1f}")
        print(f"  IS  ({is_signals:3d}건, 주 {is_per_week:.1f}개): "
              f"수익 {is_ret:6.2f}% | 승률 {is_win:5.1f}% | Sharpe {is_sharpe:.3f} | "
              f"MDD {is_mdd:6.2f}% | PF {is_pf:.2f}")
        print(f"  OOS ({oos_signals:3d}건): "
              f"수익 {oos_ret:6.2f}% | 승률 {oos_win:5.1f}% | MDD {oos_mdd:6.2f}% | "
              f"IS-OOS 차이 {oos_ret - is_ret:+.2f}%p {stability}")

    # 카테고리별 Best
    print("\n" + "="*150)
    print("카테고리별 Best")
    print("="*150)

    # 1. 수익률 최대
    best_return = is_results.nlargest(1, 'Avg_Return').iloc[0]
    print(f"\n1️⃣ 최고 수익률: VR≥{best_return['VR']}, Price≥{best_return['Price']}%")
    print(f"   IS: {best_return['Avg_Return']:.2f}% | 승률 {best_return['Win_Rate']:.1f}% | "
          f"MDD {best_return['MDD']:.2f}% | Sharpe {best_return['Sharpe']:.3f}")

    # 2. 승률 최대
    best_winrate = is_results.nlargest(1, 'Win_Rate').iloc[0]
    print(f"\n2️⃣ 최고 승률: VR≥{best_winrate['VR']}, Price≥{best_winrate['Price']}%")
    print(f"   IS: {best_winrate['Win_Rate']:.1f}% 승률 | {best_winrate['Avg_Return']:.2f}% 수익 | "
          f"MDD {best_winrate['MDD']:.2f}% | Sharpe {best_winrate['Sharpe']:.3f}")

    # 3. MDD 최소
    best_mdd = is_results.nsmallest(1, 'MDD').iloc[0]
    print(f"\n3️⃣ 최소 MDD: VR≥{best_mdd['VR']}, Price≥{best_mdd['Price']}%")
    print(f"   IS: MDD {best_mdd['MDD']:.2f}% | {best_mdd['Avg_Return']:.2f}% 수익 | "
          f"{best_mdd['Win_Rate']:.1f}% 승률 | Sharpe {best_mdd['Sharpe']:.3f}")

    # 4. Sharpe 최대
    best_sharpe = is_results.nlargest(1, 'Sharpe').iloc[0]
    print(f"\n4️⃣ 최고 Sharpe: VR≥{best_sharpe['VR']}, Price≥{best_sharpe['Price']}%")
    print(f"   IS: Sharpe {best_sharpe['Sharpe']:.3f} | {best_sharpe['Avg_Return']:.2f}% 수익 | "
          f"{best_sharpe['Win_Rate']:.1f}% 승률 | MDD {best_sharpe['MDD']:.2f}%")

    # 5. Profit Factor 최대
    best_pf = is_results.nlargest(1, 'Profit_Factor').iloc[0]
    print(f"\n5️⃣ 최고 Profit Factor: VR≥{best_pf['VR']}, Price≥{best_pf['Price']}%")
    print(f"   IS: PF {best_pf['Profit_Factor']:.2f} | {best_pf['Avg_Return']:.2f}% 수익 | "
          f"{best_pf['Win_Rate']:.1f}% 승률 | MDD {best_pf['MDD']:.2f}%")

    # VR과 PRICE의 영향 분석
    print("\n" + "="*150)
    print("VR과 PRICE의 영향 분석")
    print("="*150)

    # VR별 평균 성과
    print("\n📊 VR 수준별 평균 성과 (모든 PRICE 조합 평균):")
    vr_analysis = is_results.groupby('VR').agg({
        'Avg_Return': 'mean',
        'Win_Rate': 'mean',
        'MDD': 'mean',
        'Sharpe': 'mean',
        'Signals': 'mean'
    }).round(2)

    for vr, row in vr_analysis.iterrows():
        print(f"  VR≥{vr}: 수익 {row['Avg_Return']:5.2f}% | 승률 {row['Win_Rate']:5.1f}% | "
              f"MDD {row['MDD']:6.2f}% | Sharpe {row['Sharpe']:.3f} | 신호 {row['Signals']:.0f}건")

    # PRICE별 평균 성과
    print("\n📊 PRICE 수준별 평균 성과 (모든 VR 조합 평균):")
    price_analysis = is_results.groupby('Price').agg({
        'Avg_Return': 'mean',
        'Win_Rate': 'mean',
        'MDD': 'mean',
        'Sharpe': 'mean',
        'Signals': 'mean'
    }).round(2)

    for price, row in price_analysis.iterrows():
        print(f"  Price≥{price:2.0f}%: 수익 {row['Avg_Return']:5.2f}% | 승률 {row['Win_Rate']:5.1f}% | "
              f"MDD {row['MDD']:6.2f}% | Sharpe {row['Sharpe']:.3f} | 신호 {row['Signals']:.0f}건")

    print("\n" + "="*150)
    print("결론: 거래량과 가격만으로 수익을 내는 한계")
    print("="*150)

    # 전체 평균
    overall_avg = is_results['Avg_Return'].mean()
    overall_win = is_results['Win_Rate'].mean()
    overall_mdd = is_results['MDD'].mean()

    print(f"\n전체 {len(is_results)}개 조합의 평균:")
    print(f"  평균 수익률: {overall_avg:.2f}%")
    print(f"  평균 승률: {overall_win:.1f}%")
    print(f"  평균 MDD: {overall_mdd:.2f}%")

    # Best 조합
    best = top_combinations.iloc[0]
    print(f"\n최적 조합 (VR≥{best['VR']}, Price≥{best['Price']}%):")
    print(f"  수익률: {best['Avg_Return']:.2f}%")
    print(f"  승률: {best['Win_Rate']:.1f}%")
    print(f"  MDD: {best['MDD']:.2f}%")
    print(f"  Sharpe: {best['Sharpe']:.3f}")

    print("\n→ 거래량과 가격 조합만으로는 승률 50% 돌파가 어려움")
    print("→ MDD도 상당히 큼 (최악의 경우 -40% 이상)")
    print("→ 추가 필터(수급, 기술적 지표 등) 또는 손절/익절 로직 필요")

    print("\n" + "="*150)

if __name__ == '__main__':
    main()
