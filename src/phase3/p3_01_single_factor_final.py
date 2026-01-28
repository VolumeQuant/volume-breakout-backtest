"""
Phase 3-1: Single Supply Factor Analysis (Final Version)
========================================================

목표:
- 5개년(2021-2025) 데이터 기반 수급 주체별 단일 영향력 검증
- Duration별 (1/3/5/10/20/30/50일) 누적 효과 측정
- 조합(AND) 일절 배제, 단일 요인만 분석

요구사항:
1. 기간: 2021-01-01 ~ 2025-12-31
2. 대상: KOSPI 200, KOSDAQ 150
3. 제외: 지수 편입/편출 종목 (데이터 5년 내내 존재하는 종목만)
4. 임계치: KOSPI 1.5%, KOSDAQ 2.5%
5. Duration: 1, 3, 5, 10, 20, 30, 50일
6. 측정: 10일 및 20일 수익률, 승률
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent


def filter_stable_constituents(merged_data, stock_list, market_type, required_years=5):
    """
    지수 편입/편출 종목 제외 (5년 내내 데이터가 있는 종목만)

    Args:
        merged_data: 병합된 데이터
        stock_list: 종목 리스트
        market_type: 'KOSPI200' or 'KOSDAQ150'
        required_years: 필수 연수

    Returns:
        필터링된 데이터
    """
    # 시장별 종목 코드
    if market_type == 'KOSPI200':
        target_codes = stock_list[stock_list['Market'] == 'KOSPI'].nlargest(200, 'Marcap')['Code'].tolist()
    else:
        target_codes = stock_list[stock_list['Market'] == 'KOSDAQ'].nlargest(150, 'Marcap')['Code'].tolist()

    # 해당 종목들로 필터링
    market_data = merged_data[merged_data['Code'].isin(target_codes)].copy()

    # 각 종목의 데이터 존재 연수 계산
    market_data['Year'] = pd.to_datetime(market_data['Date']).dt.year
    years_per_stock = market_data.groupby('Code')['Year'].nunique()

    # 5년 모두 데이터가 있는 종목만 선택
    stable_codes = years_per_stock[years_per_stock >= required_years].index.tolist()

    filtered_data = market_data[market_data['Code'].isin(stable_codes)].copy()

    excluded_count = len(target_codes) - len(stable_codes)

    return filtered_data, stable_codes, excluded_count


def calculate_cumulative_flow(df, duration, investor_cols):
    """
    Duration일 누적 수급 비중 계산

    Args:
        df: 원본 데이터
        duration: 누적 기간 (일)
        investor_cols: 투자자 컬럼 리스트

    Returns:
        누적 컬럼이 추가된 데이터프레임
    """
    result = df.copy()

    for investor in investor_cols:
        cumul_col = f'{investor}_{duration}D'

        # 종목별로 rolling sum
        result[cumul_col] = result.groupby('Code')[investor].transform(
            lambda x: x.rolling(window=duration, min_periods=duration).sum()
        )

    return result


def analyze_single_factor(data, investor, duration, direction, market, threshold):
    """
    단일 수급 요인 분석

    Args:
        data: 분석 데이터
        investor: 투자자 컬럼명
        duration: 누적 기간
        direction: 'BUY' or 'SELL'
        market: 'KOSPI200' or 'KOSDAQ150'
        threshold: 임계치

    Returns:
        분석 결과 딕셔너리
    """
    cumul_col = f'{investor}_{duration}D'

    # Direction에 따라 필터링 (단일 조건만)
    if direction == 'BUY':
        signals = data[data[cumul_col] >= threshold].copy()
    else:  # SELL
        signals = data[data[cumul_col] <= -threshold].copy()

    # 시그널 수가 10개 미만이면 제외
    if len(signals) < 10:
        return None

    # NaN 제거
    signals_10d = signals.dropna(subset=['Return_10D'])
    signals_20d = signals.dropna(subset=['Return_20D'])

    if len(signals_10d) < 10:
        return None

    # 결과 계산
    result = {
        'Market': market,
        'Investor': investor,
        'Duration': duration,
        'Direction': direction,
        'Threshold': threshold,
        'Signal_Count': len(signals_10d),
        'Avg_Flow': signals[cumul_col].mean(),
        'Median_Flow': signals[cumul_col].median(),

        # 10일 수익률
        'Avg_Return_10D': signals_10d['Return_10D'].mean(),
        'Median_Return_10D': signals_10d['Return_10D'].median(),
        'Std_Return_10D': signals_10d['Return_10D'].std(),
        'Win_Rate_10D': (signals_10d['Return_10D'] > 0).sum() / len(signals_10d) * 100,
        'Max_Gain_10D': signals_10d['Return_10D'].max(),
        'Max_Loss_10D': signals_10d['Return_10D'].min(),

        # 20일 수익률
        'Avg_Return_20D': signals_20d['Return_20D'].mean() if len(signals_20d) >= 10 else np.nan,
        'Median_Return_20D': signals_20d['Return_20D'].median() if len(signals_20d) >= 10 else np.nan,
        'Std_Return_20D': signals_20d['Return_20D'].std() if len(signals_20d) >= 10 else np.nan,
        'Win_Rate_20D': (signals_20d['Return_20D'] > 0).sum() / len(signals_20d) * 100 if len(signals_20d) >= 10 else np.nan,
        'Max_Gain_20D': signals_20d['Return_20D'].max() if len(signals_20d) >= 10 else np.nan,
        'Max_Loss_20D': signals_20d['Return_20D'].min() if len(signals_20d) >= 10 else np.nan,
    }

    return result


def main():
    print("="*80)
    print("Phase 3-1: Single Supply Factor Analysis (2021-2025)")
    print("="*80)
    print("\n목표: 수급 주체별 단일 영향력 검증")
    print("기간: 2021-2025 (5개년)")
    print("방법: Duration별 누적, 조합 배제, 단일 요인만 분석\n")

    # 1. 데이터 로드
    print("[1/6] 데이터 로드 중...")
    data_dir = project_root / 'data'

    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_list = pd.read_csv(data_dir / 'stock_list.csv')

    # 수급 데이터 로드
    try:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='utf-8-sig', low_memory=False)
    except:
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)

    print(f"   주가 데이터: {len(stock_data):,}행")
    print(f"   수급 데이터: {len(investor_flow):,}행")

    # 날짜 확인
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    print(f"   데이터 기간: {stock_data['Date'].min()} ~ {stock_data['Date'].max()}")

    # 2. 수급 데이터 전처리
    print("\n[2/6] 수급 데이터 전처리 중...")

    # 컬럼명 처리 (인덱스 기반)
    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        '금융투자_비중': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        '연기금_비중': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        '외국인_비중': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })

    # 개인은 역산
    flow_data['개인_비중'] = -(flow_data['금융투자_비중'] + flow_data['연기금_비중'] + flow_data['외국인_비중'])

    print(f"   전처리 후: {len(flow_data):,}행")

    # 3. 데이터 병합
    print("\n[3/6] 주가 + 수급 데이터 병합 중...")

    merged = stock_data.merge(
        flow_data,
        on=['Code', 'Date'],
        how='inner'
    )

    print(f"   병합 후: {len(merged):,}행")

    # Return_20D 확인
    if 'Return_20D' not in merged.columns:
        print("   Return_20D 계산 중...")
        merged = merged.sort_values(['Code', 'Date'])
        merged['Return_20D'] = merged.groupby('Code')['Close'].transform(
            lambda x: (x.shift(-20) / x - 1) * 100
        )

    # 4. 지수 구성 종목 필터링 (생존자 편향 제거)
    print("\n[4/6] 지수 구성 종목 필터링 (생존자 편향 제거)...")

    all_results = []
    durations = [1, 3, 5, 10, 20, 30, 50]
    investors = ['개인_비중', '외국인_비중', '금융투자_비중', '연기금_비중']
    investor_names = ['개인', '외국인', '금융투자', '연기금']

    for market_name, threshold in [('KOSPI200', 1.5), ('KOSDAQ150', 2.5)]:
        print(f"\n   {market_name} 필터링 (임계치: {threshold}%)...")

        # 생존자 편향 제거 필터링 (5년 데이터 기준)
        market_data, stable_codes, excluded = filter_stable_constituents(
            merged, stock_list, market_name, required_years=5
        )

        print(f"      원래 종목: {200 if market_name=='KOSPI200' else 150}개")
        print(f"      5년 유지 종목: {len(stable_codes)}개")
        print(f"      제외된 종목: {excluded}개 (편입/편출 또는 데이터 부족)")
        print(f"      필터링 후 데이터: {len(market_data):,}행")

        # 5. 단일 수급 요인 분석
        print(f"\n   {market_name} 단일 수급 요인 분석 중...")

        for duration in durations:
            print(f"      Duration {duration}일...")

            # Duration별 누적 계산
            market_data = calculate_cumulative_flow(market_data, duration, investors)

            for investor_col, investor_name in zip(investors, investor_names):
                # 순매수 (BUY)
                result_buy = analyze_single_factor(
                    market_data, investor_col, duration, 'BUY', market_name, threshold
                )
                if result_buy:
                    result_buy['Investor'] = investor_name
                    all_results.append(result_buy)

                # 순매도 (SELL)
                result_sell = analyze_single_factor(
                    market_data, investor_col, duration, 'SELL', market_name, threshold
                )
                if result_sell:
                    result_sell['Investor'] = investor_name
                    all_results.append(result_sell)

    # 6. 결과 저장
    print("\n[6/6] 결과 저장 중...")

    results_df = pd.DataFrame(all_results)

    output_dir = project_root / 'results' / 'p3'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'p3_single_factor_impact.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"   저장 완료: {output_file}")
    print(f"   총 {len(results_df)}개 분석 결과")

    # 7. 요약 통계
    print("\n" + "="*80)
    print("요약 통계")
    print("="*80)

    # Duration별 10일 vs 20일 비교
    print("\nDuration별 10일 vs 20일 수익률 비교:")
    duration_compare = results_df.groupby('Duration').agg({
        'Avg_Return_10D': 'mean',
        'Win_Rate_10D': 'mean',
        'Avg_Return_20D': 'mean',
        'Win_Rate_20D': 'mean',
        'Signal_Count': 'sum'
    }).round(2)
    print(duration_compare)

    # 투자자별 Best Performance (10일 기준)
    print("\n\n투자자별 최고 성과 (10일 수익률 기준):")
    for investor in investor_names:
        investor_df = results_df[results_df['Investor'] == investor]
        if len(investor_df) > 0:
            best = investor_df.nlargest(1, 'Avg_Return_10D').iloc[0]
            print(f"\n  {investor}:")
            print(f"    Market: {best['Market']}, Duration: {best['Duration']}일, Direction: {best['Direction']}")
            print(f"    10일: {best['Avg_Return_10D']:.2f}% (승률: {best['Win_Rate_10D']:.1f}%)")
            print(f"    20일: {best['Avg_Return_20D']:.2f}% (승률: {best['Win_Rate_20D']:.1f}%)")
            print(f"    시그널 수: {best['Signal_Count']:.0f}개")

    # Top 10 전략 (10일 vs 20일)
    print("\n\nTop 10 전략 (10일 수익률 기준):")
    print("Rank | 투자자 | Dur | Dir | Market | 10D수익률 | 20D수익률 | 10D승률 | 20D승률 | 시그널")
    print("-" * 90)

    top10 = results_df.nlargest(10, 'Avg_Return_10D')
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:4d} | {row['Investor']:6s} | {row['Duration']:3d} | {row['Direction']:4s} | "
              f"{row['Market']:9s} | {row['Avg_Return_10D']:8.2f}% | {row['Avg_Return_20D']:8.2f}% | "
              f"{row['Win_Rate_10D']:6.1f}% | {row['Win_Rate_20D']:6.1f}% | {row['Signal_Count']:6.0f}")

    print("\n" + "="*80)
    print("Phase 3-1 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
