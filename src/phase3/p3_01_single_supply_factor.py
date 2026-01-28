"""
Phase 3-1: Single-Factor Foundation Analysis (수급 주체별)
==========================================================

목표:
- 5개년(2021-2025) 데이터 기반 수급 주체별 단일 영향력 검증
- 각 주체의 최적 Duration(1/3/5/10/20/30/50일) 탐색
- 조합(AND) 일절 배제, 단일 요인만 분석

분석 대상:
- 투자자: 개인, 외국인, 금융투자, 연기금
- Duration: 1, 3, 5, 10, 20, 30, 50일 누적
- Direction: 순매수(+), 순매도(-)

임계치:
- KOSPI 200: abs(순매수비중) >= 1.5%
- KOSDAQ 150: abs(순매수비중) >= 2.5%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

project_root = Path(__file__).parent.parent.parent


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


def analyze_single_supply_factor(data, investor, duration, direction, market, threshold):
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

    # Direction에 따라 필터링
    if direction == 'BUY':
        signals = data[data[cumul_col] >= threshold].copy()
    else:  # SELL
        signals = data[data[cumul_col] <= -threshold].copy()

    if len(signals) < 10:
        return None

    # 10일/20일 수익률 계산
    result = {
        'Market': market,
        'Investor': investor,
        'Duration': duration,
        'Direction': direction,
        'Threshold': threshold,
        'Signal_Count': len(signals),
        'Avg_Flow': signals[cumul_col].mean(),
        'Median_Flow': signals[cumul_col].median(),
        'Avg_Return_10D': signals['Return_10D'].mean(),
        'Median_Return_10D': signals['Return_10D'].median(),
        'Std_Return_10D': signals['Return_10D'].std(),
        'Win_Rate_10D': (signals['Return_10D'] > 0).sum() / len(signals) * 100,
        'Max_Gain_10D': signals['Return_10D'].max(),
        'Max_Loss_10D': signals['Return_10D'].min(),
    }

    # 20일 수익률 (있는 경우)
    if 'Return_20D' in signals.columns:
        result['Avg_Return_20D'] = signals['Return_20D'].mean()
        result['Median_Return_20D'] = signals['Return_20D'].median()
        result['Win_Rate_20D'] = (signals['Return_20D'] > 0).sum() / len(signals) * 100

    return result


def main():
    print("="*80)
    print("Phase 3-1: Single Supply Factor Analysis")
    print("="*80)
    print("\n목표: 수급 주체별 단일 영향력 검증 (2021-2025)")
    print("방법: Duration별 누적, 조합 배제, 단일 요인만 분석\n")

    # 1. 데이터 로드
    print("[1/6] 데이터 로드 중...")
    data_dir = project_root / 'data'

    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_list = pd.read_csv(data_dir / 'stock_list.csv')

    # 수급 데이터 로드 (인코딩 처리)
    try:
        # UTF-8 시도
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='utf-8-sig', low_memory=False)
    except:
        # CP949 시도
        investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)

    print(f"   주가 데이터: {len(stock_data):,}행")
    print(f"   수급 데이터: {len(investor_flow):,}행")

    # 2. 수급 데이터 전처리
    print("\n[2/6] 수급 데이터 전처리 중...")

    # 컬럼명이 한글이므로 인덱스로 접근
    # Date(0), 금융투자(1), 연기금(7), 외국인(10), Code(13)
    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        '금융투자_비중': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        '연기금_비중': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        '외국인_비중': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })

    # 개인은 역산
    flow_data['개인_비중'] = -(flow_data['금융투자_비중'] + flow_data['연기금_비중'] + flow_data['외국인_비중'])

    # 날짜 필터링: 2021-2025
    flow_data = flow_data[(flow_data['Date'] >= '2021-01-01') & (flow_data['Date'] <= '2025-12-31')]

    print(f"   전처리 후 수급 데이터: {len(flow_data):,}행")

    # 3. 데이터 병합
    print("\n[3/6] 주가 + 수급 데이터 병합 중...")

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[(stock_data['Date'] >= '2021-01-01') & (stock_data['Date'] <= '2025-12-31')]

    merged = stock_data.merge(
        flow_data,
        on=['Code', 'Date'],
        how='inner'
    )

    print(f"   병합 후: {len(merged):,}행")

    # Return_20D 계산 (없는 경우)
    if 'Return_20D' not in merged.columns:
        print("   Return_20D 계산 중...")
        merged = merged.sort_values(['Code', 'Date'])
        merged['Return_20D'] = merged.groupby('Code')['Close'].transform(
            lambda x: (x.shift(-20) / x - 1) * 100
        )

    # 4. 지수 구성 종목 필터링
    print("\n[4/6] 지수 구성 종목 필터링...")

    # KOSPI 200
    kospi_stocks = stock_list[stock_list['Market'] == 'KOSPI'].nlargest(200, 'Marcap')
    kospi_codes = kospi_stocks['Code'].tolist()

    # KOSDAQ 150
    kosdaq_stocks = stock_list[stock_list['Market'] == 'KOSDAQ'].nlargest(150, 'Marcap')
    kosdaq_codes = kosdaq_stocks['Code'].tolist()

    print(f"   KOSPI 200: {len(kospi_codes)}개")
    print(f"   KOSDAQ 150: {len(kosdaq_codes)}개")

    kospi_data = merged[merged['Code'].isin(kospi_codes)].copy()
    kosdaq_data = merged[merged['Code'].isin(kosdaq_codes)].copy()

    print(f"   KOSPI 데이터: {len(kospi_data):,}행")
    print(f"   KOSDAQ 데이터: {len(kosdaq_data):,}행")

    # 5. 단일 수급 요인 분석
    print("\n[5/6] 단일 수급 요인 분석 중...")

    durations = [1, 3, 5, 10, 20, 30, 50]
    investors = ['개인_비중', '외국인_비중', '금융투자_비중', '연기금_비중']
    investor_names = ['개인', '외국인', '금융투자', '연기금']

    all_results = []

    for market_name, market_data, threshold in [
        ('KOSPI200', kospi_data, 1.5),
        ('KOSDAQ150', kosdaq_data, 2.5)
    ]:
        print(f"\n   {market_name} 분석 (임계치: {threshold}%):")

        for duration in durations:
            print(f"      Duration {duration}일...")

            # Duration별 누적 계산
            market_data = calculate_cumulative_flow(market_data, duration, investors)

            for investor_col, investor_name in zip(investors, investor_names):
                # 순매수 (BUY)
                result_buy = analyze_single_supply_factor(
                    market_data, investor_col, duration, 'BUY', market_name, threshold
                )
                if result_buy:
                    result_buy['Investor'] = investor_name  # 한글 이름으로 변경
                    all_results.append(result_buy)

                # 순매도 (SELL)
                result_sell = analyze_single_supply_factor(
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

    # 투자자별 Best Performance
    print("\n투자자별 최고 성과 (10일 수익률 기준):")
    for investor in investor_names:
        investor_df = results_df[results_df['Investor'] == investor]
        if len(investor_df) > 0:
            best = investor_df.nlargest(1, 'Avg_Return_10D').iloc[0]
            print(f"\n  {investor}:")
            print(f"    Market: {best['Market']}, Duration: {best['Duration']}일, Direction: {best['Direction']}")
            print(f"    10일 수익률: {best['Avg_Return_10D']:.2f}% (승률: {best['Win_Rate_10D']:.1f}%)")
            print(f"    시그널 수: {best['Signal_Count']:.0f}개, 평균 수급: {best['Avg_Flow']:.2f}%")

    # Duration별 평균 성과
    print("\n\nDuration별 평균 성과:")
    for duration in durations:
        dur_df = results_df[results_df['Duration'] == duration]
        if len(dur_df) > 0:
            print(f"  {duration}일: 평균 수익률 {dur_df['Avg_Return_10D'].mean():.2f}%, "
                  f"평균 승률 {dur_df['Win_Rate_10D'].mean():.1f}%, "
                  f"분석 케이스 {len(dur_df)}개")

    print("\n" + "="*80)
    print("Phase 3-1 완료!")
    print("다음 단계: Golden Duration 확정 및 스코어링 시스템 설계")
    print("="*80)


if __name__ == "__main__":
    main()
