"""
Phase 3-1: Single-Factor Foundation Analysis
============================================

목표:
- 5개년(2021-2025) 데이터 기반 수급 주체별 단일 영향력 검증
- 각 주체의 최적 Duration(기간) 탐색
- 과적합 없는 견고한 기반 구축

방법론:
- 조합(AND) 일절 배제
- 단일 요인의 순수한 영향력만 측정
- Duration: 1/3/5/10/20/30/50일 누적
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def calculate_cumulative_flow(df, duration):
    """
    Duration일 누적 수급 비중 계산

    Args:
        df: 원본 데이터프레임
        duration: 누적 기간 (일)

    Returns:
        누적 수급 컬럼이 추가된 데이터프레임
    """
    result = df.copy()

    # 각 종목별로 그룹화하여 rolling sum 계산
    for investor in ['개인', '외국인', '금융투자', '연기금']:
        col_name = f'{investor}_비중'
        cumul_col_name = f'{investor}_{duration}D_Cumul'

        result[cumul_col_name] = result.groupby('Code')[col_name].transform(
            lambda x: x.rolling(window=duration, min_periods=1).sum()
        )

    return result

def analyze_single_factor(data, investor, duration, market_type, threshold):
    """
    단일 수급 주체의 영향력 분석

    Args:
        data: 데이터프레임
        investor: 투자자 유형 ('개인', '외국인', '금융투자', '연기금')
        duration: 누적 기간 (일)
        market_type: 'KOSPI200' or 'KOSDAQ150'
        threshold: 임계치 (절댓값)

    Returns:
        분석 결과 딕셔너리
    """
    cumul_col = f'{investor}_{duration}D_Cumul'

    # 순매수 (양수) 시그널
    buy_signals = data[data[cumul_col] >= threshold].copy()

    # 순매도 (음수) 시그널
    sell_signals = data[data[cumul_col] <= -threshold].copy()

    results = []

    # 순매수 분석
    if len(buy_signals) >= 10:  # 최소 10개 샘플
        buy_result = {
            'Market': market_type,
            'Investor': investor,
            'Duration': duration,
            'Direction': 'BUY',
            'Threshold': threshold,
            'Signal_Count': len(buy_signals),
            'Avg_Return_10D': buy_signals['Return_10D'].mean(),
            'Median_Return_10D': buy_signals['Return_10D'].median(),
            'Std_Return_10D': buy_signals['Return_10D'].std(),
            'Win_Rate_10D': (buy_signals['Return_10D'] > 0).sum() / len(buy_signals) * 100,
            'Avg_Return_20D': buy_signals['Return_20D'].mean() if 'Return_20D' in buy_signals.columns else np.nan,
            'Win_Rate_20D': (buy_signals['Return_20D'] > 0).sum() / len(buy_signals) * 100 if 'Return_20D' in buy_signals.columns else np.nan,
        }
        results.append(buy_result)

    # 순매도 분석
    if len(sell_signals) >= 10:
        sell_result = {
            'Market': market_type,
            'Investor': investor,
            'Duration': duration,
            'Direction': 'SELL',
            'Threshold': threshold,
            'Signal_Count': len(sell_signals),
            'Avg_Return_10D': sell_signals['Return_10D'].mean(),
            'Median_Return_10D': sell_signals['Return_10D'].median(),
            'Std_Return_10D': sell_signals['Return_10D'].std(),
            'Win_Rate_10D': (sell_signals['Return_10D'] > 0).sum() / len(sell_signals) * 100,
            'Avg_Return_20D': sell_signals['Return_20D'].mean() if 'Return_20D' in sell_signals.columns else np.nan,
            'Win_Rate_20D': (sell_signals['Return_20D'] > 0).sum() / len(sell_signals) * 100 if 'Return_20D' in sell_signals.columns else np.nan,
        }
        results.append(sell_result)

    return results

def get_index_constituents(stock_list, market_type):
    """
    지수 구성 종목 필터링

    Args:
        stock_list: 전체 종목 리스트
        market_type: 'KOSPI200' or 'KOSDAQ150'

    Returns:
        필터링된 종목 코드 리스트
    """
    if market_type == 'KOSPI200':
        # 시가총액 기준 KOSPI 상위 200개
        kospi_stocks = stock_list[stock_list['Market'] == 'KOSPI'].nsmallest(200, 'Market_Cap_Rank')
        return kospi_stocks['Code'].tolist()
    else:  # KOSDAQ150
        # 시가총액 기준 KOSDAQ 상위 150개
        kosdaq_stocks = stock_list[stock_list['Market'] == 'KOSDAQ'].nsmallest(150, 'Market_Cap_Rank')
        return kosdaq_stocks['Code'].tolist()

def main():
    print("="*80)
    print("Phase 3-1: Single-Factor Foundation Analysis")
    print("="*80)
    print("\n목표: 수급 주체별 단일 영향력 검증 (2021-2025)")
    print("방법: Duration별 누적 데이터 분석, 조합 배제\n")

    # 1. 데이터 로드
    print("[1/6] 데이터 로드 중...")

    # CSV 파일에서 직접 로드
    data_dir = project_root / 'data'
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    investor_flow_raw = pd.read_csv(data_dir / 'investor_flow_data.csv', encoding='cp949', low_memory=False)
    stock_list = pd.read_csv(data_dir / 'stock_list.csv')

    # 날짜 필터링: 2021-2025
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    investor_flow_raw['Date'] = pd.to_datetime(investor_flow_raw['Date'])

    start_date = '2021-01-01'
    end_date = '2025-12-31'

    stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    investor_flow_raw = investor_flow_raw[(investor_flow_raw['Date'] >= start_date) & (investor_flow_raw['Date'] <= end_date)]

    print(f"   기간: {start_date} ~ {end_date}")
    print(f"   주가 데이터: {len(stock_data):,}행")
    print(f"   수급 데이터: {len(investor_flow_raw):,}행")

    # 컬럼명 매핑 (인덱스 기반으로 안전하게 처리)
    # Date(0), 금융투자(1), 기타법인(6), 연기금(7), 기타외국인(8), 사모펀드(9), 외국인(10), 기타외국인(11), 전체(12), Code(13), Name(14), Market(15)
    print("\n[2/6] 데이터 전처리 중...")

    # 필요한 컬럼만 선택 (인덱스로 접근)
    investor_flow = investor_flow_raw[['Code', 'Date']].copy()

    # 개인은 전체에서 다른 모든 주체를 빼면 됨 (거래량 * 비중으로 계산)
    # 하지만 여기서는 비중 데이터를 직접 사용
    # 컬럼 1: 금융투자, 7: 연기금, 10: 외국인

    # 비중 계산을 위한 전처리
    investor_cols_mapping = {
        1: '금융투자',
        7: '연기금',
        10: '외국인'
    }

    for idx, name in investor_cols_mapping.items():
        col_data = investor_flow_raw.iloc[:, idx]
        investor_flow[f'{name}_비중'] = col_data

    # 개인은 역으로 계산 (전체 = 100%로 가정)
    # 개인 = 100 - (외국인 + 금융투자 + 연기금 + 기타)
    # 간단하게 개인 = -(외국인 + 금융투자 + 연기금)으로 근사
    investor_flow['개인_비중'] = -(investor_flow['외국인_비중'] + investor_flow['금융투자_비중'] + investor_flow['연기금_비중'])

    # 2-2. 데이터 병합
    print("   데이터 병합 중...")
    merged = stock_data.merge(
        investor_flow[['Code', 'Date', '개인_비중', '외국인_비중', '금융투자_비중', '연기금_비중']],
        on=['Code', 'Date'],
        how='inner'
    )
    print(f"   병합 후: {len(merged):,}행")

    # Return_20D 계산 (없는 경우)
    if 'Return_20D' not in merged.columns:
        print("\n[추가] Return_20D 계산 중...")
        merged = merged.sort_values(['Code', 'Date'])
        merged['Return_20D'] = merged.groupby('Code')['Close'].transform(
            lambda x: (x.shift(-20) / x - 1) * 100
        )

    # 3. 분석 설정
    durations = [1, 3, 5, 10, 20, 30, 50]
    investors = ['개인', '외국인', '금융투자', '연기금']

    all_results = []

    # 4. KOSPI 200 분석
    print("\n[3/6] KOSPI 200 분석 중...")
    kospi_codes = get_index_constituents(stock_list, 'KOSPI200')
    kospi_data = merged[merged['Code'].isin(kospi_codes)].copy()
    print(f"   KOSPI 200 종목: {len(kospi_codes)}개")
    print(f"   KOSPI 200 데이터: {len(kospi_data):,}행")

    kospi_threshold = 1.5

    for duration in durations:
        print(f"   Duration {duration}일 분석 중...")

        # Duration별 누적 계산
        kospi_data = calculate_cumulative_flow(kospi_data, duration)

        for investor in investors:
            results = analyze_single_factor(
                kospi_data, investor, duration, 'KOSPI200', kospi_threshold
            )
            all_results.extend(results)

    # 5. KOSDAQ 150 분석
    print("\n[4/6] KOSDAQ 150 분석 중...")
    kosdaq_codes = get_index_constituents(stock_list, 'KOSDAQ150')
    kosdaq_data = merged[merged['Code'].isin(kosdaq_codes)].copy()
    print(f"   KOSDAQ 150 종목: {len(kosdaq_codes)}개")
    print(f"   KOSDAQ 150 데이터: {len(kosdaq_data):,}행")

    kosdaq_threshold = 2.5

    for duration in durations:
        print(f"   Duration {duration}일 분석 중...")

        # Duration별 누적 계산
        kosdaq_data = calculate_cumulative_flow(kosdaq_data, duration)

        for investor in investors:
            results = analyze_single_factor(
                kosdaq_data, investor, duration, 'KOSDAQ150', kosdaq_threshold
            )
            all_results.extend(results)

    # 6. 결과 저장
    print("\n[5/6] 결과 저장 중...")
    results_df = pd.DataFrame(all_results)

    output_dir = project_root / 'results' / 'p3'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'p3_01_single_factor_impact.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"   저장 완료: {output_file}")
    print(f"   총 {len(results_df)}개 분석 결과")

    # 7. 요약 통계
    print("\n[6/6] 요약 통계")
    print("="*80)

    # Market별 요약
    for market in ['KOSPI200', 'KOSDAQ150']:
        market_df = results_df[results_df['Market'] == market]
        if len(market_df) > 0:
            print(f"\n{market}:")
            print(f"  총 분석 케이스: {len(market_df)}개")
            print(f"  평균 시그널 수: {market_df['Signal_Count'].mean():.0f}개")
            print(f"  평균 10일 수익률: {market_df['Avg_Return_10D'].mean():.2f}%")
            print(f"  평균 승률(10일): {market_df['Win_Rate_10D'].mean():.1f}%")

    # 투자자별 요약
    print("\n투자자별 Best Performance (10일 수익률 기준):")
    for investor in investors:
        investor_df = results_df[results_df['Investor'] == investor]
        if len(investor_df) > 0:
            best = investor_df.nlargest(1, 'Avg_Return_10D').iloc[0]
            print(f"\n  {investor}:")
            print(f"    Market: {best['Market']}, Duration: {best['Duration']}일, Direction: {best['Direction']}")
            print(f"    10일 수익률: {best['Avg_Return_10D']:.2f}% (승률: {best['Win_Rate_10D']:.1f}%)")
            print(f"    시그널 수: {best['Signal_Count']:.0f}개")

    print("\n" + "="*80)
    print("Phase 3-1 완료!")
    print(f"다음 단계: Golden Duration 확정 및 스코어링 시스템 설계")
    print("="*80)

if __name__ == "__main__":
    main()
