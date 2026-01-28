"""
2021-2025 데이터 재수집 스크립트
Phase 3 요구사항에 맞춰 5개년 데이터 수집
"""

import pandas as pd
from data_loader import DataLoader, FlowDataLoader
from indicators import VolumeIndicators
import sys

def main():
    print("="*80)
    print("2021-2025 데이터 재수집 시작")
    print("="*80)

    # 1. 주가 데이터 수집
    print("\n[1/3] 주가 데이터 수집 중...")
    print("기간: 2021-01-01 ~ 2025-12-31")

    loader = DataLoader(start_date='2021-01-01', end_date='2025-12-31')

    # 종목 리스트 가져오기 (현재 시점 KOSPI 200 + KOSDAQ 150)
    stock_list = loader.get_stock_list()
    stock_list.to_csv('data/stock_list.csv', index=False, encoding='utf-8-sig')
    print(f"종목 리스트 저장 완료: {len(stock_list)}개")

    # 전체 데이터 수집
    all_stock_data = loader.load_all_data(stock_list)

    if all_stock_data is None or len(all_stock_data) == 0:
        print("데이터 수집 실패!")
        sys.exit(1)

    print(f"\n수집 완료: {len(all_stock_data):,}행")

    # 2. 지표 계산
    print("\n[2/3] 지표 계산 중 (VR, Z-Score, Return)...")

    # 종목별로 지표 계산
    all_stock_data = all_stock_data.sort_values(['Code', 'Date'])

    # Volume_Ratio
    all_stock_data['Volume_Ratio'] = all_stock_data.groupby('Code')['Volume'].transform(
        lambda x: VolumeIndicators.calculate_volume_ratio(pd.DataFrame({'Volume': x}))
    )

    # Z_Score
    all_stock_data['Z_Score'] = all_stock_data.groupby('Code')['Volume'].transform(
        lambda x: VolumeIndicators.calculate_z_score(pd.DataFrame({'Volume': x}))
    )

    # Return 계산
    for period in [1, 3, 5, 10]:
        all_stock_data[f'Return_{period}D'] = all_stock_data.groupby('Code')['Close'].transform(
            lambda x: (x.shift(-period) / x - 1) * 100
        )

    # Return_20D 추가
    all_stock_data['Return_20D'] = all_stock_data.groupby('Code')['Close'].transform(
        lambda x: (x.shift(-20) / x - 1) * 100
    )

    # 저장
    all_stock_data.to_csv('data/stock_data_with_indicators.csv',
                          index=False, encoding='utf-8-sig')
    print(f"주가 데이터 저장 완료: data/stock_data_with_indicators.csv")

    # 3. 수급 데이터 수집
    print("\n[3/3] 수급 데이터 수집 중...")
    flow_loader = FlowDataLoader(start_date='20210101', end_date='20251231')

    all_flow_data = flow_loader.collect_flow_data('data/stock_list.csv')

    if all_flow_data is None or len(all_flow_data) == 0:
        print("수급 데이터 수집 실패!")
        sys.exit(1)

    # 저장
    all_flow_data.to_csv('data/investor_flow_data.csv', index=False, encoding='utf-8-sig')
    print(f"수급 데이터 저장 완료: {len(all_flow_data):,}행")

    print("\n" + "="*80)
    print("데이터 수집 완료!")
    print("="*80)
    print(f"주가 데이터: {len(stock_data_with_indicators):,}행")
    print(f"수급 데이터: {len(all_flow_data):,}행")
    print(f"종목 수: {len(stock_list)}개")
    print(f"기간: 2021-01-01 ~ 2025-12-31")

if __name__ == "__main__":
    main()
