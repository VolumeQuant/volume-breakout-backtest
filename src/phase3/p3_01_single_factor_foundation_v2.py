"""
Phase 3-1: Single-Factor Foundation Analysis (Simplified)
=========================================================

목표:
- 2021-2025 데이터 기반 수급 주체별 단일 영향력 검증
- Duration별 최적 기간 탐색
- 과적합 없는 견고한 기반 구축

간소화 접근:
- 기존 데이터 (2022-2024) 활용
- 1일 단위 수급 데이터로 시작
- Duration은 추후 확장
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 경로
project_root = Path(__file__).parent.parent.parent


def main():
    print("="*80)
    print("Phase 3-1: Single-Factor Foundation Analysis (Simplified)")
    print("="*80)
    print("\n목표: 수급 주체별 단일 영향력 검증")
    print("접근: 기존 데이터 활용, 단순화된 분석\n")

    # 1. 데이터 로드
    print("[1/5] 데이터 로드 중...")
    data_dir = project_root / 'data'

    # Phase 1에서 사용한 병합 데이터가 있는지 확인
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_list = pd.read_csv(data_dir / 'stock_list.csv')

    # 수급 데이터는 Phase 2 결과에서 추출
    # 대신 간단하게 Phase 1 방식 사용: 개인/외국인/금융투자/연기금 비중을 별도 계산

    print(f"   주가 데이터: {len(stock_data):,}행")

    # 2. KOSPI 200 / KOSDAQ 150 필터링
    print("\n[2/5] 지수 구성 종목 필터링...")

    # KOSPI 200 (시가총액 상위)
    kospi_stocks = stock_list[stock_list['Market'] == 'KOSPI'].nlargest(200, 'Marcap')
    kospi_codes = kospi_stocks['Code'].tolist()

    # KOSDAQ 150 (시가총액 상위)
    kosdaq_stocks = stock_list[stock_list['Market'] == 'KOSDAQ'].nlargest(150, 'Marcap')
    kosdaq_codes = kosdaq_stocks['Code'].tolist()

    print(f"   KOSPI 200: {len(kospi_codes)}개")
    print(f"   KOSDAQ 150: {len(kosdaq_codes)}개")

    # 필터링
    kospi_data = stock_data[stock_data['Code'].isin(kospi_codes)].copy()
    kosdaq_data = stock_data[stock_data['Code'].isin(kosdaq_codes)].copy()

    print(f"   KOSPI 데이터: {len(kospi_data):,}행")
    print(f"   KOSDAQ 데이터: {len(kosdaq_data):,}행")

    # 3. 간단한 분석: VR 임계치별 수익률 분포
    print("\n[3/5] VR 임계치별 분석...")

    vr_thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    results = []

    for market_name, market_data, market_codes in [
        ('KOSPI200', kospi_data, kospi_codes),
        ('KOSDAQ150', kosdaq_data, kosdaq_codes)
    ]:
        print(f"\n   {market_name} 분석:")

        for vr_threshold in vr_thresholds:
            signals = market_data[market_data['Volume_Ratio'] >= vr_threshold]

            if len(signals) >= 10:
                result = {
                    'Market': market_name,
                    'VR_Threshold': vr_threshold,
                    'Signal_Count': len(signals),
                    'Avg_Return_1D': signals['Return_1D'].mean(),
                    'Avg_Return_3D': signals['Return_3D'].mean(),
                    'Avg_Return_5D': signals['Return_5D'].mean(),
                    'Avg_Return_10D': signals['Return_10D'].mean(),
                    'Win_Rate_10D': (signals['Return_10D'] > 0).sum() / len(signals) * 100,
                }
                results.append(result)

                print(f"      VR >= {vr_threshold}: {len(signals)}건, 10일 수익률 {result['Avg_Return_10D']:.2f}%, 승률 {result['Win_Rate_10D']:.1f}%")

    # 4. 결과 저장
    print("\n[4/5] 결과 저장...")

    output_dir = project_root / 'results' / 'p3'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_file = output_dir / 'p3_01_vr_baseline.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"   저장 완료: {output_file}")

    # 5. 요약
    print("\n[5/5] 요약")
    print("="*80)
    print("\n이 분석은 Phase 3의 시작점입니다.")
    print("수급 데이터 통합 후, Duration별 세밀한 분석을 진행할 예정입니다.")
    print("\n다음 단계:")
    print("  1. 수급 데이터 전처리 및 통합")
    print("  2. Duration별 (1/3/5/10/20/30/50일) 누적 분석")
    print("  3. Golden Duration 확정")
    print("  4. 스코어링 시스템 설계")
    print("="*80)


if __name__ == "__main__":
    main()
