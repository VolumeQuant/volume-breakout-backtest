"""
메인 실행 파일

한국 주식시장 거래량 폭발 임계치 백테스팅 프로젝트를 실행합니다.

실행 순서:
1. 데이터 수집 (또는 저장된 데이터 로드)
2. 지표 계산 (Volume_Ratio, Z_Score, 미래 수익률)
3. 그리드 서치 실행
4. 결과 시각화
"""

import os
import argparse
from data_loader import DataLoader
from indicators import VolumeIndicators
from backtest import GridSearchBacktest
from visualizer import Visualizer


def main(force_download=False):
    """
    메인 실행 함수

    Parameters:
    -----------
    force_download : bool
        True이면 기존 데이터를 무시하고 새로 다운로드
    """
    print("=" * 80)
    print("한국 주식시장 거래량 폭발 임계치 백테스팅 프로젝트")
    print("=" * 80)
    print()

    # ========================================
    # 1단계: 데이터 수집 또는 로드
    # ========================================
    print("[1단계] 데이터 수집")
    print("-" * 80)

    loader = DataLoader(start_date='2022-01-01', end_date='2024-12-31')

    # 저장된 데이터가 있고 force_download가 False이면 로드
    data_file = os.path.join(loader.data_dir, 'stock_data.csv')

    if os.path.exists(data_file) and not force_download:
        print(f"저장된 데이터를 발견했습니다: {data_file}")
        print("기존 데이터를 사용합니다. (새로 다운로드하려면 --force-download 옵션 사용)\n")

        # 데이터 로드
        stock_data = loader.load_data('stock_data.csv')

        if stock_data is None:
            print("[오류] 데이터 로드 실패!")
            return

    else:
        print("데이터를 새로 다운로드합니다...\n")

        # 종목 리스트 가져오기
        stock_list = loader.get_stock_list()
        loader.save_data(stock_list, 'stock_list.csv')

        # 전체 데이터 수집
        stock_data = loader.load_all_data(stock_list)

        if stock_data is None:
            print("[오류] 데이터 수집 실패!")
            return

        # 데이터 저장
        loader.save_data(stock_data, 'stock_data.csv')

    print()

    # ========================================
    # 2단계: 지표 계산
    # ========================================
    print("[2단계] 지표 계산")
    print("-" * 80)

    # 지표가 이미 계산되어 있는지 확인
    indicators_file = os.path.join(loader.data_dir, 'stock_data_with_indicators.csv')

    if os.path.exists(indicators_file) and not force_download:
        print(f"저장된 지표 데이터를 발견했습니다: {indicators_file}")
        print("기존 지표 데이터를 사용합니다.\n")

        stock_data_with_indicators = loader.load_data('stock_data_with_indicators.csv')

        if stock_data_with_indicators is None:
            print("[오류] 지표 데이터 로드 실패!")
            return

    else:
        print("지표를 계산합니다...\n")

        # 지표 계산
        stock_data_with_indicators = VolumeIndicators.add_all_indicators(
            stock_data,
            volume_ratio_window=20,
            z_score_window=60
        )

        if stock_data_with_indicators is None:
            print("[오류] 지표 계산 실패!")
            return

        # 지표가 추가된 데이터 저장
        loader.save_data(stock_data_with_indicators, 'stock_data_with_indicators.csv')

    print()

    # ========================================
    # 3단계: 그리드 서치 백테스팅
    # ========================================
    print("[3단계] 그리드 서치 백테스팅")
    print("-" * 80)

    # 그리드 서치 실행
    backtest = GridSearchBacktest(
        stock_data_with_indicators,
        volume_ratio_values=[2.0, 3.0, 4.0, 5.0],
        z_score_values=[1.5, 2.0, 2.5, 3.0]
    )

    results = backtest.run_grid_search()

    # 결과 저장
    backtest.save_results(results, 'grid_search_results.csv')

    # 최고 조합 찾기 및 출력
    print("\n" + "=" * 80)
    print("최고 성과 조합 (익일 평균 수익률 기준)")
    print("=" * 80)

    best_1d = backtest.find_best_combination(results, metric='avg_return_1d')
    print(f"\n[익일 수익률 최고]")
    print(f"  Volume_Ratio >= {best_1d['volume_ratio_threshold']}")
    print(f"  Z_Score >= {best_1d['z_score_threshold']}")
    print(f"  시그널 수: {int(best_1d['signal_count']):,}개")
    print(f"  평균 수익률: {best_1d['avg_return_1d']:.2f}%")
    print(f"  승률: {best_1d['win_rate_1d']:.1f}%")
    print(f"  샤프지수: {best_1d['sharpe_1d']:.3f}")

    # 3일 수익률 기준
    best_3d = backtest.find_best_combination(results, metric='avg_return_3d')
    print(f"\n[3일 수익률 최고]")
    print(f"  Volume_Ratio >= {best_3d['volume_ratio_threshold']}")
    print(f"  Z_Score >= {best_3d['z_score_threshold']}")
    print(f"  시그널 수: {int(best_3d['signal_count']):,}개")
    print(f"  평균 수익률: {best_3d['avg_return_3d']:.2f}%")
    print(f"  승률: {best_3d['win_rate_3d']:.1f}%")

    print()

    # ========================================
    # 4단계: 결과 시각화
    # ========================================
    print("[4단계] 결과 시각화")
    print("-" * 80)

    visualizer = Visualizer(results)
    visualizer.create_all_plots()

    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 80)
    print("모든 작업이 완료되었습니다!")
    print("=" * 80)
    print("\n결과 파일:")
    print(f"  - CSV: results/grid_search_results.csv")
    print(f"  - 히트맵: results/heatmap_return_1d.png, heatmap_return_3d.png")
    print(f"  - 시그널 빈도: results/signal_frequency.png")
    print(f"  - 수익률 비교: results/returns_comparison_1d.png, returns_comparison_3d.png")
    print(f"  - 승률 비교: results/win_rate_comparison.png")
    print()


if __name__ == '__main__':
    # 명령줄 인수 파싱
    parser = argparse.ArgumentParser(
        description='한국 주식시장 거래량 폭발 임계치 백테스팅'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='기존 데이터를 무시하고 새로 다운로드'
    )

    args = parser.parse_args()

    # 메인 함수 실행
    try:
        main(force_download=args.force_download)
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n[오류 발생] {str(e)}")
        import traceback
        traceback.print_exc()
