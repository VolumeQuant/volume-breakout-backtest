"""
Stage 2 메인 실행 파일

가격 필터 백테스팅과 시각화를 순차적으로 실행합니다.
"""

from stage2_price_filter import PriceFilterBacktest
from stage2_visualizer import Stage2Visualizer
import pandas as pd


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("Stage 2: 가격 상승률 필터 최적화")
    print("=" * 80)
    print("\n기본 조건: VR ≥ 6.5 & ZS ≥ 3.0")
    print("추가 조건: 당일 가격 상승률 ≥ X%\n")

    # ========================================
    # 1단계: 데이터 로드
    # ========================================
    print("[1단계] 데이터 로드")
    print("-" * 80)

    df = pd.read_csv('data/stock_data_with_indicators.csv', parse_dates=[0])
    if df.columns[0] != 'Date':
        df.columns = ['Date'] + list(df.columns[1:])

    print(f"전체 데이터: {len(df):,}개 행\n")

    # ========================================
    # 2단계: 그리드 서치
    # ========================================
    print("[2단계] 가격 임계치 그리드 서치")
    print("-" * 80 + "\n")

    backtest = PriceFilterBacktest(
        df,
        volume_ratio_threshold=6.5,
        z_score_threshold=3.0
    )

    # 그리드 서치 실행
    results = backtest.run_grid_search(
        price_thresholds=[0, 1, 2, 3, 5, 7, 10, 15]
    )

    # 결과 저장
    backtest.save_results(results)

    # ========================================
    # 3단계: 최고 조합 분석
    # ========================================
    print("\n" + "=" * 80)
    print("[3단계] 최고 성과 조합 분석")
    print("=" * 80)

    # 10일 수익률 기준
    best_10d = backtest.get_best_combination(results, metric='avg_return_10d', min_monthly_signals=10)
    print(f"\n[10일 수익률 최고]")
    print(f"  가격 임계치: >= {best_10d['price_threshold']:.0f}%")
    print(f"  시그널 수: {int(best_10d['signal_count']):,}개 (월 평균 {best_10d['monthly_signals']:.1f}건)")
    print(f"  베이스라인 대비: -{best_10d['reduction_pct']:.1f}%")
    print(f"  익일 평균: {best_10d['avg_return_1d']:.3f}% | 승률: {best_10d['win_rate_1d']:.1f}%")
    print(f"  10일 평균: {best_10d['avg_return_10d']:.3f}% | 승률: {best_10d['win_rate_10d']:.1f}%")
    print(f"  샤프지수: {best_10d['sharpe_10d']:.3f}")
    print(f"  손익비: {best_10d['profit_factor']:.2f}")

    # 승률 기준
    best_winrate = backtest.get_best_combination(results, metric='win_rate_10d', min_monthly_signals=10)
    print(f"\n[10일 승률 최고]")
    print(f"  가격 임계치: >= {best_winrate['price_threshold']:.0f}%")
    print(f"  시그널 수: {int(best_winrate['signal_count']):,}개 (월 평균 {best_winrate['monthly_signals']:.1f}건)")
    print(f"  익일 평균: {best_winrate['avg_return_1d']:.3f}% | 승률: {best_winrate['win_rate_1d']:.1f}%")
    print(f"  10일 평균: {best_winrate['avg_return_10d']:.3f}% | 승률: {best_winrate['win_rate_10d']:.1f}%")

    # 손익비 기준
    best_pf = backtest.get_best_combination(results, metric='profit_factor', min_monthly_signals=10)
    print(f"\n[손익비 최고]")
    print(f"  가격 임계치: >= {best_pf['price_threshold']:.0f}%")
    print(f"  시그널 수: {int(best_pf['signal_count']):,}개 (월 평균 {best_pf['monthly_signals']:.1f}건)")
    print(f"  익일 평균: {best_pf['avg_return_1d']:.3f}% | 승률: {best_pf['win_rate_1d']:.1f}%")
    print(f"  10일 평균: {best_pf['avg_return_10d']:.3f}% | 승률: {best_pf['win_rate_10d']:.1f}%")
    print(f"  손익비: {best_pf['profit_factor']:.2f}")

    # ========================================
    # 4단계: 시각화
    # ========================================
    print("\n" + "=" * 80)
    print("[4단계] 결과 시각화")
    print("=" * 80)

    viz = Stage2Visualizer(results)
    viz.create_all_plots()

    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 80)
    print("Stage 2 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  - results/stage2_price_filter_results.csv")
    print("  - results/stage2_price_vs_return.png")
    print("  - results/stage2_signal_vs_return.png")
    print("  - results/stage2_winrate_heatmap.png")
    print("  - results/stage2_profit_factor.png")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[오류 발생] {str(e)}")
        import traceback
        traceback.print_exc()
