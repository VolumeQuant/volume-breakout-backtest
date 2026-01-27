"""
Stage 3 메인 실행 파일

세분화된 투자자별 수급 필터 백테스팅과 시각화를 순차적으로 실행합니다.
"""

from stage3_flow_data_collector import FlowDataCollector
from stage3_flow_filter import FlowFilterBacktest
from stage3_visualizer import Stage3Visualizer
import pandas as pd
import os


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("Stage 3: 세분화된 투자자별 수급 필터 최적화")
    print("=" * 80)
    print("\n기본 조건: VR >= 6.5 & ZS >= 3.0 & Price >= 10%")
    print("추가 조건: 투자자별 순매수 (금융투자, 연기금, 외국인 등)\n")

    # ========================================
    # 1단계: 수급 데이터 수집
    # ========================================
    print("[1단계] 수급 데이터 수집")
    print("-" * 80)

    flow_data_file = 'data/investor_flow_data.csv'

    # 수급 데이터가 없으면 수집
    if not os.path.exists(flow_data_file):
        print("수급 데이터가 없습니다. 데이터를 수집합니다...")
        print("(이 작업은 1-2시간 소요될 수 있습니다)\n")

        collector = FlowDataCollector(start_date='20220101', end_date='20241231')
        flow_data = collector.collect_flow_data()

        if flow_data is not None:
            collector.save_data(flow_data)
        else:
            print("\n[오류] 수급 데이터 수집 실패!")
            return
    else:
        print(f"수급 데이터 파일이 이미 존재합니다: {flow_data_file}")
        flow_data = pd.read_csv(flow_data_file, parse_dates=['Date'])
        print(f"수급 데이터 로드 완료: {len(flow_data):,}개 행\n")

    # ========================================
    # 2단계: 가격/거래량 데이터 로드
    # ========================================
    print("[2단계] 가격/거래량 데이터 로드")
    print("-" * 80)

    price_data = pd.read_csv('data/stock_data_with_indicators.csv', parse_dates=[0])
    if price_data.columns[0] != 'Date':
        price_data.columns = ['Date'] + list(price_data.columns[1:])

    print(f"가격 데이터 로드 완료: {len(price_data):,}개 행\n")

    # ========================================
    # 3단계: 그리드 서치
    # ========================================
    print("[3단계] 세분화 수급 필터 그리드 서치")
    print("-" * 80 + "\n")

    backtest = FlowFilterBacktest(
        price_data,
        flow_data,
        volume_ratio_threshold=6.5,
        z_score_threshold=3.0,
        price_threshold=10.0
    )

    # 데이터 병합
    merged_data = backtest.merge_data()

    # 그리드 서치 실행
    results = backtest.run_core_combinations(merged_data)

    # 결과 저장
    backtest.save_results(results)

    # ========================================
    # 4단계: 최고 조합 분석
    # ========================================
    print("\n" + "=" * 80)
    print("[4단계] 최고 성과 조합 분석")
    print("=" * 80)

    # 1일 수익률 기준
    best_1d = backtest.get_best_combination(results, metric='avg_return_1d', min_monthly_signals=5)
    if best_1d is not None:
        print(f"\n[1일 수익률 최고]")
        print(f"  필터: {best_1d['filter']}")
        print(f"  시그널 수: {int(best_1d['signal_count']):,}개 (월 평균 {best_1d['monthly_signals']:.1f}건)")
        print(f"  1일 평균: {best_1d['avg_return_1d']:.3f}% | 승률: {best_1d['win_rate_1d']:.1f}%")
        print(f"  10일 평균: {best_1d['avg_return_10d']:.3f}% | 승률: {best_1d['win_rate_10d']:.1f}%")

    # 10일 수익률 기준
    best_10d = backtest.get_best_combination(results, metric='avg_return_10d', min_monthly_signals=5)
    if best_10d is not None:
        print(f"\n[10일 수익률 최고]")
        print(f"  필터: {best_10d['filter']}")
        print(f"  시그널 수: {int(best_10d['signal_count']):,}개 (월 평균 {best_10d['monthly_signals']:.1f}건)")
        print(f"  1일 평균: {best_10d['avg_return_1d']:.3f}% | 승률: {best_10d['win_rate_1d']:.1f}%")
        print(f"  10일 평균: {best_10d['avg_return_10d']:.3f}% | 승률: {best_10d['win_rate_10d']:.1f}%")
        print(f"  샤프지수: {best_10d['sharpe_10d']:.3f}")
        print(f"  손익비: {best_10d['profit_factor']:.2f}")

    # 승률 기준
    best_winrate = backtest.get_best_combination(results, metric='win_rate_10d', min_monthly_signals=5)
    if best_winrate is not None:
        print(f"\n[10일 승률 최고]")
        print(f"  필터: {best_winrate['filter']}")
        print(f"  시그널 수: {int(best_winrate['signal_count']):,}개 (월 평균 {best_winrate['monthly_signals']:.1f}건)")
        print(f"  1일 평균: {best_winrate['avg_return_1d']:.3f}% | 승률: {best_winrate['win_rate_1d']:.1f}%")
        print(f"  10일 평균: {best_winrate['avg_return_10d']:.3f}% | 승률: {best_winrate['win_rate_10d']:.1f}%")

    # ========================================
    # 5단계: 핵심 인사이트 분석
    # ========================================
    print("\n" + "=" * 80)
    print("[5단계] 핵심 인사이트 분석")
    print("=" * 80)

    # 금융투자 vs 연기금 비교
    financial = results[results['filter'] == 'A1_Financial_Investment']
    pension = results[results['filter'] == 'A2_Pension']
    baseline = results[results['filter'] == 'A0_Baseline']

    if len(financial) > 0 and len(pension) > 0 and len(baseline) > 0:
        print("\n[금융투자 vs 연기금 예측력 비교]")
        print("\nBaseline (수급 필터 없음):")
        print(f"  1일: {baseline['avg_return_1d'].values[0]:.3f}% | 10일: {baseline['avg_return_10d'].values[0]:.3f}%")
        print(f"  시그널: {int(baseline['signal_count'].values[0]):,}개 (월 평균 {baseline['monthly_signals'].values[0]:.1f}건)")

        print("\n금융투자 순매수:")
        print(f"  1일: {financial['avg_return_1d'].values[0]:.3f}% | 10일: {financial['avg_return_10d'].values[0]:.3f}%")
        print(f"  시그널: {int(financial['signal_count'].values[0]):,}개 (월 평균 {financial['monthly_signals'].values[0]:.1f}건)")
        print(f"  개선: 1일 {(financial['avg_return_1d'].values[0] - baseline['avg_return_1d'].values[0]):.3f}%p | " \
              f"10일 {(financial['avg_return_10d'].values[0] - baseline['avg_return_10d'].values[0]):.3f}%p")

        print("\n연기금 순매수:")
        print(f"  1일: {pension['avg_return_1d'].values[0]:.3f}% | 10일: {pension['avg_return_10d'].values[0]:.3f}%")
        print(f"  시그널: {int(pension['signal_count'].values[0]):,}개 (월 평균 {pension['monthly_signals'].values[0]:.1f}건)")
        print(f"  개선: 1일 {(pension['avg_return_1d'].values[0] - baseline['avg_return_1d'].values[0]):.3f}%p | " \
              f"10일 {(pension['avg_return_10d'].values[0] - baseline['avg_return_10d'].values[0]):.3f}%p")

        # 가설 검증
        print("\n[가설 검증]")
        if financial['avg_return_1d'].values[0] > pension['avg_return_1d'].values[0]:
            print("  단기(1일): 금융투자가 연기금보다 우수 - 가설 지지!")
        else:
            print("  단기(1일): 연기금이 금융투자보다 우수 - 가설 불일치")

        if pension['avg_return_10d'].values[0] > financial['avg_return_10d'].values[0]:
            print("  장기(10일): 연기금이 금융투자보다 우수 - 가설 지지!")
        else:
            print("  장기(10일): 금융투자가 연기금보다 우수 - 가설 불일치")

    # ========================================
    # 6단계: 시각화
    # ========================================
    print("\n" + "=" * 80)
    print("[6단계] 결과 시각화")
    print("=" * 80)

    viz = Stage3Visualizer(results)
    viz.create_all_plots()

    # ========================================
    # 완료
    # ========================================
    print("\n" + "=" * 80)
    print("Stage 3 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  - data/investor_flow_data.csv")
    print("  - results/stage3_flow_filter_results.csv")
    print("  - results/stage3_investor_predictive_power.png")
    print("  - results/stage3_short_vs_long_term.png")
    print("  - results/stage3_multiple_buy_effect.png")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[오류 발생] {str(e)}")
        import traceback
        traceback.print_exc()
