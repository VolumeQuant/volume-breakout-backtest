"""
Stage 2: 가격 상승률 필터 최적화

거래량 조건(VR ≥ 6.5 & ZS ≥ 3.0)에 가격 상승률 필터를 추가하여
최적의 진입 조건을 찾습니다.
"""

import pandas as pd
import numpy as np
import os


class PriceFilterBacktest:
    """가격 필터 백테스팅 클래스"""

    def __init__(self, df, volume_ratio_threshold=6.5, z_score_threshold=3.0):
        """
        초기화 함수

        Parameters:
        -----------
        df : pd.DataFrame
            지표가 계산된 전체 데이터프레임
        volume_ratio_threshold : float
            Volume_Ratio 임계치 (기본값: 6.5)
        z_score_threshold : float
            Z_Score 임계치 (기본값: 3.0)
        """
        self.df = df
        self.vr_threshold = volume_ratio_threshold
        self.zs_threshold = z_score_threshold
        self.results_dir = 'results'

        # 당일 수익률 계산 (종가 - 시가) / 시가 * 100
        if 'Return_0D' not in df.columns:
            self.df['Return_0D'] = ((df['Close'] - df['Open']) / df['Open']) * 100

        # results 폴더가 없으면 생성
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def filter_by_price_return(self, price_threshold):
        """
        가격 상승률 임계치로 필터링

        Parameters:
        -----------
        price_threshold : float
            당일 수익률 임계치 (%)

        Returns:
        --------
        pd.DataFrame
            필터링된 시그널
        """
        # 기본 거래량 조건
        base_signals = self.df[
            (self.df['Volume_Ratio'] >= self.vr_threshold) &
            (self.df['Z_Score'] >= self.zs_threshold)
        ].copy()

        # 가격 조건 추가
        if price_threshold > 0:
            # 상한가(+30%) 제외
            filtered_signals = base_signals[
                (base_signals['Return_0D'] >= price_threshold) &
                (base_signals['Return_0D'] < 30.0)
            ].copy()
        else:
            # 0%는 베이스라인 (거래량만, 상한가만 제외)
            filtered_signals = base_signals[base_signals['Return_0D'] < 30.0].copy()

        return filtered_signals

    def calculate_statistics(self, signals, price_threshold):
        """
        시그널에 대한 통계를 계산합니다.

        Parameters:
        -----------
        signals : pd.DataFrame
            시그널 데이터
        price_threshold : float
            가격 임계치

        Returns:
        --------
        dict
            통계 결과
        """
        if len(signals) == 0:
            return {
                'price_threshold': price_threshold,
                'signal_count': 0,
                'monthly_signals': 0.0,
                'reduction_pct': 100.0,
            }

        # 기본 통계
        stats = {
            'price_threshold': price_threshold,
            'signal_count': len(signals),
            'monthly_signals': len(signals) / 36.0,  # 3년 = 36개월
        }

        # 베이스라인 대비 감소율 (0% 기준)
        if price_threshold == 0:
            stats['reduction_pct'] = 0.0
        else:
            baseline_count = len(self.filter_by_price_return(0))
            if baseline_count > 0:
                stats['reduction_pct'] = (1 - len(signals) / baseline_count) * 100
            else:
                stats['reduction_pct'] = 100.0

        # 각 보유기간별 통계
        for period in [1, 3, 5, 10]:
            return_col = f'Return_{period}D'

            # NaN 제거
            returns = signals[return_col].dropna()

            if len(returns) > 0:
                # 평균 수익률
                stats[f'avg_return_{period}d'] = returns.mean()

                # 중위값
                stats[f'median_return_{period}d'] = returns.median()

                # 승률
                stats[f'win_rate_{period}d'] = (returns > 0).sum() / len(returns) * 100

                # 표준편차
                stats[f'std_{period}d'] = returns.std()

                # 샤프지수
                if returns.std() > 0:
                    stats[f'sharpe_{period}d'] = returns.mean() / returns.std()
                else:
                    stats[f'sharpe_{period}d'] = 0.0

                # 최대 손실 (1일 기준)
                if period == 1:
                    stats['max_loss'] = returns.min()
                    stats['max_gain'] = returns.max()

                    # 손익비 계산
                    winning_trades = returns[returns > 0]
                    losing_trades = returns[returns < 0]

                    if len(winning_trades) > 0 and len(losing_trades) > 0:
                        avg_win = winning_trades.mean()
                        avg_loss = abs(losing_trades.mean())
                        stats['profit_factor'] = avg_win / avg_loss if avg_loss > 0 else 0
                    else:
                        stats['profit_factor'] = 0.0
            else:
                stats[f'avg_return_{period}d'] = np.nan
                stats[f'median_return_{period}d'] = np.nan
                stats[f'win_rate_{period}d'] = np.nan
                stats[f'std_{period}d'] = np.nan
                stats[f'sharpe_{period}d'] = np.nan

                if period == 1:
                    stats['max_loss'] = np.nan
                    stats['max_gain'] = np.nan
                    stats['profit_factor'] = np.nan

        return stats

    def run_grid_search(self, price_thresholds=[0, 1, 2, 3, 5, 7, 10, 15]):
        """
        가격 임계치 그리드 서치 실행

        Parameters:
        -----------
        price_thresholds : list
            테스트할 가격 임계치 리스트 (%)

        Returns:
        --------
        pd.DataFrame
            모든 조합에 대한 통계 결과
        """
        print("=" * 80)
        print("Stage 2: 가격 상승률 필터 그리드 서치")
        print("=" * 80)
        print(f"\n기본 조건: VR ≥ {self.vr_threshold}, ZS ≥ {self.zs_threshold}")
        print(f"테스트할 가격 임계치: {price_thresholds}")
        print(f"총 조합 수: {len(price_thresholds)}개\n")

        results = []

        for idx, price_threshold in enumerate(price_thresholds):
            print(f"[{idx+1}/{len(price_thresholds)}] 가격 상승률 ≥ {price_threshold}%")

            # 시그널 생성
            signals = self.filter_by_price_return(price_threshold)

            # 통계 계산
            stats = self.calculate_statistics(signals, price_threshold)

            results.append(stats)

            print(f"  시그널 수: {stats['signal_count']:,}개 (월 평균 {stats['monthly_signals']:.1f}건)")
            if stats['signal_count'] > 0:
                print(f"  베이스라인 대비: -{stats['reduction_pct']:.1f}%")
                print(f"  익일 평균: {stats['avg_return_1d']:.3f}% | 승률: {stats['win_rate_1d']:.1f}%")
                print(f"  10일 평균: {stats['avg_return_10d']:.3f}% | 승률: {stats['win_rate_10d']:.1f}%")
                print(f"  손익비: {stats['profit_factor']:.2f}")
            print()

        # 데이터프레임으로 변환
        results_df = pd.DataFrame(results)

        print("그리드 서치 완료!")
        return results_df

    def save_results(self, results_df, filename='stage2_price_filter_results.csv'):
        """
        결과를 CSV 파일로 저장

        Parameters:
        -----------
        results_df : pd.DataFrame
            결과 데이터프레임
        filename : str
            파일명
        """
        filepath = os.path.join(self.results_dir, filename)
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장 완료: {filepath}")

    def get_best_combination(self, results_df, metric='avg_return_10d', min_monthly_signals=10):
        """
        최고의 조합을 찾습니다.

        Parameters:
        -----------
        results_df : pd.DataFrame
            그리드 서치 결과
        metric : str
            평가 지표 (기본값: 10일 평균 수익률)
        min_monthly_signals : int
            최소 월 시그널 수

        Returns:
        --------
        pd.Series
            최고 성과 조합
        """
        # 최소 시그널 조건 만족하는 것만 필터링
        valid_results = results_df[results_df['monthly_signals'] >= min_monthly_signals].copy()

        if len(valid_results) == 0:
            print(f"[경고] 월 {min_monthly_signals}건 이상인 조합이 없습니다. 전체에서 선택합니다.")
            valid_results = results_df

        # 해당 metric이 가장 높은 조합 찾기
        best_idx = valid_results[metric].idxmax()
        best_combination = valid_results.loc[best_idx]

        return best_combination


def main():
    """메인 실행 함수"""
    print("데이터를 로드합니다...\n")

    # 데이터 로드
    df = pd.read_csv('data/stock_data_with_indicators.csv', parse_dates=[0])
    if df.columns[0] != 'Date':
        df.columns = ['Date'] + list(df.columns[1:])

    print(f"전체 데이터: {len(df):,}개 행\n")

    # 백테스트 실행
    backtest = PriceFilterBacktest(
        df,
        volume_ratio_threshold=6.5,
        z_score_threshold=3.0
    )

    # 그리드 서치
    results = backtest.run_grid_search(
        price_thresholds=[0, 1, 2, 3, 5, 7, 10, 15]
    )

    # 결과 저장
    backtest.save_results(results)

    # 최고 조합 출력
    print("\n" + "=" * 80)
    print("최고 성과 조합 분석")
    print("=" * 80)

    # 10일 수익률 기준
    best_10d = backtest.get_best_combination(results, metric='avg_return_10d', min_monthly_signals=10)
    print(f"\n[10일 수익률 최고]")
    print(f"  가격 임계치: ≥ {best_10d['price_threshold']:.0f}%")
    print(f"  시그널 수: {int(best_10d['signal_count']):,}개 (월 평균 {best_10d['monthly_signals']:.1f}건)")
    print(f"  익일 평균: {best_10d['avg_return_1d']:.3f}% | 승률: {best_10d['win_rate_1d']:.1f}%")
    print(f"  10일 평균: {best_10d['avg_return_10d']:.3f}% | 승률: {best_10d['win_rate_10d']:.1f}%")
    print(f"  샤프지수: {best_10d['sharpe_10d']:.3f}")
    print(f"  손익비: {best_10d['profit_factor']:.2f}")

    # 승률 기준
    best_winrate = backtest.get_best_combination(results, metric='win_rate_10d', min_monthly_signals=10)
    print(f"\n[10일 승률 최고]")
    print(f"  가격 임계치: ≥ {best_winrate['price_threshold']:.0f}%")
    print(f"  시그널 수: {int(best_winrate['signal_count']):,}개 (월 평균 {best_winrate['monthly_signals']:.1f}건)")
    print(f"  익일 평균: {best_winrate['avg_return_1d']:.3f}% | 승률: {best_winrate['win_rate_1d']:.1f}%")
    print(f"  10일 평균: {best_winrate['avg_return_10d']:.3f}% | 승률: {best_winrate['win_rate_10d']:.1f}%")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
