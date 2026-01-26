"""
백테스팅 및 그리드 서치 모듈

Volume_Ratio와 Z_Score의 다양한 조합을 테스트하여 최적의 임계치를 찾습니다.
"""

import pandas as pd
import numpy as np
from itertools import product
import os


class GridSearchBacktest:
    """그리드 서치 백테스팅 클래스"""

    def __init__(self, df, volume_ratio_values=[2.0, 3.0, 4.0, 5.0],
                 z_score_values=[1.5, 2.0, 2.5, 3.0]):
        """
        초기화 함수

        Parameters:
        -----------
        df : pd.DataFrame
            지표가 계산된 전체 데이터프레임
        volume_ratio_values : list
            테스트할 Volume_Ratio 임계치 리스트
        z_score_values : list
            테스트할 Z_Score 임계치 리스트
        """
        self.df = df
        self.volume_ratio_values = volume_ratio_values
        self.z_score_values = z_score_values
        self.results_dir = 'results'

        # results 폴더가 없으면 생성
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def generate_signals(self, volume_ratio_threshold, z_score_threshold):
        """
        특정 임계치 조합에 대한 시그널을 생성합니다.

        Parameters:
        -----------
        volume_ratio_threshold : float
            Volume_Ratio 임계치
        z_score_threshold : float
            Z_Score 임계치

        Returns:
        --------
        pd.DataFrame
            시그널이 발생한 데이터만 필터링된 데이터프레임
        """
        # 두 조건을 모두 만족하는 경우 시그널 발생
        signals = self.df[
            (self.df['Volume_Ratio'] >= volume_ratio_threshold) &
            (self.df['Z_Score'] >= z_score_threshold)
        ].copy()

        return signals

    def calculate_statistics(self, signals):
        """
        시그널에 대한 통계를 계산합니다.

        Parameters:
        -----------
        signals : pd.DataFrame
            시그널이 발생한 데이터

        Returns:
        --------
        dict
            통계 결과 딕셔너리
        """
        if len(signals) == 0:
            # 시그널이 없는 경우 빈 통계 반환
            return {
                'signal_count': 0,
                'avg_return_1d': np.nan,
                'avg_return_3d': np.nan,
                'avg_return_5d': np.nan,
                'avg_return_10d': np.nan,
                'win_rate_1d': np.nan,
                'win_rate_3d': np.nan,
                'win_rate_5d': np.nan,
                'win_rate_10d': np.nan,
                'median_return_1d': np.nan,
                'median_return_3d': np.nan,
                'median_return_5d': np.nan,
                'median_return_10d': np.nan,
                'sharpe_1d': np.nan,
                'sharpe_3d': np.nan,
                'sharpe_5d': np.nan,
                'sharpe_10d': np.nan,
                'max_return_1d': np.nan,
                'max_loss_1d': np.nan,
            }

        # 기본 통계
        stats = {
            'signal_count': len(signals),
        }

        # 각 기간별 통계 계산
        for period in [1, 3, 5, 10]:
            return_col = f'Return_{period}D'

            # NaN 제거
            returns = signals[return_col].dropna()

            if len(returns) > 0:
                # 평균 수익률
                stats[f'avg_return_{period}d'] = returns.mean()

                # 승률 (수익률 > 0인 비율)
                stats[f'win_rate_{period}d'] = (returns > 0).sum() / len(returns) * 100

                # 중위값
                stats[f'median_return_{period}d'] = returns.median()

                # 샤프지수 (평균 / 표준편차) - 일간 수익률 기준
                if returns.std() > 0:
                    stats[f'sharpe_{period}d'] = returns.mean() / returns.std()
                else:
                    stats[f'sharpe_{period}d'] = np.nan

                # 최대 수익 / 최대 손실 (1일 기준)
                if period == 1:
                    stats[f'max_return_1d'] = returns.max()
                    stats[f'max_loss_1d'] = returns.min()
            else:
                stats[f'avg_return_{period}d'] = np.nan
                stats[f'win_rate_{period}d'] = np.nan
                stats[f'median_return_{period}d'] = np.nan
                stats[f'sharpe_{period}d'] = np.nan

                if period == 1:
                    stats[f'max_return_1d'] = np.nan
                    stats[f'max_loss_1d'] = np.nan

        return stats

    def run_grid_search(self):
        """
        그리드 서치를 실행합니다.

        Returns:
        --------
        pd.DataFrame
            모든 조합에 대한 통계 결과
        """
        print("그리드 서치를 시작합니다...")
        print(f"- Volume_Ratio 테스트 값: {self.volume_ratio_values}")
        print(f"- Z_Score 테스트 값: {self.z_score_values}")
        print(f"- 전체 조합 수: {len(self.volume_ratio_values) * len(self.z_score_values)}개\n")

        results = []

        # 모든 조합 생성
        combinations = list(product(self.volume_ratio_values, self.z_score_values))

        for idx, (vr_threshold, zs_threshold) in enumerate(combinations):
            print(f"[{idx+1}/{len(combinations)}] Volume_Ratio >= {vr_threshold}, Z_Score >= {zs_threshold}")

            # 시그널 생성
            signals = self.generate_signals(vr_threshold, zs_threshold)

            # 통계 계산
            stats = self.calculate_statistics(signals)

            # 임계치 정보 추가
            stats['volume_ratio_threshold'] = vr_threshold
            stats['z_score_threshold'] = zs_threshold

            results.append(stats)

            print(f"  시그널 수: {stats['signal_count']}")
            if stats['signal_count'] > 0:
                print(f"  익일 평균 수익률: {stats['avg_return_1d']:.2f}%")
                print(f"  익일 승률: {stats['win_rate_1d']:.1f}%")
            print()

        # 데이터프레임으로 변환
        results_df = pd.DataFrame(results)

        # 컬럼 순서 정리
        first_cols = ['volume_ratio_threshold', 'z_score_threshold', 'signal_count']
        other_cols = [col for col in results_df.columns if col not in first_cols]
        results_df = results_df[first_cols + other_cols]

        print("그리드 서치 완료!")
        return results_df

    def save_results(self, results_df, filename='grid_search_results.csv'):
        """
        결과를 CSV 파일로 저장합니다.

        Parameters:
        -----------
        results_df : pd.DataFrame
            그리드 서치 결과
        filename : str
            파일명
        """
        filepath = os.path.join(self.results_dir, filename)
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장 완료: {filepath}")

    def get_detailed_signals(self, volume_ratio_threshold, z_score_threshold):
        """
        특정 조합에서 발생한 모든 시그널의 상세 데이터를 반환합니다.

        Parameters:
        -----------
        volume_ratio_threshold : float
            Volume_Ratio 임계치
        z_score_threshold : float
            Z_Score 임계치

        Returns:
        --------
        pd.DataFrame
            시그널 발생 시점의 상세 데이터
        """
        signals = self.generate_signals(volume_ratio_threshold, z_score_threshold)

        # 필요한 컬럼만 선택
        columns = ['Code', 'Name', 'Market', 'Close', 'Volume',
                   'Volume_Ratio', 'Z_Score',
                   'Return_1D', 'Return_3D', 'Return_5D', 'Return_10D']

        # 컬럼이 존재하는지 확인
        available_columns = [col for col in columns if col in signals.columns]

        return signals[available_columns]

    def find_best_combination(self, results_df, metric='avg_return_1d'):
        """
        최고의 조합을 찾습니다.

        Parameters:
        -----------
        results_df : pd.DataFrame
            그리드 서치 결과
        metric : str
            평가 지표 (기본값: 익일 평균 수익률)

        Returns:
        --------
        pd.Series
            최고 성과 조합
        """
        # 시그널이 충분히 발생한 조합만 필터링 (최소 100개)
        valid_results = results_df[results_df['signal_count'] >= 100].copy()

        if len(valid_results) == 0:
            print("[경고] 시그널이 100개 이상인 조합이 없습니다. 전체에서 최고를 선택합니다.")
            valid_results = results_df

        # 해당 metric이 가장 높은 조합 찾기
        best_idx = valid_results[metric].idxmax()
        best_combination = valid_results.loc[best_idx]

        return best_combination


def main():
    """테스트용 메인 함수"""
    # 샘플 데이터 생성 (테스트용)
    print("샘플 데이터를 생성합니다...")

    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.uniform(10000, 50000, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
        'Volume_Ratio': np.random.uniform(0.5, 6.0, len(dates)),
        'Z_Score': np.random.normal(0, 1.5, len(dates)),
        'Return_1D': np.random.normal(0, 2, len(dates)),
        'Return_3D': np.random.normal(0, 3, len(dates)),
        'Return_5D': np.random.normal(0, 4, len(dates)),
        'Return_10D': np.random.normal(0, 5, len(dates)),
        'Code': '005930',
        'Name': '삼성전자',
        'Market': 'KOSPI'
    })
    sample_data.index = dates

    print(f"샘플 데이터: {len(sample_data)}개 행\n")

    # 그리드 서치 실행
    backtest = GridSearchBacktest(
        sample_data,
        volume_ratio_values=[2.0, 3.0, 4.0],
        z_score_values=[1.5, 2.0, 2.5]
    )

    results = backtest.run_grid_search()

    # 결과 출력
    print("\n그리드 서치 결과:")
    print(results[['volume_ratio_threshold', 'z_score_threshold', 'signal_count',
                   'avg_return_1d', 'win_rate_1d', 'sharpe_1d']])

    # 최고 조합 찾기
    print("\n최고 조합 (익일 평균 수익률 기준):")
    best = backtest.find_best_combination(results, metric='avg_return_1d')
    print(best)


if __name__ == '__main__':
    main()
