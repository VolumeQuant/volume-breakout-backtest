"""
결과 시각화 모듈

그리드 서치 결과를 히트맵, 박스플롯, 막대 그래프 등으로 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


class Visualizer:
    """결과 시각화 클래스"""

    def __init__(self, results_df):
        """
        초기화 함수

        Parameters:
        -----------
        results_df : pd.DataFrame
            그리드 서치 결과 데이터프레임
        """
        self.results_df = results_df
        self.results_dir = 'results'

        # results 폴더가 없으면 생성
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # Seaborn 스타일 설정
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)

    def plot_heatmap(self, metric='avg_return_1d', title=None, filename=None):
        """
        히트맵을 그립니다.

        Parameters:
        -----------
        metric : str
            시각화할 지표 (예: 'avg_return_1d', 'win_rate_1d')
        title : str
            그래프 제목 (None이면 자동 생성)
        filename : str
            저장할 파일명 (None이면 자동 생성)
        """
        # 피벗 테이블 생성
        pivot = self.results_df.pivot(
            index='z_score_threshold',
            columns='volume_ratio_threshold',
            values=metric
        )

        # 그래프 생성
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': metric})

        # 제목 설정
        if title is None:
            metric_name = {
                'avg_return_1d': '익일 평균 수익률 (%)',
                'avg_return_3d': '3일 평균 수익률 (%)',
                'avg_return_5d': '5일 평균 수익률 (%)',
                'avg_return_10d': '10일 평균 수익률 (%)',
                'win_rate_1d': '익일 승률 (%)',
                'sharpe_1d': '샤프지수 (1일)',
            }.get(metric, metric)
            title = f'Volume_Ratio vs Z_Score 조합별 {metric_name}'

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Volume_Ratio 임계치', fontsize=12, fontweight='bold')
        plt.ylabel('Z_Score 임계치', fontsize=12, fontweight='bold')

        # 파일명 설정
        if filename is None:
            filename = f'heatmap_{metric}.png'

        # 저장
        filepath = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"히트맵 저장 완료: {filepath}")

    def plot_signal_frequency(self, filename='signal_frequency.png'):
        """
        시그널 발생 빈도를 막대 그래프로 그립니다.

        Parameters:
        -----------
        filename : str
            저장할 파일명
        """
        # 조합명 생성 (예: "VR≥2.0, ZS≥1.5")
        self.results_df['combination'] = (
            'VR≥' + self.results_df['volume_ratio_threshold'].astype(str) +
            ', ZS≥' + self.results_df['z_score_threshold'].astype(str)
        )

        # 그래프 생성
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(self.results_df)),
                       self.results_df['signal_count'],
                       color='steelblue', alpha=0.8)

        # 막대 위에 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom', fontsize=9)

        plt.xlabel('조합', fontsize=12, fontweight='bold')
        plt.ylabel('시그널 발생 횟수', fontsize=12, fontweight='bold')
        plt.title('조합별 시그널 발생 빈도', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(range(len(self.results_df)),
                   self.results_df['combination'],
                   rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # 저장
        filepath = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"시그널 빈도 그래프 저장 완료: {filepath}")

    def plot_returns_boxplot(self, period=1, filename=None):
        """
        수익률 분포를 박스플롯으로 그립니다.

        Parameters:
        -----------
        period : int
            보유 기간 (1, 3, 5, 10일)
        filename : str
            저장할 파일명
        """
        metric = f'avg_return_{period}d'

        # 조합명 생성
        self.results_df['combination'] = (
            'VR≥' + self.results_df['volume_ratio_threshold'].astype(str) +
            '\nZS≥' + self.results_df['z_score_threshold'].astype(str)
        )

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(14, 6))

        # 막대 그래프 (평균 수익률)
        bars = ax.bar(range(len(self.results_df)),
                      self.results_df[metric],
                      color=['green' if x > 0 else 'red' for x in self.results_df[metric]],
                      alpha=0.7)

        # 0선 추가
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 막대 위에 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

        ax.set_xlabel('조합', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{period}일 평균 수익률 (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'조합별 {period}일 평균 수익률 비교', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(self.results_df)))
        ax.set_xticklabels(self.results_df['combination'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # 파일명 설정
        if filename is None:
            filename = f'returns_comparison_{period}d.png'

        # 저장
        filepath = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"수익률 비교 그래프 저장 완료: {filepath}")

    def plot_win_rate_comparison(self, filename='win_rate_comparison.png'):
        """
        기간별 승률을 비교하는 그래프를 그립니다.

        Parameters:
        -----------
        filename : str
            저장할 파일명
        """
        # 조합명 생성
        self.results_df['combination'] = (
            'VR≥' + self.results_df['volume_ratio_threshold'].astype(str) +
            ', ZS≥' + self.results_df['z_score_threshold'].astype(str)
        )

        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        periods = [1, 3, 5, 10]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, period in enumerate(periods):
            ax = axes[idx]
            metric = f'win_rate_{period}d'

            bars = ax.bar(range(len(self.results_df)),
                          self.results_df[metric],
                          color=colors[idx], alpha=0.7)

            # 50% 기준선
            ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% 기준')

            # 막대 위에 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('조합', fontsize=10, fontweight='bold')
            ax.set_ylabel(f'{period}일 승률 (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{period}일 보유 승률', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(self.results_df)))
            ax.set_xticklabels(self.results_df['combination'], rotation=45, ha='right', fontsize=7)
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 100])

        plt.suptitle('조합별 기간별 승률 비교', fontsize=16, fontweight='bold', y=1.00)

        # 저장
        filepath = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"승률 비교 그래프 저장 완료: {filepath}")

    def create_all_plots(self):
        """
        모든 시각화를 생성합니다.
        """
        print("\n시각화를 생성합니다...")

        # 1. 히트맵 - 익일 평균 수익률
        self.plot_heatmap(metric='avg_return_1d',
                          title='Volume_Ratio vs Z_Score: 익일 평균 수익률 (%)',
                          filename='heatmap_return_1d.png')

        # 2. 히트맵 - 3일 평균 수익률
        self.plot_heatmap(metric='avg_return_3d',
                          title='Volume_Ratio vs Z_Score: 3일 평균 수익률 (%)',
                          filename='heatmap_return_3d.png')

        # 3. 시그널 발생 빈도
        self.plot_signal_frequency()

        # 4. 수익률 비교 (1일, 3일)
        self.plot_returns_boxplot(period=1)
        self.plot_returns_boxplot(period=3)

        # 5. 승률 비교
        self.plot_win_rate_comparison()

        print("\n모든 시각화 완료!")


def main():
    """테스트용 메인 함수"""
    # 샘플 결과 데이터 생성
    volume_ratios = [2.0, 3.0, 4.0]
    z_scores = [1.5, 2.0, 2.5]

    results = []
    for vr in volume_ratios:
        for zs in z_scores:
            results.append({
                'volume_ratio_threshold': vr,
                'z_score_threshold': zs,
                'signal_count': np.random.randint(50, 500),
                'avg_return_1d': np.random.normal(0.5, 1.0),
                'avg_return_3d': np.random.normal(1.0, 2.0),
                'avg_return_5d': np.random.normal(1.5, 3.0),
                'avg_return_10d': np.random.normal(2.0, 4.0),
                'win_rate_1d': np.random.uniform(45, 60),
                'win_rate_3d': np.random.uniform(45, 60),
                'win_rate_5d': np.random.uniform(45, 60),
                'win_rate_10d': np.random.uniform(45, 60),
                'sharpe_1d': np.random.uniform(0, 1),
            })

    results_df = pd.DataFrame(results)

    print("샘플 결과 데이터:")
    print(results_df)

    # 시각화 생성
    viz = Visualizer(results_df)
    viz.create_all_plots()


if __name__ == '__main__':
    main()
