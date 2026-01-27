"""
Stage 2 결과 시각화 모듈

가격 필터 백테스팅 결과를 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class Stage2Visualizer:
    """Stage 2 결과 시각화 클래스"""

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
        sns.set_context("notebook", font_scale=1.1)

    def plot_price_vs_return(self, filename='stage2_price_vs_return.png'):
        """
        가격 임계치별 성과 비교 (라인 차트)

        X축: 가격 임계치
        Y축 (이중): 평균 수익률 (좌) / 승률 (우)
        4개 라인: 익일, 3일, 5일, 10일
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 상단: 평균 수익률
        periods = [1, 3, 5, 10]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for idx, period in enumerate(periods):
            metric = f'avg_return_{period}d'
            ax1.plot(self.results_df['price_threshold'],
                    self.results_df[metric],
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=2,
                    markersize=8,
                    label=f'{period}일')

            # 값 표시
            for x, y in zip(self.results_df['price_threshold'], self.results_df[metric]):
                if pd.notna(y):
                    ax1.annotate(f'{y:.2f}%',
                               xy=(x, y),
                               xytext=(0, 5),
                               textcoords='offset points',
                               ha='center',
                               fontsize=8,
                               color=colors[idx])

        ax1.set_xlabel('가격 상승률 임계치 (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('평균 수익률 (%)', fontsize=12, fontweight='bold')
        ax1.set_title('가격 임계치별 보유기간별 평균 수익률', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 하단: 승률
        for idx, period in enumerate(periods):
            metric = f'win_rate_{period}d'
            ax2.plot(self.results_df['price_threshold'],
                    self.results_df[metric],
                    marker=markers[idx],
                    color=colors[idx],
                    linewidth=2,
                    markersize=8,
                    label=f'{period}일')

            # 값 표시
            for x, y in zip(self.results_df['price_threshold'], self.results_df[metric]):
                if pd.notna(y):
                    ax2.annotate(f'{y:.1f}%',
                               xy=(x, y),
                               xytext=(0, 5),
                               textcoords='offset points',
                               ha='center',
                               fontsize=8,
                               color=colors[idx])

        ax2.set_xlabel('가격 상승률 임계치 (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('승률 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('가격 임계치별 보유기간별 승률', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% 기준')
        ax2.set_ylim([40, 60])

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def plot_signal_vs_return(self, filename='stage2_signal_vs_return.png'):
        """
        시그널 빈도 vs 수익률 트레이드오프 (scatter plot)

        X축: 월 평균 시그널 수
        Y축: 10일 평균 수익률
        각 점: 가격 임계치 (레이블)
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter plot
        scatter = ax.scatter(self.results_df['monthly_signals'],
                           self.results_df['avg_return_10d'],
                           s=200,
                           c=self.results_df['price_threshold'],
                           cmap='viridis',
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=1.5)

        # 각 점에 레이블 표시 (가격 임계치)
        for idx, row in self.results_df.iterrows():
            ax.annotate(f"{row['price_threshold']:.0f}%",
                       xy=(row['monthly_signals'], row['avg_return_10d']),
                       xytext=(0, 0),
                       textcoords='offset points',
                       ha='center',
                       va='center',
                       fontsize=10,
                       fontweight='bold',
                       color='white')

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('가격 임계치 (%)', fontsize=11, fontweight='bold')

        ax.set_xlabel('월 평균 시그널 수', fontsize=12, fontweight='bold')
        ax.set_ylabel('10일 평균 수익률 (%)', fontsize=12, fontweight='bold')
        ax.set_title('시그널 빈도 vs 수익률 트레이드오프\n(각 점은 가격 임계치)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)

        # 참조선
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='목표: 2% 수익률')
        ax.axvline(x=10, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='최소: 월 10건')
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def plot_winrate_heatmap(self, filename='stage2_winrate_heatmap.png'):
        """
        승률 히트맵

        X축: 가격 임계치
        Y축: 보유기간 (1일/3일/5일/10일)
        색상: 승률
        """
        # 히트맵 데이터 준비
        periods = [1, 3, 5, 10]
        heatmap_data = []

        for period in periods:
            metric = f'win_rate_{period}d'
            heatmap_data.append(self.results_df[metric].values)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[f'{p}일' for p in periods],
            columns=[f'{int(x)}%' for x in self.results_df['price_threshold']]
        )

        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.heatmap(heatmap_df,
                   annot=True,
                   fmt='.1f',
                   cmap='RdYlGn',
                   center=50,
                   vmin=40,
                   vmax=60,
                   cbar_kws={'label': '승률 (%)'},
                   linewidths=1,
                   linecolor='white',
                   ax=ax)

        ax.set_xlabel('가격 상승률 임계치', fontsize=12, fontweight='bold')
        ax.set_ylabel('보유 기간', fontsize=12, fontweight='bold')
        ax.set_title('보유기간별 승률 히트맵 (VR ≥ 6.5 & ZS ≥ 3.0 + 가격 필터)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def plot_profit_factor(self, filename='stage2_profit_factor.png'):
        """
        손익비 비교 (막대 그래프)

        X축: 가격 임계치
        Y축: 손익비 (평균 이익 / 평균 손실)
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # 막대 그래프
        bars = ax.bar(range(len(self.results_df)),
                     self.results_df['profit_factor'],
                     color='steelblue',
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5)

        # 1.0 기준선 (손익 균형)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='손익 균형 (1.0)')

        # 막대 위에 값 표시
        for i, (bar, value) in enumerate(zip(bars, self.results_df['profit_factor'])):
            if pd.notna(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

                # 색상 변경 (1.0 이상이면 녹색, 미만이면 빨간색)
                if value >= 1.0:
                    bar.set_color('green')
                    bar.set_alpha(0.7)
                else:
                    bar.set_color('red')
                    bar.set_alpha(0.7)

        ax.set_xlabel('가격 상승률 임계치', fontsize=12, fontweight='bold')
        ax.set_ylabel('손익비 (평균 이익 / 평균 손실)', fontsize=12, fontweight='bold')
        ax.set_title('가격 임계치별 손익비 비교\n(>1.0: 이익 > 손실, <1.0: 이익 < 손실)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(self.results_df)))
        ax.set_xticklabels([f"{int(x)}%" for x in self.results_df['price_threshold']])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def create_all_plots(self):
        """모든 시각화 생성"""
        print("\nStage 2 시각화를 생성합니다...")

        self.plot_price_vs_return()
        self.plot_signal_vs_return()
        self.plot_winrate_heatmap()
        self.plot_profit_factor()

        print("\n모든 시각화 완료!")


def main():
    """테스트용 메인 함수"""
    # 결과 로드
    results_df = pd.read_csv('results/stage2_price_filter_results.csv')

    print("Stage 2 결과 데이터:")
    print(results_df[['price_threshold', 'signal_count', 'monthly_signals',
                      'avg_return_1d', 'avg_return_10d', 'win_rate_10d', 'profit_factor']])

    # 시각화 생성
    viz = Stage2Visualizer(results_df)
    viz.create_all_plots()


if __name__ == '__main__':
    main()
