"""
Stage 3 결과 시각화 모듈

세분화된 수급 필터 백테스팅 결과를 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class Stage3Visualizer:
    """Stage 3 결과 시각화 클래스"""

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

    def plot_investor_predictive_power(self, filename='stage3_investor_predictive_power.png'):
        """
        투자자 유형별 예측력 비교

        X축: 투자자 유형 (A1~A7)
        Y축: 수익률
        그룹: 보유기간 (1일, 3일, 5일, 10일)
        """
        # A그룹 필터만 추출
        a_group = self.results_df[self.results_df['filter'].str.startswith('A')].copy()

        if len(a_group) == 0:
            print("[경고] A그룹 데이터 없음")
            return

        # 그래프 설정
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('투자자 유형별 예측력 비교\n(VR>=6.5 & ZS>=3.0 & Price>=10%)',
                     fontsize=16, fontweight='bold')

        periods = [1, 3, 5, 10]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        for idx, (period, ax) in enumerate(zip(periods, axes.flatten())):
            metric_return = f'avg_return_{period}d'
            metric_winrate = f'win_rate_{period}d'

            # 막대 그래프 - 평균 수익률
            x_pos = np.arange(len(a_group))
            bars = ax.bar(x_pos, a_group[metric_return], color=colors[:len(a_group)], alpha=0.7)

            # 레이블 설정
            labels = [f.replace('A0_', '').replace('A1_', '').replace('A2_', '').replace('A3_', '')\
                        .replace('A4_', '').replace('A5_', '').replace('A7_', '')\
                        .replace('_', '\n') for f in a_group['filter']]

            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('평균 수익률 (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{period}일 보유', fontsize=12, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            # 승률을 오른쪽 Y축에 표시
            ax2 = ax.twinx()
            ax2.plot(x_pos, a_group[metric_winrate], 'ro-', linewidth=2, markersize=8, label='승률')
            ax2.set_ylabel('승률 (%)', fontsize=11, fontweight='bold', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim([40, 60])
            ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.3)

            # 막대 위에 값 표시
            for i, (bar, val) in enumerate(zip(bars, a_group[metric_return])):
                if pd.notna(val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}%',
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=8, fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def plot_short_vs_long_term(self, filename='stage3_short_vs_long_term.png'):
        """
        단기 vs 장기 투자자 효과 비교

        금융투자 (A1) vs 연기금 (A2)를 1일, 10일 수익률로 비교
        """
        # 금융투자와 연기금 필터 추출
        financial = self.results_df[self.results_df['filter'] == 'A1_Financial_Investment']
        pension = self.results_df[self.results_df['filter'] == 'A2_Pension']
        baseline = self.results_df[self.results_df['filter'] == 'A0_Baseline']

        if len(financial) == 0 or len(pension) == 0:
            print("[경고] 금융투자 또는 연기금 데이터 없음")
            return

        # 데이터 준비
        comparison_data = []

        for period in [1, 3, 5, 10]:
            metric_return = f'avg_return_{period}d'
            metric_winrate = f'win_rate_{period}d'

            if len(baseline) > 0:
                comparison_data.append({
                    'period': period,
                    'investor': 'Baseline',
                    'avg_return': baseline[metric_return].values[0],
                    'win_rate': baseline[metric_winrate].values[0]
                })

            comparison_data.append({
                'period': period,
                'investor': 'Financial\nInvestment',
                'avg_return': financial[metric_return].values[0],
                'win_rate': financial[metric_winrate].values[0]
            })

            comparison_data.append({
                'period': period,
                'investor': 'Pension',
                'avg_return': pension[metric_return].values[0],
                'win_rate': pension[metric_winrate].values[0]
            })

        comparison_df = pd.DataFrame(comparison_data)

        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('금융투자 vs 연기금 예측력 비교\n단기 스마트머니 vs 장기 방향성',
                     fontsize=16, fontweight='bold')

        # 왼쪽: 평균 수익률
        pivot_return = comparison_df.pivot(index='period', columns='investor', values='avg_return')
        pivot_return.plot(kind='bar', ax=ax1, color=['gray', '#1f77b4', '#ff7f0e'], alpha=0.8, width=0.7)

        ax1.set_xlabel('보유 기간 (일)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('평균 수익률 (%)', fontsize=12, fontweight='bold')
        ax1.set_title('보유기간별 평균 수익률', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        ax1.legend(title='투자자 유형', fontsize=10, title_fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # 값 표시
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f', fontsize=9, padding=3)

        # 오른쪽: 승률
        pivot_winrate = comparison_df.pivot(index='period', columns='investor', values='win_rate')
        pivot_winrate.plot(kind='bar', ax=ax2, color=['gray', '#1f77b4', '#ff7f0e'], alpha=0.8, width=0.7)

        ax2.set_xlabel('보유 기간 (일)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('승률 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('보유기간별 승률', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(title='투자자 유형', fontsize=10, title_fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% 기준')
        ax2.set_ylim([40, 60])

        # 값 표시
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f', fontsize=9, padding=3)

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def plot_multiple_buy_effect(self, filename='stage3_multiple_buy_effect.png'):
        """
        복수 주체 동시 매수 효과 분석

        D그룹: 2개 이상 투자자 동시 순매수 효과
        """
        # D그룹 필터 추출
        d_group = self.results_df[self.results_df['filter'].str.startswith('D')].copy()
        baseline = self.results_df[self.results_df['filter'] == 'A0_Baseline']

        if len(d_group) == 0:
            print("[경고] D그룹 데이터 없음")
            return

        # Baseline 추가
        if len(baseline) > 0:
            d_group = pd.concat([baseline, d_group], ignore_index=True)

        # 그래프 설정
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('복수 투자자 동시 매수 효과\nDouble/Triple Buy Effect',
                     fontsize=16, fontweight='bold')

        # 레이블 정리
        labels = [f.replace('A0_', '').replace('D1_', '').replace('D2_', '').replace('D4_', '')\
                    .replace('_AND_', '+').replace('_', '\n') for f in d_group['filter']]

        x_pos = np.arange(len(d_group))
        colors = ['gray'] + ['#2ca02c', '#ff7f0e', '#d62728'][:len(d_group)-1]

        # 상단: 10일 수익률
        ax1 = axes[0]
        bars1 = ax1.bar(x_pos, d_group['avg_return_10d'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax1.set_ylabel('평균 수익률 (%)', fontsize=12, fontweight='bold')
        ax1.set_title('10일 보유 수익률', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # 값 표시
        for bar in bars1:
            height = bar.get_height()
            if pd.notna(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')

        # 하단: 시그널 수 및 승률
        ax2 = axes[1]

        # 막대: 월 평균 시그널 수
        bars2 = ax2.bar(x_pos, d_group['monthly_signals'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        ax2.set_xlabel('투자자 조합', fontsize=12, fontweight='bold')
        ax2.set_ylabel('월 평균 시그널 수', fontsize=12, fontweight='bold')
        ax2.set_title('시그널 빈도 및 승률', fontsize=13, fontweight='bold', pad=15)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # 값 표시
        for bar in bars2:
            height = bar.get_height()
            if pd.notna(height):
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        # 오른쪽 Y축: 10일 승률
        ax2_right = ax2.twinx()
        ax2_right.plot(x_pos, d_group['win_rate_10d'], 'ro-', linewidth=2, markersize=10, label='10일 승률')
        ax2_right.set_ylabel('10일 승률 (%)', fontsize=12, fontweight='bold', color='red')
        ax2_right.tick_params(axis='y', labelcolor='red')
        ax2_right.set_ylim([40, 60])
        ax2_right.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.3)

        # 승률 값 표시
        for i, (x, y) in enumerate(zip(x_pos, d_group['win_rate_10d'])):
            if pd.notna(y):
                ax2_right.annotate(f'{y:.1f}%',
                                  xy=(x, y),
                                  xytext=(0, 8),
                                  textcoords='offset points',
                                  ha='center',
                                  fontsize=9,
                                  color='red',
                                  fontweight='bold')

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"그래프 저장: {filepath}")

    def create_all_plots(self):
        """모든 시각화 생성"""
        print("\nStage 3 시각화를 생성합니다...")

        self.plot_investor_predictive_power()
        self.plot_short_vs_long_term()
        self.plot_multiple_buy_effect()

        print("\n모든 시각화 완료!")


def main():
    """테스트용 메인 함수"""
    # 결과 로드
    results_df = pd.read_csv('results/stage3_flow_filter_results.csv')

    print("Stage 3 결과 데이터:")
    print(results_df[['filter', 'signal_count', 'monthly_signals',
                      'avg_return_1d', 'avg_return_10d', 'win_rate_10d', 'profit_factor']])

    # 시각화 생성
    viz = Stage3Visualizer(results_df)
    viz.create_all_plots()


if __name__ == '__main__':
    main()
