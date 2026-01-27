# -*- coding: utf-8 -*-
"""
Stage 3-1 시각화 모듈

수급 주체별 단독 영향력 분석 결과를 시각화합니다.
1. 주체별 1일/10일 수익률 비교 막대 그래프
2. 수급 주체 유무에 따른 10일 수익률 분포 Boxplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class Stage3Step1Visualizer:
    """
    Stage 3-1 시각화 클래스
    """

    def __init__(self, results_dir='results'):
        """
        초기화 함수

        Parameters:
        -----------
        results_dir : str
            결과 저장 디렉토리
        """
        self.results_dir = results_dir

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # 한글 폰트 설정
        self._setup_korean_font()

        # 스타일 설정
        plt.style.use('seaborn-v0_8-whitegrid')

    def _setup_korean_font(self):
        """한글 폰트 설정"""
        # Windows 환경 폰트
        font_candidates = [
            'Malgun Gothic',
            'NanumGothic',
            'NanumBarunGothic',
            'AppleGothic',
            'DejaVu Sans'
        ]

        for font_name in font_candidates:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path and 'DejaVu' not in font_path:
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"한글 폰트 설정: {font_name}")
                    return
            except:
                continue

        # 기본 폰트 사용
        plt.rcParams['axes.unicode_minus'] = False
        print("기본 폰트 사용 (한글이 깨질 수 있음)")

    def plot_return_comparison_bar(self, results_df, save_path=None):
        """
        주체별 1일/10일 수익률 비교 막대 그래프

        Parameters:
        -----------
        results_df : pd.DataFrame
            백테스트 결과 데이터
        save_path : str, optional
            저장 경로
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 필터 이름 및 수익률 추출
        filters = results_df['filter_name'].tolist()
        short_names = ['Baseline', 'FI(+)', 'Pension(+)', 'Individual(-)']

        # 색상 설정
        colors = ['#7f8c8d', '#3498db', '#e74c3c', '#2ecc71']

        # 1일 수익률 막대 그래프
        ax1 = axes[0]
        returns_1d = results_df['avg_return_1d'].fillna(0).tolist()
        bars1 = ax1.bar(short_names, returns_1d, color=colors, edgecolor='black', linewidth=1.2)

        # 값 표시
        for bar, val in zip(bars1, returns_1d):
            height = bar.get_height()
            ax1.annotate(f'{val:.3f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax1.set_title('1일(익일) 평균 수익률 비교', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('평균 수익률 (%)', fontsize=12)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylim(min(returns_1d) - 0.3, max(returns_1d) + 0.5)

        # 10일 수익률 막대 그래프
        ax2 = axes[1]
        returns_10d = results_df['avg_return_10d'].fillna(0).tolist()
        bars2 = ax2.bar(short_names, returns_10d, color=colors, edgecolor='black', linewidth=1.2)

        # 값 표시
        for bar, val in zip(bars2, returns_10d):
            height = bar.get_height()
            ax2.annotate(f'{val:.3f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_title('10일 평균 수익률 비교', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('평균 수익률 (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylim(min(returns_10d) - 0.5, max(returns_10d) + 1.0)

        # 범례 추가
        legend_labels = [
            f'Baseline (n={int(results_df.iloc[0]["signal_count"])})',
            f'금융투자 순매수 (n={int(results_df.iloc[1]["signal_count"])})',
            f'연기금 순매수 (n={int(results_df.iloc[2]["signal_count"])})',
            f'개인 순매도 (n={int(results_df.iloc[3]["signal_count"])})'
        ]

        fig.legend(bars1, legend_labels,
                  loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=2, fontsize=10, frameon=True)

        plt.suptitle('Stage 3-1: 수급 주체별 단독 필터 성과 비교',
                    fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout(rect=[0, 0.08, 1, 0.98])

        # 저장
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'stage3_step1_return_comparison.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"저장: {save_path}")
        plt.close()

    def plot_return_distribution_boxplot(self, merged_df, base_signals, save_path=None):
        """
        수급 주체 유무에 따른 10일 수익률 분포 Boxplot

        Parameters:
        -----------
        merged_df : pd.DataFrame
            병합된 데이터
        base_signals : pd.DataFrame
            베이스 시그널
        save_path : str, optional
            저장 경로
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # 각 필터별 10일 수익률 분포 데이터 준비
        data_for_plot = []
        labels = []

        # 1. Baseline (수급 필터 없음)
        baseline_returns = base_signals['Return_10D'].dropna()
        if len(baseline_returns) > 0:
            data_for_plot.append(baseline_returns.values)
            labels.append(f'Baseline\n(n={len(baseline_returns)})')

        # 2. 금융투자 순매수
        fi_signals = base_signals[
            (base_signals['금융투자'].notna()) &
            (base_signals['금융투자'] > 0)
        ]['Return_10D'].dropna()
        if len(fi_signals) > 0:
            data_for_plot.append(fi_signals.values)
            labels.append(f'금융투자(+)\n(n={len(fi_signals)})')

        # 3. 연기금 순매수
        pension_signals = base_signals[
            (base_signals['연기금'].notna()) &
            (base_signals['연기금'] > 0)
        ]['Return_10D'].dropna()
        if len(pension_signals) > 0:
            data_for_plot.append(pension_signals.values)
            labels.append(f'연기금(+)\n(n={len(pension_signals)})')

        # 4. 개인 순매도
        individual_sell = base_signals[
            (base_signals['개인'].notna()) &
            (base_signals['개인'] < 0)
        ]['Return_10D'].dropna()
        if len(individual_sell) > 0:
            data_for_plot.append(individual_sell.values)
            labels.append(f'개인(-)\n(n={len(individual_sell)})')

        # Boxplot 생성
        colors = ['#7f8c8d', '#3498db', '#e74c3c', '#2ecc71']
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       meanprops={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
                       medianprops={'color': 'black', 'linewidth': 2},
                       flierprops={'marker': 'o', 'markerfacecolor': 'gray',
                                  'markersize': 4, 'alpha': 0.5})

        # 색상 적용
        for patch, color in zip(bp['boxes'], colors[:len(data_for_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 평균값 표시
        for i, data in enumerate(data_for_plot):
            mean_val = np.mean(data)
            ax.annotate(f'mean: {mean_val:.2f}%',
                       xy=(i + 1, mean_val),
                       xytext=(15, 0),
                       textcoords='offset points',
                       fontsize=10, color='red', fontweight='bold')

        ax.set_title('Stage 3-1: 수급 주체별 10일 수익률 분포',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('10일 수익률 (%)', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Y축 범위 조정 (이상치 고려)
        all_data = np.concatenate(data_for_plot)
        q1, q3 = np.percentile(all_data, [5, 95])
        ax.set_ylim(q1 - 5, q3 + 10)

        # 그리드
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'stage3_step1_return_boxplot.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"저장: {save_path}")
        plt.close()

    def plot_alpha_comparison(self, results_df, save_path=None):
        """
        Alpha 비교 그래프 (베이스라인 대비 개선율)

        Parameters:
        -----------
        results_df : pd.DataFrame
            백테스트 결과 데이터
        save_path : str, optional
            저장 경로
        """
        # 베이스라인 제외
        non_baseline = results_df[results_df['filter_key'] != 'A0_Baseline'].copy()

        if len(non_baseline) == 0:
            print("베이스라인 외 데이터 없음")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # 필터 이름
        filters = ['금융투자(+)', '연기금(+)', '개인(-)']
        x = np.arange(len(filters))
        width = 0.35

        # Alpha 값 추출
        alpha_1d = non_baseline['alpha_pct_1d'].fillna(0).tolist()
        alpha_10d = non_baseline['alpha_pct_10d'].fillna(0).tolist()

        # 막대 그래프
        bars1 = ax.bar(x - width/2, alpha_1d, width, label='1일 Alpha', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, alpha_10d, width, label='10일 Alpha', color='#e74c3c', edgecolor='black')

        # 값 표시
        for bar, val in zip(bars1, alpha_1d):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{val:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords='offset points',
                       ha='center', va=va, fontsize=10, fontweight='bold')

        for bar, val in zip(bars2, alpha_10d):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{val:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords='offset points',
                       ha='center', va=va, fontsize=10, fontweight='bold')

        ax.set_title('Stage 3-1: 수급 필터별 Alpha (베이스라인 대비 개선율)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Alpha (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(filters, fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.legend(loc='upper right', fontsize=10)

        # Y축 범위
        all_alpha = alpha_1d + alpha_10d
        y_min = min(all_alpha) - 20
        y_max = max(all_alpha) + 30
        ax.set_ylim(y_min, y_max)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # 저장
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'stage3_step1_alpha_comparison.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"저장: {save_path}")
        plt.close()


def main(results_df=None, merged_df=None, base_signals=None):
    """
    시각화 메인 함수

    Parameters:
    -----------
    results_df : pd.DataFrame, optional
        백테스트 결과 (없으면 CSV에서 로드)
    merged_df : pd.DataFrame, optional
        병합된 데이터 (Boxplot용)
    base_signals : pd.DataFrame, optional
        베이스 시그널 (Boxplot용)
    """
    print("\n" + "=" * 80)
    print("Stage 3-1 시각화")
    print("=" * 80)

    visualizer = Stage3Step1Visualizer()

    # 결과 데이터 로드
    if results_df is None:
        results_path = 'results/stage3_step1_baseline_comparison.csv'
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            print(f"결과 데이터 로드: {results_path}")
        else:
            print(f"[오류] 결과 파일 없음: {results_path}")
            return

    # 1. 수익률 비교 막대 그래프
    print("\n[1] 수익률 비교 막대 그래프 생성...")
    visualizer.plot_return_comparison_bar(results_df)

    # 2. Alpha 비교 그래프
    print("\n[2] Alpha 비교 그래프 생성...")
    visualizer.plot_alpha_comparison(results_df)

    # 3. Boxplot (merged_df가 있는 경우만)
    if merged_df is not None and base_signals is not None:
        print("\n[3] 수익률 분포 Boxplot 생성...")
        visualizer.plot_return_distribution_boxplot(merged_df, base_signals)
    else:
        print("\n[3] Boxplot 생성 건너뜀 (merged_df 또는 base_signals 없음)")

    print("\n시각화 완료!")


if __name__ == '__main__':
    main()
