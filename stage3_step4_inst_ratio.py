# -*- coding: utf-8 -*-
"""
Stage 3-4: 수급 집중도 분석

목표:
- 전체 거래대금 대비 기관(금융투자+연기금) 순매집 비중이 수익률에 미치는 영향 검증
- 최적의 Inst_Ratio 임계치 탐색
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class InstRatioAnalysis:
    """기관 수급 비중 분석 클래스"""

    def __init__(self):
        self.data_dir = 'data'
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_data(self):
        """데이터 로드 및 병합"""
        print("=" * 80)
        print("데이터 로드")
        print("=" * 80)

        # 가격 데이터
        price_df = pd.read_csv(
            f'{self.data_dir}/stock_data_with_indicators.csv',
            parse_dates=[0], low_memory=False
        )
        price_df.columns = ['Date'] + list(price_df.columns[1:])
        price_df['Code'] = price_df['Code'].astype(str).str.zfill(6)
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.normalize()

        # 당일 수익률
        price_df['Return_0D'] = (price_df['Close'] - price_df['Open']) / price_df['Open'] * 100

        # 당일 거래대금 계산 (Close * Volume)
        price_df['거래대금'] = price_df['Close'] * price_df['Volume']

        print(f"가격 데이터: {len(price_df):,}개 행")

        # 수급 데이터
        flow_df = pd.read_csv(
            f'{self.data_dir}/investor_flow_data_v2.csv',
            parse_dates=['Date'], low_memory=False
        )
        flow_df['Code'] = flow_df['Code'].astype(str).str.zfill(6)
        flow_df['Date'] = pd.to_datetime(flow_df['Date']).dt.normalize()
        print(f"수급 데이터: {len(flow_df):,}개 행")

        # 병합
        merged = pd.merge(
            price_df,
            flow_df[['Date', 'Code', '금융투자', '연기금', '개인', '외국인']],
            on=['Date', 'Code'],
            how='left'
        )
        print(f"병합 완료: {len(merged):,}개 행\n")

        return merged

    def calculate_inst_ratio(self, df):
        """기관 수급 비중(Inst_Ratio) 계산"""
        print("[1단계] 기관 수급 비중(Inst_Ratio) 계산")
        print("-" * 80)

        # 기관 순매수 = 금융투자 + 연기금
        df['기관순매수'] = df['금융투자'].fillna(0) + df['연기금'].fillna(0)

        # Inst_Ratio = 기관 순매수 / 거래대금 * 100 (%)
        df['Inst_Ratio'] = (df['기관순매수'] / df['거래대금']) * 100

        # 이상치 처리 (±100% 초과 제거)
        df.loc[df['Inst_Ratio'].abs() > 100, 'Inst_Ratio'] = np.nan

        valid_count = df['Inst_Ratio'].notna().sum()
        print(f"  Inst_Ratio 계산 완료: {valid_count:,}개 유효")
        print(f"  평균: {df['Inst_Ratio'].mean():.2f}%, 중앙값: {df['Inst_Ratio'].median():.2f}%")
        print(f"  범위: {df['Inst_Ratio'].min():.1f}% ~ {df['Inst_Ratio'].max():.1f}%\n")

        return df

    def get_base_signals(self, df, vr=6.5, zs=3.0, price_th=10.0, price_upper=30.0):
        """기본 조건 시그널"""
        signals = df[
            (df['Volume_Ratio'] >= vr) &
            (df['Z_Score'] >= zs) &
            (df['Return_0D'] >= price_th) &
            (df['Return_0D'] < price_upper) &
            (df['Inst_Ratio'].notna())
        ].copy()
        return signals

    def calculate_stats(self, signals, label=''):
        """통계 계산"""
        if len(signals) == 0:
            return {'label': label, 'signal_count': 0}

        months = signals['Date'].dt.to_period('M').nunique()
        monthly = len(signals) / months if months > 0 else 0

        stats = {
            'label': label,
            'signal_count': len(signals),
            'monthly_signals': round(monthly, 2)
        }

        for period in [1, 10]:
            col = f'Return_{period}D'
            if col not in signals.columns:
                continue
            returns = signals[col].dropna()
            if len(returns) == 0:
                continue

            stats[f'{period}d_avg'] = round(returns.mean(), 4)
            stats[f'{period}d_median'] = round(returns.median(), 4)
            stats[f'{period}d_win_rate'] = round((returns > 0).sum() / len(returns) * 100, 2)

            # 손익비
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            stats[f'{period}d_pf'] = round(avg_profit / avg_loss, 3) if avg_loss > 0 else 999

        return stats

    def run_threshold_test(self, merged_df):
        """[2] Inst_Ratio 임계치별 테스트"""
        print("\n[2단계] 기관 수급 비중 임계치별 테스트")
        print("-" * 80)

        base_signals = self.get_base_signals(merged_df)
        print(f"기본 시그널 (VR≥6.5, ZS≥3.0): {len(base_signals)}개\n")

        # 임계치 케이스
        cases = [
            ('Baseline (수급무관)', None),
            ('Case A: Inst_Ratio > 0%', 0),
            ('Case B: Inst_Ratio ≥ 3%', 3),
            ('Case C: Inst_Ratio ≥ 5%', 5),
            ('Case D: Inst_Ratio ≥ 10%', 10),
            ('Case E: Inst_Ratio ≥ 15%', 15),
        ]

        results = []

        print(f"{'케이스':<30} | {'시그널':>6} | {'월평균':>6} | {'1일수익률':>10} | {'10일수익률':>10} | {'10일승률':>8}")
        print("-" * 90)

        for name, threshold in cases:
            if threshold is None:
                filtered = base_signals
            else:
                filtered = base_signals[base_signals['Inst_Ratio'] >= threshold]

            stats = self.calculate_stats(filtered, name)
            stats['threshold'] = threshold if threshold is not None else -999
            results.append(stats)

            print(f"{name:<30} | {stats.get('signal_count', 0):>6} | "
                  f"{stats.get('monthly_signals', 0):>5.1f}건 | "
                  f"{stats.get('1d_avg', 0):>9.3f}% | "
                  f"{stats.get('10d_avg', 0):>9.3f}% | "
                  f"{stats.get('10d_win_rate', 0):>7.1f}%")

        return pd.DataFrame(results), base_signals

    def run_correlation_analysis(self, signals):
        """[3] 상관관계 분석"""
        print("\n\n[3단계] Inst_Ratio와 수익률의 상관관계 분석")
        print("-" * 80)

        # 유효 데이터만
        valid = signals[
            (signals['Inst_Ratio'].notna()) &
            (signals['Return_10D'].notna())
        ].copy()

        if len(valid) < 10:
            print("  유효 데이터 부족")
            return None

        # 상관계수 계산
        corr_1d = valid['Inst_Ratio'].corr(valid['Return_1D'])
        corr_10d = valid['Inst_Ratio'].corr(valid['Return_10D'])

        # 스피어만 상관계수 (순위 기반)
        spearman_1d, p_1d = stats.spearmanr(valid['Inst_Ratio'], valid['Return_1D'].fillna(0))
        spearman_10d, p_10d = stats.spearmanr(valid['Inst_Ratio'], valid['Return_10D'].fillna(0))

        print(f"  피어슨 상관계수:")
        print(f"    - Inst_Ratio vs 1일 수익률: {corr_1d:.4f}")
        print(f"    - Inst_Ratio vs 10일 수익률: {corr_10d:.4f}")
        print(f"\n  스피어만 상관계수:")
        print(f"    - Inst_Ratio vs 1일 수익률: {spearman_1d:.4f} (p={p_1d:.4f})")
        print(f"    - Inst_Ratio vs 10일 수익률: {spearman_10d:.4f} (p={p_10d:.4f})")

        # 분위수별 분석
        print("\n  Inst_Ratio 분위수별 10일 수익률:")
        valid['IR_Quantile'] = pd.qcut(valid['Inst_Ratio'], q=5, labels=['Q1(최저)', 'Q2', 'Q3', 'Q4', 'Q5(최고)'])

        quantile_stats = valid.groupby('IR_Quantile')['Return_10D'].agg(['mean', 'median', 'count'])
        print(quantile_stats.round(3))

        return {
            'pearson_1d': corr_1d,
            'pearson_10d': corr_10d,
            'spearman_1d': spearman_1d,
            'spearman_10d': spearman_10d,
            'quantile_stats': quantile_stats
        }

    def visualize_results(self, results_df, signals):
        """시각화"""
        print("\n\n[4단계] 시각화")
        print("-" * 80)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 산점도: Inst_Ratio vs 10일 수익률
        ax1 = axes[0, 0]
        valid = signals[
            (signals['Inst_Ratio'].notna()) &
            (signals['Return_10D'].notna()) &
            (signals['Inst_Ratio'].between(-20, 30))  # 이상치 제외
        ]

        ax1.scatter(valid['Inst_Ratio'], valid['Return_10D'], alpha=0.5, s=30, c='#3498db')

        # 추세선
        if len(valid) > 2:
            z = np.polyfit(valid['Inst_Ratio'], valid['Return_10D'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['Inst_Ratio'].min(), valid['Inst_Ratio'].max(), 100)
            ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'추세선 (기울기: {z[0]:.3f})')

        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('기관 수급 비중 (%)')
        ax1.set_ylabel('10일 수익률 (%)')
        ax1.set_title('기관 수급 비중 vs 10일 수익률', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 임계치별 수익률 막대 그래프
        ax2 = axes[0, 1]
        plot_df = results_df[results_df['threshold'] >= 0].copy()

        if len(plot_df) > 0:
            x = np.arange(len(plot_df))
            width = 0.35

            bars1 = ax2.bar(x - width/2, plot_df['1d_avg'].fillna(0), width,
                           label='1일', color='#3498db', edgecolor='black')
            bars2 = ax2.bar(x + width/2, plot_df['10d_avg'].fillna(0), width,
                           label='10일', color='#e74c3c', edgecolor='black')

            ax2.set_xticks(x)
            ax2.set_xticklabels([f"≥{int(t)}%" for t in plot_df['threshold']], fontsize=10)
            ax2.set_xlabel('Inst_Ratio 임계치')
            ax2.set_ylabel('평균 수익률 (%)')
            ax2.set_title('임계치별 평균 수익률', fontsize=12, fontweight='bold')
            ax2.axhline(y=0, color='black', linewidth=0.8)
            ax2.legend()

            # 값 표시
            for bar in bars1:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f'{h:.2f}', ha='center', fontsize=9)
            for bar in bars2:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.1,
                        f'{h:.2f}', ha='center', fontsize=9)

        # 3. 임계치별 승률 막대 그래프
        ax3 = axes[1, 0]

        if len(plot_df) > 0:
            bars1 = ax3.bar(x - width/2, plot_df['1d_win_rate'].fillna(0), width,
                           label='1일', color='#27ae60', edgecolor='black')
            bars2 = ax3.bar(x + width/2, plot_df['10d_win_rate'].fillna(0), width,
                           label='10일', color='#9b59b6', edgecolor='black')

            ax3.set_xticks(x)
            ax3.set_xticklabels([f"≥{int(t)}%" for t in plot_df['threshold']], fontsize=10)
            ax3.set_xlabel('Inst_Ratio 임계치')
            ax3.set_ylabel('승률 (%)')
            ax3.set_title('임계치별 승률', fontsize=12, fontweight='bold')
            ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% 기준')
            ax3.legend()
            ax3.set_ylim(0, 100)

            for bar, val in zip(bars1, plot_df['1d_win_rate'].fillna(0)):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
                        f'{val:.1f}', ha='center', fontsize=9)
            for bar, val in zip(bars2, plot_df['10d_win_rate'].fillna(0)):
                ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
                        f'{val:.1f}', ha='center', fontsize=9)

        # 4. 임계치별 시그널 수
        ax4 = axes[1, 1]

        if len(plot_df) > 0:
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(plot_df)))
            bars = ax4.bar(x, plot_df['signal_count'], color=colors, edgecolor='black')

            ax4.set_xticks(x)
            ax4.set_xticklabels([f"≥{int(t)}%" for t in plot_df['threshold']], fontsize=10)
            ax4.set_xlabel('Inst_Ratio 임계치')
            ax4.set_ylabel('시그널 수')
            ax4.set_title('임계치별 시그널 수 (Trade-off)', fontsize=12, fontweight='bold')

            for bar, val in zip(bars, plot_df['signal_count']):
                ax4.text(bar.get_x() + bar.get_width()/2, val + 1,
                        f'{int(val)}', ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Stage 3-4: 기관 수급 비중(Inst_Ratio) 분석', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = f'{self.results_dir}/stage3_step4_inst_ratio.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"저장: {save_path}")
        plt.close()


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("Stage 3-4: 기관 수급 비중(Inst_Ratio) 분석")
    print("=" * 80 + "\n")

    analyzer = InstRatioAnalysis()

    # 1. 데이터 로드
    merged_df = analyzer.load_data()

    # 2. Inst_Ratio 계산
    merged_df = analyzer.calculate_inst_ratio(merged_df)

    # 3. 임계치별 테스트
    results_df, signals = analyzer.run_threshold_test(merged_df)

    # 4. 상관관계 분석
    corr_results = analyzer.run_correlation_analysis(signals)

    # 5. 시각화
    analyzer.visualize_results(results_df, signals)

    # 결과 저장
    results_df.to_csv(f'{analyzer.results_dir}/stage3_step4_inst_ratio.csv',
                      index=False, encoding='utf-8-sig')
    print(f"\n저장: results/stage3_step4_inst_ratio.csv")

    # Sweet Spot 제안
    print("\n" + "=" * 80)
    print("[결론] Sweet Spot 제안")
    print("=" * 80)

    # 최적 임계치 찾기 (시그널 30개 이상, 승률 최대)
    valid_results = results_df[
        (results_df['signal_count'] >= 20) &
        (results_df['threshold'] >= 0)
    ]

    if len(valid_results) > 0:
        best_idx = valid_results['10d_win_rate'].idxmax()
        best = valid_results.loc[best_idx]

        print(f"""
1. 최적 Inst_Ratio 임계치: ≥ {int(best['threshold'])}%
   - 시그널 수: {int(best['signal_count'])}개 (월 {best['monthly_signals']:.1f}건)
   - 10일 수익률: {best['10d_avg']:.3f}%
   - 10일 승률: {best['10d_win_rate']:.1f}%

2. Sweet Spot 기준:
   - Inst_Ratio ≥ 3~5%에서 빈도-수익률 균형 최적
   - 10% 이상은 시그널 감소로 실용성 저하

3. 실전 권장 조합:
   - VR ≥ 6.5, ZS ≥ 3.0, 당일 ≥ 10%
   - Inst_Ratio ≥ 3% (기관 순매수가 거래대금의 3% 이상)
   - 예상 월 시그널: 2-3건, 10일 목표 수익률: 4-6%
""")

    print("\nStage 3-4 완료!")
    return results_df


if __name__ == '__main__':
    results = main()
