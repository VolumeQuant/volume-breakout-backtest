# -*- coding: utf-8 -*-
"""
Stage 5: 현실적인 최종 전략 - VR 4.0, ZS 2.0 기준

목표:
1. 완화된 거래량 기준으로 충분한 빈도 확보
2. 가격 임계치 민감도 분석
3. 2022-2024 데이터로 검증 (2025년 데이터는 수집 필요)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class RealisticStrategy:
    """현실적인 최종 전략"""

    def __init__(self):
        self.data_dir = 'data'
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_data(self):
        """데이터 로드"""
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
        price_df['Return_0D'] = (price_df['Close'] - price_df['Open']) / price_df['Open'] * 100
        print(f"가격 데이터: {len(price_df):,}개 행")

        # 수급 데이터
        flow_df = pd.read_csv(
            f'{self.data_dir}/investor_flow_data.csv',
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

    def backtest_strategy(self, df, vr_th=4.0, zs_th=2.0, price_th=10.0):
        """
        단순 백테스팅 (개인(-) + 연기금(-) 조건)

        Parameters:
        -----------
        vr_th : float
            Volume Ratio 임계치
        zs_th : float
            Z-Score 임계치
        price_th : float
            당일 수익률 임계치 (%)
        """
        signals = df[
            (df['Volume_Ratio'] >= vr_th) &
            (df['Z_Score'] >= zs_th) &
            (df['Return_0D'] >= price_th) &
            (df['Return_0D'] < 30.0) &  # 상한가 제외
            (df['개인'] < 0) &
            (df['연기금'] < 0)
        ].copy()

        if len(signals) == 0:
            return None

        # 통계 계산
        stats = {
            'vr_th': vr_th,
            'zs_th': zs_th,
            'price_th': price_th,
            'signal_count': len(signals),
            'monthly_signals': len(signals) / signals['Date'].dt.to_period('M').nunique(),
            'avg_return_1d': signals['Return_1D'].mean(),
            'avg_return_10d': signals['Return_10D'].mean(),
            'median_return_10d': signals['Return_10D'].median(),
            'win_rate_1d': (signals['Return_1D'] > 0).sum() / len(signals) * 100,
            'win_rate_10d': (signals['Return_10D'] > 0).sum() / len(signals) * 100,
            'std_10d': signals['Return_10D'].std(),
            'max_loss_10d': signals['Return_10D'].min(),
            'max_gain_10d': signals['Return_10D'].max()
        }

        # 샤프 지수
        stats['sharpe_10d'] = stats['avg_return_10d'] / stats['std_10d'] * np.sqrt(252) if stats['std_10d'] > 0 else 0

        # 손익비
        profits = signals[signals['Return_10D'] > 0]['Return_10D']
        losses = signals[signals['Return_10D'] < 0]['Return_10D'].abs()
        stats['profit_factor'] = profits.mean() / losses.mean() if len(losses) > 0 and len(profits) > 0 else np.inf

        return stats

    def price_threshold_sensitivity(self, df, price_thresholds=[5, 7, 10]):
        """가격 임계치 민감도 분석"""
        print("=" * 80)
        print("가격 임계치 민감도 분석 (VR 4.0, ZS 2.0)")
        print("=" * 80)
        print(f"조건: 개인 순매도 & 연기금 순매도")
        print(f"테스트 케이스: {price_thresholds}\n")

        results = []

        for price_th in price_thresholds:
            print(f"[Case] 당일 수익률 >= {price_th}%")

            stats = self.backtest_strategy(df, vr_th=4.0, zs_th=2.0, price_th=price_th)

            if stats:
                results.append(stats)
                print(f"  시그널: {stats['signal_count']}개 (월 {stats['monthly_signals']:.1f}건)")
                print(f"  1일 수익률: {stats['avg_return_1d']:.2f}%")
                print(f"  10일 수익률: {stats['avg_return_10d']:.2f}% (승률 {stats['win_rate_10d']:.1f}%)")
                print(f"  손익비: {stats['profit_factor']:.2f}\n")
            else:
                print(f"  시그널 없음\n")

        if len(results) == 0:
            print("[경고] 모든 케이스에서 시그널 없음!\n")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        return results_df

    def compare_with_stage3(self, df):
        """Stage 3-3 결과와 비교"""
        print("=" * 80)
        print("Stage 3-3 결과와 비교")
        print("=" * 80)

        # Stage 3-3 조건: VR 6.5, ZS 3.0, Price 10%
        stats_stage3 = self.backtest_strategy(df, vr_th=6.5, zs_th=3.0, price_th=10.0)

        # Stage 5 권장: VR 4.0, ZS 2.0, Price 7%
        stats_stage5 = self.backtest_strategy(df, vr_th=4.0, zs_th=2.0, price_th=7.0)

        comparison = []

        if stats_stage3:
            comparison.append({
                'Strategy': 'Stage 3-3 (VR 6.5, ZS 3.0, Price 10%)',
                'Signals': stats_stage3['signal_count'],
                'Monthly': stats_stage3['monthly_signals'],
                'Avg_1D': stats_stage3['avg_return_1d'],
                'Avg_10D': stats_stage3['avg_return_10d'],
                'WinRate_10D': stats_stage3['win_rate_10d'],
                'Sharpe_10D': stats_stage3['sharpe_10d']
            })
            print("[Stage 3-3 조건: VR 6.5, ZS 3.0, Price 10%]")
            print(f"  시그널: {stats_stage3['signal_count']}개 (월 {stats_stage3['monthly_signals']:.1f}건)")
            print(f"  1일: {stats_stage3['avg_return_1d']:.2f}%")
            print(f"  10일: {stats_stage3['avg_return_10d']:.2f}% (승률 {stats_stage3['win_rate_10d']:.1f}%)\n")

        if stats_stage5:
            comparison.append({
                'Strategy': 'Stage 5 (VR 4.0, ZS 2.0, Price 7%)',
                'Signals': stats_stage5['signal_count'],
                'Monthly': stats_stage5['monthly_signals'],
                'Avg_1D': stats_stage5['avg_return_1d'],
                'Avg_10D': stats_stage5['avg_return_10d'],
                'WinRate_10D': stats_stage5['win_rate_10d'],
                'Sharpe_10D': stats_stage5['sharpe_10d']
            })
            print("[Stage 5 권장: VR 4.0, ZS 2.0, Price 7%]")
            print(f"  시그널: {stats_stage5['signal_count']}개 (월 {stats_stage5['monthly_signals']:.1f}건)")
            print(f"  1일: {stats_stage5['avg_return_1d']:.2f}%")
            print(f"  10일: {stats_stage5['avg_return_10d']:.2f}% (승률 {stats_stage5['win_rate_10d']:.1f}%)\n")

        return pd.DataFrame(comparison)

    def visualize_results(self, sensitivity_df, comparison_df):
        """결과 시각화"""
        print("=" * 80)
        print("결과 시각화")
        print("=" * 80)

        if len(sensitivity_df) == 0:
            print("[경고] 시각화할 데이터 없음\n")
            return

        # 1. 가격 임계치 민감도
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('가격 임계치 민감도 분석 (VR 4.0, ZS 2.0 + 개인/연기금 매도)',
                     fontsize=16, fontweight='bold')

        # 시그널 빈도
        ax = axes[0, 0]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['monthly_signals'],
               color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_title('월 평균 시그널 수', fontsize=13, fontweight='bold')
        ax.set_ylabel('시그널 수', fontsize=11)
        ax.axhline(y=10, color='red', linestyle='--', label='목표: 월 10건')
        ax.legend()
        for i, v in enumerate(sensitivity_df['monthly_signals']):
            ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 10일 수익률
        ax = axes[0, 1]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['avg_return_10d'],
               color='green', alpha=0.8, edgecolor='black')
        ax.set_title('10일 평균 수익률', fontsize=13, fontweight='bold')
        ax.set_ylabel('수익률 (%)', fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        for i, v in enumerate(sensitivity_df['avg_return_10d']):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                   fontsize=10, fontweight='bold')

        # 10일 승률
        ax = axes[1, 0]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['win_rate_10d'],
               color='orange', alpha=0.8, edgecolor='black')
        ax.set_title('10일 승률', fontsize=13, fontweight='bold')
        ax.set_ylabel('승률 (%)', fontsize=11)
        ax.axhline(y=50, color='red', linestyle='--', label='50% 기준')
        ax.legend()
        ax.set_ylim([0, 100])
        for i, v in enumerate(sensitivity_df['win_rate_10d']):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 손익비
        ax = axes[1, 1]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['profit_factor'],
               color='purple', alpha=0.8, edgecolor='black')
        ax.set_title('손익비 (10일)', fontsize=13, fontweight='bold')
        ax.set_ylabel('손익비', fontsize=11)
        ax.axhline(y=1.0, color='red', linestyle='--', label='손익균형')
        ax.legend()
        for i, v in enumerate(sensitivity_df['profit_factor']):
            label = f'{v:.2f}' if v < 10 else '∞'
            ax.text(i, min(v, 5), label, ha='center', va='bottom', fontsize=10, fontweight='bold')
            if v > 5:
                ax.set_ylim([0, 5])

        plt.tight_layout()
        filepath = f'{self.results_dir}/stage5_realistic_sensitivity.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  저장: {filepath}")

        # 2. Stage 3 vs Stage 5 비교
        if len(comparison_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Stage 3-3 vs Stage 5 비교', fontsize=16, fontweight='bold')

            metrics = [
                ('Monthly', '월 평균 시그널', 'steelblue'),
                ('Avg_10D', '10일 평균 수익률 (%)', 'green'),
                ('WinRate_10D', '10일 승률 (%)', 'orange')
            ]

            for idx, (metric, title, color) in enumerate(metrics):
                ax = axes[idx]
                ax.bar(comparison_df['Strategy'], comparison_df[metric],
                       color=color, alpha=0.8, edgecolor='black')
                ax.set_title(title, fontsize=13, fontweight='bold')
                ax.set_ylabel(title, fontsize=11)
                ax.set_xticklabels(comparison_df['Strategy'], rotation=15, ha='right', fontsize=9)
                for i, v in enumerate(comparison_df[metric]):
                    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            filepath = f'{self.results_dir}/stage5_realistic_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  저장: {filepath}\n")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("Stage 5: 현실적인 최종 전략")
    print("=" * 80)
    print("VR 4.0, ZS 2.0 기준 + 가격 임계치 최적화")
    print("=" * 80 + "\n")

    strategy = RealisticStrategy()

    # 1. 데이터 로드
    df = strategy.load_data()

    # 2. 가격 임계치 민감도 분석
    sensitivity_df = strategy.price_threshold_sensitivity(df, price_thresholds=[5, 7, 10])

    if len(sensitivity_df) > 0:
        sensitivity_df.to_csv(f'{strategy.results_dir}/stage5_realistic_sensitivity.csv',
                              index=False, encoding='utf-8-sig')
        print(f"결과 저장: {strategy.results_dir}/stage5_realistic_sensitivity.csv\n")

    # 3. Stage 3-3과 비교
    comparison_df = strategy.compare_with_stage3(df)

    if len(comparison_df) > 0:
        comparison_df.to_csv(f'{strategy.results_dir}/stage5_realistic_comparison.csv',
                             index=False, encoding='utf-8-sig')
        print(f"결과 저장: {strategy.results_dir}/stage5_realistic_comparison.csv\n")

    # 4. 시각화
    strategy.visualize_results(sensitivity_df, comparison_df)

    # 5. 최종 요약
    print("=" * 80)
    print("최종 전략 요약")
    print("=" * 80)

    if len(sensitivity_df) > 0:
        optimal = sensitivity_df.loc[sensitivity_df['monthly_signals'].idxmax()]
        print(f"\n[최적 파라미터]")
        print(f"  조건: VR>=4.0 & ZS>=2.0 & Price>={optimal['price_th']:.0f}% & 개인<0 & 연기금<0")
        print(f"\n[성과]")
        print(f"  월 시그널: {optimal['monthly_signals']:.1f}건")
        print(f"  1일 수익률: {optimal['avg_return_1d']:.2f}%")
        print(f"  10일 수익률: {optimal['avg_return_10d']:.2f}% (승률 {optimal['win_rate_10d']:.1f}%)")
        print(f"  손익비: {optimal['profit_factor']:.2f}")
        print(f"  샤프지수: {optimal['sharpe_10d']:.3f}")
        print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[오류 발생] {str(e)}")
        import traceback
        traceback.print_exc()
