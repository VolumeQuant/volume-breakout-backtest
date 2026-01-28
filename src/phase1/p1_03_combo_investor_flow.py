# -*- coding: utf-8 -*-
"""
Stage 3-2: 수급 필터 조합 백테스팅

목표:
- 단독 필터가 아닌 조합 필터의 성과 검증
- 1일, 3일, 5일, 10일 보유 기간별 최적 조합 탐색

조합 필터:
1. 금융투자(+) AND 개인(-): 스마트머니 + 개미털기
2. 금융투자(+) AND 연기금(-): 증권사 매수 + 연기금 매도
3. 금융투자(+) AND 연기금(+): 기관 동반 매수
4. 개인(-) AND 연기금(-): 기관 전체가 개인에게서 매집
5. 금융투자(+) AND 개인(-) AND 연기금(-): 트리플 조건
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader, FlowDataLoader


class ComboFilterBacktest:
    """조합 필터 백테스트 클래스"""

    # 조합 필터 설정
    COMBO_FILTERS = {
        # 단독 필터 (비교용)
        'S0_Baseline': {
            'name': 'Baseline (필터없음)',
            'conditions': []
        },
        'S1_FI_Only': {
            'name': '금융투자(+) 단독',
            'conditions': [('금융투자', '>', 0)]
        },
        'S2_Ind_Sell_Only': {
            'name': '개인(-) 단독',
            'conditions': [('개인', '<', 0)]
        },

        # 2개 조합
        'C1_FI_IndSell': {
            'name': '금융투자(+) + 개인(-)',
            'conditions': [('금융투자', '>', 0), ('개인', '<', 0)]
        },
        'C2_FI_PensionSell': {
            'name': '금융투자(+) + 연기금(-)',
            'conditions': [('금융투자', '>', 0), ('연기금', '<', 0)]
        },
        'C3_FI_PensionBuy': {
            'name': '금융투자(+) + 연기금(+)',
            'conditions': [('금융투자', '>', 0), ('연기금', '>', 0)]
        },
        'C4_IndSell_PensionSell': {
            'name': '개인(-) + 연기금(-)',
            'conditions': [('개인', '<', 0), ('연기금', '<', 0)]
        },
        'C5_FI_ForeignBuy': {
            'name': '금융투자(+) + 외국인(+)',
            'conditions': [('금융투자', '>', 0), ('외국인', '>', 0)]
        },

        # 3개 조합
        'C6_FI_IndSell_PensionSell': {
            'name': '금융투자(+) + 개인(-) + 연기금(-)',
            'conditions': [('금융투자', '>', 0), ('개인', '<', 0), ('연기금', '<', 0)]
        },
        'C7_FI_IndSell_ForeignBuy': {
            'name': '금융투자(+) + 개인(-) + 외국인(+)',
            'conditions': [('금융투자', '>', 0), ('개인', '<', 0), ('외국인', '>', 0)]
        },
    }

    def __init__(self,
                 volume_ratio_threshold=6.5,
                 z_score_threshold=3.0,
                 price_threshold=10.0,
                 price_upper_limit=30.0):
        """초기화"""
        self.vr_threshold = volume_ratio_threshold
        self.zs_threshold = z_score_threshold
        self.price_threshold = price_threshold
        self.price_upper_limit = price_upper_limit

        self.data_dir = 'data'
        self.results_dir = 'results'

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_and_merge_data(self):
        """데이터 로드 및 병합"""
        print("=" * 80)
        print("데이터 로드 및 병합")
        print("=" * 80)

        # 가격 데이터
        price_df = pd.read_csv(
            f'{self.data_dir}/stock_data_with_indicators.csv',
            parse_dates=[0],
            low_memory=False
        )
        if price_df.columns[0] != 'Date':
            price_df = price_df.rename(columns={price_df.columns[0]: 'Date'})
        print(f"가격 데이터: {len(price_df):,}개 행")

        # 수급 데이터
        flow_df = pd.read_csv(
            f'{self.data_dir}/investor_flow_data_v2.csv',
            parse_dates=['Date'],
            low_memory=False
        )
        print(f"수급 데이터: {len(flow_df):,}개 행")

        # 타입 통일
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df['Code'] = price_df['Code'].astype(str).str.zfill(6)
        flow_df['Code'] = flow_df['Code'].astype(str).str.zfill(6)

        # 병합
        merged = pd.merge(
            price_df,
            flow_df[['Date', 'Code', '금융투자', '연기금', '개인', '외국인']],
            on=['Date', 'Code'],
            how='left'
        )

        # 당일 수익률 계산
        if 'Return_0D' not in merged.columns:
            merged['Return_0D'] = (merged['Close'] - merged['Open']) / merged['Open'] * 100

        print(f"병합 완료: {len(merged):,}개 행")
        print()

        return merged

    def get_base_signals(self, df):
        """베이스 시그널 추출"""
        base = df[
            (df['Volume_Ratio'] >= self.vr_threshold) &
            (df['Z_Score'] >= self.zs_threshold) &
            (df['Return_0D'] >= self.price_threshold) &
            (df['Return_0D'] < self.price_upper_limit)
        ].copy()
        return base

    def apply_combo_filter(self, signals, filter_key):
        """조합 필터 적용"""
        config = self.COMBO_FILTERS[filter_key]
        conditions = config['conditions']

        if len(conditions) == 0:
            return signals.copy()

        # 수급 데이터가 있는 행만 대상
        mask = signals['금융투자'].notna()

        for col, op, val in conditions:
            if op == '>':
                mask = mask & (signals[col] > val)
            elif op == '<':
                mask = mask & (signals[col] < val)
            elif op == '>=':
                mask = mask & (signals[col] >= val)
            elif op == '<=':
                mask = mask & (signals[col] <= val)

        return signals[mask].copy()

    def calculate_period_stats(self, signals, periods=[1, 3, 5, 10]):
        """보유 기간별 통계 계산"""
        if len(signals) == 0:
            return {f'{p}d': {} for p in periods}

        stats = {}

        for period in periods:
            col = f'Return_{period}D'
            if col not in signals.columns:
                stats[f'{period}d'] = {}
                continue

            returns = signals[col].dropna()

            if len(returns) == 0:
                stats[f'{period}d'] = {}
                continue

            # 통계 계산
            avg_ret = returns.mean()
            median_ret = returns.median()
            std_ret = returns.std()
            win_rate = (returns > 0).sum() / len(returns) * 100
            max_ret = returns.max()
            min_ret = returns.min()

            # 손익비
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            profit_factor = avg_profit / avg_loss if avg_loss > 0 else float('inf')

            # 샤프 비율 (연율화)
            sharpe = (avg_ret / std_ret * np.sqrt(252 / period)) if std_ret > 0 else 0

            stats[f'{period}d'] = {
                'avg_return': round(avg_ret, 4),
                'median_return': round(median_ret, 4),
                'std_return': round(std_ret, 4),
                'win_rate': round(win_rate, 2),
                'max_return': round(max_ret, 2),
                'mdd': round(min_ret, 2),
                'profit_factor': round(profit_factor, 3),
                'sharpe': round(sharpe, 3)
            }

        return stats

    def run_backtest(self, merged_df):
        """전체 백테스트 실행"""
        print("=" * 80)
        print("Stage 3-2: 조합 필터 백테스트")
        print("=" * 80)
        print(f"조건: VR >= {self.vr_threshold}, ZS >= {self.zs_threshold}, "
              f"Price >= {self.price_threshold}%")
        print()

        # 베이스 시그널
        base_signals = self.get_base_signals(merged_df)
        print(f"베이스 시그널: {len(base_signals):,}개\n")

        results = []

        for filter_key, config in self.COMBO_FILTERS.items():
            print(f"[{filter_key}] {config['name']}")

            # 필터 적용
            filtered = self.apply_combo_filter(base_signals, filter_key)
            signal_count = len(filtered)

            if signal_count == 0:
                print(f"  → 시그널 없음\n")
                continue

            # 월평균 시그널
            months = filtered['Date'].dt.to_period('M').nunique()
            monthly = signal_count / months if months > 0 else 0

            # 기간별 통계
            period_stats = self.calculate_period_stats(filtered)

            # 결과 저장
            row = {
                'filter_key': filter_key,
                'filter_name': config['name'],
                'signal_count': signal_count,
                'monthly_signals': round(monthly, 2)
            }

            for period in [1, 3, 5, 10]:
                key = f'{period}d'
                if key in period_stats and period_stats[key]:
                    for stat_name, stat_val in period_stats[key].items():
                        row[f'{key}_{stat_name}'] = stat_val

            results.append(row)

            # 출력
            print(f"  시그널: {signal_count:,}개 (월 {monthly:.1f}건)")
            for period in [1, 3, 5, 10]:
                key = f'{period}d'
                if key in period_stats and period_stats[key]:
                    s = period_stats[key]
                    print(f"  {period}일: {s['avg_return']:.3f}% | "
                          f"승률 {s['win_rate']:.1f}% | "
                          f"손익비 {s['profit_factor']:.2f}")
            print()

        return pd.DataFrame(results)

    def calculate_alpha(self, results_df):
        """베이스라인 대비 Alpha 계산"""
        df = results_df.copy()

        baseline = df[df['filter_key'] == 'S0_Baseline']
        if len(baseline) == 0:
            return df

        baseline = baseline.iloc[0]

        for period in [1, 3, 5, 10]:
            avg_col = f'{period}d_avg_return'
            if avg_col in df.columns:
                base_val = baseline[avg_col]
                df[f'{period}d_alpha'] = df[avg_col] - base_val
                if base_val != 0:
                    df[f'{period}d_alpha_pct'] = ((df[avg_col] - base_val) / abs(base_val) * 100).round(1)
                else:
                    df[f'{period}d_alpha_pct'] = 0

        return df

    def save_results(self, results_df, filename='stage3_step2_combo_results.csv'):
        """결과 저장"""
        filepath = os.path.join(self.results_dir, filename)
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"저장: {filepath}")

    def print_summary(self, results_df):
        """결과 요약 출력"""
        print("\n" + "=" * 80)
        print("보유 기간별 최적 조합")
        print("=" * 80)

        for period in [1, 3, 5, 10]:
            avg_col = f'{period}d_avg_return'
            alpha_col = f'{period}d_alpha_pct'

            if avg_col not in results_df.columns:
                continue

            # 베이스라인 제외하고 최고 성과 찾기
            non_baseline = results_df[~results_df['filter_key'].str.startswith('S0')]

            if len(non_baseline) == 0:
                continue

            # 시그널 30개 이상만
            valid = non_baseline[non_baseline['signal_count'] >= 30]

            if len(valid) == 0:
                valid = non_baseline

            best_idx = valid[avg_col].idxmax()
            best = valid.loc[best_idx]

            print(f"\n[{period}일 보유] 최적: {best['filter_name']}")
            print(f"  수익률: {best[avg_col]:.3f}% (Alpha: {best.get(alpha_col, 0):+.1f}%)")
            print(f"  시그널: {int(best['signal_count'])}개, "
                  f"승률: {best.get(f'{period}d_win_rate', 0):.1f}%, "
                  f"손익비: {best.get(f'{period}d_profit_factor', 0):.2f}")


def visualize_results(results_df, results_dir='results'):
    """시각화"""
    # 한글 폰트
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 보유 기간별 수익률 비교 (상위 필터만)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    periods = [1, 3, 5, 10]

    for idx, period in enumerate(periods):
        ax = axes[idx // 2, idx % 2]
        avg_col = f'{period}d_avg_return'

        if avg_col not in results_df.columns:
            continue

        # 데이터 준비 (시그널 있는 것만)
        valid = results_df[results_df['signal_count'] > 0].copy()
        valid = valid.sort_values(avg_col, ascending=True)

        # 색상 (베이스라인 회색, 나머지 파랑/빨강)
        colors = []
        for _, row in valid.iterrows():
            if row['filter_key'].startswith('S0'):
                colors.append('#7f8c8d')
            elif row[avg_col] > valid[valid['filter_key'] == 'S0_Baseline'][avg_col].values[0]:
                colors.append('#27ae60')
            else:
                colors.append('#e74c3c')

        # 가로 막대
        bars = ax.barh(range(len(valid)), valid[avg_col], color=colors, edgecolor='black')

        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels(valid['filter_name'], fontsize=9)
        ax.set_xlabel('평균 수익률 (%)')
        ax.set_title(f'{period}일 보유 수익률', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)

        # 값 표시
        for bar, val in zip(bars, valid[avg_col]):
            x_pos = val + 0.05 if val >= 0 else val - 0.05
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}%', va='center', ha=ha, fontsize=8)

    plt.suptitle('Stage 3-2: 조합 필터별 보유 기간 수익률', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(results_dir, 'stage3_step2_period_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"저장: {save_path}")
    plt.close()

    # 2. 시그널 수 vs 수익률 산점도
    fig, ax = plt.subplots(figsize=(10, 6))

    valid = results_df[results_df['signal_count'] > 0].copy()

    scatter = ax.scatter(
        valid['signal_count'],
        valid['1d_avg_return'],
        s=valid['1d_win_rate'] * 3,  # 승률에 비례한 크기
        c=valid['1d_profit_factor'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )

    # 라벨
    for _, row in valid.iterrows():
        ax.annotate(
            row['filter_name'][:15],
            (row['signal_count'], row['1d_avg_return']),
            fontsize=8,
            alpha=0.8
        )

    ax.set_xlabel('시그널 수')
    ax.set_ylabel('1일 평균 수익률 (%)')
    ax.set_title('시그널 수 vs 수익률 (크기=승률, 색상=손익비)', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.colorbar(scatter, label='손익비')
    plt.tight_layout()

    save_path = os.path.join(results_dir, 'stage3_step2_signal_vs_return.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"저장: {save_path}")
    plt.close()


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("Stage 3-2: 수급 필터 조합 백테스팅")
    print("=" * 80 + "\n")

    # 백테스트 실행
    backtest = ComboFilterBacktest()
    merged_df = backtest.load_and_merge_data()
    results_df = backtest.run_backtest(merged_df)

    # Alpha 계산
    results_df = backtest.calculate_alpha(results_df)

    # 결과 저장
    backtest.save_results(results_df)

    # 요약 출력
    backtest.print_summary(results_df)

    # 시각화
    print("\n시각화 생성 중...")
    visualize_results(results_df)

    print("\nStage 3-2 완료!")

    return results_df


if __name__ == '__main__':
    results = main()
