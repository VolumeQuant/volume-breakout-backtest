# -*- coding: utf-8 -*-
"""
Stage 5: 최종 실전 전략 - 완화된 기준 + 시장 필터 + Trailing Stop

목표:
1. 완화된 거래량 기준 (VR 4.0, ZS 2.0)으로 빈도 확보
2. 시장 필터 추가 (지수 > 20일 이평선)
3. Trailing Stop -3.5% 매도 로직 구현
4. 가격 임계치 민감도 테스트 (5%, 7%, 10%)
5. 2025년 OOS 최종 검증
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import timedelta
import FinanceDataReader as fdr

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class FinalStrategyBacktest:
    """최종 실전 전략 백테스팅"""

    def __init__(self, start_date='2022-01-01', end_date='2025-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = 'data'
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_index_data(self):
        """지수 데이터 로드 (KOSPI 200, KOSDAQ Composite)"""
        print("=" * 80)
        print("지수 데이터 수집")
        print("=" * 80)

        # KOSPI 200
        print("KOSPI 200 데이터 수집 중...")
        kospi200 = fdr.DataReader('KS200', self.start_date, self.end_date)
        kospi200['MA20'] = kospi200['Close'].rolling(window=20).mean()
        kospi200 = kospi200[['Close', 'MA20']].reset_index()
        kospi200.columns = ['Date', 'KOSPI200', 'KOSPI200_MA20']
        print(f"  KOSPI 200: {len(kospi200)}개 행")

        # KOSDAQ Composite (KQ11)
        print("KOSDAQ Composite 데이터 수집 중...")
        kosdaq = fdr.DataReader('KQ11', self.start_date, self.end_date)
        kosdaq['MA20'] = kosdaq['Close'].rolling(window=20).mean()
        kosdaq = kosdaq[['Close', 'MA20']].reset_index()
        kosdaq.columns = ['Date', 'KOSDAQ', 'KOSDAQ_MA20']
        print(f"  KOSDAQ Composite: {len(kosdaq)}개 행\n")

        return kospi200, kosdaq

    def load_data(self):
        """가격 및 수급 데이터 로드"""
        print("=" * 80)
        print("주가 및 수급 데이터 로드")
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

        # Inst_Ratio 계산 (거래대금 대비 기관 순매수 비율)
        merged['거래대금'] = merged['Close'] * merged['Volume']
        merged['Inst_Ratio'] = (
            (merged['금융투자'].fillna(0) + merged['연기금'].fillna(0)) /
            (merged['거래대금'] + 1) * 100
        )

        print(f"병합 완료: {len(merged):,}개 행\n")

        return merged

    def add_market_filter(self, df, kospi200, kosdaq):
        """시장 필터 추가 (지수 > 20일 이평선)"""
        print("시장 필터 추가 중...")

        # 지수 데이터 병합
        df = pd.merge(df, kospi200, on='Date', how='left')
        df = pd.merge(df, kosdaq, on='Date', how='left')

        # 시장 필터 적용
        df['Market_Filter'] = False
        df.loc[
            (df['Market'] == 'KOSPI') &
            (df['KOSPI200'] > df['KOSPI200_MA20']),
            'Market_Filter'
        ] = True
        df.loc[
            (df['Market'] == 'KOSDAQ') &
            (df['KOSDAQ'] > df['KOSDAQ_MA20']),
            'Market_Filter'
        ] = True

        filter_rate = df['Market_Filter'].sum() / len(df) * 100
        print(f"  시장 필터 통과: {df['Market_Filter'].sum():,}개 행 ({filter_rate:.1f}%)\n")

        return df

    def trailing_stop_simulation(self, signals_df):
        """Trailing Stop -3.5% 매도 로직 시뮬레이션"""
        print("Trailing Stop 시뮬레이션 중...")

        results = []

        for idx, row in signals_df.iterrows():
            entry_date = row['Date']
            entry_price = row['Close']
            code = row['Code']

            # 향후 10거래일 데이터 가져오기
            future_data = signals_df[
                (signals_df['Code'] == code) &
                (signals_df['Date'] > entry_date)
            ].head(10).copy()

            if len(future_data) == 0:
                continue

            # Trailing Stop 계산
            highest_price = entry_price
            exit_date = None
            exit_price = None
            exit_reason = 'MaxHold'  # 기본값: 10일 만기

            for i, (_, day) in enumerate(future_data.iterrows(), 1):
                current_price = day['Close']
                highest_price = max(highest_price, current_price)

                # Trailing Stop 발동 확인 (고점 대비 -3.5%)
                if current_price <= highest_price * 0.965:
                    exit_date = day['Date']
                    exit_price = current_price
                    exit_reason = 'TrailingStop'
                    break

            # Exit 없으면 10일 후 종가로 청산
            if exit_date is None:
                exit_date = future_data.iloc[-1]['Date']
                exit_price = future_data.iloc[-1]['Close']

            # 수익률 계산
            return_pct = (exit_price - entry_price) / entry_price * 100
            hold_days = (exit_date - entry_date).days

            results.append({
                'Date': entry_date,
                'Code': code,
                'Name': row['Name'],
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Exit_Date': exit_date,
                'Hold_Days': hold_days,
                'Return': return_pct,
                'Exit_Reason': exit_reason
            })

        results_df = pd.DataFrame(results)
        print(f"  완료: {len(results_df)}개 거래 시뮬레이션\n")

        return results_df

    def backtest_strategy(self, df, vr_th=4.0, zs_th=2.0, price_th=10.0,
                          inst_ratio_th=3.0, use_market_filter=True, use_trailing_stop=True):
        """
        전략 백테스팅

        Parameters:
        -----------
        vr_th : float
            Volume Ratio 임계치
        zs_th : float
            Z-Score 임계치
        price_th : float
            당일 수익률 임계치 (%)
        inst_ratio_th : float
            Inst_Ratio 임계치 (%)
        use_market_filter : bool
            시장 필터 사용 여부
        use_trailing_stop : bool
            Trailing Stop 사용 여부
        """
        # 기본 필터
        signals = df[
            (df['Volume_Ratio'] >= vr_th) &
            (df['Z_Score'] >= zs_th) &
            (df['Return_0D'] >= price_th) &
            (df['Return_0D'] < 30.0) &  # 상한가 제외
            (df['개인'] < 0) &
            (df['연기금'] < 0) &
            (df['Inst_Ratio'] >= inst_ratio_th)
        ].copy()

        # 시장 필터 적용
        if use_market_filter:
            signals = signals[signals['Market_Filter'] == True].copy()

        if len(signals) == 0:
            return None, None

        # Trailing Stop 적용
        if use_trailing_stop:
            trade_results = self.trailing_stop_simulation(signals)

            # 빈 결과 처리
            if len(trade_results) == 0:
                return None, None

            # 통계 계산
            stats = {
                'vr_th': vr_th,
                'zs_th': zs_th,
                'price_th': price_th,
                'signal_count': len(trade_results),
                'monthly_signals': len(trade_results) / signals['Date'].dt.to_period('M').nunique() if signals['Date'].dt.to_period('M').nunique() > 0 else 0,
                'avg_return': trade_results['Return'].mean(),
                'median_return': trade_results['Return'].median(),
                'win_rate': (trade_results['Return'] > 0).sum() / len(trade_results) * 100,
                'avg_hold_days': trade_results['Hold_Days'].mean(),
                'trailing_stop_rate': (trade_results['Exit_Reason'] == 'TrailingStop').sum() / len(trade_results) * 100,
                'std': trade_results['Return'].std(),
                'sharpe': trade_results['Return'].mean() / trade_results['Return'].std() * np.sqrt(252) if trade_results['Return'].std() > 0 else 0,
                'max_loss': trade_results['Return'].min(),
                'max_gain': trade_results['Return'].max()
            }

            # 손익비
            profits = trade_results[trade_results['Return'] > 0]['Return']
            losses = trade_results[trade_results['Return'] < 0]['Return'].abs()
            stats['profit_factor'] = profits.mean() / losses.mean() if len(losses) > 0 and len(profits) > 0 else np.inf

        else:
            # 기존 방식 (10일 고정 보유)
            stats = {
                'vr_th': vr_th,
                'zs_th': zs_th,
                'price_th': price_th,
                'signal_count': len(signals),
                'monthly_signals': len(signals) / signals['Date'].dt.to_period('M').nunique(),
                'avg_return': signals['Return_10D'].mean(),
                'median_return': signals['Return_10D'].median(),
                'win_rate': (signals['Return_10D'] > 0).sum() / len(signals) * 100,
                'avg_hold_days': 10,
                'trailing_stop_rate': 0,
                'std': signals['Return_10D'].std(),
                'sharpe': signals['Return_10D'].mean() / signals['Return_10D'].std() * np.sqrt(252) if signals['Return_10D'].std() > 0 else 0,
                'max_loss': signals['Return_10D'].min(),
                'max_gain': signals['Return_10D'].max()
            }

            profits = signals[signals['Return_10D'] > 0]['Return_10D']
            losses = signals[signals['Return_10D'] < 0]['Return_10D'].abs()
            stats['profit_factor'] = profits.mean() / losses.mean() if len(losses) > 0 else np.inf

            trade_results = None

        return stats, trade_results

    def price_threshold_sensitivity(self, df, price_thresholds=[5, 7, 10]):
        """가격 임계치 민감도 테스트"""
        print("=" * 80)
        print("가격 임계치 민감도 분석")
        print("=" * 80)
        print(f"테스트 케이스: {price_thresholds}\n")

        results = []

        for price_th in price_thresholds:
            print(f"[Case] 당일 수익률 >= {price_th}%")

            stats, _ = self.backtest_strategy(
                df,
                vr_th=4.0,
                zs_th=2.0,
                price_th=price_th,
                inst_ratio_th=3.0,
                use_market_filter=True,
                use_trailing_stop=True
            )

            if stats:
                results.append(stats)
                print(f"  시그널: {stats['signal_count']}개 (월 {stats['monthly_signals']:.1f}건)")
                print(f"  평균 수익률: {stats['avg_return']:.2f}%")
                print(f"  승률: {stats['win_rate']:.1f}%")
                print(f"  평균 보유일: {stats['avg_hold_days']:.1f}일")
                print(f"  Trailing Stop 발동: {stats['trailing_stop_rate']:.1f}%\n")
            else:
                print(f"  시그널 없음\n")

        results_df = pd.DataFrame(results)
        return results_df

    def oos_validation_2025(self, df):
        """2025년 OOS 검증"""
        print("=" * 80)
        print("2025년 Out-of-Sample 검증")
        print("=" * 80)

        # In-Sample (2022-2024)
        df_is = df[df['Date'].dt.year.isin([2022, 2023, 2024])].copy()

        # OOS (2025)
        df_oos = df[df['Date'].dt.year == 2025].copy()

        print(f"In-Sample: {len(df_is):,}개 행 (2022-2024)")
        print(f"OOS: {len(df_oos):,}개 행 (2025)\n")

        # 최적 전략으로 백테스팅
        print("[In-Sample 성과]")
        stats_is, trades_is = self.backtest_strategy(
            df_is, vr_th=4.0, zs_th=2.0, price_th=7.0,
            inst_ratio_th=3.0, use_market_filter=True, use_trailing_stop=True
        )

        if stats_is:
            print(f"  시그널: {stats_is['signal_count']}개 (월 {stats_is['monthly_signals']:.1f}건)")
            print(f"  평균 수익률: {stats_is['avg_return']:.2f}%")
            print(f"  승률: {stats_is['win_rate']:.1f}%")
            print(f"  손익비: {stats_is['profit_factor']:.2f}\n")

        print("[Out-of-Sample 성과 (2025)]")
        stats_oos, trades_oos = self.backtest_strategy(
            df_oos, vr_th=4.0, zs_th=2.0, price_th=7.0,
            inst_ratio_th=3.0, use_market_filter=True, use_trailing_stop=True
        )

        if stats_oos:
            print(f"  시그널: {stats_oos['signal_count']}개 (월 {stats_oos['monthly_signals']:.1f}건)")
            print(f"  평균 수익률: {stats_oos['avg_return']:.2f}%")
            print(f"  승률: {stats_oos['win_rate']:.1f}%")
            print(f"  손익비: {stats_oos['profit_factor']:.2f}\n")

        comparison = pd.DataFrame({
            'Period': ['In-Sample (22-24)', 'OOS (2025)'],
            'Signals': [stats_is['signal_count'] if stats_is else 0,
                       stats_oos['signal_count'] if stats_oos else 0],
            'Monthly_Signals': [stats_is['monthly_signals'] if stats_is else 0,
                               stats_oos['monthly_signals'] if stats_oos else 0],
            'Avg_Return': [stats_is['avg_return'] if stats_is else 0,
                          stats_oos['avg_return'] if stats_oos else 0],
            'Win_Rate': [stats_is['win_rate'] if stats_is else 0,
                        stats_oos['win_rate'] if stats_oos else 0],
            'Profit_Factor': [stats_is['profit_factor'] if stats_is else 0,
                             stats_oos['profit_factor'] if stats_oos else 0]
        })

        return comparison, trades_is, trades_oos

    def visualize_results(self, sensitivity_df, oos_comparison):
        """결과 시각화"""
        print("=" * 80)
        print("결과 시각화")
        print("=" * 80)

        # 1. 가격 임계치 민감도
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('가격 임계치 민감도 분석 (VR 4.0, ZS 2.0 + 시장 필터 + Trailing Stop)',
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

        # 평균 수익률
        ax = axes[0, 1]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['avg_return'],
               color='green', alpha=0.8, edgecolor='black')
        ax.set_title('평균 수익률', fontsize=13, fontweight='bold')
        ax.set_ylabel('수익률 (%)', fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        for i, v in enumerate(sensitivity_df['avg_return']):
            ax.text(i, v, f'{v:.2f}%', ha='center', va='bottom' if v > 0 else 'top',
                   fontsize=10, fontweight='bold')

        # 승률
        ax = axes[1, 0]
        ax.bar(sensitivity_df['price_th'].astype(str) + '%', sensitivity_df['win_rate'],
               color='orange', alpha=0.8, edgecolor='black')
        ax.set_title('승률', fontsize=13, fontweight='bold')
        ax.set_ylabel('승률 (%)', fontsize=11)
        ax.axhline(y=50, color='red', linestyle='--', label='50% 기준')
        ax.legend()
        ax.set_ylim([0, 100])
        for i, v in enumerate(sensitivity_df['win_rate']):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 평균 보유일 vs Trailing Stop 발동률
        ax = axes[1, 1]
        x = np.arange(len(sensitivity_df))
        width = 0.35
        ax.bar(x - width/2, sensitivity_df['avg_hold_days'], width, label='평균 보유일',
               color='purple', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, sensitivity_df['trailing_stop_rate'], width, label='Trailing Stop 발동률',
                color='red', alpha=0.8)
        ax.set_title('보유 기간 및 Trailing Stop 발동률', fontsize=13, fontweight='bold')
        ax.set_ylabel('평균 보유일', fontsize=11)
        ax2.set_ylabel('Trailing Stop 발동률 (%)', fontsize=11, color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(sensitivity_df['price_th'].astype(str) + '%')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        filepath = f'{self.results_dir}/stage5_price_sensitivity.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  저장: {filepath}")

        # 2. OOS 비교
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('2025년 Out-of-Sample 검증 결과', fontsize=16, fontweight='bold')

        metrics = [
            ('Monthly_Signals', '월 평균 시그널', 'steelblue'),
            ('Avg_Return', '평균 수익률 (%)', 'green'),
            ('Win_Rate', '승률 (%)', 'orange')
        ]

        for idx, (metric, title, color) in enumerate(metrics):
            ax = axes[idx]
            ax.bar(oos_comparison['Period'], oos_comparison[metric],
                   color=color, alpha=0.8, edgecolor='black')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylabel(title, fontsize=11)
            for i, v in enumerate(oos_comparison[metric]):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filepath = f'{self.results_dir}/stage5_oos_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  저장: {filepath}\n")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("Stage 5: 최종 실전 전략")
    print("=" * 80)
    print("VR 4.0, ZS 2.0 + 시장 필터 + Trailing Stop -3.5%")
    print("=" * 80 + "\n")

    backtest = FinalStrategyBacktest()

    # 1. 지수 데이터 로드
    kospi200, kosdaq = backtest.load_index_data()

    # 2. 주가 및 수급 데이터 로드
    df = backtest.load_data()

    # 3. 시장 필터 추가
    df = backtest.add_market_filter(df, kospi200, kosdaq)

    # 4. 가격 임계치 민감도 분석
    sensitivity_df = backtest.price_threshold_sensitivity(df, price_thresholds=[5, 7, 10])
    sensitivity_df.to_csv(f'{backtest.results_dir}/stage5_price_sensitivity.csv',
                          index=False, encoding='utf-8-sig')
    print(f"결과 저장: {backtest.results_dir}/stage5_price_sensitivity.csv\n")

    # 5. 2025년 OOS 검증
    oos_comparison, trades_is, trades_oos = backtest.oos_validation_2025(df)
    oos_comparison.to_csv(f'{backtest.results_dir}/stage5_oos_validation.csv',
                          index=False, encoding='utf-8-sig')
    print(f"결과 저장: {backtest.results_dir}/stage5_oos_validation.csv\n")

    # 6. 시각화
    backtest.visualize_results(sensitivity_df, oos_comparison)

    # 7. 최종 요약
    print("=" * 80)
    print("최종 전략 요약")
    print("=" * 80)

    optimal_case = sensitivity_df.loc[sensitivity_df['monthly_signals'].idxmax()]
    print(f"\n[최적 파라미터: 당일 수익률 >= {optimal_case['price_th']:.0f}%]")
    print(f"  조건: VR>=4.0 & ZS>=2.0 & Price>={optimal_case['price_th']:.0f}% & 개인<0 & 연기금<0 & Inst_Ratio>=3%")
    print(f"  시장 필터: ON (지수 > 20일 이평선)")
    print(f"  매도: Trailing Stop -3.5% (최대 10일)")
    print(f"\n[성과]")
    print(f"  월 시그널: {optimal_case['monthly_signals']:.1f}건")
    print(f"  평균 수익률: {optimal_case['avg_return']:.2f}%")
    print(f"  승률: {optimal_case['win_rate']:.1f}%")
    print(f"  평균 보유: {optimal_case['avg_hold_days']:.1f}일")
    print(f"  손익비: {optimal_case['profit_factor']:.2f}")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[오류 발생] {str(e)}")
        import traceback
        traceback.print_exc()
