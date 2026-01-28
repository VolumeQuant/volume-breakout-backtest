# -*- coding: utf-8 -*-
"""
Stage 4-1: Out-of-Sample (OOS) 검증

목표:
- 2025년 데이터를 활용한 전략 강건성 평가
- In-Sample (2022-2024) vs Out-of-Sample (2025) 성과 비교

검증 조건 (Best Case):
- VR ≥ 6.5, ZS ≥ 3.0, 당일 ≥ 10%
- 개인 순매도 (< 0)
- 연기금 순매도 (< 0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import time
from datetime import datetime

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class OOSValidation:
    """Out-of-Sample 검증 클래스"""

    def __init__(self):
        self.data_dir = 'data'
        self.results_dir = 'results'

        # 기간 설정
        self.in_sample_start = '2022-01-01'
        self.in_sample_end = '2024-12-31'
        self.oos_start = '2025-01-01'
        self.oos_end = '2025-12-31'

        # Best Case 필터 조건
        self.vr_threshold = 6.5
        self.zs_threshold = 3.0
        self.price_threshold = 10.0
        self.price_upper_limit = 30.0

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def collect_extended_data(self):
        """2025년까지 확장된 데이터 수집"""
        import FinanceDataReader as fdr
        from pykrx import stock

        print("=" * 80)
        print("[1단계] 2025년 데이터 수집")
        print("=" * 80)

        # 종목 리스트 로드
        stock_list = pd.read_csv(f'{self.data_dir}/stock_list.csv')
        print(f"대상 종목: {len(stock_list)}개")

        # 1. 가격 데이터 수집 (2025년)
        print("\n[1-1] 2025년 가격 데이터 수집...")
        price_2025 = []
        total = len(stock_list)

        for idx, row in stock_list.iterrows():
            code = str(row['Code']).zfill(6)
            name = row['Name']
            market = row['Market']

            if (idx + 1) % 50 == 0:
                print(f"  진행: {idx+1}/{total}")

            try:
                df = fdr.DataReader(code, '2025-01-01', '2025-12-31')
                if df is not None and len(df) > 0:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df['Code'] = code
                    df['Name'] = name
                    df['Market'] = market
                    df = df[df['Volume'] > 0]
                    price_2025.append(df)
            except Exception as e:
                pass

            time.sleep(0.05)

        if len(price_2025) > 0:
            price_2025_df = pd.concat(price_2025, ignore_index=False)
            price_2025_df = price_2025_df.reset_index()
            price_2025_df = price_2025_df.rename(columns={'index': 'Date'})
            print(f"  2025년 가격 데이터: {len(price_2025_df):,}개 행, {len(price_2025)}개 종목")
        else:
            print("  [오류] 2025년 가격 데이터 수집 실패")
            return None, None

        # 2. 수급 데이터 수집 (2025년)
        print("\n[1-2] 2025년 수급 데이터 수집...")
        flow_2025 = []

        for idx, row in stock_list.iterrows():
            code = str(row['Code']).zfill(6)
            name = row['Name']
            market = row['Market']

            if (idx + 1) % 50 == 0:
                print(f"  진행: {idx+1}/{total}")

            try:
                df = stock.get_market_trading_value_by_date(
                    '20250101', '20251231', code, detail=True
                )
                if df is not None and len(df) > 0:
                    df = df.reset_index()
                    df = df.rename(columns={'날짜': 'Date'})
                    df['Code'] = code
                    df['Name'] = name
                    df['Market'] = market
                    flow_2025.append(df)
            except Exception as e:
                pass

            time.sleep(0.1)

        if len(flow_2025) > 0:
            flow_2025_df = pd.concat(flow_2025, ignore_index=True)
            print(f"  2025년 수급 데이터: {len(flow_2025_df):,}개 행, {len(flow_2025)}개 종목")
        else:
            print("  [오류] 2025년 수급 데이터 수집 실패")
            return price_2025_df, None

        return price_2025_df, flow_2025_df

    def load_existing_data(self):
        """기존 2022-2024 데이터 로드"""
        print("\n[2단계] 기존 데이터 로드 (2022-2024)")
        print("-" * 80)

        # 가격 데이터
        price_df = pd.read_csv(
            f'{self.data_dir}/stock_data_with_indicators.csv',
            parse_dates=[0], low_memory=False
        )
        price_df.columns = ['Date'] + list(price_df.columns[1:])
        price_df['Code'] = price_df['Code'].astype(str).str.zfill(6)
        print(f"가격 데이터: {len(price_df):,}개 행")

        # 수급 데이터
        flow_df = pd.read_csv(
            f'{self.data_dir}/investor_flow_data_v2.csv',
            parse_dates=['Date'], low_memory=False
        )
        flow_df['Code'] = flow_df['Code'].astype(str).str.zfill(6)
        print(f"수급 데이터: {len(flow_df):,}개 행")

        return price_df, flow_df

    def calculate_indicators(self, price_df):
        """기술적 지표 계산 (Volume_Ratio, Z_Score, Returns)"""
        print("\n지표 계산 중...")

        result = []

        for code in price_df['Code'].unique():
            df = price_df[price_df['Code'] == code].copy()
            df = df.sort_values('Date')

            if len(df) < 60:
                continue

            # Volume_Ratio: 당일 거래량 / 20일 평균 거래량
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']

            # Z_Score: (당일 거래량 - 60일 평균) / 60일 표준편차
            df['Volume_MA60'] = df['Volume'].rolling(window=60).mean()
            df['Volume_STD60'] = df['Volume'].rolling(window=60).std()
            df['Z_Score'] = (df['Volume'] - df['Volume_MA60']) / df['Volume_STD60']

            # 당일 수익률
            df['Return_0D'] = (df['Close'] - df['Open']) / df['Open'] * 100

            # 미래 수익률 (1일, 3일, 5일, 10일)
            df['Next_Open'] = df['Open'].shift(-1)
            for period in [1, 3, 5, 10]:
                future_close = df['Close'].shift(-period)
                df[f'Return_{period}D'] = (future_close - df['Next_Open']) / df['Next_Open'] * 100

            result.append(df)

        return pd.concat(result, ignore_index=True) if result else None

    def merge_and_prepare_data(self, price_df, flow_df, price_2025, flow_2025):
        """데이터 병합 및 준비"""
        print("\n[3단계] 데이터 병합 및 준비")
        print("-" * 80)

        # 날짜/코드 형식 통일 (기존 데이터)
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.normalize()
        price_df['Code'] = price_df['Code'].astype(str).str.zfill(6)
        flow_df['Date'] = pd.to_datetime(flow_df['Date']).dt.normalize()
        flow_df['Code'] = flow_df['Code'].astype(str).str.zfill(6)

        # 2025년 가격 데이터에 지표 계산
        if price_2025 is not None:
            price_2025['Date'] = pd.to_datetime(price_2025['Date']).dt.normalize()
            price_2025['Code'] = price_2025['Code'].astype(str).str.zfill(6)
            price_2025_with_ind = self.calculate_indicators(price_2025)
            if price_2025_with_ind is not None:
                print(f"2025년 지표 계산 완료: {len(price_2025_with_ind):,}개 행")

                # 기존 데이터와 합치기
                all_price = pd.concat([price_df, price_2025_with_ind], ignore_index=True)
                all_price = all_price.drop_duplicates(subset=['Date', 'Code'], keep='last')
            else:
                all_price = price_df
        else:
            all_price = price_df

        # 수급 데이터 합치기
        if flow_2025 is not None:
            flow_2025['Date'] = pd.to_datetime(flow_2025['Date']).dt.normalize()
            flow_2025['Code'] = flow_2025['Code'].astype(str).str.zfill(6)
            all_flow = pd.concat([flow_df, flow_2025], ignore_index=True)
            all_flow = all_flow.drop_duplicates(subset=['Date', 'Code'], keep='last')
        else:
            all_flow = flow_df

        print(f"전체 가격 데이터: {len(all_price):,}개 행")
        print(f"전체 수급 데이터: {len(all_flow):,}개 행")

        # 병합
        merged = pd.merge(
            all_price,
            all_flow[['Date', 'Code', '금융투자', '연기금', '개인', '외국인']],
            on=['Date', 'Code'],
            how='left'
        )

        # 수급 데이터 병합 확인
        flow_coverage = merged['개인'].notna().sum() / len(merged) * 100
        print(f"수급 데이터 커버리지: {flow_coverage:.1f}%")

        # 당일 수익률 계산 (없거나 NaN인 경우)
        if 'Return_0D' not in merged.columns:
            merged['Return_0D'] = (merged['Close'] - merged['Open']) / merged['Open'] * 100
        else:
            # NaN인 행만 계산
            mask = merged['Return_0D'].isna()
            if mask.any():
                merged.loc[mask, 'Return_0D'] = (
                    (merged.loc[mask, 'Close'] - merged.loc[mask, 'Open']) /
                    merged.loc[mask, 'Open'] * 100
                )
                print(f"Return_0D 계산 (NaN 채움): {mask.sum():,}개 행")

        print(f"병합 완료: {len(merged):,}개 행")

        # 기간별 분리
        in_sample = merged[
            (merged['Date'] >= self.in_sample_start) &
            (merged['Date'] <= self.in_sample_end)
        ].copy()

        oos = merged[
            (merged['Date'] >= self.oos_start) &
            (merged['Date'] <= self.oos_end)
        ].copy()

        # 기간별 수급 커버리지 확인
        is_coverage = in_sample['개인'].notna().sum() / len(in_sample) * 100 if len(in_sample) > 0 else 0
        oos_coverage = oos['개인'].notna().sum() / len(oos) * 100 if len(oos) > 0 else 0

        print(f"\nIn-Sample (2022-2024): {len(in_sample):,}개 행 (수급: {is_coverage:.1f}%)")
        print(f"Out-of-Sample (2025): {len(oos):,}개 행 (수급: {oos_coverage:.1f}%)")

        return merged, in_sample, oos

    def apply_best_case_filter(self, df, verbose=False):
        """Best Case 필터 적용"""
        # 기본 조건
        base = df[
            (df['Volume_Ratio'] >= self.vr_threshold) &
            (df['Z_Score'] >= self.zs_threshold) &
            (df['Return_0D'] >= self.price_threshold) &
            (df['Return_0D'] < self.price_upper_limit)
        ].copy()

        if verbose:
            print(f"  기본 필터 통과: {len(base)}개")
            if len(base) > 0:
                has_flow = base['개인'].notna().sum()
                print(f"  수급 데이터 있음: {has_flow}개")
                if has_flow > 0:
                    ind_sell = (base['개인'] < 0).sum()
                    pension_sell = (base['연기금'] < 0).sum()
                    print(f"  개인 순매도: {ind_sell}개, 연기금 순매도: {pension_sell}개")

        # 수급 조건: 개인(-) + 연기금(-)
        best_case = base[
            (base['개인'].notna()) &
            (base['연기금'].notna()) &
            (base['개인'] < 0) &
            (base['연기금'] < 0)
        ].copy()

        return best_case

    def calculate_statistics(self, signals, label):
        """통계 계산"""
        if len(signals) == 0:
            return {
                'period': label,
                'signal_count': 0,
                'monthly_signals': 0
            }

        # 기본 통계
        signal_count = len(signals)
        months = signals['Date'].dt.to_period('M').nunique()
        monthly = signal_count / months if months > 0 else 0

        stats = {
            'period': label,
            'signal_count': signal_count,
            'monthly_signals': round(monthly, 2)
        }

        # 보유 기간별 통계
        for period in [1, 3, 5, 10]:
            col = f'Return_{period}D'
            if col not in signals.columns:
                continue

            returns = signals[col].dropna()
            if len(returns) == 0:
                continue

            avg_ret = returns.mean()
            median_ret = returns.median()
            win_rate = (returns > 0).sum() / len(returns) * 100
            max_ret = returns.max()
            min_ret = returns.min()

            # 손익비
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
            pf = avg_profit / avg_loss if avg_loss > 0 else float('inf')

            stats[f'{period}d_avg_return'] = round(avg_ret, 4)
            stats[f'{period}d_median_return'] = round(median_ret, 4)
            stats[f'{period}d_win_rate'] = round(win_rate, 2)
            stats[f'{period}d_max'] = round(max_ret, 2)
            stats[f'{period}d_mdd'] = round(min_ret, 2)
            stats[f'{period}d_profit_factor'] = round(pf, 3) if pf != float('inf') else 999

        return stats

    def calculate_yearly_stats(self, merged_df):
        """연도별 통계 계산"""
        yearly_stats = []

        for year in [2022, 2023, 2024, 2025]:
            year_data = merged_df[merged_df['Date'].dt.year == year]

            if len(year_data) == 0:
                continue

            signals = self.apply_best_case_filter(year_data)
            stats = self.calculate_statistics(signals, str(year))
            stats['year'] = year
            yearly_stats.append(stats)

        return pd.DataFrame(yearly_stats)

    def run_validation(self):
        """OOS 검증 실행"""
        print("\n" + "=" * 80)
        print("Stage 4-1: Out-of-Sample (OOS) 검증")
        print("=" * 80)
        print(f"In-Sample: {self.in_sample_start} ~ {self.in_sample_end}")
        print(f"Out-of-Sample: {self.oos_start} ~ {self.oos_end}")
        print(f"필터: VR≥{self.vr_threshold}, ZS≥{self.zs_threshold}, "
              f"Price≥{self.price_threshold}%, 개인(-), 연기금(-)")

        # 1. 기존 데이터 로드
        price_df, flow_df = self.load_existing_data()

        # 2. 2025년 데이터 수집
        price_2025, flow_2025 = self.collect_extended_data()

        # 3. 데이터 병합
        merged, in_sample, oos = self.merge_and_prepare_data(
            price_df, flow_df, price_2025, flow_2025
        )

        # 4. Best Case 필터 적용
        print("\n[4단계] Best Case 필터 적용")
        print("-" * 80)

        print("\nIn-Sample 필터링:")
        is_signals = self.apply_best_case_filter(in_sample, verbose=True)
        print(f"  → 최종 시그널: {len(is_signals)}개")

        print("\nOOS 필터링:")
        oos_signals = self.apply_best_case_filter(oos, verbose=True)
        print(f"  → 최종 시그널: {len(oos_signals)}개")

        # 5. 통계 계산
        print("\n[5단계] 성과 비교")
        print("-" * 80)

        is_stats = self.calculate_statistics(is_signals, 'In-Sample (2022-2024)')
        oos_stats = self.calculate_statistics(oos_signals, 'Out-of-Sample (2025)')

        results_df = pd.DataFrame([is_stats, oos_stats])

        # 연도별 통계
        yearly_df = self.calculate_yearly_stats(merged)

        # 6. 결과 출력
        self.print_comparison(is_stats, oos_stats)

        # 7. 저장
        results_df.to_csv(
            f'{self.results_dir}/stage4_oos_validation_results.csv',
            index=False, encoding='utf-8-sig'
        )
        print(f"\n저장: {self.results_dir}/stage4_oos_validation_results.csv")

        yearly_df.to_csv(
            f'{self.results_dir}/stage4_yearly_stats.csv',
            index=False, encoding='utf-8-sig'
        )

        # 8. 시각화
        print("\n[6단계] 시각화")
        print("-" * 80)
        self.visualize_results(merged, is_signals, oos_signals, yearly_df)

        # 9. 결론
        self.print_conclusion(is_stats, oos_stats)

        return results_df, yearly_df, merged

    def print_comparison(self, is_stats, oos_stats):
        """성과 비교 출력"""
        print("\n" + "=" * 80)
        print("In-Sample vs Out-of-Sample 성과 비교")
        print("=" * 80)

        print(f"\n{'지표':<20} | {'In-Sample':>15} | {'OOS (2025)':>15} | {'변화':>10}")
        print("-" * 70)

        # 시그널 수
        is_sig = is_stats.get('signal_count', 0)
        oos_sig = oos_stats.get('signal_count', 0)
        print(f"{'시그널 수':<20} | {is_sig:>15} | {oos_sig:>15} | {'-':>10}")

        # 월평균 시그널
        is_monthly = is_stats.get('monthly_signals', 0)
        oos_monthly = oos_stats.get('monthly_signals', 0)
        print(f"{'월평균 시그널':<20} | {is_monthly:>14.1f}건 | {oos_monthly:>14.1f}건 | {'-':>10}")

        # 수익률 비교
        for period in [1, 10]:
            is_ret = is_stats.get(f'{period}d_avg_return', 0)
            oos_ret = oos_stats.get(f'{period}d_avg_return', 0)
            change = ((oos_ret - is_ret) / abs(is_ret) * 100) if is_ret != 0 else 0
            change_str = f"{change:+.1f}%" if is_ret != 0 else "-"
            print(f"{f'{period}일 수익률':<20} | {is_ret:>14.3f}% | {oos_ret:>14.3f}% | {change_str:>10}")

        # 승률 비교
        for period in [1, 10]:
            is_wr = is_stats.get(f'{period}d_win_rate', 0)
            oos_wr = oos_stats.get(f'{period}d_win_rate', 0)
            change = oos_wr - is_wr
            change_str = f"{change:+.1f}%p"
            print(f"{f'{period}일 승률':<20} | {is_wr:>14.1f}% | {oos_wr:>14.1f}% | {change_str:>10}")

        # 손익비
        for period in [1, 10]:
            is_pf = is_stats.get(f'{period}d_profit_factor', 0)
            oos_pf = oos_stats.get(f'{period}d_profit_factor', 0)
            print(f"{f'{period}일 손익비':<20} | {is_pf:>15.2f} | {oos_pf:>15.2f} | {'-':>10}")

        # MDD
        is_mdd = is_stats.get('10d_mdd', 0)
        oos_mdd = oos_stats.get('10d_mdd', 0)
        print(f"{'10일 MDD':<20} | {is_mdd:>14.1f}% | {oos_mdd:>14.1f}% | {'-':>10}")

    def visualize_results(self, merged, is_signals, oos_signals, yearly_df):
        """시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 누적 수익률 곡선
        ax1 = axes[0, 0]

        # 전체 시그널 합치기
        all_signals = pd.concat([is_signals, oos_signals], ignore_index=True)
        all_signals = all_signals.sort_values('Date')

        if len(all_signals) > 0 and 'Return_1D' in all_signals.columns:
            all_signals['cum_return'] = (1 + all_signals['Return_1D'].fillna(0) / 100).cumprod() - 1
            all_signals['cum_return'] *= 100

            ax1.plot(all_signals['Date'], all_signals['cum_return'], 'b-', linewidth=1.5)
            ax1.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--',
                       linewidth=2, label='OOS 시작 (2025)')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            ax1.set_title('누적 수익률 (1일 보유 기준)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('날짜')
            ax1.set_ylabel('누적 수익률 (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 연도별 수익률 막대 그래프
        ax2 = axes[0, 1]

        if len(yearly_df) > 0:
            years = yearly_df['year'].astype(str).tolist()
            x = np.arange(len(years))
            width = 0.35

            ret_1d = yearly_df['1d_avg_return'].fillna(0).tolist()
            ret_10d = yearly_df['10d_avg_return'].fillna(0).tolist()

            bars1 = ax2.bar(x - width/2, ret_1d, width, label='1일', color='#3498db')
            bars2 = ax2.bar(x + width/2, ret_10d, width, label='10일', color='#e74c3c')

            # 값 표시
            for bar, val in zip(bars1, ret_1d):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', fontsize=9)
            for bar, val in zip(bars2, ret_10d):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', fontsize=9)

            ax2.set_xticks(x)
            ax2.set_xticklabels(years)
            ax2.set_title('연도별 평균 수익률', fontsize=12, fontweight='bold')
            ax2.set_ylabel('평균 수익률 (%)')
            ax2.legend()
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 3. 연도별 승률 막대 그래프
        ax3 = axes[1, 0]

        if len(yearly_df) > 0:
            wr_1d = yearly_df['1d_win_rate'].fillna(0).tolist()
            wr_10d = yearly_df['10d_win_rate'].fillna(0).tolist()

            bars1 = ax3.bar(x - width/2, wr_1d, width, label='1일', color='#27ae60')
            bars2 = ax3.bar(x + width/2, wr_10d, width, label='10일', color='#9b59b6')

            for bar, val in zip(bars1, wr_1d):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=9)
            for bar, val in zip(bars2, wr_10d):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=9)

            ax3.set_xticks(x)
            ax3.set_xticklabels(years)
            ax3.set_title('연도별 승률', fontsize=12, fontweight='bold')
            ax3.set_ylabel('승률 (%)')
            ax3.axhline(y=50, color='red', linestyle='--', linewidth=1, label='50% 기준선')
            ax3.legend()
            ax3.set_ylim(0, 100)

        # 4. 연도별 시그널 수
        ax4 = axes[1, 1]

        if len(yearly_df) > 0:
            signals = yearly_df['signal_count'].fillna(0).tolist()
            colors = ['#3498db', '#3498db', '#3498db', '#e74c3c']  # 2025년은 빨간색

            bars = ax4.bar(years, signals, color=colors[:len(years)], edgecolor='black')

            for bar, val in zip(bars, signals):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{int(val)}', ha='center', fontsize=10, fontweight='bold')

            ax4.set_title('연도별 시그널 수', fontsize=12, fontweight='bold')
            ax4.set_ylabel('시그널 수')

        plt.suptitle('Stage 4-1: Out-of-Sample 검증 결과', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = f'{self.results_dir}/stage4_oos_validation.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"저장: {save_path}")
        plt.close()

    def print_conclusion(self, is_stats, oos_stats):
        """결론 출력"""
        print("\n" + "=" * 80)
        print("결론: 이 전략은 2025년에도 유효한가?")
        print("=" * 80)

        # 핵심 지표 추출
        oos_ret_1d = oos_stats.get('1d_avg_return', 0)
        oos_ret_10d = oos_stats.get('10d_avg_return', 0)
        oos_wr_1d = oos_stats.get('1d_win_rate', 0)
        oos_wr_10d = oos_stats.get('10d_win_rate', 0)
        oos_signals = oos_stats.get('signal_count', 0)

        is_ret_1d = is_stats.get('1d_avg_return', 0)
        is_ret_10d = is_stats.get('10d_avg_return', 0)

        # 평가 기준
        issues = []
        positives = []

        # 1. 수익률 평가
        if oos_ret_1d > 0:
            positives.append(f"1일 수익률 양수 유지 ({oos_ret_1d:.3f}%)")
        else:
            issues.append(f"1일 수익률 음수 전환 ({oos_ret_1d:.3f}%)")

        if oos_ret_10d > 0:
            positives.append(f"10일 수익률 양수 유지 ({oos_ret_10d:.3f}%)")
        else:
            issues.append(f"10일 수익률 음수 전환 ({oos_ret_10d:.3f}%)")

        # 2. 승률 평가
        if oos_wr_1d >= 50:
            positives.append(f"1일 승률 50% 이상 ({oos_wr_1d:.1f}%)")
        else:
            issues.append(f"1일 승률 50% 미만 ({oos_wr_1d:.1f}%)")

        if oos_wr_10d >= 50:
            positives.append(f"10일 승률 50% 이상 ({oos_wr_10d:.1f}%)")
        else:
            issues.append(f"10일 승률 50% 미만 ({oos_wr_10d:.1f}%)")

        # 3. Alpha Decay 평가
        if is_ret_1d > 0:
            decay_1d = (oos_ret_1d - is_ret_1d) / is_ret_1d * 100
            if decay_1d < -50:
                issues.append(f"1일 Alpha Decay 심각 ({decay_1d:.1f}%)")
            elif decay_1d < 0:
                issues.append(f"1일 Alpha Decay 발생 ({decay_1d:.1f}%)")
            else:
                positives.append(f"1일 Alpha 유지/개선 ({decay_1d:+.1f}%)")

        # 4. 시그널 수 평가
        if oos_signals >= 10:
            positives.append(f"충분한 시그널 수 ({oos_signals}개)")
        elif oos_signals > 0:
            issues.append(f"시그널 수 적음 ({oos_signals}개)")
        else:
            issues.append("시그널 없음")

        # 결론 출력
        print("\n[긍정적 신호]")
        for p in positives:
            print(f"  ✓ {p}")

        print("\n[우려 사항]")
        for i in issues:
            print(f"  ✗ {i}")

        # 최종 판정
        print("\n[최종 판정]")
        if len(issues) == 0:
            print("  ★★★ 전략 유효: 2025년에도 In-Sample과 유사한 성과 유지")
        elif len(issues) <= 2 and oos_ret_1d > 0:
            print("  ★★☆ 전략 부분 유효: 성과 하락 있으나 여전히 수익 가능")
        else:
            print("  ★☆☆ 전략 재검토 필요: 유의미한 Alpha Decay 또는 성과 저하 발생")


def main():
    """메인 함수"""
    validator = OOSValidation()
    results_df, yearly_df, merged = validator.run_validation()

    print("\n" + "=" * 80)
    print("Stage 4-1 완료!")
    print("=" * 80)

    return results_df, yearly_df


if __name__ == '__main__':
    results, yearly = main()
