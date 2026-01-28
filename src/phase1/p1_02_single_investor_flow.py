# -*- coding: utf-8 -*-
"""
Stage 3-1: 데이터 인프라 정상화 및 수급 주체별 단독 영향력 정밀 검증

목표:
- 연기금 데이터 오류 해결 ('연기금등' → '연기금')
- 개인 수급 데이터 신규 추가
- 수급 주체별 단독 성과 백테스팅 (금융투자, 연기금, 개인매도)

분석 대상:
- Stage 2 최적 조건: VR >= 6.5, ZS >= 3.0, 당일 수익률 >= 10%
- 분석 기간: 2022년 1월 ~ 2024년 12월

비교 그룹:
1) 베이스라인: 수급 필터 없음
2) 금융투자 필터: 금융투자 순매수 > 0
3) 연기금 필터: 연기금 순매수 > 0
4) 개인 매도 필터: 개인 순매수 < 0 (개미 털기 효과)
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_loader import DataLoader, FlowDataLoader


class Stage3Step1Backtest:
    """
    Stage 3-1 백테스트 클래스

    수급 주체별 단독 영향력을 정밀 검증합니다.
    """

    # 투자자 필터 설정
    FILTER_CONFIGS = {
        'A0_Baseline': {
            'name': '베이스라인 (수급 필터 없음)',
            'column': None,
            'condition': None
        },
        'A1_FinancialInvestment': {
            'name': '금융투자 순매수',
            'column': '금융투자',
            'condition': 'positive'  # > 0
        },
        'A2_Pension': {
            'name': '연기금 순매수',
            'column': '연기금',
            'condition': 'positive'  # > 0
        },
        'A3_IndividualSell': {
            'name': '개인 순매도 (개미 털기)',
            'column': '개인',
            'condition': 'negative'  # < 0
        }
    }

    def __init__(self,
                 volume_ratio_threshold=6.5,
                 z_score_threshold=3.0,
                 price_threshold=10.0,
                 price_upper_limit=30.0):
        """
        초기화 함수

        Parameters:
        -----------
        volume_ratio_threshold : float
            Volume_Ratio 임계치 (기본값: 6.5)
        z_score_threshold : float
            Z_Score 임계치 (기본값: 3.0)
        price_threshold : float
            당일 가격 상승률 임계치 % (기본값: 10.0)
        price_upper_limit : float
            당일 가격 상승률 상한 % (기본값: 30.0, 상한가 제외)
        """
        self.vr_threshold = volume_ratio_threshold
        self.zs_threshold = z_score_threshold
        self.price_threshold = price_threshold
        self.price_upper_limit = price_upper_limit

        self.data_dir = 'data'
        self.results_dir = 'results'

        # 결과 디렉토리 생성
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_price_data(self, filename='stock_data_with_indicators.csv'):
        """
        가격/지표 데이터 로드

        Returns:
        --------
        pd.DataFrame
            가격 데이터 (Volume_Ratio, Z_Score 포함)
        """
        filepath = os.path.join(self.data_dir, filename)

        print("=" * 80)
        print("[1단계] 가격/지표 데이터 로드")
        print("=" * 80)

        df = pd.read_csv(filepath, parse_dates=[0])

        # 첫 번째 컬럼이 Date가 아니면 이름 변경
        if df.columns[0] != 'Date':
            df = df.rename(columns={df.columns[0]: 'Date'})

        print(f"파일: {filepath}")
        print(f"행 수: {len(df):,}개")
        print(f"컬럼: {df.columns.tolist()[:10]}...")
        print()

        return df

    def collect_or_load_flow_data(self, force_collect=False):
        """
        수급 데이터 수집 또는 로드

        기존 데이터가 있으면 로드, 없으면 새로 수집합니다.

        Parameters:
        -----------
        force_collect : bool
            True면 기존 데이터 무시하고 새로 수집

        Returns:
        --------
        pd.DataFrame
            수급 데이터
        """
        print("=" * 80)
        print("[2단계] 수급 데이터 수집/로드")
        print("=" * 80)

        flow_file = os.path.join(self.data_dir, 'investor_flow_data_v2.csv')

        if os.path.exists(flow_file) and not force_collect:
            # 기존 데이터 로드
            print(f"기존 수급 데이터 로드: {flow_file}")
            flow_df = pd.read_csv(flow_file, parse_dates=['Date'])
            print(f"행 수: {len(flow_df):,}개")
            print(f"컬럼: {flow_df.columns.tolist()}")
            print()
            return flow_df

        # 새로 수집
        print("새로운 수급 데이터 수집 시작...")
        loader = FlowDataLoader(start_date='20220101', end_date='20241231')

        flow_df = loader.collect_flow_data(stock_list_file='data/stock_list.csv')

        if flow_df is not None:
            loader.save_data(flow_df, 'investor_flow_data_v2.csv')

        print()
        return flow_df

    def merge_data(self, price_df, flow_df):
        """
        가격 데이터와 수급 데이터 병합

        Parameters:
        -----------
        price_df : pd.DataFrame
            가격/지표 데이터
        flow_df : pd.DataFrame
            수급 데이터

        Returns:
        --------
        pd.DataFrame
            병합된 데이터
        """
        print("=" * 80)
        print("[3단계] 데이터 병합")
        print("=" * 80)

        # Date 타입 통일
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        flow_df['Date'] = pd.to_datetime(flow_df['Date'])

        # Code 타입 통일 (문자열, 6자리)
        price_df['Code'] = price_df['Code'].astype(str).str.zfill(6)
        flow_df['Code'] = flow_df['Code'].astype(str).str.zfill(6)

        # 병합 (Date + Code 기준)
        merged = pd.merge(
            price_df,
            flow_df[['Date', 'Code', '금융투자', '연기금', '개인', '외국인']],
            on=['Date', 'Code'],
            how='left'
        )

        # 당일 수익률 계산 (Return_0D)
        if 'Return_0D' not in merged.columns:
            if 'Open' in merged.columns and 'Close' in merged.columns:
                merged['Return_0D'] = (merged['Close'] - merged['Open']) / merged['Open'] * 100
                print("당일 수익률(Return_0D) 계산 완료")

        # 수급 데이터 매칭 통계
        total = len(merged)
        matched = merged['금융투자'].notna().sum()
        match_rate = matched / total * 100

        print(f"병합 완료: {len(merged):,}개 행")
        print(f"수급 데이터 매칭: {matched:,}개 / {total:,}개 ({match_rate:.1f}%)")
        print()

        return merged

    def get_base_signals(self, df):
        """
        Stage 2 최적 조건 만족 시그널 추출

        조건: VR >= 6.5 & ZS >= 3.0 & 10% <= Price < 30%

        Returns:
        --------
        pd.DataFrame
            베이스 시그널
        """
        # 필수 컬럼 확인
        required_cols = ['Volume_Ratio', 'Z_Score', 'Return_0D']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")

        # 조건 적용
        base = df[
            (df['Volume_Ratio'] >= self.vr_threshold) &
            (df['Z_Score'] >= self.zs_threshold) &
            (df['Return_0D'] >= self.price_threshold) &
            (df['Return_0D'] < self.price_upper_limit)
        ].copy()

        return base

    def apply_flow_filter(self, signals, filter_key):
        """
        수급 필터 적용

        Parameters:
        -----------
        signals : pd.DataFrame
            베이스 시그널
        filter_key : str
            필터 키 (FILTER_CONFIGS의 키)

        Returns:
        --------
        pd.DataFrame
            필터링된 시그널
        """
        config = self.FILTER_CONFIGS[filter_key]
        column = config['column']
        condition = config['condition']

        # 베이스라인은 필터 없음
        if column is None:
            return signals.copy()

        # 수급 데이터가 있는 행만 대상
        valid_signals = signals[signals[column].notna()].copy()

        if condition == 'positive':
            # 순매수 > 0
            filtered = valid_signals[valid_signals[column] > 0]
        elif condition == 'negative':
            # 순매수 < 0 (순매도)
            filtered = valid_signals[valid_signals[column] < 0]
        else:
            filtered = valid_signals

        return filtered.copy()

    def calculate_statistics(self, signals, filter_key):
        """
        시그널 통계 계산 (정밀 분석 지표)

        Parameters:
        -----------
        signals : pd.DataFrame
            시그널 데이터
        filter_key : str
            필터 키

        Returns:
        --------
        dict
            통계 결과
        """
        config = self.FILTER_CONFIGS[filter_key]
        label = config['name']

        # 시그널이 없는 경우
        if len(signals) == 0:
            return {
                'filter_key': filter_key,
                'filter_name': label,
                'signal_count': 0,
                'monthly_signals': 0
            }

        # 기본 통계
        signal_count = len(signals)
        unique_months = signals['Date'].dt.to_period('M').nunique()
        monthly_signals = signal_count / unique_months if unique_months > 0 else 0

        stats = {
            'filter_key': filter_key,
            'filter_name': label,
            'signal_count': signal_count,
            'monthly_signals': round(monthly_signals, 2)
        }

        # 보유기간별 수익률 통계
        periods = [1, 3, 5, 10]

        for period in periods:
            col = f'Return_{period}D'
            if col not in signals.columns:
                continue

            returns = signals[col].dropna()

            if len(returns) == 0:
                continue

            # 평균 수익률
            avg_return = returns.mean()
            stats[f'avg_return_{period}d'] = round(avg_return, 4)

            # 중위값
            median_return = returns.median()
            stats[f'median_return_{period}d'] = round(median_return, 4)

            # 승률
            win_rate = (returns > 0).sum() / len(returns) * 100
            stats[f'win_rate_{period}d'] = round(win_rate, 2)

            # 최대 수익
            max_return = returns.max()
            stats[f'max_return_{period}d'] = round(max_return, 2)

            # 최대 손실 (MDD)
            min_return = returns.min()
            stats[f'mdd_{period}d'] = round(min_return, 2)

            # 손익비 (Profit Factor)
            profits = returns[returns > 0]
            losses = returns[returns < 0]

            avg_profit = profits.mean() if len(profits) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

            if avg_loss > 0:
                profit_factor = avg_profit / avg_loss
            else:
                profit_factor = float('inf') if avg_profit > 0 else 1.0

            stats[f'profit_factor_{period}d'] = round(profit_factor, 3)

        return stats

    def calculate_alpha(self, results_df):
        """
        베이스라인 대비 Alpha 계산

        Parameters:
        -----------
        results_df : pd.DataFrame
            결과 데이터

        Returns:
        --------
        pd.DataFrame
            Alpha 컬럼이 추가된 결과
        """
        df = results_df.copy()

        # 베이스라인 수익률
        baseline = df[df['filter_key'] == 'A0_Baseline'].iloc[0]

        for period in [1, 3, 5, 10]:
            avg_col = f'avg_return_{period}d'
            alpha_col = f'alpha_{period}d'

            if avg_col in df.columns:
                baseline_return = baseline[avg_col]
                df[alpha_col] = df[avg_col] - baseline_return

                # Alpha 비율 (%)
                if baseline_return != 0:
                    df[f'alpha_pct_{period}d'] = ((df[avg_col] - baseline_return) / abs(baseline_return) * 100).round(1)
                else:
                    df[f'alpha_pct_{period}d'] = 0

        return df

    def run_backtest(self, merged_df):
        """
        백테스트 실행

        Parameters:
        -----------
        merged_df : pd.DataFrame
            병합된 데이터

        Returns:
        --------
        pd.DataFrame
            백테스트 결과
        """
        print("=" * 80)
        print("[4단계] 수급 주체별 백테스트 실행")
        print("=" * 80)
        print(f"베이스 조건: VR >= {self.vr_threshold}, ZS >= {self.zs_threshold}, "
              f"Price >= {self.price_threshold}%")
        print()

        # 베이스 시그널 추출
        base_signals = self.get_base_signals(merged_df)
        print(f"베이스 시그널: {len(base_signals):,}개")
        print()

        # 각 필터별 백테스트
        results = []

        for filter_key, config in self.FILTER_CONFIGS.items():
            print(f"  [{filter_key}] {config['name']}")

            # 필터 적용
            filtered_signals = self.apply_flow_filter(base_signals, filter_key)

            # 통계 계산
            stats = self.calculate_statistics(filtered_signals, filter_key)
            results.append(stats)

            # 결과 출력
            if stats['signal_count'] > 0:
                print(f"    시그널: {stats['signal_count']:,}개 (월 평균 {stats['monthly_signals']:.1f}건)")
                if 'avg_return_1d' in stats:
                    print(f"    1일: {stats['avg_return_1d']:.3f}% | "
                          f"10일: {stats.get('avg_return_10d', 'N/A')}")
            else:
                print(f"    시그널 없음")
            print()

        # DataFrame 변환
        results_df = pd.DataFrame(results)

        # Alpha 계산
        results_df = self.calculate_alpha(results_df)

        print("백테스트 완료!")
        print()

        return results_df

    def save_results(self, results_df, filename='stage3_step1_baseline_comparison.csv'):
        """
        결과 저장

        Parameters:
        -----------
        results_df : pd.DataFrame
            결과 데이터
        filename : str
            파일명
        """
        filepath = os.path.join(self.results_dir, filename)
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {filepath}")

    def print_summary(self, results_df):
        """
        결과 요약 출력

        Parameters:
        -----------
        results_df : pd.DataFrame
            결과 데이터
        """
        print("\n" + "=" * 80)
        print("Stage 3-1 백테스트 결과 요약")
        print("=" * 80)
        print()

        # 표 형식으로 출력
        print(f"{'필터':^25} | {'시그널':>8} | {'1일':>8} | {'10일':>8} | "
              f"{'승률(1일)':>8} | {'Alpha(1일)':>10}")
        print("-" * 85)

        for _, row in results_df.iterrows():
            name = row['filter_name'][:25]
            signals = row['signal_count']
            ret_1d = row.get('avg_return_1d', 0)
            ret_10d = row.get('avg_return_10d', 0)
            win_1d = row.get('win_rate_1d', 0)
            alpha_1d = row.get('alpha_pct_1d', 0)

            print(f"{name:<25} | {signals:>8,} | {ret_1d:>7.3f}% | {ret_10d:>7.3f}% | "
                  f"{win_1d:>7.1f}% | {alpha_1d:>+9.1f}%")

        print()

        # 최고 성과 필터 (베이스라인 제외)
        non_baseline = results_df[results_df['filter_key'] != 'A0_Baseline']
        if len(non_baseline) > 0 and 'avg_return_1d' in non_baseline.columns:
            best_1d = non_baseline.loc[non_baseline['avg_return_1d'].idxmax()]
            print(f"[익일 수익률 최고] {best_1d['filter_name']}: "
                  f"{best_1d['avg_return_1d']:.3f}% (Alpha: {best_1d.get('alpha_pct_1d', 0):+.1f}%)")


def main():
    """메인 함수"""
    print("\n" + "=" * 80)
    print("Stage 3-1: 데이터 인프라 정상화 및 수급 주체별 단독 영향력 정밀 검증")
    print("=" * 80)
    print()

    # 백테스트 인스턴스 생성
    backtest = Stage3Step1Backtest(
        volume_ratio_threshold=6.5,
        z_score_threshold=3.0,
        price_threshold=10.0,
        price_upper_limit=30.0
    )

    # 1. 가격 데이터 로드
    price_df = backtest.load_price_data()

    # 2. 수급 데이터 수집/로드
    flow_df = backtest.collect_or_load_flow_data(force_collect=False)

    if flow_df is None:
        print("[오류] 수급 데이터를 로드할 수 없습니다.")
        return

    # 3. 데이터 병합
    merged_df = backtest.merge_data(price_df, flow_df)

    # 4. 백테스트 실행
    results_df = backtest.run_backtest(merged_df)

    # 5. 결과 저장
    backtest.save_results(results_df)

    # 6. 결과 요약 출력
    backtest.print_summary(results_df)

    print("\nStage 3-1 완료!")

    return results_df, merged_df


if __name__ == '__main__':
    results, merged = main()
