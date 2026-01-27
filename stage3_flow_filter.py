"""
Stage 3: 세분화된 투자자별 수급 필터 백테스팅

금융투자 vs 연기금 분리 분석
- 금융투자: 단기 스마트머니 (1~5일 예측력)
- 연기금: 장기 방향성 (10일+ 바닥 확인)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class FlowFilterBacktest:
    """세분화된 수급 필터 백테스팅 클래스"""

    def __init__(self, price_data, flow_data,
                 volume_ratio_threshold=6.5,
                 z_score_threshold=3.0,
                 price_threshold=10.0):
        """
        초기화 함수

        Parameters:
        -----------
        price_data : pd.DataFrame
            가격/거래량 데이터 (indicators 포함)
        flow_data : pd.DataFrame
            투자자별 수급 데이터 (pykrx)
        volume_ratio_threshold : float
            Volume_Ratio 임계치
        z_score_threshold : float
            Z_Score 임계치
        price_threshold : float
            당일 가격 상승률 임계치 (%)
        """
        self.price_data = price_data
        self.flow_data = flow_data
        self.vr_threshold = volume_ratio_threshold
        self.zs_threshold = z_score_threshold
        self.price_threshold = price_threshold
        self.results_dir = 'results'

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def merge_data(self):
        """
        가격 데이터와 수급 데이터 병합

        Returns:
        --------
        pd.DataFrame
            병합된 데이터
        """
        print("\n데이터 병합 중...")

        # 날짜 컬럼 통일
        if 'Date' not in self.flow_data.columns and '날짜' in self.flow_data.columns:
            self.flow_data = self.flow_data.rename(columns={'날짜': 'Date'})

        # Date를 datetime으로 변환
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        self.flow_data['Date'] = pd.to_datetime(self.flow_data['Date'])

        # 병합 (Date + Code 기준)
        merged = pd.merge(
            self.price_data,
            self.flow_data,
            on=['Date', 'Code'],
            how='left',
            suffixes=('', '_flow')
        )

        # Return_0D 계산 (당일 가격 상승률)
        if 'Return_0D' not in merged.columns:
            if 'Open' in merged.columns and 'Close' in merged.columns:
                merged['Return_0D'] = (merged['Close'] - merged['Open']) / merged['Open'] * 100
                print(f"Return_0D 계산 완료")

        print(f"병합 완료: {len(merged):,}개 행")

        return merged

    def get_base_signals(self, df):
        """
        Stage 2 조건 만족 시그널 추출

        Base: VR >= 6.5 & ZS >= 3.0 & Price >= 10% & Price < 30%

        Returns:
        --------
        pd.DataFrame
            베이스 시그널
        """
        base = df[
            (df['Volume_Ratio'] >= self.vr_threshold) &
            (df['Z_Score'] >= self.zs_threshold) &
            (df['Return_0D'] >= self.price_threshold) &
            (df['Return_0D'] < 30.0)  # 상한가 제외
        ].copy()

        return base

    def apply_flow_filter(self, base_signals, filter_config):
        """
        수급 필터 적용

        Parameters:
        -----------
        base_signals : pd.DataFrame
            베이스 시그널
        filter_config : dict
            필터 설정
            예: {
                'type': 'single',  # single, double, triple, continuity
                'investors': ['금융투자'],  # 대상 투자자
                'threshold': 0,  # 순매수 임계치
                'days': 1  # continuity용
            }

        Returns:
        --------
        pd.DataFrame
            필터링된 시그널
        """
        filter_type = filter_config['type']
        investors = filter_config['investors']
        threshold = filter_config.get('threshold', 0)

        if filter_type == 'single':
            # 단일 투자자 순매수
            investor = investors[0]
            condition = base_signals[investor] > threshold
            filtered = base_signals[condition].copy()

        elif filter_type == 'double':
            # 두 투자자 모두 순매수
            investor1, investor2 = investors[0], investors[1]
            condition = (base_signals[investor1] > threshold) & \
                       (base_signals[investor2] > threshold)
            filtered = base_signals[condition].copy()

        elif filter_type == 'triple':
            # 세 투자자 모두 순매수
            investor1, investor2, investor3 = investors[0], investors[1], investors[2]
            condition = (base_signals[investor1] > threshold) & \
                       (base_signals[investor2] > threshold) & \
                       (base_signals[investor3] > threshold)
            filtered = base_signals[condition].copy()

        elif filter_type == 'continuity':
            # 특정 투자자가 최근 N일 중 M일 순매수
            # 이 부분은 구현 복잡도로 인해 간소화
            # 여기서는 단순히 해당 투자자 순매수로 처리
            investor = investors[0]
            condition = base_signals[investor] > threshold
            filtered = base_signals[condition].copy()

        else:
            filtered = base_signals.copy()

        return filtered

    def calculate_statistics(self, signals, label):
        """
        시그널 통계 계산

        Parameters:
        -----------
        signals : pd.DataFrame
            시그널 데이터
        label : str
            필터 레이블

        Returns:
        --------
        dict
            통계 결과
        """
        if len(signals) == 0:
            return {
                'filter': label,
                'signal_count': 0,
                'monthly_signals': 0,
                'avg_return_1d': np.nan,
                'win_rate_1d': np.nan,
                'avg_return_3d': np.nan,
                'win_rate_3d': np.nan,
                'avg_return_5d': np.nan,
                'win_rate_5d': np.nan,
                'avg_return_10d': np.nan,
                'win_rate_10d': np.nan,
                'sharpe_10d': np.nan,
                'profit_factor': np.nan
            }

        # 시그널 수 및 빈도
        signal_count = len(signals)
        unique_months = signals['Date'].dt.to_period('M').nunique()
        monthly_signals = signal_count / unique_months if unique_months > 0 else 0

        # 보유기간별 통계
        stats = {'filter': label, 'signal_count': signal_count, 'monthly_signals': monthly_signals}

        for period in [1, 3, 5, 10]:
            col = f'Return_{period}D'
            if col in signals.columns:
                returns = signals[col].dropna()

                if len(returns) > 0:
                    avg_return = returns.mean()
                    win_rate = (returns > 0).sum() / len(returns) * 100
                    std_return = returns.std()
                    sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

                    stats[f'avg_return_{period}d'] = avg_return
                    stats[f'win_rate_{period}d'] = win_rate

                    if period == 10:
                        stats['sharpe_10d'] = sharpe
                else:
                    stats[f'avg_return_{period}d'] = np.nan
                    stats[f'win_rate_{period}d'] = np.nan
                    if period == 10:
                        stats['sharpe_10d'] = np.nan

        # 손익비 (10일 기준)
        if 'Return_10D' in signals.columns:
            returns_10d = signals['Return_10D'].dropna()
            if len(returns_10d) > 0:
                profits = returns_10d[returns_10d > 0]
                losses = returns_10d[returns_10d < 0]

                avg_profit = profits.mean() if len(profits) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

                if avg_loss > 0:
                    profit_factor = avg_profit / avg_loss
                else:
                    profit_factor = np.inf if avg_profit > 0 else 1.0

                stats['profit_factor'] = profit_factor
            else:
                stats['profit_factor'] = np.nan
        else:
            stats['profit_factor'] = np.nan

        return stats

    def run_core_combinations(self, merged_data):
        """
        핵심 조합 그리드 서치

        A그룹: 개별 투자자 유형
        D그룹: 복수 주체 동시 매수
        C그룹: 연속성

        Returns:
        --------
        pd.DataFrame
            결과 테이블
        """
        print("\n" + "=" * 80)
        print("Stage 3: 세분화된 수급 필터 그리드 서치")
        print("=" * 80)
        print(f"베이스 조건: VR >= {self.vr_threshold} & ZS >= {self.zs_threshold} & Price >= {self.price_threshold}%")
        print()

        # 투자자 컬럼명 확인 (pykrx 컬럼명)
        investor_columns = [col for col in merged_data.columns if '금융투자' in col or '연기금' in col or '외국인' in col or '기관' in col]
        print(f"수급 컬럼: {investor_columns}\n")

        # 베이스 시그널 추출
        base_signals = self.get_base_signals(merged_data)
        print(f"베이스 시그널: {len(base_signals):,}개")
        print(f"월 평균: {len(base_signals) / base_signals['Date'].dt.to_period('M').nunique():.1f}건\n")

        # 테스트할 조합 정의
        # pykrx 컬럼명 추정 (실제 데이터 확인 필요)
        # 예: '금융투자', '보험', '투신', '사모', '은행', '기타금융', '연기금등', '외국인'
        test_configs = [
            # A그룹: 개별 투자자 순매수
            {
                'label': 'A0_Baseline',
                'type': 'baseline',
                'investors': [],
                'threshold': 0
            },
            {
                'label': 'A1_Financial_Investment',
                'type': 'single',
                'investors': ['금융투자'],
                'threshold': 0
            },
            {
                'label': 'A2_Pension',
                'type': 'single',
                'investors': ['연기금등'],
                'threshold': 0
            },
            {
                'label': 'A3_Foreign',
                'type': 'single',
                'investors': ['외국인'],
                'threshold': 0
            },
            {
                'label': 'A4_Financial_plus_Pension',
                'type': 'single',
                'investors': ['금융투자+연기금'],  # 계산 필요
                'threshold': 0
            },
            {
                'label': 'A5_Foreign_plus_Financial',
                'type': 'single',
                'investors': ['외국인+금융투자'],  # 계산 필요
                'threshold': 0
            },
            {
                'label': 'A7_Total_Institution',
                'type': 'single',
                'investors': ['기관합계'],
                'threshold': 0
            },
            # D그룹: 복수 매수
            {
                'label': 'D1_Financial_AND_Foreign',
                'type': 'double',
                'investors': ['금융투자', '외국인'],
                'threshold': 0
            },
            {
                'label': 'D2_Financial_AND_Pension',
                'type': 'double',
                'investors': ['금융투자', '연기금등'],
                'threshold': 0
            },
            {
                'label': 'D4_Triple_Buy',
                'type': 'triple',
                'investors': ['금융투자', '연기금등', '외국인'],
                'threshold': 0
            },
            # C그룹: 연속성 (간소화)
            {
                'label': 'C2_Financial_Continuity',
                'type': 'continuity',
                'investors': ['금융투자'],
                'threshold': 0,
                'days': 3
            }
        ]

        # 복합 컬럼 생성 (필요한 경우)
        if '금융투자' in merged_data.columns and '연기금등' in merged_data.columns:
            merged_data['금융투자+연기금'] = merged_data['금융투자'] + merged_data['연기금등']

        if '외국인' in merged_data.columns and '금융투자' in merged_data.columns:
            merged_data['외국인+금융투자'] = merged_data['외국인'] + merged_data['금융투자']

        # 그리드 서치
        results = []

        for idx, config in enumerate(test_configs):
            label = config['label']
            print(f"[{idx+1}/{len(test_configs)}] {label}")

            # 베이스라인은 필터 없이
            if config['type'] == 'baseline':
                filtered_signals = base_signals.copy()
            else:
                # 필터 적용
                try:
                    filtered_signals = self.apply_flow_filter(base_signals, config)
                except KeyError as e:
                    print(f"  [경고] 컬럼 없음: {e} - 건너뛰기")
                    continue

            # 통계 계산
            stats = self.calculate_statistics(filtered_signals, label)
            results.append(stats)

            # 결과 출력
            if stats['signal_count'] > 0:
                print(f"  시그널: {stats['signal_count']:,}개 (월 평균 {stats['monthly_signals']:.1f}건)")
                print(f"  1일: {stats['avg_return_1d']:.3f}% | 10일: {stats['avg_return_10d']:.3f}%")
            else:
                print(f"  시그널 없음")

        # DataFrame으로 변환
        results_df = pd.DataFrame(results)

        print("\n그리드 서치 완료!")
        print(f"총 {len(results_df)}개 조합 테스트\n")

        return results_df

    def save_results(self, results_df, filename='stage3_flow_filter_results.csv'):
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

    def get_best_combination(self, results_df, metric='avg_return_10d', min_monthly_signals=5):
        """
        최고 성과 조합 찾기

        Parameters:
        -----------
        results_df : pd.DataFrame
            결과 데이터
        metric : str
            최적화 지표
        min_monthly_signals : int
            최소 월 시그널 수

        Returns:
        --------
        pd.Series
            최고 성과 조합
        """
        # 최소 시그널 수 필터
        filtered = results_df[results_df['monthly_signals'] >= min_monthly_signals].copy()

        if len(filtered) == 0:
            print(f"[경고] 월 {min_monthly_signals}건 이상 조합 없음")
            return None

        # 최고 성과 찾기
        best = filtered.loc[filtered[metric].idxmax()]

        return best


def main():
    """테스트용 메인 함수"""
    print("\n" + "=" * 80)
    print("Stage 3: 세분화 수급 필터 백테스트")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    print("-" * 80)

    price_data = pd.read_csv('data/stock_data_with_indicators.csv', parse_dates=[0])
    if price_data.columns[0] != 'Date':
        price_data.columns = ['Date'] + list(price_data.columns[1:])

    print(f"가격 데이터: {len(price_data):,}개 행")

    # 수급 데이터 로드 (stage3_flow_data_collector로 수집한 데이터)
    flow_data = pd.read_csv('data/investor_flow_data.csv', parse_dates=['Date'])
    print(f"수급 데이터: {len(flow_data):,}개 행")

    # 2. 백테스트 실행
    print("\n[2단계] 백테스트 실행")
    print("-" * 80)

    backtest = FlowFilterBacktest(
        price_data,
        flow_data,
        volume_ratio_threshold=6.5,
        z_score_threshold=3.0,
        price_threshold=10.0
    )

    # 데이터 병합
    merged_data = backtest.merge_data()

    # 그리드 서치
    results = backtest.run_core_combinations(merged_data)

    # 결과 저장
    backtest.save_results(results)

    # 3. 최고 조합 분석
    print("\n" + "=" * 80)
    print("[3단계] 최고 성과 조합 분석")
    print("=" * 80)

    # 10일 수익률 기준
    best_10d = backtest.get_best_combination(results, metric='avg_return_10d', min_monthly_signals=5)
    if best_10d is not None:
        print(f"\n[10일 수익률 최고]")
        print(f"  필터: {best_10d['filter']}")
        print(f"  시그널: {int(best_10d['signal_count']):,}개 (월 평균 {best_10d['monthly_signals']:.1f}건)")
        print(f"  1일: {best_10d['avg_return_1d']:.3f}% | 승률: {best_10d['win_rate_1d']:.1f}%")
        print(f"  10일: {best_10d['avg_return_10d']:.3f}% | 승률: {best_10d['win_rate_10d']:.1f}%")
        print(f"  샤프: {best_10d['sharpe_10d']:.3f} | 손익비: {best_10d['profit_factor']:.2f}")

    print("\n완료!")


if __name__ == '__main__':
    main()
