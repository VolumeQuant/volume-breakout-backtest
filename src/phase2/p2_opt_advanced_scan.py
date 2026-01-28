"""
Phase 2: Advanced Multi-Dimensional Grid Search
목표: 수급 비중의 임계치와 과열권을 식별하는 전수 조사

분석 차원:
- VR: [3.0, 4.0, 5.0, 6.0, 7.0]
- Price: [5, 7, 10, 12, 15]
- 개인/외국인/금융투자/연기금: 각각 5단계
  [-1(매도), 0(0~1.5%), 1(1.5~3.5%), 2(3.5~5.5%), 3(5.5%+)]

핵심 검증 포인트:
1. 5.5% 이상에서 '피크 아웃' 현상 존재하는가?
2. 금융투자+연기금 합산 ~3%에서 수익률 최대화되는가?
3. 대형주 vs 중소형주 최적 수급 구간은?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class Phase2GridSearch:
    def __init__(self):
        self.data_dir = Path('data')
        self.results_dir = Path('results/p2')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 고정 조건
        self.fixed_zs = 2.0

        # 그리드 차원
        self.vr_range = [3.0, 4.0, 5.0, 6.0, 7.0]
        self.price_range = [5, 7, 10, 12, 15]

        # 수급 주체별 5단계 (비중 기준)
        # -1: 순매도, 0: 0~1.5%, 1: 1.5~3.5%, 2: 3.5~5.5%, 3: 5.5%+
        self.flow_levels = [-1, 0, 1, 2, 3]

        print("="*80)
        print("Phase 2: Advanced Multi-Dimensional Grid Search")
        print("="*80)
        print(f"VR: {self.vr_range}")
        print(f"Price: {self.price_range}")
        print(f"Flow Levels: {self.flow_levels}")
        print(f"Total Combinations: {len(self.vr_range) * len(self.price_range) * len(self.flow_levels)**4:,}")
        print("="*80)
        print()

    def load_data(self):
        """데이터 로드 및 전처리"""
        print("="*80)
        print("데이터 로드")
        print("="*80)

        # 주가 데이터
        price_df = pd.read_csv(
            f'{self.data_dir}/stock_data_with_indicators.csv',
            parse_dates=['Date']
        )
        print(f"가격 데이터: {len(price_df):,}개 행")

        # 수급 데이터
        flow_df = pd.read_csv(
            f'{self.data_dir}/investor_flow_data.csv',
            parse_dates=['Date']
        )
        print(f"수급 데이터: {len(flow_df):,}개 행")

        # 병합
        df = price_df.merge(
            flow_df[['Date', 'Code', '개인', '외국인', '금융투자', '연기금']],
            on=['Date', 'Code'],
            how='left'
        )
        print(f"병합 완료: {len(df):,}개 행")

        # 거래대금 계산 및 수급 비중 계산
        df['거래대금'] = df['Close'] * df['Volume']

        # 각 투자자 비중 계산 (%)
        for investor in ['개인', '외국인', '금융투자', '연기금']:
            df[f'{investor}_비중'] = (df[investor] / df['거래대금'] * 100).fillna(0)

        # Inst_Ratio 계산
        df['Inst_Ratio'] = df['금융투자_비중'] + df['연기금_비중']

        # 당일 수익률 계산 (Return_0D가 없으므로 직접 계산)
        df['Return_0D'] = ((df['Close'] - df['Open']) / df['Open'] * 100).fillna(0)

        # 시가총액 분류 (주가 × 거래량으로 추정)
        # 각 종목의 평균 거래대금 기준으로 대형/중소형 구분
        market_cap = df.groupby('Code')['거래대금'].mean().sort_values(ascending=False)
        large_caps = market_cap.head(100).index.tolist()

        df['Cap_Type'] = df['Code'].apply(
            lambda x: 'Large' if x in large_caps else 'Small'
        )

        print(f"\n대형주: {df[df['Cap_Type']=='Large']['Code'].nunique()}개")
        print(f"중소형주: {df[df['Cap_Type']=='Small']['Code'].nunique()}개")
        print()

        return df

    def classify_flow_level(self, ratio):
        """수급 비중을 5단계로 분류"""
        if ratio < 0:
            return -1  # 순매도
        elif ratio < 1.5:
            return 0   # 0~1.5%
        elif ratio < 3.5:
            return 1   # 1.5~3.5%
        elif ratio < 5.5:
            return 2   # 3.5~5.5%
        else:
            return 3   # 5.5%+

    def backtest_combination(self, df, vr_th, price_th,
                            indiv_level, foreign_level, fi_level, pension_level,
                            cap_type=None):
        """특정 조합에 대한 백테스팅"""

        # 체급 필터
        if cap_type:
            df = df[df['Cap_Type'] == cap_type].copy()

        # 거래량 + 가격 필터
        signals = df[
            (df['Volume_Ratio'] >= vr_th) &
            (df['Z_Score'] >= self.fixed_zs) &
            (df['Return_0D'] >= price_th) &
            (df['Return_0D'] < 30.0)  # 상한가 제외
        ].copy()

        if len(signals) == 0:
            return None

        # 수급 레벨 분류
        signals['개인_level'] = signals['개인_비중'].apply(self.classify_flow_level)
        signals['외국인_level'] = signals['외국인_비중'].apply(self.classify_flow_level)
        signals['금융투자_level'] = signals['금융투자_비중'].apply(self.classify_flow_level)
        signals['연기금_level'] = signals['연기금_비중'].apply(self.classify_flow_level)

        # 수급 레벨 필터 적용
        signals = signals[
            (signals['개인_level'] == indiv_level) &
            (signals['외국인_level'] == foreign_level) &
            (signals['금융투자_level'] == fi_level) &
            (signals['연기금_level'] == pension_level)
        ]

        if len(signals) < 5:  # 최소 시그널 수
            return None

        # 성과 계산
        return_10d = signals['Return_10D'].dropna()

        if len(return_10d) < 5:
            return None

        # 연도별 수익률 계산 (강건성 측정용)
        signals['Year'] = signals['Date'].dt.year
        yearly_returns = signals.groupby('Year')['Return_10D'].mean()

        stats = {
            'VR': vr_th,
            'Price': price_th,
            '개인_level': indiv_level,
            '외국인_level': foreign_level,
            '금융투자_level': fi_level,
            '연기금_level': pension_level,
            'Cap_Type': cap_type if cap_type else 'All',

            'Signal_Count': len(return_10d),
            'Monthly_Signals': len(return_10d) / signals['Date'].dt.to_period('M').nunique(),

            'Avg_Return_10D': return_10d.mean(),
            'Median_Return_10D': return_10d.median(),
            'Std_Return_10D': return_10d.std(),
            'Win_Rate_10D': (return_10d > 0).sum() / len(return_10d) * 100,

            'Sharpe_Ratio': return_10d.mean() / return_10d.std() if return_10d.std() > 0 else 0,

            'Max_Gain': return_10d.max(),
            'Max_Loss': return_10d.min(),

            # 강건성 지표
            'Yearly_Return_Std': yearly_returns.std(),  # 연도별 수익률 편차
            'Yearly_Count': len(yearly_returns),

            # 수급 비중 평균 (분석용)
            'Avg_개인_비중': signals['개인_비중'].mean(),
            'Avg_외국인_비중': signals['외국인_비중'].mean(),
            'Avg_금융투자_비중': signals['금융투자_비중'].mean(),
            'Avg_연기금_비중': signals['연기금_비중'].mean(),
            'Avg_Inst_Ratio': signals['Inst_Ratio'].mean(),
        }

        # Expectancy (기대값) 계산
        if stats['Win_Rate_10D'] > 0:
            avg_win = return_10d[return_10d > 0].mean()
            avg_loss = abs(return_10d[return_10d <= 0].mean()) if len(return_10d[return_10d <= 0]) > 0 else 0
            win_rate = stats['Win_Rate_10D'] / 100
            stats['Expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            stats['Expectancy'] = 0

        return stats

    def run_grid_search(self, df):
        """전수 그리드 서치 실행"""
        print("="*80)
        print("다차원 그리드 서치 실행")
        print("="*80)

        results = []

        # 체급별 분석 (전체, 대형, 중소형)
        cap_types = [None, 'Large', 'Small']

        total_combinations = (
            len(self.vr_range) *
            len(self.price_range) *
            len(self.flow_levels) ** 4 *
            len(cap_types)
        )

        print(f"총 조합 수: {total_combinations:,}")
        print()

        with tqdm(total=total_combinations, desc="Grid Search") as pbar:
            for cap_type in cap_types:
                for vr, price in product(self.vr_range, self.price_range):
                    for indiv, foreign, fi, pension in product(
                        self.flow_levels, self.flow_levels,
                        self.flow_levels, self.flow_levels
                    ):
                        stats = self.backtest_combination(
                            df, vr, price,
                            indiv, foreign, fi, pension,
                            cap_type
                        )

                        if stats:
                            results.append(stats)

                        pbar.update(1)

        results_df = pd.DataFrame(results)
        print(f"\n유효한 조합: {len(results_df):,}개")
        print()

        return results_df

    def analyze_results(self, results_df):
        """결과 분석 및 인사이트 도출"""
        print("="*80)
        print("핵심 인사이트 분석")
        print("="*80)

        # 강건성 필터: 시그널 ≥ 30건
        robust_df = results_df[results_df['Signal_Count'] >= 30].copy()
        print(f"강건성 필터 (시그널 ≥ 30건): {len(robust_df):,}개 조합\n")

        # 1. 전체 최고 성과 (Expectancy 기준)
        print("[1] 전체 최고 성과 (Expectancy 기준)")
        print("-" * 80)
        top_all = robust_df.nlargest(10, 'Expectancy')[
            ['VR', 'Price', '개인_level', '외국인_level', '금융투자_level', '연기금_level',
             'Cap_Type', 'Signal_Count', 'Avg_Return_10D', 'Win_Rate_10D',
             'Sharpe_Ratio', 'Expectancy', 'Avg_Inst_Ratio']
        ]
        print(top_all.to_string())
        print()

        # 2. 대형주 vs 중소형주 비교
        print("[2] 대형주 vs 중소형주 최적 전략")
        print("-" * 80)

        large_best = robust_df[robust_df['Cap_Type']=='Large'].nlargest(1, 'Expectancy')
        small_best = robust_df[robust_df['Cap_Type']=='Small'].nlargest(1, 'Expectancy')

        if len(large_best) > 0:
            print("\n대형주 최적:")
            print(large_best[['VR', 'Price', 'Avg_Inst_Ratio', 'Signal_Count',
                             'Avg_Return_10D', 'Win_Rate_10D', 'Expectancy']].to_string())

        if len(small_best) > 0:
            print("\n중소형주 최적:")
            print(small_best[['VR', 'Price', 'Avg_Inst_Ratio', 'Signal_Count',
                             'Avg_Return_10D', 'Win_Rate_10D', 'Expectancy']].to_string())
        print()

        # 3. 피크 아웃 분석
        print("[3] 수급 비중 '피크 아웃' 분석")
        print("-" * 80)

        # Inst_Ratio 구간별 평균 성과
        robust_df['Inst_Ratio_Bin'] = pd.cut(
            robust_df['Avg_Inst_Ratio'],
            bins=[0, 3, 5.5, 100],
            labels=['<3%', '3~5.5%', '5.5%+']
        )

        peakout_analysis = robust_df.groupby('Inst_Ratio_Bin').agg({
            'Signal_Count': 'sum',
            'Avg_Return_10D': 'mean',
            'Win_Rate_10D': 'mean',
            'Expectancy': 'mean'
        }).round(2)

        print(peakout_analysis)
        print()

        # 4. 수급 조합 패턴 분석
        print("[4] 최고 성과 수급 조합 패턴")
        print("-" * 80)

        top_20 = robust_df.nlargest(20, 'Expectancy')

        flow_pattern = top_20.groupby(
            ['개인_level', '외국인_level', '금융투자_level', '연기금_level']
        ).size().sort_values(ascending=False).head(5)

        print("가장 빈번한 수급 패턴 (Top 20 전략 중):")
        for (indiv, foreign, fi, pension), count in flow_pattern.items():
            print(f"개인:{indiv:2} 외국인:{foreign:2} 금융투자:{fi:2} 연기금:{pension:2} → {count}회 등장")
        print()

        return robust_df

    def save_results(self, results_df):
        """결과 저장"""
        output_file = self.results_dir / 'p2_opt_comprehensive_matrix.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {output_file}")
        print()


def main():
    searcher = Phase2GridSearch()

    # 1. 데이터 로드
    df = searcher.load_data()

    # 2. 그리드 서치 실행
    results_df = searcher.run_grid_search(df)

    # 3. 결과 분석
    robust_df = searcher.analyze_results(results_df)

    # 4. 저장
    searcher.save_results(results_df)

    print("="*80)
    print("Phase 2 완료!")
    print("="*80)


if __name__ == '__main__':
    main()
