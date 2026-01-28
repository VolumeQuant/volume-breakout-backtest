"""
Phase 3-2: Hybrid Strategy & Paper Trading System (2025 OOS)
============================================================

목표:
- Phase 2의 과적합 문제 해결
- 2단계 필터링: Rough Signal + Quality Filter
- 2021-2024 IS vs 2025 OOS 비교
- 2025년 모의투자 매매일지 생성

철학:
- 가격/거래량: 넓은 그물 (시그널 생성기)
- 수급 조건: 진짜를 골라내는 품질 필터
- 단일 최적 수급 필터로 과적합 방지

수급 필터 (20일 수익률 Top 3):
1. 연기금 30일 순매도 (2.69%)
2. 금융투자 10일 순매도 (2.69%)
3. 개인 30일 순매수 (2.66%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent


# =============================================================================
# 수급 필터 정의 (Phase 3-1 결과 기반)
# =============================================================================

QUALITY_FILTERS = {
    'Filter_1': {
        'name': '연기금 30D SELL',
        'investor': '연기금_비중',
        'duration': 30,
        'direction': 'SELL',
        'threshold_kospi': 1.5,
        'threshold_kosdaq': 2.5,
    },
    'Filter_2': {
        'name': '금융투자 10D SELL',
        'investor': '금융투자_비중',
        'duration': 10,
        'direction': 'SELL',
        'threshold_kospi': 1.5,
        'threshold_kosdaq': 2.5,
    },
    'Filter_3': {
        'name': '개인 30D BUY',
        'investor': '개인_비중',
        'duration': 30,
        'direction': 'BUY',
        'threshold_kospi': 1.5,
        'threshold_kosdaq': 2.5,
    },
}


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("[1/5] 데이터 로드 중...")

    data_dir = project_root / 'data'

    # 주가 데이터
    stock_data = pd.read_csv(data_dir / 'stock_data_with_indicators.csv', low_memory=False)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # 수급 데이터
    investor_flow = pd.read_csv(data_dir / 'investor_flow_data.csv',
                                 encoding='utf-8-sig', low_memory=False)

    # 수급 데이터 전처리 (인덱스 기반)
    flow_data = pd.DataFrame({
        'Code': investor_flow.iloc[:, 13],
        'Date': pd.to_datetime(investor_flow.iloc[:, 0]),
        '금융투자_비중': pd.to_numeric(investor_flow.iloc[:, 1], errors='coerce'),
        '연기금_비중': pd.to_numeric(investor_flow.iloc[:, 7], errors='coerce'),
        '외국인_비중': pd.to_numeric(investor_flow.iloc[:, 10], errors='coerce'),
    })
    flow_data['개인_비중'] = -(flow_data['금융투자_비중'] + flow_data['연기금_비중'] + flow_data['외국인_비중'])

    # 종목 리스트
    stock_list = pd.read_csv(data_dir / 'stock_list.csv')

    print(f"   주가 데이터: {len(stock_data):,}행")
    print(f"   수급 데이터: {len(flow_data):,}행")
    print(f"   기간: {stock_data['Date'].min()} ~ {stock_data['Date'].max()}")

    return stock_data, flow_data, stock_list


def merge_and_calculate_cumulative(stock_data, flow_data, stock_list):
    """데이터 병합 및 누적 수급 계산"""
    print("\n[2/5] 데이터 병합 및 누적 수급 계산 중...")

    # 병합
    merged = stock_data.merge(flow_data, on=['Code', 'Date'], how='inner')

    # 당일 수익률 계산 (시가→종가)
    merged['Return_0D'] = (merged['Close'] / merged['Open'] - 1) * 100

    # Market 정보 추가
    code_to_market = stock_list.set_index('Code')['Market'].to_dict()
    merged['Market_Type'] = merged['Code'].map(code_to_market)

    # 정렬
    merged = merged.sort_values(['Code', 'Date']).reset_index(drop=True)

    # 누적 수급 계산 (Duration별)
    investors = ['금융투자_비중', '연기금_비중', '외국인_비중', '개인_비중']
    durations = [10, 30, 50]

    for investor in investors:
        for duration in durations:
            col_name = f'{investor}_{duration}D'
            merged[col_name] = merged.groupby('Code')[investor].transform(
                lambda x: x.rolling(window=duration, min_periods=duration).sum()
            )

    print(f"   병합 후: {len(merged):,}행")

    return merged


def apply_rough_signal(df, vr_threshold=3.0, price_threshold=5.0):
    """1단계: Rough Signal (넓은 그물)"""
    signals = df[
        (df['Volume_Ratio'] >= vr_threshold) &
        (df['Return_0D'] >= price_threshold) &
        (df['Return_0D'] < 30.0)  # 이상치 제외
    ].copy()

    return signals


def apply_quality_filter(signals, filter_config, market_type=None):
    """2단계: Quality Filter (수급 조건)"""
    investor = filter_config['investor']
    duration = filter_config['duration']
    direction = filter_config['direction']

    cumul_col = f'{investor}_{duration}D'

    if cumul_col not in signals.columns:
        return pd.DataFrame()

    # 시장별 임계치 적용
    if market_type == 'KOSPI':
        threshold = filter_config['threshold_kospi']
    else:
        threshold = filter_config['threshold_kosdaq']

    # 방향에 따라 필터링
    if direction == 'BUY':
        filtered = signals[signals[cumul_col] >= threshold].copy()
    else:  # SELL
        filtered = signals[signals[cumul_col] <= -threshold].copy()

    return filtered


def calculate_performance(df, period='10D'):
    """성과 계산"""
    col = f'Return_{period}'
    if col not in df.columns or len(df) == 0:
        return {
            'signal_count': 0,
            'avg_return': np.nan,
            'median_return': np.nan,
            'win_rate': np.nan,
            'sharpe': np.nan,
            'max_gain': np.nan,
            'max_loss': np.nan,
        }

    valid = df.dropna(subset=[col])
    if len(valid) == 0:
        return {
            'signal_count': 0,
            'avg_return': np.nan,
            'median_return': np.nan,
            'win_rate': np.nan,
            'sharpe': np.nan,
            'max_gain': np.nan,
            'max_loss': np.nan,
        }

    returns = valid[col]

    return {
        'signal_count': len(valid),
        'avg_return': returns.mean(),
        'median_return': returns.median(),
        'win_rate': (returns > 0).sum() / len(returns) * 100,
        'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'max_gain': returns.max(),
        'max_loss': returns.min(),
    }


def run_hybrid_backtest(merged_data):
    """하이브리드 백테스트 실행"""
    print("\n[3/5] 하이브리드 백테스트 실행 중...")

    results = []

    # 연도별 분리
    merged_data['Year'] = merged_data['Date'].dt.year

    # 기간 정의
    is_data = merged_data[merged_data['Year'].between(2021, 2024)]
    oos_data = merged_data[merged_data['Year'] == 2025]

    print(f"   In-Sample (2021-2024): {len(is_data):,}행")
    print(f"   Out-of-Sample (2025): {len(oos_data):,}행")

    # 1단계: Rough Signal
    is_rough = apply_rough_signal(is_data)
    oos_rough = apply_rough_signal(oos_data)

    print(f"\n   [1단계] Rough Signal (VR≥3.0, Price≥5%)")
    print(f"     IS: {len(is_rough):,}건, OOS: {len(oos_rough):,}건")

    # Rough Signal 성과 기록
    for period in ['10D', '20D']:
        is_perf = calculate_performance(is_rough, period)
        oos_perf = calculate_performance(oos_rough, period)

        results.append({
            'Filter': 'Rough Signal Only',
            'Period': period,
            'IS_Signals': is_perf['signal_count'],
            'IS_Return': is_perf['avg_return'],
            'IS_WinRate': is_perf['win_rate'],
            'OOS_Signals': oos_perf['signal_count'],
            'OOS_Return': oos_perf['avg_return'],
            'OOS_WinRate': oos_perf['win_rate'],
        })

    # 2단계: Quality Filter 적용
    print(f"\n   [2단계] Quality Filter 적용")

    for filter_key, filter_config in QUALITY_FILTERS.items():
        filter_name = filter_config['name']

        # KOSDAQ150에만 적용 (Phase 3-1 결과에 따라)
        is_kosdaq = is_rough[is_rough['Market_Type'] == 'KOSDAQ']
        oos_kosdaq = oos_rough[oos_rough['Market_Type'] == 'KOSDAQ']

        is_filtered = apply_quality_filter(is_kosdaq, filter_config, 'KOSDAQ')
        oos_filtered = apply_quality_filter(oos_kosdaq, filter_config, 'KOSDAQ')

        print(f"     {filter_name}: IS {len(is_filtered):,}건, OOS {len(oos_filtered):,}건")

        for period in ['10D', '20D']:
            is_perf = calculate_performance(is_filtered, period)
            oos_perf = calculate_performance(oos_filtered, period)

            results.append({
                'Filter': filter_name,
                'Period': period,
                'IS_Signals': is_perf['signal_count'],
                'IS_Return': is_perf['avg_return'],
                'IS_WinRate': is_perf['win_rate'],
                'OOS_Signals': oos_perf['signal_count'],
                'OOS_Return': oos_perf['avg_return'],
                'OOS_WinRate': oos_perf['win_rate'],
            })

    return pd.DataFrame(results), oos_rough


def generate_trading_log(oos_data, oos_rough):
    """2025년 모의투자 매매일지 생성"""
    print("\n[4/5] 2025년 매매일지 생성 중...")

    trading_logs = []

    # 각 필터별로 매매일지 생성
    for filter_key, filter_config in QUALITY_FILTERS.items():
        filter_name = filter_config['name']

        # KOSDAQ만 필터링
        oos_kosdaq = oos_rough[oos_rough['Market_Type'] == 'KOSDAQ']
        filtered = apply_quality_filter(oos_kosdaq, filter_config, 'KOSDAQ')

        if len(filtered) == 0:
            continue

        for _, row in filtered.iterrows():
            trading_logs.append({
                '매수일자': row['Date'].strftime('%Y-%m-%d'),
                '종목코드': row['Code'],
                '종목명': row['Name'],
                '시장': row['Market_Type'],
                '수급필터': filter_name,
                '진입가(종가)': row['Close'],
                '당일수익률(%)': round(row['Return_0D'], 2),
                'VR': round(row['Volume_Ratio'], 2),
                '10일후수익률(%)': round(row['Return_10D'], 2) if pd.notna(row['Return_10D']) else None,
                '20일후수익률(%)': round(row['Return_20D'], 2) if pd.notna(row['Return_20D']) else None,
            })

    trading_log_df = pd.DataFrame(trading_logs)

    if len(trading_log_df) > 0:
        trading_log_df = trading_log_df.sort_values('매수일자')
        print(f"   총 {len(trading_log_df)}건의 매매 기록 생성")

    return trading_log_df


def visualize_results(results_df):
    """결과 시각화"""
    print("\n[5/5] 결과 시각화 중...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 20일 수익률 데이터만 사용
    df_20d = results_df[results_df['Period'] == '20D']

    # 1. IS vs OOS 수익률 비교
    ax1 = axes[0, 0]
    x = range(len(df_20d))
    width = 0.35
    ax1.bar([i - width/2 for i in x], df_20d['IS_Return'], width, label='IS (2021-2024)', color='steelblue')
    ax1.bar([i + width/2 for i in x], df_20d['OOS_Return'], width, label='OOS (2025)', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_20d['Filter'], rotation=45, ha='right')
    ax1.set_ylabel('20D Return (%)')
    ax1.set_title('IS vs OOS: 20D Return Comparison')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # 2. IS vs OOS 승률 비교
    ax2 = axes[0, 1]
    ax2.bar([i - width/2 for i in x], df_20d['IS_WinRate'], width, label='IS (2021-2024)', color='steelblue')
    ax2.bar([i + width/2 for i in x], df_20d['OOS_WinRate'], width, label='OOS (2025)', color='coral')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_20d['Filter'], rotation=45, ha='right')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('IS vs OOS: Win Rate Comparison')
    ax2.legend()
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # 3. 시그널 수 비교
    ax3 = axes[1, 0]
    ax3.bar([i - width/2 for i in x], df_20d['IS_Signals'], width, label='IS (2021-2024)', color='steelblue')
    ax3.bar([i + width/2 for i in x], df_20d['OOS_Signals'], width, label='OOS (2025)', color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_20d['Filter'], rotation=45, ha='right')
    ax3.set_ylabel('Signal Count')
    ax3.set_title('IS vs OOS: Signal Count')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. OOS 수익률 vs 승률 산점도
    ax4 = axes[1, 1]
    colors = ['gray', 'blue', 'green', 'red']
    for i, (_, row) in enumerate(df_20d.iterrows()):
        ax4.scatter(row['OOS_WinRate'], row['OOS_Return'],
                   s=100, c=colors[i], label=row['Filter'], alpha=0.7)
    ax4.set_xlabel('OOS Win Rate (%)')
    ax4.set_ylabel('OOS 20D Return (%)')
    ax4.set_title('2025 OOS: Return vs Win Rate')
    ax4.legend(loc='best', fontsize=8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axvline(x=50, color='black', linestyle='--', linewidth=0.5)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    output_path = project_root / 'results' / 'p3' / 'p3_02_hybrid_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   시각화 저장: {output_path}")


def main():
    print("="*80)
    print("Phase 3-2: Hybrid Strategy & Paper Trading System (2025 OOS)")
    print("="*80)
    print("\n철학: 넓은 그물(가격/거래량) + 품질 필터(수급)")
    print("목표: 과적합 방지 + 실전 유연성 + 통계적 유의성")

    # 1. 데이터 로드
    stock_data, flow_data, stock_list = load_and_prepare_data()

    # 2. 데이터 병합 및 누적 수급 계산
    merged = merge_and_calculate_cumulative(stock_data, flow_data, stock_list)

    # 3. 하이브리드 백테스트
    results_df, oos_rough = run_hybrid_backtest(merged)

    # 4. 2025년 매매일지 생성
    oos_data = merged[merged['Date'].dt.year == 2025]
    trading_log = generate_trading_log(oos_data, oos_rough)

    # 5. 시각화
    visualize_results(results_df)

    # 6. 결과 저장
    output_dir = project_root / 'results' / 'p3'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'p3_02_hybrid_is_vs_oos.csv',
                      index=False, encoding='utf-8-sig')

    if len(trading_log) > 0:
        trading_log.to_csv(output_dir / 'p3_02_trading_log_2025.csv',
                          index=False, encoding='utf-8-sig')

    # 7. 결과 출력
    print("\n" + "="*80)
    print("결과 요약")
    print("="*80)

    print("\n[IS vs OOS 성과 비교 (20일 수익률)]")
    print("-"*70)
    df_20d = results_df[results_df['Period'] == '20D']
    for _, row in df_20d.iterrows():
        print(f"{row['Filter']:20s} | IS: {row['IS_Return']:6.2f}% ({row['IS_Signals']:4.0f}건) | "
              f"OOS: {row['OOS_Return']:6.2f}% ({row['OOS_Signals']:4.0f}건)")

    # 2025년 최고 성과 필터 찾기
    best_filter = df_20d.loc[df_20d['OOS_Return'].idxmax()]

    print("\n" + "="*80)
    print("2025년 최고 성과 필터")
    print("="*80)
    print(f"필터: {best_filter['Filter']}")
    print(f"20일 수익률: {best_filter['OOS_Return']:.2f}%")
    print(f"승률: {best_filter['OOS_WinRate']:.1f}%")
    print(f"시그널 수: {best_filter['OOS_Signals']:.0f}건")

    if len(trading_log) > 0:
        print("\n" + "="*80)
        print("2025년 매매일지 미리보기 (최근 10건)")
        print("="*80)
        print(trading_log.tail(10).to_string(index=False))

    print("\n" + "="*80)
    print("Phase 3-2 완료!")
    print("="*80)
    print(f"\n저장된 파일:")
    print(f"  - {output_dir / 'p3_02_hybrid_is_vs_oos.csv'}")
    print(f"  - {output_dir / 'p3_02_trading_log_2025.csv'}")
    print(f"  - {output_dir / 'p3_02_hybrid_analysis.png'}")

    return results_df, trading_log


if __name__ == "__main__":
    results_df, trading_log = main()
