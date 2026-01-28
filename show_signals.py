"""
특정 조합의 시그널 상세 조회
"""

import pandas as pd
from backtest import GridSearchBacktest

# 데이터 로드
print("데이터를 로드합니다...")
df = pd.read_csv('data/stock_data_with_indicators.csv', index_col=0, parse_dates=True)

print(f"전체 데이터: {len(df):,}개 행\n")

# 백테스트 객체 생성
backtest = GridSearchBacktest(df)

# VR ≥5.0, ZS ≥2.0 조합의 상세 시그널 가져오기
print("=" * 100)
print("Volume_Ratio ≥5.0, Z_Score ≥2.0 조합의 실제 시그널")
print("=" * 100)

signals = backtest.get_detailed_signals(
    volume_ratio_threshold=5.0,
    z_score_threshold=2.0
)

print(f"\n총 시그널 수: {len(signals):,}개\n")

# 날짜 인덱스를 컬럼으로 변환
signals_with_date = signals.reset_index()
signals_with_date.columns = ['Date'] + list(signals.columns)

# 최근 50개 시그널 보기
print("최근 50개 시그널:")
print("-" * 100)
recent_signals = signals_with_date.tail(50)

# 보기 좋게 포맷팅
for idx, row in recent_signals.iterrows():
    print(f"\n날짜: {row['Date'].strftime('%Y-%m-%d')}")
    print(f"종목: {row['Name']} ({row['Code']}) - {row['Market']}")
    print(f"종가: {row['Close']:,.0f}원")
    print(f"거래량: {row['Volume']:,.0f}주")
    print(f"Volume_Ratio: {row['Volume_Ratio']:.2f}배 (20일 평균 대비)")
    print(f"Z_Score: {row['Z_Score']:.2f} (표준편차)")

    if pd.notna(row['Return_1D']):
        print(f"익일 수익률: {row['Return_1D']:.2f}%")
    if pd.notna(row['Return_3D']):
        print(f"3일 수익률: {row['Return_3D']:.2f}%")
    if pd.notna(row['Return_5D']):
        print(f"5일 수익률: {row['Return_5D']:.2f}%")
    if pd.notna(row['Return_10D']):
        print(f"10일 수익률: {row['Return_10D']:.2f}%")

# 통계 정보
print("\n" + "=" * 100)
print("종목별 시그널 발생 빈도 (Top 20)")
print("=" * 100)
signal_count_by_stock = signals_with_date.groupby(['Code', 'Name']).size().sort_values(ascending=False)
print(signal_count_by_stock.head(20))

# 수익률이 가장 높았던 시그널
print("\n" + "=" * 100)
print("익일 수익률 Top 10")
print("=" * 100)
top_returns = signals_with_date.nlargest(10, 'Return_1D')[['Date', 'Name', 'Code', 'Close', 'Volume_Ratio', 'Z_Score', 'Return_1D', 'Return_10D']]
for idx, row in top_returns.iterrows():
    print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['Name']:10s} | VR:{row['Volume_Ratio']:5.1f} ZS:{row['Z_Score']:4.1f} | 익일:{row['Return_1D']:6.2f}% | 10일:{row['Return_10D']:6.2f}%")

# 손실이 가장 컸던 시그널
print("\n" + "=" * 100)
print("익일 손실 Top 10")
print("=" * 100)
worst_returns = signals_with_date.nsmallest(10, 'Return_1D')[['Date', 'Name', 'Code', 'Close', 'Volume_Ratio', 'Z_Score', 'Return_1D', 'Return_10D']]
for idx, row in worst_returns.iterrows():
    print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['Name']:10s} | VR:{row['Volume_Ratio']:5.1f} ZS:{row['Z_Score']:4.1f} | 익일:{row['Return_1D']:6.2f}% | 10일:{row['Return_10D']:6.2f}%")

# CSV로 저장
output_file = 'results/signals_vr5.0_zs2.0.csv'
signals_with_date.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n\n전체 시그널이 저장되었습니다: {output_file}")
