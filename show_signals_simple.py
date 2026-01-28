"""
특정 조합의 시그널 상세 조회 (간단 버전)
"""

import pandas as pd

# 데이터 로드
print("데이터를 로드합니다...")
df = pd.read_csv('data/stock_data_with_indicators.csv', parse_dates=[0])
df.columns = ['Date'] + list(df.columns[1:])

print(f"전체 데이터: {len(df):,}개 행\n")

# VR ≥5.0, ZS ≥2.0 필터링
print("=" * 100)
print("Volume_Ratio ≥5.0, Z_Score ≥2.0 조합의 실제 시그널")
print("=" * 100)

signals = df[(df['Volume_Ratio'] >= 5.0) & (df['Z_Score'] >= 2.0)].copy()

print(f"\n총 시그널 수: {len(signals):,}개\n")

# 최근 30개 시그널 보기
print("최근 30개 시그널:")
print("-" * 100)
recent_signals = signals.sort_values('Date').tail(30)

for idx, row in recent_signals.iterrows():
    print(f"\n날짜: {row['Date']}")
    print(f"종목: {row['Name']} ({row['Code']}) - {row['Market']}")
    print(f"종가: {row['Close']:,.0f}원")
    print(f"거래량: {row['Volume']:,.0f}주")
    print(f"Volume_Ratio: {row['Volume_Ratio']:.2f}배")
    print(f"Z_Score: {row['Z_Score']:.2f}")

    if pd.notna(row['Return_1D']):
        print(f"익일: {row['Return_1D']:+.2f}% | 3일: {row['Return_3D']:+.2f}% | 5일: {row['Return_5D']:+.2f}% | 10일: {row['Return_10D']:+.2f}%")

# 종목별 빈도
print("\n" + "=" * 100)
print("종목별 시그널 발생 빈도 (Top 20)")
print("=" * 100)
signal_count = signals.groupby(['Name', 'Code']).size().sort_values(ascending=False).head(20)
for (name, code), count in signal_count.items():
    print(f"{name:15s} ({code}) : {count:3d}회")

# 수익률 Top 10
print("\n" + "=" * 100)
print("익일 수익률 Top 10")
print("=" * 100)
top_returns = signals.nlargest(10, 'Return_1D')
for idx, row in top_returns.iterrows():
    date_str = str(row['Date'])[:10]
    print(f"{date_str} | {row['Name']:12s} | VR:{row['Volume_Ratio']:5.1f} ZS:{row['Z_Score']:4.1f} | 익일:{row['Return_1D']:+6.2f}% | 10일:{row['Return_10D']:+6.2f}%")

# 손실 Top 10
print("\n" + "=" * 100)
print("익일 손실 Top 10")
print("=" * 100)
worst_returns = signals.nsmallest(10, 'Return_1D')
for idx, row in worst_returns.iterrows():
    date_str = str(row['Date'])[:10]
    print(f"{date_str} | {row['Name']:12s} | VR:{row['Volume_Ratio']:5.1f} ZS:{row['Z_Score']:4.1f} | 익일:{row['Return_1D']:+6.2f}% | 10일:{row['Return_10D']:+6.2f}%")

# CSV 저장
output_file = 'results/signals_vr5.0_zs2.0.csv'
signals.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n\n전체 {len(signals):,}개 시그널이 저장되었습니다: {output_file}")
