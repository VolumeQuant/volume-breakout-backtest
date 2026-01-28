"""
CSV 파일을 엑셀에서 읽을 수 있도록 변환
"""

import pandas as pd

print("CSV 파일을 엑셀용으로 변환합니다...\n")

# 1. 시그널 파일 변환
print("[1] 시그널 파일 (VR≥5.0, ZS≥2.0)")
signals_df = pd.read_csv('results/signals_vr5.0_zs2.0.csv', encoding='utf-8')
signals_df.to_csv('results/signals_vr5.0_zs2.0_excel.csv', index=False, encoding='utf-8-sig')
print(f"   → results/signals_vr5.0_zs2.0_excel.csv ({len(signals_df):,}개 행)")

# 2. 그리드 서치 결과 변환
print("\n[2] 그리드 서치 결과")
grid_df = pd.read_csv('results/grid_search_results.csv', encoding='utf-8')
grid_df.to_csv('results/grid_search_results_excel.csv', index=False, encoding='utf-8-sig')
print(f"   → results/grid_search_results_excel.csv ({len(grid_df):,}개 행)")

# 3. 종목 리스트 변환
print("\n[3] 종목 리스트")
stocks_df = pd.read_csv('data/stock_list.csv', encoding='utf-8')
stocks_df.to_csv('data/stock_list_excel.csv', index=False, encoding='utf-8-sig')
print(f"   → data/stock_list_excel.csv ({len(stocks_df):,}개 행)")

print("\n" + "="*80)
print("✅ 변환 완료! 이제 엑셀에서 열면 한글이 정상적으로 보입니다.")
print("="*80)
print("\n파일 위치:")
print("  - results/signals_vr5.0_zs2.0_excel.csv")
print("  - results/grid_search_results_excel.csv")
print("  - data/stock_list_excel.csv")
