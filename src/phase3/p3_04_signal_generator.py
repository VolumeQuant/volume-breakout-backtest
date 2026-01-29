"""
Phase 3-4: 실전용 Triple-Core 시그널 생성기
============================================

목적: Phase 3-3에서 검증된 3가지 최적 전략과 통계적 폭발 필터(ATR, Z-Score)를 
     결합한 실전용 종목 발굴기

작성일: 2026-01-29
작성자: 조 (HTS 개발 8년차)

사용법:
    python p3_04_signal_generator.py              # 최신 날짜 분석
    python p3_04_signal_generator.py 2025-12-30   # 특정 날짜 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
import io

warnings.filterwarnings('ignore')

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# ============================================================
# 설정
# ============================================================

# 프로젝트 루트 경로 (실제 환경에 맞게 조정)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# 파일 경로
STOCK_DATA_PATH = DATA_DIR / 'stock_data_with_indicators.csv'
INVESTOR_FLOW_PATH = DATA_DIR / 'investor_flow_data.csv'

# 기본 필터 임계치
VR_THRESHOLD = 3.0          # Volume Ratio >= 300%
PRICE_CHANGE_THRESHOLD = 5.0  # 등락률 >= 5%

# 폭발 품질 필터 임계치
ATR_MULTIPLIER = 1.5        # ATR 배수
ZSCORE_VOL_THRESHOLD = 2.0  # 거래량 Z-Score 임계치

# 수급 임계치 (억 원 단위)
FLOW_THRESHOLD_KOSPI = 1.5   # KOSPI: 1.5억 이상
FLOW_THRESHOLD_KOSDAQ = 2.5  # KOSDAQ: 2.5억 이상


# ============================================================
# 데이터 로딩
# ============================================================

def load_data():
    """데이터 로딩 및 전처리"""
    print("📂 데이터 로딩 중...")
    
    # 주가 데이터
    stock_df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # 수급 데이터
    flow_df = pd.read_csv(INVESTOR_FLOW_PATH, parse_dates=['Date'])
    flow_df['Date'] = pd.to_datetime(flow_df['Date'])
    
    print(f"   ✅ 주가 데이터: {len(stock_df):,}건")
    print(f"   ✅ 수급 데이터: {len(flow_df):,}건")
    
    return stock_df, flow_df


def get_available_dates(stock_df):
    """사용 가능한 날짜 목록 반환"""
    return sorted(stock_df['Date'].unique())


# ============================================================
# ATR 계산 (Average True Range)
# ============================================================

def calculate_atr(df, period=20):
    """
    ATR (Average True Range) 계산
    
    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    ATR = TR의 이동평균
    """
    df = df.copy()
    df = df.sort_values(['Code', 'Date'])
    
    # 이전 종가
    df['PrevClose'] = df.groupby('Code')['Close'].shift(1)
    
    # True Range
    df['TR'] = df.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['PrevClose']) if pd.notna(row['PrevClose']) else 0,
            abs(row['Low'] - row['PrevClose']) if pd.notna(row['PrevClose']) else 0
        ), axis=1
    )
    
    # ATR (20일 이동평균)
    df['ATR'] = df.groupby('Code')['TR'].transform(
        lambda x: x.rolling(window=period, min_periods=period).mean()
    )
    
    # 당일 등락폭 (절대값)
    df['DailyRange'] = abs(df['Close'] - df['Open'])
    
    return df


def calculate_volume_zscore(df, period=20):
    """
    거래량 Z-Score 계산
    
    Z = (당일 거래량 - 20일 평균) / 20일 표준편차
    """
    df = df.copy()
    df = df.sort_values(['Code', 'Date'])
    
    # 20일 거래량 평균 및 표준편차
    df['Vol_Mean'] = df.groupby('Code')['Volume'].transform(
        lambda x: x.rolling(window=period, min_periods=period).mean()
    )
    df['Vol_Std'] = df.groupby('Code')['Volume'].transform(
        lambda x: x.rolling(window=period, min_periods=period).std()
    )
    
    # Z-Score
    df['Vol_ZScore'] = (df['Volume'] - df['Vol_Mean']) / df['Vol_Std']
    df['Vol_ZScore'] = df['Vol_ZScore'].replace([np.inf, -np.inf], np.nan)
    
    return df


# ============================================================
# 수급 데이터 전처리
# ============================================================

def prepare_flow_data(flow_df, target_date):
    """
    수급 데이터 전처리: 1일/3일 누적 수급 계산
    """
    flow_df = flow_df.copy()
    flow_df = flow_df.sort_values(['Code', 'Date'])

    # 투자자 유형별 컬럼 직접 매핑
    investor_cols = {
        '개인': '개인',
        '외국인': '외국인',
        '금융투자': '금융투자',
        '연기금': '연기금'
    }

    # 컬럼 존재 확인
    for investor, col in investor_cols.items():
        if col not in flow_df.columns:
            print(f"⚠️ 경고: '{col}' 컬럼을 찾을 수 없습니다.")
            investor_cols[investor] = None

    # 원 단위 → 억 원 단위 변환
    for investor, col in investor_cols.items():
        if col and col in flow_df.columns:
            # 1D 수급 (당일, 억 원)
            flow_df[f'{investor}_1D'] = flow_df[col] / 100_000_000

            # 3D 누적 수급 (억 원)
            flow_df[f'{investor}_3D'] = flow_df.groupby('Code')[f'{investor}_1D'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )

    return flow_df


# ============================================================
# Triple-Core 전략 분류
# ============================================================

def classify_triple_core(row, flow_threshold):
    """
    Triple-Core 등급 분류
    
    🥇 SS (Sniper): 3일 누적 [개인/금투 BUY] AND [외인/연기금 SELL]
    🥈 S (Tactical): 당일 [외인 SELL] AND [연기금 SELL]
    🥉 A (Base): 3일 누적 [개인 SELL] AND [외인 OR 연기금 BUY]
    
    Returns:
        (등급, 기대수익률, 기대승률)
    """
    grades = []
    
    # SS-Sniper 조건 (3일 누적)
    ss_condition = (
        row.get('개인_3D', 0) > flow_threshold and      # 개인 순매수
        row.get('금융투자_3D', 0) > flow_threshold and  # 금투 순매수
        row.get('외국인_3D', 0) < -flow_threshold and   # 외인 순매도
        row.get('연기금_3D', 0) < -flow_threshold       # 연기금 순매도
    )
    if ss_condition:
        grades.append(('SS', 16.91, 63.5))
    
    # S-Tactical 조건 (당일)
    s_condition = (
        row.get('외국인_1D', 0) < -flow_threshold and   # 외인 순매도
        row.get('연기금_1D', 0) < -flow_threshold       # 연기금 순매도
    )
    if s_condition:
        grades.append(('S', 14.30, 63.0))
    
    # A-Base 조건 (3일 누적) - Handover
    a_condition = (
        row.get('개인_3D', 0) < -flow_threshold and     # 개인 순매도
        (row.get('외국인_3D', 0) > flow_threshold or    # 외인 순매수 OR
         row.get('연기금_3D', 0) > flow_threshold)      # 연기금 순매수
    )
    if a_condition:
        grades.append(('A', 8.48, 56.2))
    
    # 최고 등급 반환 (SS > S > A)
    if not grades:
        return None, None, None
    
    # 등급 우선순위
    priority = {'SS': 0, 'S': 1, 'A': 2}
    grades.sort(key=lambda x: priority[x[0]])
    
    return grades[0]


def get_explosion_badges(row):
    """
    폭발 품질 뱃지 생성
    
    🔥ATR: 당일 등락률 > (ATR × 1.5)
    ⚡Z-VOL: 거래량 Z-Score > 2.0
    """
    badges = []
    
    # ATR 뱃지
    if pd.notna(row.get('ATR')) and pd.notna(row.get('DailyRange')):
        if row['DailyRange'] > row['ATR'] * ATR_MULTIPLIER:
            badges.append('🔥ATR')
    
    # Z-Score 뱃지
    if pd.notna(row.get('Vol_ZScore')):
        if row['Vol_ZScore'] > ZSCORE_VOL_THRESHOLD:
            badges.append('⚡Z-VOL')
    
    return ' '.join(badges) if badges else '-'


# ============================================================
# 링크 생성
# ============================================================

def generate_naver_news_link(code):
    """네이버 증권 뉴스 링크 생성"""
    return f"https://finance.naver.com/item/news.naver?code={code}"


def generate_dart_link(code):
    """DART 공시 링크 생성"""
    return f"https://dart.fss.or.kr/dsab001/search.ax?textCrpNM={code}"


def generate_naver_chart_link(code):
    """네이버 증권 차트 링크 생성"""
    return f"https://finance.naver.com/item/main.naver?code={code}"


# ============================================================
# 메인 시그널 생성
# ============================================================

def generate_signals(target_date=None):
    """
    메인 시그널 생성 함수
    
    Args:
        target_date: 분석 대상 날짜 (None이면 최신 날짜)
    
    Returns:
        DataFrame: 시그널 종목 리스트
    """
    # 데이터 로딩
    stock_df, flow_df = load_data()
    
    # 날짜 결정
    available_dates = get_available_dates(stock_df)
    
    if target_date is None:
        target_date = available_dates[-1]
    else:
        target_date = pd.to_datetime(target_date)
        if target_date not in available_dates:
            print(f"⚠️ {target_date.strftime('%Y-%m-%d')}는 데이터에 없습니다.")
            closest = min(available_dates, key=lambda x: abs(x - target_date))
            print(f"   가장 가까운 날짜: {closest.strftime('%Y-%m-%d')}")
            target_date = closest
    
    print(f"\n📅 분석 대상 날짜: {target_date.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    # ATR 및 Z-Score 계산
    print("\n📊 통계 지표 계산 중...")
    stock_df = calculate_atr(stock_df)
    stock_df = calculate_volume_zscore(stock_df)
    
    # 수급 데이터 전처리
    print("📈 수급 데이터 전처리 중...")
    flow_df = prepare_flow_data(flow_df, target_date)
    
    # 당일 데이터 필터링
    daily_stock = stock_df[stock_df['Date'] == target_date].copy()
    daily_flow = flow_df[flow_df['Date'] == target_date].copy()
    
    print(f"   📌 당일 종목 수: {len(daily_stock)}개")
    
    # 데이터 병합
    merged = daily_stock.merge(daily_flow, on=['Date', 'Code'], how='left', suffixes=('', '_flow'))

    # 컬럼명 매핑 (VR, Change 계산)
    if 'Volume_Ratio' in merged.columns:
        merged['VR'] = merged['Volume_Ratio']

    # 등락률 계산
    if 'Change' not in merged.columns:
        merged['Change'] = ((merged['Close'] - merged['Open']) / merged['Open'] * 100)

    # ============================================================
    # Stage 1: 기본 필터 (VR >= 3.0 AND Change >= 5%)
    # ============================================================
    print(f"\n🔍 Stage 1: 기본 필터 적용 (VR≥{VR_THRESHOLD}, Change≥{PRICE_CHANGE_THRESHOLD}%)")

    # VR 컬럼 확인
    vr_col = 'VR' if 'VR' in merged.columns else 'Volume_Ratio'

    stage1 = merged[
        (merged[vr_col] >= VR_THRESHOLD) &
        (merged['Change'] >= PRICE_CHANGE_THRESHOLD)
    ].copy()

    # VR 컬럼 통일
    if vr_col == 'Volume_Ratio':
        stage1['VR'] = stage1['Volume_Ratio']
    
    print(f"   ✅ Stage 1 통과: {len(stage1)}개 종목")
    
    if len(stage1) == 0:
        print("\n⚠️ 조건을 만족하는 종목이 없습니다.")
        return pd.DataFrame()
    
    # ============================================================
    # Stage 2: Triple-Core 등급 분류
    # ============================================================
    print("\n🏆 Stage 2: Triple-Core 등급 분류")
    
    results = []
    
    for idx, row in stage1.iterrows():
        # 종목 코드로 시장 구분 (KOSPI: 0~5로 시작, KOSDAQ: 그 외)
        code = str(row['Code']).zfill(6)
        is_kospi = code[0] in ['0', '1', '2', '3', '4', '5']
        flow_threshold = FLOW_THRESHOLD_KOSPI if is_kospi else FLOW_THRESHOLD_KOSDAQ
        
        # Triple-Core 등급 분류
        grade, expected_return, expected_winrate = classify_triple_core(row, flow_threshold)
        
        # 폭발 품질 뱃지
        badges = get_explosion_badges(row)
        
        # 결과 저장 (등급이 있는 경우만)
        if grade:
            results.append({
                'Code': code,
                'Name': row.get('Name', row.get('Name_flow', 'N/A')),
                'Market': 'KOSPI' if is_kospi else 'KOSDAQ',
                'Grade': grade,
                'Badges': badges,
                'VR': row['VR'],
                'Change': row['Change'],
                'Vol_ZScore': row.get('Vol_ZScore', np.nan),
                'ATR_Ratio': row['DailyRange'] / row['ATR'] if pd.notna(row.get('ATR')) and row['ATR'] > 0 else np.nan,
                '개인_1D': row.get('개인_1D', 0),
                '개인_3D': row.get('개인_3D', 0),
                '외국인_1D': row.get('외국인_1D', 0),
                '외국인_3D': row.get('외국인_3D', 0),
                '금융투자_1D': row.get('금융투자_1D', 0),
                '금융투자_3D': row.get('금융투자_3D', 0),
                '연기금_1D': row.get('연기금_1D', 0),
                '연기금_3D': row.get('연기금_3D', 0),
                'Expected_Return': expected_return,
                'Expected_WinRate': expected_winrate,
                'News_Link': generate_naver_news_link(code),
                'DART_Link': generate_dart_link(code),
                'Chart_Link': generate_naver_chart_link(code),
            })
    
    # DataFrame 생성
    result_df = pd.DataFrame(results)
    
    if len(result_df) == 0:
        print("\n⚠️ Triple-Core 조건을 만족하는 종목이 없습니다.")
        return pd.DataFrame()
    
    # 등급순 정렬 (SS > S > A)
    grade_order = {'SS': 0, 'S': 1, 'A': 2}
    result_df['GradeOrder'] = result_df['Grade'].map(grade_order)
    result_df = result_df.sort_values(['GradeOrder', 'Expected_Return'], ascending=[True, False])
    result_df = result_df.drop(columns=['GradeOrder'])
    
    return result_df


# ============================================================
# 결과 출력
# ============================================================

def print_results(df, target_date):
    """결과를 포맷팅하여 출력"""
    
    if df.empty:
        return
    
    print("\n" + "=" * 100)
    print(f"🎯 Triple-Core 시그널 결과 ({target_date.strftime('%Y-%m-%d')})")
    print("=" * 100)
    
    grade_emoji = {'SS': '🥇', 'S': '🥈', 'A': '🥉'}
    
    for grade in ['SS', 'S', 'A']:
        grade_df = df[df['Grade'] == grade]
        
        if len(grade_df) == 0:
            continue
        
        print(f"\n{grade_emoji[grade]} [{grade}] 등급 ({len(grade_df)}건) - 기대수익률: {grade_df.iloc[0]['Expected_Return']:.1f}%, 승률: {grade_df.iloc[0]['Expected_WinRate']:.1f}%")
        print("-" * 100)
        
        for idx, row in grade_df.iterrows():
            print(f"\n  📌 {row['Name']} ({row['Code']}) [{row['Market']}]")
            print(f"     ├─ 폭발력: VR={row['VR']:.1f}x | 등락률={row['Change']:+.1f}% | {row['Badges']}")
            
            # ATR/Z-Score 상세
            atr_str = f"ATR배수={row['ATR_Ratio']:.2f}" if pd.notna(row['ATR_Ratio']) else "ATR=N/A"
            zscore_str = f"Z-VOL={row['Vol_ZScore']:.2f}" if pd.notna(row['Vol_ZScore']) else "Z-VOL=N/A"
            print(f"     ├─ 통계: {atr_str} | {zscore_str}")
            
            # 수급 상세 (억 원 단위)
            print(f"     ├─ 수급 [1D]: 개인={row['개인_1D']:+.1f}억 | 외인={row['외국인_1D']:+.1f}억 | 금투={row['금융투자_1D']:+.1f}억 | 연기금={row['연기금_1D']:+.1f}억")
            print(f"     ├─ 수급 [3D]: 개인={row['개인_3D']:+.1f}억 | 외인={row['외국인_3D']:+.1f}억 | 금투={row['금융투자_3D']:+.1f}억 | 연기금={row['연기금_3D']:+.1f}억")
            
            # 링크
            print(f"     ├─ 📰 뉴스: {row['News_Link']}")
            print(f"     └─ 📋 공시: {row['DART_Link']}")
    
    # 요약 통계
    print("\n" + "=" * 100)
    print("📊 요약 통계")
    print("-" * 100)
    
    for grade in ['SS', 'S', 'A']:
        count = len(df[df['Grade'] == grade])
        if count > 0:
            print(f"  {grade_emoji[grade]} {grade}: {count}건")
    
    print(f"\n  📌 총 시그널: {len(df)}건")
    
    # 폭발 품질 뱃지 통계
    atr_count = df['Badges'].str.contains('🔥ATR').sum()
    zvol_count = df['Badges'].str.contains('⚡Z-VOL').sum()
    both_count = df['Badges'].str.contains('🔥ATR').multiply(df['Badges'].str.contains('⚡Z-VOL')).sum()
    
    print(f"  🔥 ATR 폭발: {atr_count}건")
    print(f"  ⚡ Z-VOL 폭발: {zvol_count}건")
    print(f"  🔥⚡ 양쪽 모두: {both_count}건")


def save_results(df, target_date, output_dir=None):
    """결과를 CSV로 저장"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'results' / 'p3'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"signals_{target_date.strftime('%Y%m%d')}.csv"
    filepath = output_dir / filename
    
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {filepath}")
    
    return filepath


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🚀 Phase 3-4: Triple-Core 시그널 생성기")
    print("   VR≥3.0 + Change≥5% + 통계적 폭발 필터 + Triple-Core 수급")
    print("=" * 70)
    
    # 날짜 파싱
    target_date = None
    if len(sys.argv) > 1:
        try:
            target_date = pd.to_datetime(sys.argv[1])
            print(f"\n📅 입력된 날짜: {target_date.strftime('%Y-%m-%d')}")
        except:
            print(f"\n⚠️ 잘못된 날짜 형식: {sys.argv[1]}")
            print("   올바른 형식: YYYY-MM-DD (예: 2025-12-30)")
            sys.exit(1)
    
    # 시그널 생성
    try:
        result_df = generate_signals(target_date)
        
        if not result_df.empty:
            # 결과 출력
            actual_date = pd.to_datetime(result_df['Code'].iloc[0]) if False else target_date
            # 실제 분석 날짜는 데이터에서 가져옴
            stock_df, _ = load_data()
            available_dates = get_available_dates(stock_df)
            if target_date is None:
                actual_date = available_dates[-1]
            else:
                actual_date = target_date
            
            print_results(result_df, actual_date)
            
            # 결과 저장
            save_results(result_df, actual_date)
        
    except FileNotFoundError as e:
        print(f"\n❌ 데이터 파일을 찾을 수 없습니다: {e}")
        print(f"   예상 경로:")
        print(f"   - {STOCK_DATA_PATH}")
        print(f"   - {INVESTOR_FLOW_PATH}")
        print(f"\n   PROJECT_ROOT 경로를 확인하세요: {PROJECT_ROOT}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
