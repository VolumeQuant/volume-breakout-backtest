"""
Stage 3: 투자자별 수급 데이터 수집

pykrx를 사용하여 세분화된 기관별 수급 데이터를 수집합니다.
"""

import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
import time
import os


class FlowDataCollector:
    """투자자별 수급 데이터 수집 클래스"""

    def __init__(self, start_date='20220101', end_date='20241231'):
        """
        초기화 함수

        Parameters:
        -----------
        start_date : str
            데이터 수집 시작일 (YYYYMMDD)
        end_date : str
            데이터 수집 종료일 (YYYYMMDD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = 'data'

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_investor_flow(self, ticker, name):
        """
        개별 종목의 투자자별 순매수 데이터를 가져옵니다.

        Parameters:
        -----------
        ticker : str
            종목 코드
        name : str
            종목명

        Returns:
        --------
        pd.DataFrame
            투자자별 순매수 데이터
        """
        try:
            # pykrx로 투자자별 거래대금 가져오기
            df = stock.get_market_trading_value_by_date(
                self.start_date,
                self.end_date,
                ticker,
                detail=True  # 기관 세부 정보 포함
            )

            if df is None or len(df) == 0:
                print(f"  [경고] {name} ({ticker}): 수급 데이터 없음")
                return None

            # 컬럼명 정리 (pykrx 버전에 따라 다를 수 있음)
            df = df.reset_index()
            df['Code'] = ticker
            df['Name'] = name

            # 날짜 컬럼 통일
            if '날짜' in df.columns:
                df = df.rename(columns={'날짜': 'Date'})

            return df

        except Exception as e:
            print(f"  [오류] {name} ({ticker}): {str(e)}")
            return None

    def collect_flow_data(self, stock_list_file='data/stock_list.csv'):
        """
        전체 종목의 수급 데이터를 수집합니다.

        Parameters:
        -----------
        stock_list_file : str
            종목 리스트 CSV 파일 경로

        Returns:
        --------
        pd.DataFrame
            전체 종목의 수급 데이터
        """
        # 종목 리스트 로드
        stock_list = pd.read_csv(stock_list_file)

        print("=" * 80)
        print("투자자별 수급 데이터 수집")
        print("=" * 80)
        print(f"대상 종목: {len(stock_list)}개")
        print(f"기간: {self.start_date} ~ {self.end_date}\n")

        all_data = []
        total = len(stock_list)

        for idx, row in stock_list.iterrows():
            code = row['Code']
            name = row['Name']
            market = row['Market']

            print(f"[{idx+1}/{total}] {name} ({code}) - {market}")

            # 수급 데이터 가져오기
            df = self.get_investor_flow(code, name)

            if df is not None:
                df['Market'] = market
                all_data.append(df)

            # API 호출 제한 (0.2초)
            time.sleep(0.2)

            # 진행상황 출력 (10%마다)
            if (idx + 1) % max(1, total // 10) == 0:
                progress = (idx + 1) / total * 100
                print(f"  진행률: {progress:.1f}% ({idx+1}/{total})\n")

        # 데이터 합치기
        if len(all_data) == 0:
            print("\n[오류] 수집된 수급 데이터가 없습니다!")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\n수급 데이터 수집 완료!")
        print(f"- 성공한 종목: {len(all_data)}개")
        print(f"- 전체 데이터: {len(combined_df):,}개 행")

        return combined_df

    def save_data(self, df, filename='investor_flow_data.csv'):
        """
        수급 데이터를 CSV 파일로 저장합니다.

        Parameters:
        -----------
        df : pd.DataFrame
            저장할 데이터프레임
        filename : str
            파일명
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n수급 데이터 저장 완료: {filepath}")

    def load_data(self, filename='investor_flow_data.csv'):
        """
        저장된 수급 데이터를 로드합니다.

        Parameters:
        -----------
        filename : str
            파일명

        Returns:
        --------
        pd.DataFrame
            로드한 데이터
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"[경고] 파일이 존재하지 않습니다: {filepath}")
            return None

        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"수급 데이터 로드 완료: {filepath}")
        print(f"- 전체 데이터: {len(df):,}개 행")

        return df


def main():
    """테스트용 메인 함수"""
    collector = FlowDataCollector(start_date='20220101', end_date='20241231')

    # 수급 데이터 수집
    flow_data = collector.collect_flow_data()

    if flow_data is not None:
        # 데이터 저장
        collector.save_data(flow_data)

        # 컬럼 확인
        print("\n컬럼 목록:")
        print(flow_data.columns.tolist())

        print("\n샘플 데이터:")
        print(flow_data.head(10))


if __name__ == '__main__':
    main()
