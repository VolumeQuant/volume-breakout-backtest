"""
데이터 수집 모듈

FinanceDataReader를 사용하여 한국 주식시장 데이터를 수집합니다.
KOSPI/KOSDAQ 시가총액 상위 종목의 일봉 데이터를 가져옵니다.
"""

import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
import os
import time


class DataLoader:
    """한국 주식 데이터를 수집하는 클래스"""

    def __init__(self, start_date='2022-01-01', end_date='2024-12-31'):
        """
        초기화 함수

        Parameters:
        -----------
        start_date : str
            데이터 수집 시작일 (YYYY-MM-DD)
        end_date : str
            데이터 수집 종료일 (YYYY-MM-DD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = 'data'

        # data 폴더가 없으면 생성
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_stock_list(self):
        """
        KOSPI/KOSDAQ 시가총액 상위 종목 리스트를 가져옵니다.

        Returns:
        --------
        pd.DataFrame
            종목 코드, 종목명, 시장, 시가총액 정보
        """
        print("종목 리스트를 가져오는 중...")

        # KOSPI 전체 종목
        kospi = fdr.StockListing('KOSPI')
        kospi['Market'] = 'KOSPI'

        # KOSDAQ 전체 종목
        kosdaq = fdr.StockListing('KOSDAQ')
        kosdaq['Market'] = 'KOSDAQ'

        # 시가총액 기준 정렬 및 상위 종목 선택
        kospi_top = kospi.nlargest(200, 'Marcap')  # KOSPI 상위 200개
        kosdaq_top = kosdaq.nlargest(150, 'Marcap')  # KOSDAQ 상위 150개

        # 합치기
        stock_list = pd.concat([kospi_top, kosdaq_top], ignore_index=True)

        # 필요한 컬럼만 선택
        stock_list = stock_list[['Code', 'Name', 'Market', 'Marcap']]

        print(f"총 {len(stock_list)}개 종목을 선택했습니다.")
        print(f"- KOSPI: {len(kospi_top)}개")
        print(f"- KOSDAQ: {len(kosdaq_top)}개")

        return stock_list

    def fetch_stock_data(self, code, name):
        """
        개별 종목의 일봉 데이터를 가져옵니다.

        Parameters:
        -----------
        code : str
            종목 코드
        name : str
            종목명 (로그 출력용)

        Returns:
        --------
        pd.DataFrame
            일봉 데이터 (시가/고가/저가/종가/거래량)
        """
        try:
            # 데이터 가져오기
            df = fdr.DataReader(code, self.start_date, self.end_date)

            if df is None or len(df) == 0:
                print(f"  [경고] {name} ({code}): 데이터 없음")
                return None

            # 필요한 컬럼만 선택
            if 'Open' in df.columns:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            else:
                print(f"  [경고] {name} ({code}): 필요한 컬럼이 없습니다")
                return None

            # 종목 정보 추가
            df['Code'] = code
            df['Name'] = name

            # 거래량 0인 날 제거
            df = df[df['Volume'] > 0]

            # 60일 미만 데이터는 제외 (Z-Score 계산을 위해)
            if len(df) < 60:
                print(f"  [경고] {name} ({code}): 데이터가 60일 미만입니다")
                return None

            return df

        except Exception as e:
            print(f"  [오류] {name} ({code}): {str(e)}")
            return None

    def load_all_data(self, stock_list):
        """
        모든 종목의 데이터를 수집합니다.

        Parameters:
        -----------
        stock_list : pd.DataFrame
            종목 리스트

        Returns:
        --------
        pd.DataFrame
            전체 종목의 일봉 데이터
        """
        print("\n데이터 수집을 시작합니다...")
        all_data = []

        total = len(stock_list)
        for idx, row in stock_list.iterrows():
            code = row['Code']
            name = row['Name']
            market = row['Market']

            print(f"[{idx+1}/{total}] {name} ({code}) - {market}")

            # 데이터 가져오기
            df = self.fetch_stock_data(code, name)

            if df is not None:
                df['Market'] = market
                all_data.append(df)

            # API 호출 제한을 위한 대기 (0.1초)
            time.sleep(0.1)

        # 데이터 합치기
        if len(all_data) == 0:
            print("\n[오류] 수집된 데이터가 없습니다!")
            return None

        combined_df = pd.concat(all_data, ignore_index=False)

        print(f"\n데이터 수집 완료!")
        print(f"- 성공한 종목: {len(all_data)}개")
        print(f"- 전체 데이터: {len(combined_df):,}개 행")

        return combined_df

    def save_data(self, df, filename='stock_data.csv'):
        """
        데이터를 CSV 파일로 저장합니다.

        Parameters:
        -----------
        df : pd.DataFrame
            저장할 데이터프레임
        filename : str
            파일명
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        print(f"\n데이터 저장 완료: {filepath}")

    def load_data(self, filename='stock_data.csv'):
        """
        저장된 CSV 파일을 불러옵니다.

        Parameters:
        -----------
        filename : str
            파일명

        Returns:
        --------
        pd.DataFrame
            불러온 데이터
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"[오류] 파일이 존재하지 않습니다: {filepath}")
            return None

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"데이터 로드 완료: {filepath}")
        print(f"- 전체 데이터: {len(df):,}개 행")

        return df


def main():
    """테스트용 메인 함수"""
    # DataLoader 초기화
    loader = DataLoader(start_date='2022-01-01', end_date='2024-12-31')

    # 종목 리스트 가져오기
    stock_list = loader.get_stock_list()

    # 종목 리스트 저장
    loader.save_data(stock_list, 'stock_list.csv')

    # 전체 데이터 수집
    all_data = loader.load_all_data(stock_list)

    if all_data is not None:
        # 데이터 저장
        loader.save_data(all_data, 'stock_data.csv')


if __name__ == '__main__':
    main()
