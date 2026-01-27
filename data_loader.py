"""
데이터 수집 모듈

FinanceDataReader를 사용하여 한국 주식시장 데이터를 수집합니다.
KOSPI/KOSDAQ 시가총액 상위 종목의 일봉 데이터를 가져옵니다.

pykrx를 사용하여 투자자별 수급 데이터(금융투자, 연기금, 개인 등)를 수집합니다.
"""

import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock
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


class FlowDataLoader:
    """
    투자자별 수급 데이터 수집 클래스

    pykrx를 사용하여 금융투자, 연기금, 개인, 외국인 등
    세분화된 투자자별 순매수 데이터를 수집합니다.

    주요 컬럼:
    - 금융투자: 증권사 자기매매
    - 연기금: 국민연금, 사학연금 등 (기존 '연기금등' 오류 수정)
    - 개인: 개인 투자자
    - 외국인: 외국인 투자자
    """

    # pykrx 투자자 컬럼 매핑 (실제 컬럼명 기준)
    INVESTOR_COLUMNS = {
        '금융투자': '금융투자',      # 증권사 자기매매
        '보험': '보험',
        '투신': '투신',              # 투자신탁
        '사모': '사모',              # 사모펀드
        '은행': '은행',
        '기타금융': '기타금융',
        '연기금': '연기금',          # 연기금 (연기금등 X)
        '기타법인': '기타법인',
        '개인': '개인',              # 개인 투자자
        '외국인': '외국인',
        '기타외국인': '기타외국인',
        '전체': '전체'
    }

    def __init__(self, start_date='20220101', end_date='20241231'):
        """
        초기화 함수

        Parameters:
        -----------
        start_date : str
            데이터 수집 시작일 (YYYYMMDD 형식)
        end_date : str
            데이터 수집 종료일 (YYYYMMDD 형식)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = 'data'

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_investor_flow(self, ticker, name):
        """
        개별 종목의 투자자별 순매수 데이터를 가져옵니다.

        pykrx의 get_market_trading_value_by_date를 사용하여
        금융투자, 연기금, 개인, 외국인 등의 순매수 데이터를 수집합니다.

        Parameters:
        -----------
        ticker : str
            종목 코드 (6자리)
        name : str
            종목명 (로그 출력용)

        Returns:
        --------
        pd.DataFrame
            투자자별 순매수 데이터
            컬럼: Date, Code, Name, 금융투자, 연기금, 개인, 외국인 등
        """
        try:
            # pykrx로 투자자별 거래대금 가져오기 (detail=True로 세부 기관 구분)
            df = stock.get_market_trading_value_by_date(
                self.start_date,
                self.end_date,
                ticker,
                detail=True  # 기관 세부 정보 포함 (금융투자, 연기금, 개인 등)
            )

            if df is None or len(df) == 0:
                print(f"  [경고] {name} ({ticker}): 수급 데이터 없음")
                return None

            # 인덱스(날짜)를 컬럼으로 변환
            df = df.reset_index()
            df = df.rename(columns={'날짜': 'Date'})

            # 종목 정보 추가
            df['Code'] = ticker
            df['Name'] = name

            # Date를 datetime 형식으로 변환
            df['Date'] = pd.to_datetime(df['Date'])

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
        print("투자자별 수급 데이터 수집 (pykrx)")
        print("=" * 80)
        print(f"대상 종목: {len(stock_list)}개")
        print(f"기간: {self.start_date} ~ {self.end_date}")
        print(f"수집 컬럼: 금융투자, 연기금, 개인, 외국인 등")
        print()

        all_data = []
        total = len(stock_list)
        success_count = 0

        for idx, row in stock_list.iterrows():
            code = str(row['Code']).zfill(6)  # 6자리 맞추기
            name = row['Name']
            market = row['Market']

            print(f"[{idx+1}/{total}] {name} ({code}) - {market}")

            # 수급 데이터 가져오기
            df = self.get_investor_flow(code, name)

            if df is not None:
                df['Market'] = market
                all_data.append(df)
                success_count += 1

            # API 호출 제한 (0.15초 대기)
            time.sleep(0.15)

            # 진행상황 출력 (10% 단위)
            if (idx + 1) % max(1, total // 10) == 0:
                progress = (idx + 1) / total * 100
                print(f"  진행률: {progress:.1f}% ({idx+1}/{total})\n")

        # 데이터 합치기
        if len(all_data) == 0:
            print("\n[오류] 수집된 수급 데이터가 없습니다!")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"\n수급 데이터 수집 완료!")
        print(f"- 성공한 종목: {success_count}개 / {total}개")
        print(f"- 전체 데이터: {len(combined_df):,}개 행")

        # 컬럼 확인
        print(f"\n수집된 컬럼:")
        for col in combined_df.columns:
            print(f"  - {col}")

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
            로드한 수급 데이터
        """
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"[오류] 파일이 존재하지 않습니다: {filepath}")
            return None

        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"수급 데이터 로드 완료: {filepath}")
        print(f"- 전체 데이터: {len(df):,}개 행")

        # 컬럼 확인
        print(f"- 컬럼: {df.columns.tolist()}")

        return df

    def verify_data(self, ticker='005930', name='삼성전자', sample_date='20240115'):
        """
        특정 종목/날짜의 수급 데이터를 샘플링하여 검증합니다.

        금융투자, 연기금, 개인 데이터가 정상적으로 로드되는지 확인합니다.

        Parameters:
        -----------
        ticker : str
            검증할 종목 코드 (기본값: 삼성전자)
        name : str
            종목명
        sample_date : str
            샘플링할 날짜 (YYYYMMDD)

        Returns:
        --------
        bool
            검증 성공 여부
        """
        print("\n" + "=" * 80)
        print(f"수급 데이터 검증: {name} ({ticker})")
        print("=" * 80)

        # 샘플 기간 설정 (지정일 기준 전후 5일)
        from datetime import datetime, timedelta
        sample_dt = datetime.strptime(sample_date, '%Y%m%d')
        start_dt = (sample_dt - timedelta(days=7)).strftime('%Y%m%d')
        end_dt = (sample_dt + timedelta(days=7)).strftime('%Y%m%d')

        try:
            # pykrx로 수급 데이터 가져오기
            df = stock.get_market_trading_value_by_date(
                start_dt,
                end_dt,
                ticker,
                detail=True
            )

            if df is None or len(df) == 0:
                print("[오류] 수급 데이터를 가져올 수 없습니다.")
                return False

            print(f"\n기간: {start_dt} ~ {end_dt}")
            print(f"행 수: {len(df)}개")
            print()

            # 주요 투자자 컬럼 확인
            key_investors = ['금융투자', '연기금', '개인', '외국인']

            print("=== 주요 투자자별 순매수 데이터 (억원 단위) ===")
            print()

            # 헤더 출력
            header = f"{'날짜':^12}"
            for inv in key_investors:
                header += f" | {inv:^12}"
            print(header)
            print("-" * 60)

            # 데이터 출력 (억원 단위로 변환)
            for date, row in df.iterrows():
                date_str = date.strftime('%Y-%m-%d')
                line = f"{date_str:^12}"
                for inv in key_investors:
                    if inv in df.columns:
                        value = row[inv] / 100_000_000  # 억원 단위
                        line += f" | {value:>10,.0f}억"
                    else:
                        line += f" | {'N/A':>12}"
                print(line)

            print()

            # 데이터 유효성 검증
            print("=== 데이터 유효성 검증 ===")
            all_valid = True

            for inv in key_investors:
                if inv in df.columns:
                    non_zero = (df[inv] != 0).sum()
                    total = len(df)
                    non_zero_pct = non_zero / total * 100

                    status = "✓ 정상" if non_zero > 0 else "✗ 오류 (모두 0)"
                    print(f"  {inv}: {status} (비영 데이터 {non_zero}/{total}개, {non_zero_pct:.1f}%)")

                    if non_zero == 0:
                        all_valid = False
                else:
                    print(f"  {inv}: ✗ 오류 (컬럼 없음)")
                    all_valid = False

            print()

            if all_valid:
                print("✓ 검증 성공: 모든 주요 투자자 데이터가 정상적으로 로드됩니다.")
            else:
                print("✗ 검증 실패: 일부 투자자 데이터에 문제가 있습니다.")

            return all_valid

        except Exception as e:
            print(f"[오류] 검증 중 오류 발생: {str(e)}")
            return False


if __name__ == '__main__':
    main()
