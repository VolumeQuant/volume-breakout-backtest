"""
거래량 지표 계산 모듈

Volume_Ratio와 Z_Score를 계산합니다.
"""

import pandas as pd
import numpy as np


class VolumeIndicators:
    """거래량 지표를 계산하는 클래스"""

    @staticmethod
    def calculate_volume_ratio(df, window=20):
        """
        Volume_Ratio 계산: 당일 거래량 / 20일 평균 거래량

        Parameters:
        -----------
        df : pd.DataFrame
            종목별 데이터프레임 (Volume 컬럼 필요)
        window : int
            이동평균 기간 (기본값: 20일)

        Returns:
        --------
        pd.Series
            Volume_Ratio 값
        """
        # 20일 이동평균 계산
        volume_ma = df['Volume'].rolling(window=window, min_periods=window).mean()

        # Volume_Ratio 계산
        volume_ratio = df['Volume'] / volume_ma

        return volume_ratio

    @staticmethod
    def calculate_z_score(df, window=60):
        """
        Z_Score 계산: (당일 거래량 - 60일 평균) / 60일 표준편차

        Parameters:
        -----------
        df : pd.DataFrame
            종목별 데이터프레임 (Volume 컬럼 필요)
        window : int
            통계 계산 기간 (기본값: 60일)

        Returns:
        --------
        pd.Series
            Z_Score 값
        """
        # 60일 이동평균 계산
        volume_mean = df['Volume'].rolling(window=window, min_periods=window).mean()

        # 60일 표준편차 계산
        volume_std = df['Volume'].rolling(window=window, min_periods=window).std()

        # Z_Score 계산 (표준편차가 0인 경우 NaN 처리)
        z_score = (df['Volume'] - volume_mean) / volume_std

        return z_score

    @staticmethod
    def calculate_future_returns(df, holding_periods=[1, 3, 5, 10]):
        """
        미래 수익률 계산

        Parameters:
        -----------
        df : pd.DataFrame
            종목별 데이터프레임 (Close 컬럼 필요)
        holding_periods : list
            보유 기간 리스트 (일)

        Returns:
        --------
        pd.DataFrame
            각 기간별 미래 수익률이 추가된 데이터프레임
        """
        result_df = df.copy()

        for period in holding_periods:
            # 미래 종가 가져오기
            future_close = df['Close'].shift(-period)

            # 수익률 계산 (%)
            returns = ((future_close - df['Close']) / df['Close']) * 100

            # 컬럼 추가
            result_df[f'Return_{period}D'] = returns

        return result_df

    @staticmethod
    def add_all_indicators(df, volume_ratio_window=20, z_score_window=60):
        """
        모든 지표를 계산하여 데이터프레임에 추가합니다.

        Parameters:
        -----------
        df : pd.DataFrame
            전체 데이터프레임
        volume_ratio_window : int
            Volume_Ratio 계산 기간
        z_score_window : int
            Z_Score 계산 기간

        Returns:
        --------
        pd.DataFrame
            모든 지표가 추가된 데이터프레임
        """
        print("지표 계산 중...")

        # 종목별로 그룹화하여 계산
        result_list = []

        # 종목 코드 리스트
        if 'Code' in df.columns:
            unique_codes = df['Code'].unique()
        else:
            # Code 컬럼이 없으면 전체를 하나의 종목으로 처리
            unique_codes = [None]

        total = len(unique_codes)

        for idx, code in enumerate(unique_codes):
            if code is not None:
                # 해당 종목 데이터만 추출
                stock_df = df[df['Code'] == code].copy()
            else:
                stock_df = df.copy()

            # 데이터가 최소 기간 이상인지 확인
            if len(stock_df) < z_score_window:
                print(f"  [경고] 종목 {code}: 데이터가 {z_score_window}일 미만입니다 (건너뜀)")
                continue

            # Volume_Ratio 계산
            stock_df['Volume_Ratio'] = VolumeIndicators.calculate_volume_ratio(
                stock_df, window=volume_ratio_window
            )

            # Z_Score 계산
            stock_df['Z_Score'] = VolumeIndicators.calculate_z_score(
                stock_df, window=z_score_window
            )

            # 미래 수익률 계산
            stock_df = VolumeIndicators.calculate_future_returns(
                stock_df, holding_periods=[1, 3, 5, 10]
            )

            result_list.append(stock_df)

            # 진행상황 출력 (10%마다)
            if (idx + 1) % max(1, total // 10) == 0:
                progress = (idx + 1) / total * 100
                print(f"  진행률: {progress:.1f}% ({idx+1}/{total})")

        # 모든 데이터 합치기
        if len(result_list) == 0:
            print("\n[오류] 지표 계산에 실패했습니다!")
            return None

        result_df = pd.concat(result_list, ignore_index=False)

        # NaN 값 제거
        initial_count = len(result_df)
        result_df = result_df.dropna(subset=['Volume_Ratio', 'Z_Score'])
        final_count = len(result_df)

        print(f"\n지표 계산 완료!")
        print(f"- 초기 데이터: {initial_count:,}개 행")
        print(f"- 최종 데이터: {final_count:,}개 행")
        print(f"- 제거된 행: {initial_count - final_count:,}개")

        return result_df


def main():
    """테스트용 메인 함수"""
    # 샘플 데이터 생성 (테스트용)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.uniform(10000, 50000, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
        'Code': '005930',
        'Name': '삼성전자'
    })
    sample_data.set_index('Date', inplace=True)

    print("샘플 데이터:")
    print(sample_data.head())

    # 지표 계산
    result = VolumeIndicators.add_all_indicators(sample_data)

    if result is not None:
        print("\n지표 계산 결과:")
        print(result[['Close', 'Volume', 'Volume_Ratio', 'Z_Score',
                      'Return_1D', 'Return_3D', 'Return_5D', 'Return_10D']].head(70))

        # 통계 출력
        print("\nVolume_Ratio 통계:")
        print(result['Volume_Ratio'].describe())

        print("\nZ_Score 통계:")
        print(result['Z_Score'].describe())


if __name__ == '__main__':
    main()
