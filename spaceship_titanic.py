import pandas as pd

def load_data(filepath: str):
    """
    CSV 파일 로드 함수
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ {filepath} 불러오기 성공!")
        print("  데이터 크기:", df.shape)
        return df

    except FileNotFoundError:
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("예상치 못한 오류 발생:", e)
