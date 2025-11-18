import pandas as pd

def load_data(filepath: str):
   
    try:
        df = pd.read_csv(filepath)
        print(f"\n- {filepath} 불러오기 성공!")
        print("  데이터 크기:", df.shape)
        return df

    except FileNotFoundError:
        print(f"오류: {filepath} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print("예상치 못한 오류 발생:", e)
