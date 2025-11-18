import pandas as pd

def load_train_data(filepath="train.csv"):
    try:
        df = pd.read_csv(filepath)
        print("✔ train.csv 파일 불러오기 성공!")
        print("데이터 크기:", df.shape)
        print("컬럼 목록:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print("오류: train.csv 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    except Exception as e:
        print("예상치 못한 오류 발생:", e)

train_df = load_train_data()
