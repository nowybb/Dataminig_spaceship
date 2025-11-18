import pandas as pd

def preprocess(df):
    """
    Spaceship Titanic 데이터 전처리 함수.
    train/test에 모두 공통 적용 가능.
    """

    # ------------------------------
    # 1. Cabin 컬럼 분해
    # ------------------------------
    # Cabin 구조: Deck/Num/Side → "B/0/P"
    df['Cabin'] = df['Cabin'].fillna('Missing/Missing/Missing')
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # CabinNum은 숫자 → float 변환
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    # ------------------------------
    # 2. Boolean 컬럼 처리
    # ------------------------------
    bool_cols = ["CryoSleep", "VIP"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    # ------------------------------
    # 3. 범주형(Categorical) 결측치 처리
    # ------------------------------
    cat_cols = ["HomePlanet", "Destination", "Deck", "Side", "Name"]
    for col in cat_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    # ------------------------------
    # 4. 수치형(Numerical) 결측치 처리 → median
    # ------------------------------
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # ------------------------------
    # 5. PassengerId 분리 (그룹별 feature)
    # ------------------------------
    # PassengerId 구조: "0001_01" → group, number
    df[['Group', 'GroupMember']] = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = df['Group'].astype(int)
    df['GroupMember'] = df['GroupMember'].astype(int)

    # ------------------------------
    # 6. 사용하지 않을 컬럼 제거
    # ------------------------------
    drop_cols = ["Cabin"]   # 원본 Cabin은 제거 (Deck/CabinNum/Side로 분해했기 때문)
    df = df.drop(columns=drop_cols)

    return df
