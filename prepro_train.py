import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def preprocess(df):

    # 1. Cabin → Deck / Num / Side
    df['Cabin'] = df['Cabin'].fillna("Missing/Missing/Missing")
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
    df['Deck'] = df['Deck'].fillna("Missing")
    df['Side'] = df['Side'].map({'P':0, 'S':1}).fillna(-1)

    # CabinNum 구간화
    df["CabinNum_bin"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 0, 50, 100, 150, 200, 300],
        labels=False
    ).astype("float")



    # 2. Boolean 처리
    bool_cols = ["CryoSleep", "VIP"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(int)   # bool → int 가 XGBoost에 더 좋음


   
    # 3. 범주형 missing
    cat_cols = ["HomePlanet", "Destination", "Deck", "Name"]
    for col in cat_cols:
        df[col] = df[col].fillna("Missing").astype(str)



    # 4. 숫자형 기본 결측치 처리
    num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())



    # 5. Spending Feature 강화
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]

    df["Spend_Total"] = df[spend_cols].sum(axis=1)
    df["Spend_Log"] = np.log1p(df["Spend_Total"])      

    # 항목별로 binary flag 추가
    for col in spend_cols:
        df[col + "_bool"] = (df[col] > 0).astype(int)



    # 6. Age Feature 강화
    df["Age_bin"] = pd.cut(
        df["Age"],
        bins=[0,12,18,30,50,100],
        labels=False
    ).astype("float")



    # 7. Group Feature 강화
    df[['Group', 'GroupMember']] = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = df['Group'].astype(int)
    df['GroupMember'] = df['GroupMember'].astype(int)
    df["Group_Size"] = df.groupby("Group")["Group"].transform("count")
    df["Group_Age_Mean"] = df.groupby("Group")["Age"].transform("mean")
    df["Group_Spend_Mean"] = df.groupby("Group")["Spend_Total"].transform("mean")
    df["Is_Alone"] = (df["Group_Size"] == 1).astype(int)



    # 8. 최종 불필요 컬럼 제거
    drop_cols = ["Cabin", "Name"]     
    df = df.drop(columns=drop_cols)

    return df
