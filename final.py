import pandas as pd
from xgboost import XGBClassifier

# 1. 데이터 로드
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# PassengerId 백업
test["PassengerId_original"] = test["PassengerId"]

# 2. 전처리 함수 적용
train = preprocess(train)
test = preprocess(test)

# 3. Train/Label 분리
X = train.drop("Transported", axis=1)
y = train["Transported"].astype(int)  # XGBoost는 라벨을 0/1로 필요해

# 4. 모델 생성
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

# 5. 모델 학습
print("✔ 모델 학습 중...")
model.fit(X, y)

# 6. 예측
pred = model.predict(test)

# Bool로 변환 (과제에서 Bool로 요구)
pred = pred.astype(bool)

# 7. submission.csv 생성
submission = pd.DataFrame({
    "PassengerId": test["PassengerId_original"],
    "Transported": pred
})

submission.to_csv("submission.csv", index=False)
print("✔ submission.csv 저장 완료!")
