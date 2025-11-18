import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from spaceship_titanic import load_data
from prepro_train import preprocess


# =====================================================
# 1. 데이터 로드
# =====================================================
train = load_data("train.csv")
test = load_data("test.csv")

# test PassengerId 따로 저장 (align 전에 보관해야 함)
test_passenger_ids = test["PassengerId"].copy()


# =====================================================
# 2. 전처리
# =====================================================
print("\n✓ 전처리 중...")
train = preprocess(train)
test = preprocess(test)


# =====================================================
# 3. Train / Label 분리
# =====================================================
X = train.drop("Transported", axis=1)
y = train["Transported"].astype(int)


# =====================================================
# 4. 원-핫 인코딩
# =====================================================
print("\n✓ One-hot Encoding 중...")

X = pd.get_dummies(X)
test = pd.get_dummies(test)

# 컬럼을 일치시켜줌
X, test = X.align(test, join='left', axis=1)

# test에서 없는 값(NaN) 생기면 0으로 채움
test = test.fillna(0)


# =====================================================
# 5. Train / Validation 분리
# =====================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 6. 모델 생성
# =====================================================
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)


# =====================================================
# 7. 학습 — tqdm 프로그레스바 추가
# =====================================================
print("\n✓ 모델 학습 중...(검증용 split)")

for _ in tqdm(range(1), desc="Training (validation)"):
    model.fit(X_train, y_train)

# 검증 정확도 출력
valid_pred = model.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_pred)
print(f"\n✓ 검증 정확도: {valid_acc:.4f}\n")


# =====================================================
# 8. 전체 train 데이터로 다시 학습
# =====================================================
print("✓ 전체 train 데이터로 최종 학습 중...")

final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

for _ in tqdm(range(1), desc="Training (final)"):
    final_model.fit(X, y)


# =====================================================
# 9. test 예측
# =====================================================
print("\n✓ test 데이터 예측 중...")

test_pred = final_model.predict(test)
test_pred = test_pred.astype(bool)


# =====================================================
# 10. submission.csv 생성
# =====================================================
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Transported": test_pred
})

submission.to_csv("submission.csv", index=False)

print("\n✓ submission.csv 저장 완료!")
