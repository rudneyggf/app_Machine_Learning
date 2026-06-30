import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df_mental = pd.read_csv('mental_health.csv')

ATRIBUTOS = [
    'sleep_hours', 'depression_score', 'panic_attack_history',
    'anxiety_score', 'family_history_mental_illness', 'social_support_score',
    'financial_stress_level', 'work_stress_level'
]

X = df_mental[ATRIBUTOS]
y = df_mental['mental_health_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=777,
    stratify=y
)

modelo_dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=777
)

modelo_dt.fit(X_train, y_train)


joblib.dump(
    {'model': modelo_dt, 'atributos': ATRIBUTOS},
    'modelo_mental_health.joblib'
)

