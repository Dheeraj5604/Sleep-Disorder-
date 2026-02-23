import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_clean(file_path):
    df = pd.read_csv(file_path)
    df['sleep_disorder'] = df['sleep_disorder'].fillna('None')

    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_target = LabelEncoder()
    df['sleep_disorder'] = le_target.fit_transform(df['sleep_disorder'])

    X = df.drop('sleep_disorder', axis=1)
    y = df['sleep_disorder']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_target, 'label_encoder.pkl')
    joblib.dump(le_gender, 'gender_encoder.pkl')

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, le_target