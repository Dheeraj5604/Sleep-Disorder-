import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
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
    
    # Split BEFORE SMOTE to prevent data leakage into the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to balance and augment the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_target, 'label_encoder.pkl')
    joblib.dump(le_gender, 'gender_encoder.pkl')

    return X_train_scaled, X_test_scaled, y_train_resampled.values, y_test.values, le_target