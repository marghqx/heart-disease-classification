import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prepare_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df.fillna(df.median(), inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df[['age', 'trestbps', 'chol']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_df = pd.DataFrame(X_train_scaled, columns=['age', 'trestbps', 'chol'])
    train_df['target'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=['age', 'trestbps', 'chol'])
    test_df['target'] = y_test.values
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(data_dir, 'train_scaled.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_scaled.csv'), index=False)
    
    print(f"The files have been saved in: {data_dir}")

if __name__ == "__main__":
    prepare_data()