import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train_scaled.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_scaled.csv'))
    
    X_train = train_df[['age', 'trestbps', 'chol']]
    y_train = train_df['target']
    X_test = test_df[['age', 'trestbps', 'chol']]
    y_test = test_df['target']
    
    return X_train, X_test, y_train, y_test

def train_base_models():
    X_train, X_test, y_train, y_test = load_data()
    
    models = {
        'RandomForest (Base)': RandomForestClassifier(random_state=42),
        'SVC (Base)': SVC(random_state=42),
        'KNN (Base)': KNeighborsClassifier(),
        'MLP Sklearn (Base)': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    print("Trenowanie bazowych modeli:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} - Baseline Accuracy: {acc*100:.2f}%")

    #GridSearchCV, feature_importances, tensroflowmlp 


if __name__ == "__main__":
    train_base_models()