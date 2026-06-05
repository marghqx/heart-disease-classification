import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=columns, na_values='?')
    df.fillna(df.median(), inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    plots_dir = os.path.join(project_root, 'plots')
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Rozkład cech
    plt.figure(figsize=(15, 5))
    features = ['age', 'trestbps', 'chol']
    for i, col in enumerate(features, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data=df, x=col, hue='target', kde=True, multiple="stack", palette="Set2")
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'features_distribution.png'), dpi=300)
    plt.close()
    
    # Macierz korelacji
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[['age', 'trestbps', 'chol', 'target']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    print(f"The plots have been saved in: {plots_dir}")

if __name__ == "__main__":
    generate_plots()