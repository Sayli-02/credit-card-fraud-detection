import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def main():
    print("Loading a sample of fraudTrain.csv...")
    # Load a sample to keep memory usage and execution time low
    df = pd.read_csv('data/raw/fraudTrain.csv', nrows=200000)
    
    print("Dataset shape (sample):", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Identify target
    target = 'is_fraud'
    if target not in df.columns:
        for col in df.columns:
            if 'fraud' in col.lower() or 'class' in col.lower() or 'target' in col.lower():
                target = col
                break
    
    print(f"\nTarget column identified as: {target}")
    print(f"Target distribution in sample:\n{df[target].value_counts(normalize=True)}")
    
    # Drop identifiers and date columns that need complex parsing for a quick run
    cols_to_drop = ['Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'dob', 'trans_date_trans_time']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Label encode categorical features
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str)
        df_clean[col] = le.fit_transform(df_clean[col])
        
    df_clean = df_clean.fillna(0)
    
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    
    print("\nTraining quick RandomForest for feature importance...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Important Features:")
    for i in range(min(15, X.shape[1])):
        print(f"{i+1}. {X.columns[indices[i]]} ({importances[indices[i]]:.4f})")

if __name__ == "__main__":
    main()
