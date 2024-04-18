import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import RobustScaler
import pandas as pd
import sys
import os
import pickle
from pathlib import Path

model = Path("artifacts/model/voting.pkl")
balanced_df = Path('artifacts/balanced_df.csv')

voting_classifier = pickle.load(open(model, 'rb'))
df = pd.read_csv(balanced_df)
rob_scaler = RobustScaler()
df.insert(0, 'scaled_amount', rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1)))
df.insert(1, 'scaled_time', rob_scaler.fit_transform(df['Time'].values.reshape(-1,1)))
df.drop(['Time','Amount'], axis=1, inplace=True)
X = df.drop(['Class'], axis = 1)
y = df['Class']

original_ypred_test_pred = voting_classifier.predict_proba(X)[:, 1]
original_ypred_test = [1 if x >= 0.69 else 0 for x in original_ypred_test_pred]
cm = confusion_matrix(y, original_ypred_test)
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
sns.heatmap(cm, annot=True, ax=ax1, fmt='d', cmap="Blues")
ax1.set_title('Testing Confusion Matrix', fontsize=14)
plt.show()