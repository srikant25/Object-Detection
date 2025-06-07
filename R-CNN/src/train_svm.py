from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
import os
from utils import evaluate_mAP,NMS

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()
root_dir = os.getcwd()
data_path = os.path.join(root_dir,'R-CNN/data')


# Load saved features
X = np.load(os.path.join(current_dir,'fc2_features.npy'))
y_class = np.load(os.path.join(data_path,'train_label.npy'))
y_bbox = np.load(os.path.join(data_path,'target_box.npy'))



# Train-test split
X_train, X_test, y_train_cls, y_test_cls, y_train_box, y_test_box = train_test_split(
    X, y_class, y_bbox, test_size=0.2, random_state=42
)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train_cls)

# Regressor (bbox correction)
reg = MultiOutputRegressor(SVR())
reg.fit(X_train[y_train_cls == 1], y_train_box[y_train_cls == 1])  # only for object class

import joblib

# Save classifier
joblib.dump(clf, os.path.join(current_dir,'svm_classifier.pkl'))

# Save regressor
joblib.dump(reg, os.path.join(current_dir,'svr_regressor.pkl'))

# testing
y_pred_class  = clf.predict(X_test)
print(y_test_cls[:5])
print(y_pred_class[:5])
y_pred_box = reg.predict(X_test[y_test_cls == 1])

cls_ac , mean_ac=evaluate_mAP(y_test_cls, y_pred_class, y_test_box, y_pred_box)
print('accuracy on class besis',cls_ac)
print('accuracy of boundary box',mean_ac)


