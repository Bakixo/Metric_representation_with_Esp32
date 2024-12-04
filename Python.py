import serial
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBClassifier

# Seri port bağlantısı
ser = serial.Serial('/dev/ttyUSB0', 9600)

np.random.seed(0)
X = np.random.rand(200, 10)
y = np.random.randint(0, 2, 200)

# K-fold Cross-Validation
kf = KFold(n_splits=5)
fold = 1

def send_metrics_to_lcd(fold, mse, acc):
    metrics = f"{fold};{mse:.4f};{acc:.2f}"
    ser.write(metrics.encode())
    ser.write(b'\n')

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # MSE ve Accuracy hesapla
    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Terminalde MSE ve Accuracy değerlerini yazdır
    print(f"Fold {fold}: MSE = {mse:.4f}, ACC = {acc:.2f}")

    # Veriyi LCD'ye gönder
    send_metrics_to_lcd(fold, mse, acc)

    # Bir sonraki fold'a geç
    fold += 1
    time.sleep(2)  # Ekranda her fold için 2 saniye bekle
