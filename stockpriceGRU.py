import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import zscore
import streamlit as st
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import os

# Zaman adımına uygun model dosya adını oluştur
def get_model_filename(time_step):
    return f'grumodel_time_step_{time_step}.h5'

# Streamlit başlık ve ayarlar
st.title("Hisse Senedi Fiyat Tahmini")
st.write("Bu uygulama, GridSearchCV kullanarak optimize edilen GRU modeli ile hisse senedi fiyat tahmini yapar.")

# Kullanıcıdan hisse senedi sembolü ve tarih aralığı alalım
ticker = st.text_input("Hisse Sembolü (Örn: AAPL)", "AAPL")
start_date = st.date_input("Başlangıç Tarihi", pd.to_datetime("2015-01-01"))
end_date = st.date_input("Bitiş Tarihi", pd.to_datetime(datetime.today()))

# Kullanıcının tahmin etmek istediği gün sayısını seçmesine olanak tanıyalım
num_days = st.slider("Gelecek gün sayısını seçin", min_value=1, max_value=60, value=30)

# Zaman adımını belirleme
def determine_time_step(data, min_time_step=2, average_time_step=5, max_time_step=20):
    """
    Zaman adımını veri boyutuna göre dinamik olarak belirler.

    :param data: Verinin tamamı (liste, dizi, vb.)
    :param min_time_step: Zaman adımının alabileceği minimum değer
    :param average_time_step: Orta düzey veri boyutları için varsayılan zaman adımı
    :param max_time_step: Zaman adımının alabileceği maksimum değer
    :return: Seçilen zaman adımı
    """
    # Veri uzunluğunu al
    data_length = len(data)

    # Hata kontrolü
    if min_time_step <= 0 or max_time_step <= 0:
        raise ValueError("Zaman adımları sıfırdan büyük olmalıdır!")
    if min_time_step > max_time_step:
        raise ValueError("Minimum zaman adımı maksimumdan büyük olamaz!")

    # Zaman adımını belirle
    if data_length < 500:
        time_step = min_time_step
        message = f"Veri boyutu ({data_length}) küçük. Zaman adımı {min_time_step} olarak ayarlandı."
    elif 500 <= data_length < 1000:
        time_step = average_time_step
        message = f"Veri boyutu ({data_length}) orta düzey. Zaman adımı {average_time_step} olarak ayarlandı."
    else:
        time_step = max_time_step
        message = f"Veri boyutu ({data_length}) büyük. Zaman adımı {max_time_step} olarak belirlendi."

    # Uyarıyı döndür
    try:
        import streamlit as st
        st.warning(message)
    except ImportError:
        print(message)

    return time_step

# Aykırı değerleri temizleme fonksiyonu
def remove_outliers(data, threshold=3):
    """
    Z-Skoru yöntemiyle aykırı değerleri temizler.
    :param data: Veriler (pandas DataFrame veya numpy array)
    :param threshold: Aykırı değer eşik değeri (örneğin, 3)
    :return: Aykırı değerler temizlenmiş veri
    """
    z_scores = zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)  # Tüm sütunlarda eşik değerini kontrol eder
    return data[filtered_entries]

# GRU modelini oluşturma fonksiyonu
from tensorflow.keras.regularizers import l2

def create_model(units=256, dropout_rate=0.02, l2_rate=0.001, learning_rate=0.0001):
    model = Sequential()

    # İlk GRU Katmanı
    model.add(GRU(units=units, return_sequences=True, kernel_regularizer=l2(l2_rate)))
    model.add(LayerNormalization())
    model.add(Dropout(dropout_rate))

    # İkinci GRU Katmanı
    model.add(GRU(units=units, return_sequences=False, kernel_regularizer=l2(l2_rate)))
    model.add(LayerNormalization())
    model.add(Dropout(dropout_rate))

    # Çıkış Katmanı
    model.add(Dense(1, kernel_regularizer=l2(l2_rate)))

    # Modeli Derleme
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Veriyi çekme düğmesi
if st.button("Veriyi Çek ve Modeli Eğit"):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        st.error("Seçilen tarihler arasında veri bulunamadı.")
    else:
        st.write("İlk birkaç veri:")
        st.write(stock_data.head())

        # Kapanış fiyatı grafiği
        st.subheader("Kapanış Fiyatı")
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Close'], label='Kapanış Fiyatı', marker='o', markersize=3)
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat ($)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.title(f"{ticker} Kapanış Fiyatları")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Kapanış fiyatlarını aykırı değerlerden arındırma
        st.write("Aykırı değerleri temizliyoruz...")
        filtered_close_prices = remove_outliers(stock_data[['Close']].values)

        if len(filtered_close_prices) < len(stock_data):
            st.success(
                f"Aykırı değerler başarıyla temizlendi. Orijinal veri boyutu: {len(stock_data)}, Temizlenmiş veri boyutu: {len(filtered_close_prices)}")
        else:
            st.info("Veride aykırı değer bulunamadı.")

        # Kapanış fiyatını normalize etme
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(filtered_close_prices)

        # Eğitim ve test verilerini ayırma (%80 eğitim, %20 test)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Zaman serisi için veri oluşturma fonksiyonu
        def create_dataset(data, time_step):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                Y.append(data[i + time_step, 0])
            # Eğer veriyi oluştururken eksiklik varsa uyarı veriyoruz
            if len(X) == 0:
                st.warning("Insufficient data for the specified time_step. Try using a smaller time_step or more data.")
            return np.array(X), np.array(Y)

        # Zaman adımını belirleme
        time_step = determine_time_step(stock_data)

        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        # Veriyi LSTM modeline uygun şekle dönüştürme
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Model dosya adını belirle
        model_filename = get_model_filename(time_step)

        # Modelin var olup olmadığını kontrol et
        if os.path.exists(model_filename):
            # Mevcut model yükleniyor
            best_model = load_model(model_filename)
            st.write(f"Zaman adımına uygun mevcut model ({model_filename}) yüklendi.")

            # `X_test` ve `Y_test` daha önce eğitimde kullanıldı ve mevcut
            predictions = best_model.predict(X_test)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            Y_test_original = scaler.inverse_transform(
                Y_test.reshape(-1, 1))  # Y_test_original her durumda tanımlanmalı
        else:

            # Model yok, eğitimi başlat
            st.write(f"Zaman adımı ({time_step}) için uygun model bulunamadı. Yeni model eğitiliyor...")

            # Zaman adımını belirleme
            time_step = determine_time_step(stock_data)

            X_train, Y_train = create_dataset(train_data, time_step)
            X_test, Y_test = create_dataset(test_data, time_step)

            # Veriyi GRU modeline uygun şekle dönüştürme
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # GridSearchCV için KerasRegressor kullanımı
            model = KerasRegressor(build_fn=create_model, verbose=1)

            # Hiperparametre grid tanımı
            param_grid = {
                'batch_size': [32],
                'epochs': [50],
            }

            # GridSearchCV uygulaması
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_result = grid.fit(X_train, Y_train)

            # En iyi parametreler
            best_params = grid_result.best_params_
            st.write("En İyi Parametreler:", best_params)

            # En iyi model ile tahmin yapma
            best_model = grid_result.best_estimator_

            # EarlyStopping ve ModelCheckpoint kullanımı
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_filename = get_model_filename(time_step)
            checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, mode='min')

            predictions = best_model.predict(X_test)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            Y_test_original = scaler.inverse_transform(
                Y_test.reshape(-1, 1))  # Y_test_original her durumda tanımlanmalı
            history = best_model.fit(X_train, Y_train, epochs=best_params['epochs'],
                                     batch_size=best_params['batch_size'], validation_data=(X_test, Y_test),
                                     callbacks=[early_stop, checkpoint])

            # Eğitim ve test kaybını görselleştirme
            if hasattr(history, 'history'):
                st.subheader("Eğitim ve Test Kaybı")
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Eğitim Kaybı', marker='o', markersize=3)
                plt.plot(history.history['val_loss'], label='Test Kaybı', marker='o', markersize=3)
                plt.xlabel('Epoch')
                plt.ylabel('Kayıp Değeri')
                plt.title('Model Kaybı')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)

        # Gerçek ve tahmin edilen fiyatları görselleştirme
        st.subheader("Gerçek ve Tahmin Edilen Fiyatlar")
        plt.figure(figsize=(10, 6))
        plt.plot(Y_test_original, label='Gerçek Fiyat', marker='o', markersize=3)
        plt.plot(predictions, label='Tahmin Edilen Fiyat', marker='o', markersize=3)
        plt.xlabel('Zaman')
        plt.ylabel('Fiyat ($)')
        plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Gelecek günler için tahmin yapma
        last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)
        future_predictions = []

        for _ in range(num_days):
            next_pred = best_model.predict(last_60_days)

            # next_pred'in boyutunu kontrol et ve uygun şekilde şekillendir
            if next_pred.ndim == 2:  # Eğer 2D ise, 3D'ye dönüştür
                next_pred = next_pred.reshape(1, 1, 1)
            elif next_pred.ndim == 1:  # Eğer 1D ise, 3D'ye dönüştür
                next_pred = next_pred.reshape(1, 1, 1)
            elif next_pred.ndim == 0:  # Eğer 0D ise, tek bir değeri doğrudan al
                next_pred = np.array([next_pred]).reshape(1, 1, 1)

            # Tek bir değeri alıyoruz
            future_predictions.append(next_pred[0, 0, 0])  # 3D diziden ilk değeri alıyoruz

            # Sonraki tahminleri hazırlamak için last_60_days'ı güncelle
            last_60_days = np.append(last_60_days[:, 1:, :], next_pred, axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Gelecek günler için tarihleri oluşturma
        future_dates = [end_date + timedelta(days=i) for i in range(1, num_days + 1)]

        # Gelecek fiyatları görselleştirme
        st.subheader(f"Gelecek {num_days} Gün için Tahmin Edilen Fiyatlar")
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, future_predictions, label='Tahmin Edilen Gelecek Fiyatlar', marker='o', markersize=3)
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat ($)')
        plt.title(f"Gelecek {num_days} Gün İçin Tahmin Edilen Fiyatlar")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
 # Hata metrikleri
        rmse = np.sqrt(mean_squared_error(Y_test_original, predictions))
        mse = mean_squared_error(Y_test_original, predictions)
        mae = mean_absolute_error(Y_test_original, predictions)
        mape_value = mean_absolute_percentage_error(Y_test_original, predictions)
        r2 = r2_score(Y_test_original, predictions)

        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"MAPE: {mape_value * 100:.2f}%")
        st.write(f"R^2 Skoru: {r2:.2f}")