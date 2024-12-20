import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Streamlit başlık ve ayarlar
st.title("Hisse Senedi Fiyat Tahmini")
st.write("Bu uygulama, LSTM modeli kullanarak seçtiğiniz hisse senedinin gelecekteki fiyatlarını tahmin eder.")

# Kullanıcıdan hisse senedi sembolü ve tarih aralığı alalım
ticker = st.text_input("Hisse Sembolü (Örn: AAPL)", "AAPL")
start_date = st.date_input("Başlangıç Tarihi", pd.to_datetime("2015-01-01"))
end_date = st.date_input("Bitiş Tarihi", pd.to_datetime(datetime.today()))

# Kullanıcının tahmin etmek istediği gün sayısını seçmesine olanak tanıyalım
num_days = st.slider("Gelecek gün sayısını seçin", min_value=1, max_value=60, value=30)

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

        # Y eksenini 50'şer birim artırma
        if not stock_data.empty:  # Boş olup olmadığını kontrol et
            max_price = stock_data['Close'].max().item()  # Tek bir sayı olarak al
            max_price_with_offset = max_price + 100  # 100 ekle
            plt.yticks(np.arange(0, max_price_with_offset + 50, 50))  # Y eksenini ayarla
        else:
            st.write("Seçilen tarihler arasında veri bulunamadı.")

        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Kapanış fiyatını normalize etme
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

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
            return np.array(X), np.array(Y)

        # Zaman adımını kontrol et
        if len(train_data) < 1 or len(test_data) < 1:
            st.warning("Veri yetersiz. Eğitim ve test verileri için yeterli sayıda veri yok.")
        else:
            if len(train_data) < 15:
                time_step = 1  # Çok az veri varsa, zaman adımını 1 olarak ayarlayın
            else:
                # Verinin uzunluğuna bağlı olarak 5 ile 15 arasında bir zaman adımı belirleyin
                time_step = max(10, min(30, len(train_data) // 10))

            st.success(f"Zaman adımı {time_step} olarak ayarlandı.")

        # Dataset oluşturma
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        # Veriyi LSTM modeline uygun şekle dönüştürme
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # LSTM modelini tanımlama ve eğitme
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.1),
            LSTM(50, return_sequences=False),
            Dropout(0.1),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, Y_train, epochs=20, batch_size=512, validation_data=(X_test, Y_test), verbose=1)

        # Eğitim ve test kaybını görselleştirme
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

        # Tahminleri geri ölçeklendirme
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))

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

        # Gelecek gün için tahmin yapma
        all_future_predictions = []  # Tüm tahminleri saklamak için bir liste oluştur

        for _ in range(10):  # Tahmin işlemini 10 defa yap
            future_predictions = []
            last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)

            for _ in range(num_days):  # Kullanıcıdan alınan gün sayısı
                next_pred = model.predict(last_60_days)
                future_predictions.append(next_pred[0][0])
                last_60_days = np.append(last_60_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

            all_future_predictions.append(future_predictions)  # Her bir tahmin listesini sakla

        # Tüm tahminlerin ortalamasını hesapla
        average_future_predictions = np.mean(all_future_predictions, axis=0)

        # Gelecek tahminleri geri ölçeklendirme
        average_future_predictions = scaler.inverse_transform(np.array(average_future_predictions).reshape(-1, 1))

        # Gelecek günler için tarihleri oluşturma
        future_dates = [end_date + timedelta(days=i) for i in range(1, num_days + 1)]

        # Gelecek fiyatları görselleştirme
        st.subheader(f"Gelecek {num_days} Gün için Tahmin Edilen Fiyatlar (Ortalama)")
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, average_future_predictions, label='Ortalama Tahmin Edilen Gelecek Fiyatlar', marker='o', markersize=3)
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat ($)')
        plt.title(f"Gelecek {num_days} Gün İçin Tahmin Edilen Fiyatlar (Ortalama)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Test seti için tahminlerin geri ölçeklendirilmiş halleri ile gerçek değerleri karşılaştırma
        rmse = np.sqrt(mean_squared_error(Y_test_original, predictions))
        mae = mean_absolute_error(Y_test_original, predictions)

        # Ortalama kapanış fiyatını alarak doğruluk oranı hesaplama
        mean_price = Y_test_original.mean()
        accuracy = 100 - (rmse / mean_price * 100)

        # Sonuçları Streamlit ile gösterme
        st.subheader("Model Doğruluk ve Hata Değerleri")
        st.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
        st.write(f"MAE (Mean Absolute Error): {mae:.2f}")
        st.write(f"Model Doğruluğu: %{accuracy:.2f}")
