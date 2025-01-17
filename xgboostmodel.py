import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from scipy.stats import zscore
import streamlit as st
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import os

# Zaman adımına uygun model dosya adını oluştur
def get_model_filename(time_step):
    return f'xgboost_model_time_step_{time_step}.h5'

# Streamlit başlık ve ayarlar
st.title("Hisse Senedi Fiyat Tahmini")
st.write("Bu uygulama, Xgboost modeli ile hisse senedi fiyat tahmini yapar.")

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
       # close_prices = stock_data['Close'].values.reshape(-1, 1)
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
            return np.array(X), np.array(Y)


        # Zaman adımını belirleme
        time_step = determine_time_step(stock_data)

        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        # Veriyi LSTM modeline uygun şekle dönüştürme
        X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the time steps into features
        Y_train = Y_train.reshape(Y_train.shape[0], 1)  # Ensure Y_train is a 2D array

        # Model dosya adını belirle
        model_filename = get_model_filename(time_step)

        # Modelin var olup olmadığını kontrol et
        if os.path.exists(model_filename):
            # Mevcut modeli yükle
            bst = xgb.Booster()  # Modeli yüklemek için Booster nesnesi oluşturuyoruz
            bst.load_model(model_filename)
            st.write(f"Zaman adımına uygun mevcut model ({model_filename}) yüklendi.")

            # Tahmin için veriyi hazırlama
            dtest = xgb.DMatrix(X_test, label=Y_test)

            # Tahmin yapma
            predictions = bst.predict(dtest)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))

        else:
            # Model yoksa, yeni bir model eğit
            st.write(f"Zaman adımı ({time_step}) için uygun model bulunamadı. Yeni model eğitiliyor...")

            # Zaman adımını belirleme
            time_step = determine_time_step(stock_data)

            X_train, Y_train = create_dataset(train_data, time_step)
            X_test, Y_test = create_dataset(test_data, time_step)

            # Veriyi XGBoost için uygun hale getirme
            X_train = X_train.reshape(X_train.shape[0], -1)  # Zaman adımlarını özelliklere dönüştürme
            Y_train = Y_train.reshape(Y_train.shape[0], 1)  # Y_train'in 2D bir dizi olmasını sağlama

            # XGBoost model parametrelerini ayarlama
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 5,
                'eval_metric': 'rmse'
            }

            # Parametre grid'i
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.001],
                'max_depth': [3, 5, 7]
            }

            # GridSearchCV kullanarak en iyi parametreyi bulma
            grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=param_grid, cv=3,
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            grid_search.fit(X_train, Y_train)

            # En iyi parametreleri al
            best_params = grid_search.best_params_
            st.write("En İyi Parametreler:", best_params)

            best_model = grid_search.best_estimator_

            # Veriyi DMatrix formatına dönüştürme
            dtrain = xgb.DMatrix(X_train, label=Y_train)
            dtest = xgb.DMatrix(X_test, label=Y_test)

            # Eval seti oluşturma
            eval_set = [(dtrain, 'train'), (dtest, 'test')]

            # Modeli eğitme
            evals_result = {}
            bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=1000, evals=eval_set,
                            evals_result=evals_result, verbose_eval=True)

            # Modeli kaydetme
            model_filename = get_model_filename(time_step)
            bst.save_model(model_filename)
            st.write(f"Yeni model kaydedildi: {model_filename}")

            # Test seti üzerinde tahmin yapma
            predictions = bst.predict(dtest)
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1))

            # Eğitim ve test kaybını görselleştirme
            if evals_result:
                st.subheader("Eğitim ve Test Kaybı")
                plt.figure(figsize=(10, 6))

                plt.plot(evals_result['train']['rmse'], label='Eğitim Kaybı', marker='o', markersize=3)
                plt.plot(evals_result['test']['rmse'], label='Test Kaybı', marker='o', markersize=3)

                plt.xlabel('Epoch')
                plt.ylabel('RMSE (Root Mean Squared Error)')
                plt.title('XGBoost Model Kaybı')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # Streamlit ile görselleştirme
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

        # Gelecek tahminleri için giriş verilerini hazırlama
        future_input = scaled_data[-time_step:].reshape(1, -1)
        future_input = future_input.flatten()

        future_predictions = []
        for _ in range(num_days):
            input_data = future_input[-time_step:]  # Son zaman adımına göre veri al
            input_data = input_data.reshape(1, -1)  # 2D diziye dönüştür
            pred = bst.predict(xgb.DMatrix(input_data))  # Modeli kullanarak tahmin yap
            future_predictions.append(pred[0])
            future_input = np.append(future_input, pred)

        # Tahminleri orijinal ölçekte geri dönüştür
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Gelecek günler için tarihleri oluşturma
        future_dates = [end_date + timedelta(days=i) for i in range(1, num_days + 1)]

        # Gelecek fiyatları görselleştirme
        st.subheader(f"Gelecek {num_days} Gün için Tahminler")
        future_dates_str = [date.strftime('%Y-%m-%d') for date in future_dates]
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates_str, future_predictions, label='Gelecek Tahmin Edilen Fiyat', marker='o', markersize=3)
        plt.xlabel('Tarih')
        plt.ylabel('Fiyat ($)')
        plt.title(f"{ticker} Gelecek {num_days} Gün Tahmini")
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
