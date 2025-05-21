from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Memuat model dan ringkasan statistik
model_data = joblib.load('naive_bayes_modelGG.pkl')
summaries = model_data['summaries']

# Fungsi menghitung distribusi normal
def calculate_probability(x, mean, std):
    if std == 0:
        std = 1e-6  # Hindari pembagian dengan nol
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Fungsi menghitung probabilitas setiap kelas dengan prior
def calculate_class_probabilities(summaries, input_data):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        # Gunakan probabilitas prior
        probabilities[class_value] = class_summaries['prior']
        for i in range(len(input_data)):
            mean = class_summaries['mean'].iloc[i]
            std = class_summaries['std'].iloc[i]
            probabilities[class_value] *= calculate_probability(input_data[i], mean, std)
    return probabilities

# Fungsi prediksi
def predict(summaries, input_data):
    probabilities = calculate_class_probabilities(summaries, input_data)
    return max(probabilities, key=probabilities.get)

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        print("Data yang diterima:", data)

        # Konversi input
        imt = float(data['imt']) if data.get('imt') is not None else None
        umur = float(data['umur']) if data.get('umur') is not None else None
        gdp = float(data['gdp']) if data.get('gdp') is not None else None
        tekanan_darah = float(data['tekanan_darah']) if data.get('tekanan_darah') is not None else None

        if None in [gdp, tekanan_darah, imt, umur]:
            return jsonify({'success': False, 'message': 'Data yang diterima tidak lengkap atau salah tipe'})

        input_data = [gdp, tekanan_darah, imt, umur]
        prediction = predict(summaries, input_data)

        return jsonify({'success': True, 'outcome': int(prediction)})

    except Exception as e:
        print("Error dalam prediksi:", e)
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
