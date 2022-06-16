import os
import pandas as pd
from flask import Flask, jsonify, request, render_template
from keras.preprocessing import image
from keras.models import load_model
import tensorflow_hub as tfhub
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = load_model("deeplearning.h5", custom_objects={
                       'KerasLayer': tfhub.KerasLayer})



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    wilayah = str(request.form['wilayah'])
    waktu = str(request.form['waktu'])
    kelembaban_persen = str(request.form['kelembaban_persen'])
    suhu_derajat_celsius = str(request.form['suhu_derajat_celsius'])
    banyakkotarawan = int(request.form['banyakkotarawan'])
    banyakkotarawan = (banyakkotarawan) // 25
    items = ['wilayah', 'waktu', 'kelembaban_persen', 'suhu_derajat_celcius', 'BanyakKotaRawan']
    data = [[wilayah, waktu, kelembaban_persen, suhu_derajat_celsius, banyakkotarawan]]
    data_df = pd.DataFrame(data=data, columns=items)

    X_df = pd.read_csv("https://raw.githubusercontent.com/ammaresok/Dataset/main/X_data.csv", usecols=items)
    predict_df = pd.concat([data_df, X_df])

    col_cat = [x for x in predict_df.columns if x not in ["BanyakKotaRawan"]]
    for var in col_cat:
        catlist = 'var'+''+var
        cat_list = pd.get_dummies(predict_df[var], prefix=var)
        data1= predict_df.join(cat_list)
        predict_df=data1
    
    data_vars = predict_df.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in col_cat]
    predict_df = predict_df[to_keep].copy()

    sc = StandardScaler()
    predict_df = pd.DataFrame(StandardScaler().fit_transform(predict_df), columns=predict_df.columns, index=predict_df.index)
    predictions = model.predict(predict_df.iloc[[0], :])
    predicted_class_indices=np.argmax(predictions)

    target = ['Berawan', 'Berawan Tebal', 'Cerah', 'Cerah Berawan', 
          'Hujan', 'Hujan Ringan', 'Hujan Lokal', 'Hujan Petir', 
          'Hujan Sedang', 'Udara Kabur']

    output = target[predicted_class_indices]
    

    return render_template("index.html", prediction_text='Cuaca di daerah tersebut kemungkinan {}'. format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)