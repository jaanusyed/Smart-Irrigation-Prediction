from ast import parse
from urllib import response
import numpy as np
from flask import Flask, render_template, request
import pickle
import requests
import json
from datetime import date, datetime


apiKey = "5047949cd60ed4752224b61d58fd8c19"


app = Flask(__name__)

model = pickle.load(open("D:\\c data\\Downloads\\Intelligent-farming-system-main\\Intelligent-farming-system-main\\RandomForest.pkl", 'rb'))


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    sowing = request.form['sowing']
    today = str(date.today())
    place = request.form['place']
    inputval = f'{place},IN'

    ans = fetchingApi(inputval)

    crop = int(request.form['crop'])
    days = days_between(sowing, today)
    moisture = int(request.form['moisture'])
    temperature = int(request.form['temp'])
    humidity = int(request.form['humidity'])
    output = ""

    if ans == False:
        output = "Irrigation not needed!"
    else:
        data = np.array([[crop, days, moisture, temperature, humidity]])
        prediction = model.predict(data)
        if moisture>=500:
            output = 'Irrigation not needed!'
        elif prediction[0] == 1:
            output = 'Start your Irrigation!'
        else:
            output = 'Irrigation not needed!'

    return render_template('index.html', prediction_text=output)


def fetchingApi(inputVal):
    heavy_rains_id = [201, 202, 211, 212, 232,
                      314, 501, 502, 503, 504, 511, 522]

    response_API = requests.get(
        f'https://api.openweathermap.org/data/2.5/weather?q={inputVal}&APPID={apiKey}')

    data = response_API.text
    parse_json = json.loads(data)

    # print(parse_json)
    loc_id = parse_json['weather'][0]['id']

    if loc_id in heavy_rains_id:
        return False
    else:
        return True


if __name__ == "__main__":
    app.run()
