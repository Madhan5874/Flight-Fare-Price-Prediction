import pickle
from flask import Flask, render_template,request

app=Flask(__name__)
model=pickle.load(open('flight_fare_prediction.pkl','rb'))

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def house_prediction():
    airline_company=int(request.form['airline_company'])
    source=int(request.form['source'])
    destination=int(request.form['destination'])
    stops=int(request.form['no_of_stops'])
    result=model.predict([[airline_company,source,destination,stops]])[0]
    prediction=round(result)

    return render_template('index.html', fare_prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)