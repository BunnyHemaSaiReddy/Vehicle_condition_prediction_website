from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open("vehicle.pkl", "rb"))


app = Flask(__name__)


columns = ['Engine Health', 'Brake Performance', 'Fuel System Condition',
           'Transmission System Health', 'Steering System Condition',
           'Tire Health', 'Battery Condition', 'Suspension System Health',
           'Cooling System Performance', 'Exhaust System Condition',
           'Headlight Condition', 'Tail Light Condition', 'AC Performance',
           'Wiper Performance', 'Window Operation', 'Door Lock Functionality',
           'Dashboard Electronics', 'Rearview Camera', 'Seat Adjustment Mechanism',
           'Horn Functionality']


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get values from form input
            values = [float(request.form[col]) for col in columns]
            values = np.array(values).reshape(1, -1)  

           
            prediction = model.predict(values)[0] 
            prediction = round(prediction, 2) 
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, columns=columns)

if __name__ == "__main__":
    app.run(debug=True)
