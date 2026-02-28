from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('Tips_Encoder.pkl', 'rb') as oe_file:
    oe = pickle.load(oe_file)

with open('tips_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 1. Get data from the HTML form (and convert numbers from strings to floats/ints)
        totalbill = float(request.form.get('totalbill'))
        gender = request.form.get('gender')
        smoker = request.form.get('smoker')
        day = request.form.get('day')
        time = request.form.get('time')
        size = int(request.form.get('size'))


        input_df = pd.DataFrame({
            'total_bill': [totalbill],
            'size': [size],
            'sex': [gender],
            'smoker': [smoker],
            'day': [day],
            'time': [time]
        })

        # 3. Encode the categorical text into numbers
        cat_cols = ['sex', 'smoker', 'day', 'time']
        input_df[cat_cols] = oe.transform(input_df[cat_cols])
        # 4. Predict
        prediction = model.predict(input_df)[0]

        # 5. Return the result to the user
        return f"""
            <h3>Prediction Complete!</h3>
            <p>Based on a bill of ${totalbill:.2f} for a party of {size}, the predicted tip is: <strong>${prediction:.2f}</strong></p>
            <a href="/">Try another prediction</a>
        """

    return render_template('home.html')

@app.route('/predict')
def predict():
    return 'Model Prediction Page'

if __name__ == '__main__':
    app.run()