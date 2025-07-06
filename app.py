from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and preprocessor
dtr = pickle.load(open('./dtr.pkl', 'rb'))
preprocessor = pickle.load(open('./preprocessor.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get form data
            Year = request.form['Year']
            average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
            pesticides_tonnes = request.form['pesticides_tonnes']
            avg_temp = request.form['avg_temp']
            Area = request.form['Area']
            Item = request.form['Item']

            # Format input for prediction
            features = np.array([[Year, average_rain_fall_mm_per_year,
                                pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            transformed_features = preprocessor.transform(features)
            prediction = dtr.predict(transformed_features).reshape(1, -1)

            # Return result page with prediction value
            return render_template('index.html', prediction=prediction)

    except ValueError:
        error_message = "⚠️ Please enter valid input values."
        return render_template("index.html", error=error_message)
    except Exception as e:
        error_message = f"⚠️ An error occurred: {str(e)}"
        return render_template("index.html", error=error_message)


if __name__ == '__main__':
    app.run(debug=True)
