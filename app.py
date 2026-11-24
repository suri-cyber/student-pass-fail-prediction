
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the fitted scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':

        # Get values from the HTML form
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        previous_marks = float(request.form['previous_marks'])
        assignment_score = float(request.form['assignment_score'])

        # Arrange the actual values in the correct order
        student_data = [[
            study_hours,
            attendance,
            previous_marks,
            assignment_score
        ]]

        # Scale the new student data
        student_data_scaled = scaler.transform(student_data)

        # Predict the result
        prediction = model.predict(student_data_scaled)[0]

        # Display the prediction
        return render_template(
            'index.html',
            prediction=prediction
        )

    # Display the form when the page opens
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
