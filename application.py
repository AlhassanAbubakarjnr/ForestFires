from flask import Flask, render_template, request
import pickle

application = Flask(__name__)
app = application

# Load model + scaler
model = pickle.load(open("models/ridge.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Debug check
print("Model type:", type(model))
print("Scaler type:", type(scaler))

@app.route("/")
def index():
    return render_template("index.html", results=None)

@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoints():
    try:
        # Collect form inputs
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Scale + predict
        new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        new_data_scaled = scaler.transform(new_data)
        result = model.predict(new_data_scaled)

        return render_template("home.html", results=float(result[0]))
    
    except Exception as e:
        print("Prediction error:", str(e))
        return render_template("home.html", results="Error: " + str(e))

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
