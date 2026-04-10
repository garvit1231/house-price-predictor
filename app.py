from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
columns = json.load(open("columns.json", "r"))["data_columns"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/locations")
def get_locations():
    return jsonify({
        "locations": columns[3:]
    })

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))

    x[0] = float(sqft)
    x[1] = float(bath)
    x[2] = float(bhk)

    location = location.lower()

    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1

    return model.predict([x])[0]
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        price = predict_price(
            data["location"],
            float(data["sqft"]),
            float(data["bath"]),
            float(data["bhk"])
        )

        return jsonify({"price": round(price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/graph-data", methods=["POST"])
def graph_data():
    try:
        data = request.get_json()

        location = data["location"]
        bath = float(data["bath"])
        bhk = float(data["bhk"])
        sqft = float(data["sqft"])

        sqft_points = list(range(500, 2000, 100))
        price_points = []

        for s in sqft_points:
            pred = predict_price(location, s, bath, bhk)
            price_points.append(round(pred, 2))

        return jsonify({
            "labels": sqft_points,
            "prices": price_points
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
