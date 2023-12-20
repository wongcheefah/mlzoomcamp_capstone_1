import pandas as pd
import pickle
from flask import Flask, request, jsonify

with open(f"./model/ccpp_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask("ccpp")


@app.route("/predict", methods=["POST"])
def predict():
    test_case = request.get_json()
    test_df = pd.DataFrame([test_case])
    y_pred = model.predict(test_df)

    result = {"PE": float(y_pred)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
