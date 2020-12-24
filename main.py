import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg


app = Flask('mpg_prediction')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    vehicle = request.get_json()
    print(vehicle)
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = predict_mpg(vehicle, model)

    result = {
        'mpg_prediction': list(predictions)
    }
    return jsonify(result)

@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"

# # define a route for our url,checking if our code is running or not
# @app.route('/ping', methods = ['GET'])

# def ping():
#     return "Pinging model application!"

# to start our appliacation we need this code:-

if __name__ == "__main__":
    app.run(debug = True)
    
