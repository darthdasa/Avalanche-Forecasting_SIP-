import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model_above = load('dt_above.h5')
model_near = load('dt_near.h5')
model_below = load('dt_below.h5')
scaler= load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
   
    x= request.form.values()
    x_test=[[]]

    for i in x:
        if i=="":
            x_test[0].append(0)
        else:
            x_test[0].append(i)
    
    print(x_test)
    print(len(x_test[0]))
    
    x_test_scaled= scaler.transform(x_test)
    
    
    prediction_above = model_above.predict(x_test_scaled)
    print(prediction_above)
    output_above=(prediction_above[0])
    chance_above = ''
    if output_above == 0:
        chance_above = "Considerable"
    elif output_above == 1:
        chance_above = "High"
    elif output_above == 2:
        chance_above = "Low"
    else:
        chance_above = "Moderate"
    
    
    prediction_near = model_near.predict(x_test_scaled)
    print(prediction_near)
    output_near=(prediction_near[0])
    chance_near = ''
    if output_near == 0:
        chance_near = "Considerable"
    elif output_near == 1:
        chance_near = "High"
    elif output_near == 2:
        chance_near= "Low"
    else:
        chance_near = "Moderate"
    
    
    prediction_below = model_below.predict(x_test_scaled)
    print(prediction_below)
    output_below=(prediction_below[0])
    chance_below = ''
    if output_below == 0:
        chance_below = "Considerable"
    elif output_below == 1:
        chance_below = "High"
    elif output_below == 2:
        chance_below= "Low"
    else:
        chance_below = "Moderate"
    
    return render_template('./index.html', 
                           prediction_text='The Danger level Above the Treeline is -{}, Near the Treeline is-{} and Below the Treeline is-{}'
                           .format(chance_above,chance_near,chance_below))



@app.route('/predict_api',methods=['POST'])
def predict_api():
  
    data = request.get_json(force=True)
    
    prediction_above = model_above.y_predict([np.array(list(data.values()))])
    output_above = prediction_above[0]
    
    prediction_near = model_near.y_predict([np.array(list(data.values()))])
    output_near = prediction_near[0]
    
    prediction_below = model_below.y_predict([np.array(list(data.values()))])
    output_below = prediction_below[0]
    
    return jsonify(output_above, output_near, output_below)

if __name__ == "__main__":
    app.run(debug=True)
    