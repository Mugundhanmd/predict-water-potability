import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import requests

app = Flask(__name__)

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_purification_advice(outcome, ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    if not GROQ_API_KEY:
        return "Groq API key not configured. Mock Advice: Based on your metrics, ensure proper filtration and chemical balancing to secure safe water potability."
    
    prompt = f"""
    You are an expert water quality and sanitation advisor. A machine learning model has evaluated the water quality metrics and classified the water as: '{outcome}'.
    The sample has the following specific metrics:
    - pH Value: {ph}
    - Hardness (mg/L): {hardness}
    - Solids (ppm): {solids}
    - Chloramines (ppm): {chloramines}
    - Sulfate (mg/L): {sulfate}
    - Conductivity (uS/cm): {conductivity}
    - Organic Carbon (ppm): {organic_carbon}
    - Trihalomethanes (ug/L): {trihalomethanes}
    - Turbidity (NTU): {turbidity}

    Aligned with SDG 6: Clean Water and Sanitation, please provide actionable advice.
    - If the water is potable, confirm its safety and suggest regular monitoring metrics.
    - If it's not potable, outline the primary risks and provide a brief remediation or filtration step to make it potable.
    Keep the advice concise, positive, and direct. Add a touch of micro-tips where applicable.
    """
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Purification Tip: Ensure adequate filtration. (Groq API error code {response.status_code})"
    except Exception as e:
        return f"Error fetching purification advice from Groq: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        try:
            ph = float(request.form['ph value'])
            Hardness = float(request.form['Hardness'])
            Solids = float(request.form['Solids'])
            Chloramines = float(request.form['Chloramines'])
            Sulfate = float(request.form['Sulfate'])
            Conductivity = float(request.form['Conductivity'])
            Organic_carbon = float(request.form['Organic carbon'])
            Trihalomethanes = float(request.form['Trihalomethanes'])
            Turbidity = float(request.form['Turbidity'])

            val = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])

            model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost.sav')
            scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.sav')
            model = pickle.load(open(model_path, 'rb'))
            scc = pickle.load(open(scaler_path, 'rb'))

            data = scc.transform(val)
            res = model.predict(data)

            if res[0] == 1:
                outcome = 'Potable'
            else:
                outcome = 'Not Potable'
            
            advice = get_purification_advice(outcome, ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
            return render_template('index.html', result=outcome, advice=advice,
                                   ph=ph, hardness=Hardness, solids=Solids, chloramines=Chloramines,
                                   sulfate=Sulfate, conductivity=Conductivity, organic_carbon=Organic_carbon,
                                   trihalomethanes=Trihalomethanes, turbidity=Turbidity)
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        ph = float(data['ph'])
        Hardness = float(data['Hardness'])
        Solids = float(data['Solids'])
        Chloramines = float(data['Chloramines'])
        Sulfate = float(data['Sulfate'])
        Conductivity = float(data['Conductivity'])
        Organic_carbon = float(data['Organic_carbon'])
        Trihalomethanes = float(data['Trihalomethanes'])
        Turbidity = float(data['Turbidity'])

        val = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost.sav')
        scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.sav')

        model = pickle.load(open(model_path, 'rb'))
        scc = pickle.load(open(scaler_path, 'rb'))

        res = model.predict(scc.transform(val))
        outcome = 'Potable' if res[0] == 1 else 'Not Potable'
        
        advice = get_purification_advice(outcome, ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
        return jsonify({"success": True, "potability": outcome, "advice": advice})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
