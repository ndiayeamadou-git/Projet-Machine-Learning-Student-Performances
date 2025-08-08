from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pickle

# Charger le modèle et les outils
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Créer l'application Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        input_data = request.form.to_dict()
        print("Données reçues :", input_data)  # Debugging: Affichez les données reçues
        
        # Convertir les données en un tableau compatible avec le modèle
        data = []
        for col in label_encoders.keys():  # Parcourez toutes les colonnes nécessaires
            if col in input_data:
                encoded_value = label_encoders[col].transform([input_data[col]])[0]
                data.append(encoded_value)
            else:
                data.append(0)  # Gérer les colonnes manquantes avec une valeur par défaut
         
        # Ajoutez les colonnes numériques (comme celles du formulaire)
        numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'freetime', 'goout', 
                           'failures', 'famrel', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
        for col in numeric_columns:
            if col in input_data:
                data.append(float(input_data[col]))
            else:
                data.append(0)  # Valeur par défaut si manquante
                
        # Vérifiez les données collectées
        print("Données préparées :", data)
        
        # Transformez les données en une forme compatible
        data = np.array(data).reshape(1, -1)
        
        # Normaliser les données avec le scaler préentraîné
        data_scaled = scaler.transform(data)
        
        # Faire la prédiction
        prediction = model.predict(data_scaled)[0][0]
        
        # Retourner la prédiction arrondie
        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        print("Erreur :", e)
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)


