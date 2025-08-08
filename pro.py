import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# Charger le dataset
df = pd.read_csv("C:/Users/2SH/Downloads/student+performance/student/student-mat.csv", sep=";") 

# Affichage de l'aperçu des données
print(df.head())

# Conversion des variables catégoriques en numériques
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Définir les variables X (entrées) et y (cible)
X = df.drop('G3', axis=1).values
y = df['G3'].values

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construction du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='linear')
])


# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)

# Évaluation
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Erreur Absolue Moyenne (MAE) sur le jeu de test : {test_mae:.2f}')

# Affichage des courbes
plt.figure(figsize=(12, 5))

# Perte
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perte Entraînement')
plt.plot(history.history['val_loss'], label='Perte Validation')
plt.title('Courbe de Perte')
plt.xlabel('Époques')
plt.ylabel('Perte (MSE)')
plt.legend()

plt.show()

# Prédictions
y_pred = model.predict(X_test)

# Visualisation des prédictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title('Comparaison des valeurs réelles et prédites')
plt.xlabel('Valeurs Réelles (G3)')
plt.ylabel('Valeurs Prédites (G3)')
plt.show()
print("Nombre total d'échantillons :", len(df))
print(df.corr()['G3'])

# Sauvegarde du modèle et des outils
model.save('model.h5')  # Sauvegarder le modèle

# Sauvegarder le scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Sauvegarder les label encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


