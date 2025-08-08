import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Charger et prétraiter les données
df = pd.read_csv("C:/Users/2SH/Downloads/student+performance/student/student-mat.csv", sep=";")

# Encodage des colonnes catégoriques
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Séparer les entrées (X) et la cible (y)
X = df.drop('G3', axis=1).values
y = df['G3'].values  # Note finale (cible pour régression)

# Normalisation des données
X = StandardScaler().fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 1 : Modèle de régression
regression_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Sortie pour régression
])

regression_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Entraînement du modèle de régression
history_reg = regression_model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Évaluation du modèle de régression
reg_loss, reg_mae = regression_model.evaluate(X_test, y_test)
print(f"MAE (Régression) : {reg_mae:.2f}")

# Prédictions du modèle de régression
y_pred_reg = regression_model.predict(X_test).flatten()

# Étape 2 : Modèle de classification
# Conversion des notes en classes binaires
y_class = (y_test >= 10).astype(int)  # Succès (1) ou Échec (0)

classification_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Sortie pour classification binaire
])

classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape des prédictions pour la classification
X_class = y_pred_reg.reshape(-1, 1)

# Entraînement du modèle de classification
history_class = classification_model.fit(X_class, y_class, epochs=50, validation_split=0.2, batch_size=16)

# Évaluation du modèle de classification
class_loss, class_accuracy = classification_model.evaluate(X_class, y_class)
print(f"Précision (Classification) : {class_accuracy:.2f}")

# Matrice de confusion
y_pred_class = (classification_model.predict(X_class) >= 0.5).astype(int)
conf_matrix = confusion_matrix(y_class, y_pred_class)
print("Matrice de confusion :")
print(conf_matrix)

# Visualisation des résultats
plt.figure(figsize=(12, 5))

# Courbe de perte (Régression)
plt.subplot(1, 2, 1)
plt.plot(history_reg.history['loss'], label='Entraînement')
plt.plot(history_reg.history['val_loss'], label='Validation')
plt.title("Perte (Régression)")
plt.xlabel("Époques")
plt.ylabel("MSE")
plt.legend()

# Courbe de précision (Classification)
plt.subplot(1, 2, 2)
plt.plot(history_class.history['accuracy'], label='Entraînement')
plt.plot(history_class.history['val_accuracy'], label='Validation')
plt.title("Précision (Classification)")
plt.xlabel("Époques")
plt.ylabel("Précision")
plt.legend()

plt.show()
# Comparaison des valeurs réelles et prédites pour la régression
plt.figure(figsize=(12, 6))

plt.scatter(y_test, y_pred_reg, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.title("Comparaison des valeurs réelles et prédites (Régression)")
plt.xlabel('Valeurs Réelles (G3)')
plt.ylabel('Valeurs Prédites (G3)')

plt.show()

# Visualisation avec t-SNE pour la classification
# Réduction de la dimensionnalité des données d'entrée
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_test)

plt.figure(figsize=(8, 6))
# Classes réelles
plt.scatter(X_tsne[y_class == 0, 0], X_tsne[y_class == 0, 1], label="Échec (Classe réelle)", alpha=0.6, color='red')
plt.scatter(X_tsne[y_class == 1, 0], X_tsne[y_class == 1, 1], label="Succès (Classe réelle)", alpha=0.6, color='green')


plt.title("Visualisation t-SNE des données classifiées")
plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.show()

