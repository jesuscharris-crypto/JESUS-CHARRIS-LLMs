import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# FASE 1 - PASO 2: GENERADOR DE CASOS DE USO
# =====================================================

def generar_caso_uso_data_drift():
    """
    Genera un caso de prueba aleatorio para la función detectar_data_drift
    """

    np.random.seed()

    # ---------------------------------------------------
    # 1. Generar datos aleatorios
    # ---------------------------------------------------
    n_samples = np.random.randint(60, 150)
    n_features = np.random.randint(4, 10)

    # Dataset original
    X_train = np.random.randn(n_samples, n_features)

    # Dataset nuevo (puede tener drift o no)
    drift_flag = np.random.choice([0, 1])

    if drift_flag == 1:
        # Introducir cambio en la media (drift)
        X_new = np.random.randn(n_samples, n_features) + np.random.uniform(1, 3)
    else:
        # Sin drift (misma distribución)
        X_new = np.random.randn(n_samples, n_features)

    columns = [f"feature_{i}" for i in range(n_features)]
    X_train_df = pd.DataFrame(X_train, columns=columns)
    X_new_df = pd.DataFrame(X_new, columns=columns)

    # ---------------------------------------------------
    # 2. Calcular OUTPUT esperado (ground truth)
    # ---------------------------------------------------

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_new_scaled = scaler.transform(X_new_df)

    # Reducir dimensionalidad
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_new_pca = pca.transform(X_new_scaled)

    # Calcular centroides
    centroide_train = np.mean(X_train_pca, axis=0)
    centroide_new = np.mean(X_new_pca, axis=0)

    # Distancia entre centroides
    distancia = np.linalg.norm(centroide_train - centroide_new)

    # Regla simple de decisión
    drift_detectado = bool(distancia > 0.5)

    expected_output = {
        "distancia_centroides": float(distancia),
        "drift_detectado": drift_detectado
    }

    # ---------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------

    inputs = {
        "X_train": X_train_df.copy(),
        "X_new": X_new_df.copy()
    }

    return inputs, expected_output


# =====================================================
# FASE 1 - PASO 3: PRUEBA DEL GENERADOR
# =====================================================

if __name__ == "__main__":

    inputs, expected = generar_caso_uso_data_drift()

    print("======== INPUTS ========")
    for k, v in inputs.items():
        print(f"{k}:\n{v}\n")

    print("======== EXPECTED OUTPUT ========")
    print("Distancia centroides:", expected["distancia_centroides"])
    print("Drift detectado:", expected["drift_detectado"])
