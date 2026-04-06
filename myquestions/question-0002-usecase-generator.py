import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# FASE 1 - PASO 2: GENERADOR DE CASOS DE USO
# =====================================================

def generar_caso_uso_pipeline_seleccion():
    """
    Genera un caso de prueba aleatorio para la función pipeline_seleccion_modelo
    """

    np.random.seed()

    # ---------------------------------------------------
    # 1. Generar datos aleatorios
    # ---------------------------------------------------
    n_samples = np.random.randint(60, 150)
    n_features = np.random.randint(5, 12)

    X = np.random.randn(n_samples, n_features)

    # Variable objetivo binaria
    y = np.random.choice([0, 1], size=n_samples)

    columns = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=columns)

    # Número de variables a seleccionar
    k = np.random.randint(2, n_features)

    # ---------------------------------------------------
    # 2. Calcular OUTPUT esperado (ground truth)
    # ---------------------------------------------------

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Selección de características
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    # Índices seleccionados
    selected_indices = selector.get_support(indices=True)
    selected_features = [columns[i] for i in selected_indices]

    # Modelo
    model = RandomForestClassifier()
    model.fit(X_selected, y)

    importancias = model.feature_importances_

    expected_output = {
        "variables_seleccionadas": selected_features,
        "importancias": importancias
    }

    # ---------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------

    inputs = {
        "X": X_df.copy(),
        "y": y.copy(),
        "k": k
    }

    return inputs, expected_output


# =====================================================
# FASE 1 - PASO 3: PRUEBA DEL GENERADOR
# =====================================================

if __name__ == "__main__":

    inputs, expected = generar_caso_uso_pipeline_seleccion()

    print("======== INPUTS ========")
    for k, v in inputs.items():
        print(f"{k}:\n{v}\n")

    print("======== EXPECTED OUTPUT ========")
    print("Variables seleccionadas:", expected["variables_seleccionadas"])
    print("Importancias:", expected["importancias"])
