import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# =====================================================
# FASE 1 - PASO 2: GENERADOR DE CASOS DE USO
# =====================================================

def generar_caso_uso_optimizacion():
    """
    Genera un caso de prueba aleatorio para la función optimizar_modelo_clasificacion
    """

    np.random.seed()

    # ---------------------------------------------------
    # 1. Generar datos aleatorios
    # ---------------------------------------------------
    n_samples = np.random.randint(80, 180)
    n_features = np.random.randint(4, 10)

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)

    columns = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=columns)

    # ---------------------------------------------------
    # 2. Calcular OUTPUT esperado (ground truth)
    # ---------------------------------------------------

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Modelo base
    model = GradientBoostingClassifier()

    # Espacio de búsqueda
    param_dist = {
        "n_estimators": np.arange(50, 150),
        "learning_rate": np.linspace(0.01, 0.3, 20),
        "max_depth": np.arange(2, 6)
    }

    # Búsqueda aleatoria
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        random_state=42
    )

    search.fit(X_scaled, y)

    expected_output = {
        "mejores_parametros": search.best_params_,
        "mejor_score": float(search.best_score_)
    }

    # ---------------------------------------------------
    # 3. Construir INPUT
    # ---------------------------------------------------

    inputs = {
        "X": X_df.copy(),
        "y": y.copy()
    }

    return inputs, expected_output


# =====================================================
# FASE 1 - PASO 3: PRUEBA DEL GENERADOR
# =====================================================

if __name__ == "__main__":

    inputs, expected = generar_caso_uso_optimizacion()

    print("======== INPUTS ========")
    for k, v in inputs.items():
        print(f"{k}:\n{v}\n")

    print("======== EXPECTED OUTPUT ========")
    print("Mejores parámetros:", expected["mejores_parametros"])
    print("Mejor score:", expected["mejor_score"])
