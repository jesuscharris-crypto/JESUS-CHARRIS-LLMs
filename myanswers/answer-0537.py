import pandas as pd


def calcular_totales(df: pd.DataFrame, categoria: str) -> pd.DataFrame:
    resultado = df[df['categoria'] == categoria].copy()
    resultado['total'] = resultado['cantidad'] * resultado['precio_unitario']
    return resultado[['producto', 'cantidad', 'precio_unitario', 'total']].reset_index(drop=True)
