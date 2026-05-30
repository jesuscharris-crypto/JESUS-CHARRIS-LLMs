import pandas as pd


def suavizar_temperaturas(df, ventana):
    df = df.copy()
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    return (
        df
        .set_index('fecha_hora')[['temperatura']]
        .rolling(ventana)
        .mean()
        .dropna()
    )
