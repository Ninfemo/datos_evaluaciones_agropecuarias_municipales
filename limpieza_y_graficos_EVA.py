# -*- coding: utf-8 -*-
"""
EVA (Evaluaciones Agropecuarias Municipales) - Cultivo de CAFÉ, Colombia
=========================================================================
Pipeline completo y reproducible:
  1. Carga y normalización de nombres de columnas
  2. Conversión de tipos (punto = miles, coma = decimal)
  3. Estandarización de texto (mayúsculas, sin tildes, conserva la Ñ)
  4. Filtro del cultivo objetivo (CAFE)
  5. Validación de la lectura numérica (Rendimiento reportado vs. recalculado)
  6. Faltantes, duplicados e inconsistencias lógicas (con auditoría)
  7. Estadísticas descriptivas (tendencia central, dispersión, posición)
  8. Gráficos descriptivos
  9. Taller 3: correlación, boxplot por grupos, tabla cruzada, pronósticos

Dependencias: pandas, numpy, scipy, matplotlib, seaborn>=0.13, statsmodels
    pip install pandas numpy scipy matplotlib "seaborn>=0.13" statsmodels
"""
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no abre ventanas; solo guarda los PNG
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
    HAY_STATSMODELS = True
except ImportError:
    HAY_STATSMODELS = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# ==========================================================================
# CONFIGURACIÓN
# ==========================================================================
CSV_PATH = "Evaluaciones_Agropecuarias_Municipales_EVA_20260927.csv"
CULTIVO_OBJETIVO = "CAFE"          # None -> usa TODOS los cultivos
RENDIMIENTO_MAX_PLAUSIBLE = 3.0    # t/ha; None para no marcar atípicos
EXCLUIR_ATIPICOS = False           # False: se marcan pero NO se eliminan
N_DEPTOS_BOXPLOT = 8
N_DEPTOS_TABLA_CRUZADA = 6
ANIOS_PRUEBA = 3                   # años finales reservados para evaluar pronósticos
CORRER_TALLER3 = True
OUT = Path("salida")
OUT.mkdir(exist_ok=True)

COL_SEMB = "Área Sembrada (ha)"
COL_COSE = "Área Cosechada (ha)"
COL_PROD = "Producción (t)"
COL_REND = "Rendimiento (t/ha)"
VARS_CUANT = [COL_SEMB, COL_COSE, COL_PROD, COL_REND]
COLS_TEXTO = ["DEPARTAMENTO", "MUNICIPIO", "GRUPO DE CULTIVO", "SUBGRUPO DE CULTIVO",
              "CULTIVO", "DESAGREGACION_REGIONAL", "PERIODO", "ESTADO FISICO PRODUCCION",
              "NOMBRE CIENTIFICO", "CICLO DE CULTIVO"]
COLS_ESPERADAS = (["COD_DEP", "COD_MUN", "AÑO"] + VARS_CUANT + COLS_TEXTO)


# ==========================================================================
# FUNCIONES AUXILIARES
# ==========================================================================
def titulo(txt):
    print("\n" + "=" * 80 + f"\n{txt}\n" + "=" * 80)


def quitar_tildes(t):
    """Quita tildes/diéresis pero conserva la Ñ (CAFÉ -> CAFE, NARIÑO -> NARIÑO)."""
    if not isinstance(t, str):
        return t
    t = t.replace("Ñ", "\uE000").replace("ñ", "\uE001")
    t = "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))
    return t.replace("\uE000", "Ñ").replace("\uE001", "ñ")


def renombrar_columna(col):
    """Mapea el nombre original a un nombre estándar (robusto a tildes y saltos de línea)."""
    c = quitar_tildes(" ".join(str(col).split())).upper()
    if re.match(r"^C\S{0,2}D\W", c) and "DEP" in c:
        return "COD_DEP"
    if re.match(r"^C\S{0,2}D\W", c) and "MUN" in c:
        return "COD_MUN"
    if re.fullmatch(r"A\S{0,2}O", c):
        return "AÑO"
    if "DESAGREG" in c:
        return "DESAGREGACION_REGIONAL"
    if "SEMBRADA" in c:
        return COL_SEMB
    if "COSECHADA" in c:
        return COL_COSE
    if "RENDIMIENTO" in c:
        return COL_REND
    if c.startswith("ESTADO"):
        return "ESTADO FISICO PRODUCCION"
    if c.startswith("PRODUCC"):
        return COL_PROD
    if "NOMBRE" in c and "CIENT" in c:
        return "NOMBRE CIENTIFICO"
    if "CICLO" in c:
        return "CICLO DE CULTIVO"
    if c.startswith("SUBGRUPO"):
        return "SUBGRUPO DE CULTIVO"
    if c.startswith("GRUPO"):
        return "GRUPO DE CULTIVO"
    return c


def a_numero(serie):
    """Texto -> número. El archivo usa '.' como separador de MILES y ',' como DECIMAL."""
    s = serie.astype("string").str.strip()
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def formatear_codigo(serie, ancho):
    num = a_numero(serie)
    return num.map(lambda x: f"{int(x):0{ancho}d}" if pd.notna(x) else np.nan)


def normalizar_texto(serie):
    s = serie.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    s = s.map(quitar_tildes)
    return s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA}).astype(object)


def leer_csv(ruta):
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(ruta, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("No se pudo decodificar el CSV (probado utf-8 y latin-1).")


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" • {OUT / nombre}")


def p_fmt(p):
    return "< .001" if p < 0.001 else f"{p:.3f}"


# ==========================================================================
# 1. CARGA Y NORMALIZACIÓN DE COLUMNAS
# ==========================================================================
titulo("1. CARGA DE DATOS")
df = leer_csv(CSV_PATH)
df.columns = [renombrar_columna(c) for c in df.columns]
faltan = [c for c in COLS_ESPERADAS if c not in df.columns]
if faltan:
    raise KeyError(f"Columnas no encontradas tras renombrar: {faltan}\nColumnas actuales: {list(df.columns)}")
total_archivo = len(df)
print(f"Registros en el archivo (todos los cultivos): {total_archivo:,}")
print(f"Variables: {len(df.columns)}")

# ==========================================================================
# 2. TIPOS DE DATOS
# ==========================================================================
for col in [COL_SEMB, COL_COSE, COL_PROD, COL_REND]:
    df[col] = a_numero(df[col])
df["AÑO"] = a_numero(df["AÑO"]).round()
df["COD_DEP"] = formatear_codigo(df["COD_DEP"], 2)
df["COD_MUN"] = formatear_codigo(df["COD_MUN"], 5)

# ==========================================================================
# 3. TEXTO (PERIODO se conserva como TEXTO: antes se volvía "2007.0")
# ==========================================================================
for col in COLS_TEXTO:
    df[col] = normalizar_texto(df[col])

# ==========================================================================
# 4. FILTRO DEL CULTIVO OBJETIVO
# ==========================================================================
if CULTIVO_OBJETIVO:
    candidatos = sorted(v for v in df["CULTIVO"].dropna().unique() if CULTIVO_OBJETIVO in v)
    print(f"\nValores de CULTIVO que contienen '{CULTIVO_OBJETIVO}': {candidatos}")
    df = df[df["CULTIVO"] == CULTIVO_OBJETIVO].copy()
    print(f"Registros de {CULTIVO_OBJETIVO}: {len(df):,} (de {total_archivo:,})")
    if df.empty:
        raise ValueError("El filtro de cultivo dejó 0 registros.")

# ==========================================================================
# 5. VALIDACIÓN DE LA LECTURA NUMÉRICA
#    Compara Rendimiento reportado vs Producción / Área Cosechada. Una tasa alta
#    de discrepancias delata un error de formato (como el de los miles).
# ==========================================================================
titulo("5. VALIDACIÓN DE LECTURA NUMÉRICA")
rend_calc = df[COL_PROD] / df[COL_COSE].where(df[COL_COSE] > 0)
comparable = df[COL_REND].notna() & rend_calc.notna()
discrepa = comparable & ((df[COL_REND] - rend_calc).abs() > 0.01 + 0.01 * rend_calc.abs())
pct_disc = discrepa.sum() / max(comparable.sum(), 1)
print(f"Rendimiento reportado vs recalculado: {int(discrepa.sum()):,} discrepancias "
      f"de {int(comparable.sum()):,} comparables ({pct_disc:.2%})")
if pct_disc > 0.01:
    print("  ⚠ Más de 1% de discrepancias: revisar separadores decimal/miles del CSV.")

mismatch_cod = df["COD_MUN"].notna() & df["COD_DEP"].notna() & (df["COD_MUN"].str[:2] != df["COD_DEP"])
print(f"Códigos municipio/departamento incoherentes: {int(mismatch_cod.sum()):,}")

print("\nValores nulos por columna (antes de imputar):")
nulos = df.isna().sum()
print(nulos[nulos > 0].to_string() if (nulos > 0).any() else "  Sin nulos.")

# ==========================================================================
# 6. FALTANTES, DUPLICADOS E INCONSISTENCIAS
# ==========================================================================
titulo("6. LIMPIEZA")
n_inicial = len(df)

# 6.1 Imputación de nombre científico (moda por cultivo)
mapa_nc = (df.dropna(subset=["NOMBRE CIENTIFICO", "CULTIVO"])
             .groupby("CULTIVO")["NOMBRE CIENTIFICO"].agg(lambda s: s.mode().iat[0]))
df["NOMBRE CIENTIFICO"] = (df["NOMBRE CIENTIFICO"]
                           .fillna(df["CULTIVO"].map(mapa_nc)).fillna("NO REGISTRADO"))

# 6.2 Duplicados exactos
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
clave = ["COD_MUN", "CULTIVO", "AÑO", "PERIODO", "DESAGREGACION_REGIONAL",
         "ESTADO FISICO PRODUCCION", "CICLO DE CULTIVO"]
n_dup_clave = int(df.duplicated(subset=clave).sum())
print(f"Duplicados exactos eliminados: {n_dup:,}")
print(f"Filas con clave repetida (municipio-año-periodo-desagregación...), solo informativo: {n_dup_clave:,}")

# 6.3 Reglas de inconsistencia (un registro puede incumplir varias)
sem, cos, prod = df[COL_SEMB], df[COL_COSE], df[COL_PROD]
reglas = pd.DataFrame({
    "Datos numéricos faltantes": sem.isna() | cos.isna() | prod.isna(),
    "Valores negativos": (sem < 0) | (cos < 0) | (prod < 0),
    "Año faltante o fuera de rango": ~df["AÑO"].between(1990, 2100),
    "Área Cosechada > Área Sembrada": cos > sem,
    "Producción > 0 con Área Cosechada = 0": (prod > 0) & (cos == 0),
})
descartar = reglas.any(axis=1)

df_descartados = df[descartar].copy()
if df_descartados.empty:
    df_descartados["MOTIVO_DESCARTADO"] = pd.Series(dtype=str)
else:
    df_descartados["MOTIVO_DESCARTADO"] = reglas[descartar].apply(lambda f: " | ".join(f.index[f]), axis=1)

df_limpio = df[~descartar].copy()
df_limpio["AÑO"] = df_limpio["AÑO"].astype(int)

# 6.4 Rendimiento: se recalcula con precisión completa.
#     Con Área Cosechada = 0 el rendimiento NO está definido -> NaN (antes se rellenaba con 0,
#     lo que sesgaba media y mediana hacia abajo).
df_limpio[COL_REND] = df_limpio[COL_PROD] / df_limpio[COL_COSE].where(df_limpio[COL_COSE] > 0)

# 6.5 Atípicos de rendimiento: se MARCAN para verificación contra la fuente
if RENDIMIENTO_MAX_PLAUSIBLE is not None:
    df_limpio["ATIPICO_RENDIMIENTO"] = df_limpio[COL_REND] > RENDIMIENTO_MAX_PLAUSIBLE
    atip = df_limpio[df_limpio["ATIPICO_RENDIMIENTO"]]
    atip.to_csv(OUT / "EVA_atipicos_rendimiento.csv", index=False, encoding="utf-8-sig")
    print(f"\nAtípicos de rendimiento (> {RENDIMIENTO_MAX_PLAUSIBLE} t/ha): {len(atip):,}")
    if len(atip):
        print(atip.groupby(["DEPARTAMENTO", "AÑO"]).size().to_string())
    if EXCLUIR_ATIPICOS:
        df_limpio = df_limpio[~df_limpio["ATIPICO_RENDIMIENTO"]].copy()
        print("  -> excluidos del dataset limpio (EXCLUIR_ATIPICOS=True)")

df_limpio.to_csv(OUT / "EVA_limpio.csv", index=False, encoding="utf-8-sig")
df_descartados.to_csv(OUT / "EVA_descartados.csv", index=False, encoding="utf-8-sig")

print("\n--- AUDITORÍA ---")
print(f"Registros del cultivo antes de limpiar:   {n_inicial:,}")
print(f"Registros en dataset limpio:              {len(df_limpio):,}")
print(f"Registros descartados:                    {len(df_descartados):,} "
      f"({len(df_descartados) / max(n_inicial, 1):.2%})")
print("Incumplimiento por regla (no excluyentes):")
for k, v in reglas.sum().items():
    print(f"  • {k}: {int(v):,}")
info_cero = int(((df_limpio[COL_COSE] > 0) & (df_limpio[COL_PROD] == 0)).sum())
print(f"Informativo (NO descartados): Área Cosechada > 0 con Producción = 0: {info_cero:,}")

# ==========================================================================
# 7. ESTADÍSTICAS DESCRIPTIVAS
# ==========================================================================
titulo("7. ESTADÍSTICAS DESCRIPTIVAS")


def resumen_cuantitativo(d, cols):
    filas = []
    for c in cols:
        s = d[c].dropna()
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        filas.append({
            "Variable": c, "n": len(s), "Mínimo": s.min(), "Q1": q1, "Mediana": med, "Q3": q3,
            "Máximo": s.max(), "Promedio": s.mean(), "Desv. estándar": s.std(), "RIC": q3 - q1,
            "CV (%)": s.std() / s.mean() * 100 if s.mean() else np.nan,
            "Asimetría": s.skew(), "Curtosis": s.kurt(),
        })
    return pd.DataFrame(filas).set_index("Variable")


desc = resumen_cuantitativo(df_limpio, VARS_CUANT)
print(desc.round(2).to_string())
desc.round(4).to_csv(OUT / "estadisticas_cuantitativas.csv", encoding="utf-8-sig")
print("\n(Rendimiento: solo registros con Área Cosechada > 0)")

for c in ["DEPARTAMENTO", "CICLO DE CULTIVO", "GRUPO DE CULTIVO", "ESTADO FISICO PRODUCCION"]:
    vc = df_limpio[c].value_counts(dropna=False)
    tabla = pd.DataFrame({"Cantidad": vc, "%": (vc / vc.sum() * 100).round(2)})
    print(f"\n{c} (Top 10):")
    print(tabla.head(10).to_string())
    tabla.to_csv(OUT / f"frecuencias_{c.replace(' ', '_')}.csv", encoding="utf-8-sig")

anual = (df_limpio.groupby("AÑO")
         .agg(Produccion_t=(COL_PROD, "sum"), Registros=("COD_MUN", "size"), Municipios=("COD_MUN", "nunique")))
print("\nProducción anual y cobertura (si Municipios cambia mucho, la caída puede ser de cobertura):")
print(anual.to_string())
anual.to_csv(OUT / "produccion_anual.csv", encoding="utf-8-sig")

# ==========================================================================
# 8. GRÁFICOS DESCRIPTIVOS
# ==========================================================================
titulo("8. GRÁFICOS DESCRIPTIVOS")

# 8.1 Histogramas (escala log por el sesgo; log no admite 0)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), VARS_CUANT):
    datos = df_limpio.loc[df_limpio[col] > 0, col]
    sns.histplot(datos, bins=50, log_scale=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribución de {col} (escala log)")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
fig.suptitle("Histogramas de variables cuantitativas", fontsize=14)
fig.tight_layout()
guardar(fig, "01_histogramas_cuantitativas.png")

# 8.2 Boxplots (escala log)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), VARS_CUANT):
    datos = df_limpio.loc[df_limpio[col] > 0, col]
    sns.boxplot(x=datos, ax=ax, color="lightcoral")
    ax.set_xscale("log")
    ax.set_title(f"Boxplot de {col} (escala log)")
    ax.set_xlabel(col)
fig.suptitle("Boxplots de variables cuantitativas", fontsize=14)
fig.tight_layout()
guardar(fig, "02_boxplots_cuantitativas.png")

# 8.3 Barras cualitativas: solo variables con más de una categoría
#     (en café CULTIVO/GRUPO/CICLO/ESTADO tienen una sola categoría y no informan)
cands = ["DEPARTAMENTO", "MUNICIPIO", "GRUPO DE CULTIVO", "CICLO DE CULTIVO", "ESTADO FISICO PRODUCCION"]
cuali = [c for c in cands if df_limpio[c].nunique() > 1][:3]
if cuali:
    fig, axes = plt.subplots(1, len(cuali), figsize=(6 * len(cuali), 6))
    for ax, col in zip(np.atleast_1d(axes), cuali):
        top = df_limpio[col].value_counts().head(10)
        sns.barplot(x=top.values, y=top.index.astype(str), hue=top.index.astype(str),
                    palette="viridis", legend=False, ax=ax)
        ax.set_title(f"{col} (Top 10)")
        ax.set_xlabel("Cantidad de registros")
        ax.set_ylabel("")
    fig.suptitle("Distribución de variables cualitativas", fontsize=14)
    fig.tight_layout()
    guardar(fig, "03_barras_cualitativas.png")

# 8.4 Producción total por año
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=anual.index, y=anual["Produccion_t"], marker="o", ax=ax, color="darkgreen")
ax.set_title("Producción total (t) por año")
ax.set_xlabel("Año")
ax.set_ylabel("Producción total (t)")
ax.ticklabel_format(style="plain", axis="y")
fig.tight_layout()
guardar(fig, "04_produccion_por_anio.png")

# 8.5 Matrices de correlación (Pearson y Spearman; pares completos)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, met in zip(axes, ["pearson", "spearman"]):
    sns.heatmap(df_limpio[VARS_CUANT].corr(method=met), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Matriz de correlación ({met.capitalize()})")
fig.tight_layout()
guardar(fig, "05_matriz_correlacion.png")

# ==========================================================================
# 9. TALLER 3: ANÁLISIS CON DOS VARIABLES Y PRONÓSTICOS
# ==========================================================================
if CORRER_TALLER3:
    # ---------- 9.1 Correlación (Pearson + IC Fisher, R², Spearman, log-log) ----------
    titulo("9.1 CORRELACIÓN ENTRE VARIABLES CUANTITATIVAS")

    def correlacion(d, x, y):
        d = d[[x, y]].dropna()
        n = len(d)
        r, p = stats.pearsonr(d[x], d[y])
        zc = stats.norm.ppf(0.975)
        lo, hi = np.tanh(np.arctanh(r) + np.array([-zc, zc]) / np.sqrt(n - 3))
        rho = stats.spearmanr(d[x], d[y])[0]
        dp = d[(d[x] > 0) & (d[y] > 0)]
        r_log = stats.pearsonr(np.log10(dp[x]), np.log10(dp[y]))[0]
        return {"Par": f"{x} – {y}", "n": n, "r": r, "IC95 inf": lo, "IC95 sup": hi,
                "R²": r ** 2, "ρ Spearman": rho, "r log-log": r_log, "p": p_fmt(p)}

    pares = [(COL_COSE, COL_PROD), (COL_SEMB, COL_COSE), (COL_COSE, COL_REND)]
    tabla_corr = pd.DataFrame([correlacion(df_limpio, x, y) for x, y in pares]).set_index("Par")
    print(tabla_corr.round(3).to_string())
    tabla_corr.round(4).to_csv(OUT / "tabla3_correlaciones.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    d1 = df_limpio[[COL_COSE, COL_PROD]].dropna()
    sns.regplot(data=d1, x=COL_COSE, y=COL_PROD, ci=None, ax=axes[0],
                scatter_kws=dict(s=6, alpha=0.3, color="green"), line_kws=dict(color="red"))
    axes[0].set_title(f"Producción vs Área cosechada (r = {tabla_corr.iloc[0]['r']:.3f})")
    d2 = d1[(d1 > 0).all(axis=1)]
    axes[1].scatter(d2[COL_COSE], d2[COL_PROD], s=6, alpha=0.3, color="green")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Área cosechada (ha, log)"); axes[1].set_ylabel("Producción (t, log)")
    axes[1].set_title(f"Escala log-log (r = {tabla_corr.iloc[0]['r log-log']:.3f})")
    d3 = df_limpio[[COL_COSE, COL_REND]].dropna()
    sns.regplot(data=d3, x=COL_COSE, y=COL_REND, ci=None, ax=axes[2],
                scatter_kws=dict(s=6, alpha=0.3, color="purple"), line_kws=dict(color="red"))
    n_omit = int((d3[COL_REND] > 3).sum())
    axes[2].set_ylim(0, 3)
    axes[2].set_title(f"Rendimiento vs Área cosechada (r = {tabla_corr.iloc[2]['r']:.3f})\n"
                      f"eje limitado a 3 t/ha; {n_omit} puntos fuera de la vista")
    fig.tight_layout()
    guardar(fig, "06_dispersion_correlacion.png")

    # ---------- 9.2 Boxplot de rendimiento por departamento ----------
    titulo("9.2 RENDIMIENTO POR DEPARTAMENTO")
    dd = df_limpio.dropna(subset=[COL_REND, "DEPARTAMENTO"])
    top = dd["DEPARTAMENTO"].value_counts().head(N_DEPTOS_BOXPLOT).index
    dd = dd[dd["DEPARTAMENTO"].isin(top)]
    orden = dd.groupby("DEPARTAMENTO")[COL_REND].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=dd, x="DEPARTAMENTO", y=COL_REND, order=orden, hue="DEPARTAMENTO",
                palette="Set2", legend=False, fliersize=3, ax=ax)
    ax.set_title("Diagrama de cajas: Rendimiento por departamento")
    ax.set_xlabel("Departamento"); ax.set_ylabel("Rendimiento (t/ha)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    guardar(fig, "07_boxplot_rendimiento_departamento.png")

    g = dd.groupby("DEPARTAMENTO")[COL_REND]
    tabla4 = pd.DataFrame({
        "n": g.size(), "Media": g.mean(), "Mediana": g.median(),
        "Q1": g.quantile(0.25), "Q3": g.quantile(0.75), "Desv. est.": g.std(), "Máx.": g.max(),
    })
    tabla4["RIC"] = tabla4["Q3"] - tabla4["Q1"]
    tabla4 = tabla4.loc[orden]
    print(tabla4.round(2).to_string())
    tabla4.round(4).to_csv(OUT / "tabla4_rendimiento_departamento.csv", encoding="utf-8-sig")

    grupos = [x.values for _, x in dd.groupby("DEPARTAMENTO")[COL_REND]]
    k, N = len(grupos), len(dd)
    W, p_lev = stats.levene(*grupos)            # scipy usa center='median' (Brown-Forsythe)
    H, p_kw = stats.kruskal(*grupos)
    F, p_f = stats.f_oneway(*grupos)
    print(f"\nLevene: W = {W:.2f}, p {p_fmt(p_lev)}")
    print(f"Kruskal-Wallis: H({k - 1}) = {H:.2f}, p {p_fmt(p_kw)}, ε² = {H / (N - 1):.3f}")
    print(f"ANOVA: F({k - 1}, {N - k}) = {F:.2f}, p {p_fmt(p_f)}")

    # ---------- 9.3 Tabla cruzada Departamento x Nivel de rendimiento ----------
    titulo("9.3 TABLA CRUZADA Y CHI-CUADRADO")
    dc = df_limpio.dropna(subset=[COL_REND, "DEPARTAMENTO"]).copy()
    q1, q3 = dc[COL_REND].quantile([0.25, 0.75])
    dc["NIVEL"] = pd.cut(dc[COL_REND], [-np.inf, q1, q3, np.inf], labels=["Bajo", "Medio", "Alto"])
    print(f"Cuartiles: Bajo ≤ {q1:.2f} < Medio ≤ {q3:.2f} < Alto")
    top_c = dc["DEPARTAMENTO"].value_counts().head(N_DEPTOS_TABLA_CRUZADA).index
    dc["GRUPO_DEP"] = np.where(dc["DEPARTAMENTO"].isin(top_c), dc["DEPARTAMENTO"], "OTROS")
    ct = pd.crosstab(dc["GRUPO_DEP"], dc["NIVEL"])
    ct = ct.loc[list(top_c) + ["OTROS"]]
    ct["Total"] = ct.sum(axis=1)
    pct = (ct[["Bajo", "Medio", "Alto"]].div(ct["Total"], axis=0) * 100).round(1).add_prefix("% ")
    print(pd.concat([ct, pct], axis=1).to_string())
    pd.concat([ct, pct], axis=1).to_csv(OUT / "tabla5_cruzada.csv", encoding="utf-8-sig")

    obs = ct[["Bajo", "Medio", "Alto"]].values
    chi2, p_chi, dof, esp = stats.chi2_contingency(obs)
    n_tot = obs.sum()
    cramer = np.sqrt(chi2 / (n_tot * (min(obs.shape) - 1)))
    print(f"\nχ²({dof}, N = {n_tot:,}) = {chi2:.2f}, p {p_fmt(p_chi)}; V de Cramér = {cramer:.3f}; "
          f"esperada mínima = {esp.min():.1f}")
    res_pearson = (obs - esp) / np.sqrt(esp)
    fil = obs.sum(axis=1, keepdims=True) / n_tot
    col = obs.sum(axis=0, keepdims=True) / n_tot
    res_ajust = (obs - esp) / np.sqrt(esp * (1 - fil) * (1 - col))
    idx = ct.index
    print("\nResiduos de Pearson:")
    print(pd.DataFrame(res_pearson, index=idx, columns=["Bajo", "Medio", "Alto"]).round(1).to_string())
    print("\nResiduos estandarizados AJUSTADOS (más apropiados para el criterio |z| > 3):")
    print(pd.DataFrame(res_ajust, index=idx, columns=["Bajo", "Medio", "Alto"]).round(1).to_string())

    pct_plot = (ct[["Bajo", "Medio", "Alto"]].div(ct["Total"], axis=0) * 100).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    pct_plot.plot(kind="barh", stacked=True, color=["#c62828", "#f9a825", "#2e7d32"], ax=ax, width=0.8)
    ax.set_xlabel("Porcentaje de registros (%)"); ax.set_ylabel("Departamento")
    ax.set_title("Nivel de rendimiento por departamento (barras apiladas al 100 %)")
    ax.legend(title="Nivel de rendimiento", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    guardar(fig, "08_barras_apiladas_nivel_rendimiento.png")

    # ---------- 9.4 Pronósticos exploratorios (serie anual) ----------
    titulo("9.4 PRONÓSTICOS EXPLORATORIOS")
    serie = anual["Produccion_t"].astype(float)
    n_per = len(serie)
    print(f"Periodos disponibles: {n_per} (el taller exige ≥ 30 para SES/Holt-Winters/GRU/LSTM)")
    if not HAY_STATSMODELS:
        print("statsmodels no instalado: se omiten los pronósticos (pip install statsmodels).")
    elif n_per <= ANIOS_PRUEBA + 3:
        print("Serie demasiado corta para separar entrenamiento y prueba.")
    else:
        train, test = serie.iloc[:-ANIOS_PRUEBA], serie.iloc[-ANIOS_PRUEBA:]

        def mape(real, pred):
            real, pred = np.asarray(real, float), np.asarray(pred, float)
            return np.mean(np.abs((real - pred) / real)) * 100

        ses = SimpleExpSmoothing(train.values, initialization_method="estimated").fit()
        holt = Holt(train.values, initialization_method="estimated").fit()
        f_naive = np.repeat(train.iloc[-1], ANIOS_PRUEBA)
        f_ses = ses.forecast(ANIOS_PRUEBA)
        f_holt = holt.forecast(ANIOS_PRUEBA)
        res = pd.DataFrame({
            "Método": ["Ingenuo (último valor)", "Suavización exponencial simple", "Holt (tendencia)"],
            "Parámetros": ["—", f"α = {ses.params['smoothing_level']:.2f}",
                           f"α = {holt.params['smoothing_level']:.2f}; β = {holt.params['smoothing_trend']:.2f}"],
            f"MAPE prueba ({test.index[0]}–{test.index[-1]}) %": [mape(test, f_naive), mape(test, f_ses), mape(test, f_holt)],
        })
        print(res.round(2).to_string(index=False))
        res.round(4).to_csv(OUT / "tabla6_pronosticos.csv", index=False, encoding="utf-8-sig")

        holt_full = Holt(serie.values, initialization_method="estimated").fit()
        fut_idx = np.arange(serie.index[-1] + 1, serie.index[-1] + 4)
        fut = holt_full.forecast(3)
        print("Holt (serie completa), solo ilustrativo:", {int(a): round(float(b) / 1000, 1) for a, b in zip(fut_idx, fut)}, "(miles de t)")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(serie.index, serie.values / 1000, "o-", color="darkgreen", label="Producción observada")
        ax.axvspan(test.index[0] - 0.5, test.index[-1] + 0.5, color="grey", alpha=0.12)
        ax.plot(test.index, f_ses / 1000, "s--", color="tab:blue", label="Pronóstico SES (prueba)")
        ax.plot(test.index, f_holt / 1000, "^--", color="tab:orange", label="Pronóstico Holt (prueba)")
        ax.plot(fut_idx, fut / 1000, "D:", color="purple", label="Holt (ilustrativo)")
        ax.set_xlabel("Año"); ax.set_ylabel("Producción total (miles de t)")
        ax.set_title(f"Producción total por año y pronósticos exploratorios ({n_per} periodos)")
        ax.legend()
        fig.tight_layout()
        guardar(fig, "09_pronosticos.png")

titulo(f"PROCESO TERMINADO. Resultados en: {OUT.resolve()}")