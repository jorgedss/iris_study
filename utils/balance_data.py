"""
Generates SMOTE-NC balanced training datasets from data/features/baseline/.
Runs three sampling ratios (minority:majority) — 1:1, 1:5, 1:10 — saving each
to its own output directory (smote_nc_1_1 / smote_nc_1_5 / smote_nc_1_10).
Applies the same preprocessing decisions from the EDA before oversampling:
ALRM_*/GRAV_* NaN filled with 0 (informative absence), age clipped at 120,
categoricals encoded. SMOTE-NC requires complete numeric data, so imputation
is applied here. Test set is copied unchanged.
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

INPUT_DIR    = "data/features/baseline"
RANDOM_STATE = 42
YEAR_COL     = "year"

# sampling_strategy float = minority / majority after resampling
# None → default 'auto' (1:1)
RATIOS = {
    "1_1":  None,   # 1:1  — padrão SMOTE
    "1_5":  0.2,    # 1:5
    "1_10": 0.1,    # 1:10
}

ALRM_COLS = [
    "ALRM_HIPOT", "ALRM_PLAQ", "ALRM_VOM", "ALRM_SANG", "ALRM_HEMAT",
    "ALRM_ABDOM", "ALRM_LETAR", "ALRM_HEPAT", "ALRM_LIQ",
]
GRAV_COLS = [
    "GRAV_PULSO", "GRAV_CONV", "GRAV_ENCH", "GRAV_INSUF", "GRAV_TAQUI",
    "GRAV_EXTRE", "GRAV_HIPOT", "GRAV_HEMAT", "GRAV_MELEN", "GRAV_METRO",
    "GRAV_SANG", "GRAV_AST", "GRAV_MIOC", "GRAV_CONSC", "GRAV_ORGAO",
]
SYMP_COLS = [
    "FEBRE", "MIALGIA", "CEFALEIA", "EXANTEMA", "VOMITO", "NAUSEA",
    "DOR_COSTAS", "CONJUNTVIT", "ARTRITE", "ARTRALGIA", "PETEQUIA_N",
    "LEUCOPENIA", "LACO", "DOR_RETRO", "DIABETES", "HEMATOLOG",
    "HEPATOPAT", "RENAL", "HIPERTENSA", "AUTO_IMUNE",
]
CONTINUOUS_COLS = ["age_years", "epi_week", "CS_GESTANT", "CS_RACA", "CS_ESCOL_N"]


def preprocess(X: pd.DataFrame, enc_sexo=None, enc_uf=None, imp_cont=None, imp_symp=None, fit=True):
    X = X.copy()

    # Dropa coluna de ano — não é feature
    X = X.drop(columns=[YEAR_COL], errors="ignore")

    # age_years: outliers → NaN
    if "age_years" in X.columns:
        X.loc[X["age_years"] > 120, "age_years"] = np.nan

    # ALRM_* e GRAV_*: ausência informativa → 0
    for col in ALRM_COLS + GRAV_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # CS_SEXO: F→0, M→1
    if "CS_SEXO" in X.columns:
        if enc_sexo is None:
            enc_sexo = OrdinalEncoder(
                categories=[["F", "M"]],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            )
        vals = X[["CS_SEXO"]]
        X["CS_SEXO"] = enc_sexo.fit_transform(vals) if fit else enc_sexo.transform(vals)
        mode_val = 0.0
        X["CS_SEXO"] = X["CS_SEXO"].fillna(mode_val)

    # SG_UF: encode ordinal
    if "SG_UF" in X.columns:
        if enc_uf is None:
            enc_uf = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        vals = X[["SG_UF"]]
        X["SG_UF"] = enc_uf.fit_transform(vals) if fit else enc_uf.transform(vals)

    # Colunas contínuas: imputa com mediana
    cont_present = [c for c in CONTINUOUS_COLS if c in X.columns]
    if cont_present:
        if imp_cont is None:
            imp_cont = SimpleImputer(strategy="median")
        X[cont_present] = imp_cont.fit_transform(X[cont_present]) if fit else imp_cont.transform(X[cont_present])

    # Sintomas binários: imputa com moda
    symp_present = [c for c in SYMP_COLS if c in X.columns]
    if symp_present:
        if imp_symp is None:
            imp_symp = SimpleImputer(strategy="most_frequent")
        X[symp_present] = imp_symp.fit_transform(X[symp_present]) if fit else imp_symp.transform(X[symp_present])

    return X, enc_sexo, enc_uf, imp_cont, imp_symp


def main():
    print("=" * 55)
    print("Gerando datasets balanceados com SMOTE-NC")
    print("=" * 55)

    X_train = pd.read_parquet(os.path.join(INPUT_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(INPUT_DIR, "y_train.parquet")).squeeze()
    X_test  = pd.read_parquet(os.path.join(INPUT_DIR, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(INPUT_DIR, "y_test.parquet")).squeeze()

    # Remove NaN do target
    mask = y_train.notna()
    X_train, y_train = X_train[mask], y_train[mask]

    print(f"\nTreino original: {len(X_train):,} | Óbitos: {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)")

    # Preprocessamento (fit no treino)
    X_tr, enc_sexo, enc_uf, imp_cont, imp_symp = preprocess(X_train, fit=True)

    # Preprocessamento (transform no teste — encoders ajustados no treino)
    X_te, *_ = preprocess(X_test, enc_sexo=enc_sexo, enc_uf=enc_uf,
                           imp_cont=imp_cont, imp_symp=imp_symp, fit=False)

    # Colunas categóricas para SMOTE-NC: binárias (0/1) + SG_UF (ordinal discreta)
    binary_cols = [c for c in ALRM_COLS + GRAV_COLS + SYMP_COLS if c in X_tr.columns]
    cat_cols    = binary_cols + (["CS_SEXO", "SG_UF"] if "CS_SEXO" in X_tr.columns else [])
    cat_cols    = [c for c in cat_cols if c in X_tr.columns]

    print(f"Colunas categóricas para SMOTE-NC: {len(cat_cols)}")

    for ratio_label, sampling_strategy in RATIOS.items():
        output_dir = f"data/features/smote_nc_{ratio_label}"
        print(f"\n--- Ratio {ratio_label.replace('_', ':')} | sampling_strategy={sampling_strategy or 'auto (1:1)'} ---")

        smote = SMOTENC(
            categorical_features=cat_cols,
            sampling_strategy=sampling_strategy if sampling_strategy is not None else "auto",
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )

        X_res, y_res = smote.fit_resample(X_tr, y_train)
        X_res = pd.DataFrame(X_res, columns=X_tr.columns)
        y_res = pd.Series(y_res, name="target")

        n_obitos = int(y_res.sum())
        n_total  = len(y_res)
        print(f"Treino balanceado: {n_total:,} | Óbitos: {n_obitos:,} ({y_res.mean()*100:.2f}%)")

        os.makedirs(output_dir, exist_ok=True)

        X_res.to_parquet(os.path.join(output_dir, "X_train.parquet"), index=False, engine="pyarrow")
        y_res.to_frame().to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False, engine="pyarrow")
        X_te.to_parquet(os.path.join(output_dir, "X_test.parquet"),  index=False, engine="pyarrow")
        y_test.to_frame().to_parquet(os.path.join(output_dir, "y_test.parquet"), index=False, engine="pyarrow")

        config = {
            "strategy":             "smote_nc",
            "ratio":                ratio_label.replace("_", ":"),
            "sampling_strategy":    sampling_strategy,
            "preprocessed":         True,
            "categorical_cols":     cat_cols,
            "continuous_cols":      [c for c in CONTINUOUS_COLS if c in X_tr.columns],
            "train_size_original":  len(X_train),
            "train_size_resampled": n_total,
            "obitos_original":      int(y_train.sum()),
            "obitos_resampled":     n_obitos,
            "test_size":            len(X_te),
            "random_state":         RANDOM_STATE,
            "created_at":           str(date.today()),
            "note": "X_train e X_test já estão pré-processados (sem NaN, codificados). Pipelines de modelo devem aplicar apenas escala para colunas contínuas.",
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Artefatos salvos em: {output_dir}/")


if __name__ == "__main__":
    main()
