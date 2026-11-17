"""Lab 4 pipeline. Run everything with:  python run_pipeline.py"""
import hashlib, json, logging, datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd, yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import Cleaner, add_features

log = logging.getLogger("lab4")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_binary(s, positive):
    return (s.astype(str).str.strip() == positive).astype(int)


# ---------------- Step 2: ingestion (raw data is saved untouched) ----------------
def ingest(cfg):
    o = cfg["openml"]
    raw = fetch_openml(name=o["name"], version=o["version"], as_frame=True).frame
    log.info("INGEST rows=%d cols=%d (expected %d x %d)", raw.shape[0], raw.shape[1],
             cfg["expected"]["rows"], cfg["expected"]["cols"])
    if raw.shape != (cfg["expected"]["rows"], cfg["expected"]["cols"]):
        log.warning("Shape differs from the lab's expected shape!")
    p = Path(cfg["paths"]["raw"]); p.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(p, index=False)                       # exact copy, no edits
    log.info("Raw copy saved to %s (sha256 %s)", p, sha256_file(p)[:12])
    return raw


# ---------------- Step 3: validation (reports, never drops) ----------------
def is_missing(s):
    return s.isna() | (s.astype(object) == "?")


def validate(df, cfg):
    rep = {"rows": int(len(df)), "cols": int(df.shape[1]), "type_violations": {}, "range_violations": {},
           "missing": {}, "exact_duplicate_rows": int(df.duplicated().sum()),
           "class_counts": df[cfg["target"]].astype(str).value_counts().to_dict()}
    for col, t in cfg["schema"].items():
        if col not in df.columns:
            rep["type_violations"][col] = "column missing"; continue
        s = df[col]
        if t == "int":
            ok = pd.api.types.is_numeric_dtype(s) and bool((s.dropna() % 1 == 0).all())
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s)
        if not ok:
            rep["type_violations"][col] = f"expected {t}, got {s.dtype}"
    for col, (lo, hi) in cfg["ranges"].items():
        s = pd.to_numeric(df[col], errors="coerce")
        bad = s.notna() & ~s.between(lo, hi)
        rep["range_violations"][col] = {"range": [lo, hi], "count": int(bad.sum()),
                                        "example_row_index": [int(i) for i in df.index[bad][:10]]}
    for col in df.columns:
        n = int(is_missing(df[col]).sum())
        if n:
            rep["missing"][col] = n
    return rep


def write_report(rep, path):
    L = ["# Validation report", "",
         f"Rows: {rep['rows']}  |  Columns: {rep['cols']}  |  Exact duplicate rows: {rep['exact_duplicate_rows']}", "",
         "## Type violations", ""]
    L += [f"- {k}: {v}" for k, v in rep["type_violations"].items()] or ["- none"]
    L += ["", "## Range violations", ""]
    L += [f"- {k} {v['range']}: {v['count']} rows" for k, v in rep["range_violations"].items()]
    L += ["", "## Missing values ('?' or NaN; kept in raw data, imputed later on train statistics)", ""]
    L += [f"- {k}: {v}" for k, v in rep["missing"].items()] or ["- none"]
    L += ["", "## Class counts", ""] + [f"- {k}: {v}" for k, v in rep["class_counts"].items()]
    Path(path).write_text("\n".join(L) + "\n")


# ---------------- Step 5: baseline ----------------
def baseline(train, test, feats, num_cols, cat_cols, target, positive, seed):
    nums = [c for c in feats if c in num_cols]
    cats = [c for c in feats if c in cat_cols]
    pre = ColumnTransformer([("num", StandardScaler(), nums),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    model = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, random_state=seed))])
    ytr, yte = to_binary(train[target], positive), to_binary(test[target], positive)
    model.fit(train[feats], ytr)
    pred = model.predict(test[feats])
    return {"accuracy": float(accuracy_score(yte, pred)), "f1": float(f1_score(yte, pred))}


# ---------------- Step 6: label quality ----------------
def label_quality(df, feature_cols, cfg, results):
    target = cfg["target"]
    sample = df.sample(cfg["audit_sample_size"], random_state=cfg["seed"])
    sample.to_csv(results / "audit_sample_50.csv")      # <- you re-check these 50 rows by hand
    work = df[feature_cols + [target]].astype(str)
    g = work.groupby(feature_cols)[target].agg(["nunique", "size"])
    conflicts = g[g["nunique"] > 1]
    edu = df.assign(_e=pd.to_numeric(df["education-num"], errors="coerce")).groupby(
        df["education"].astype(str))["_e"].nunique()
    sam_bad = 0
    for col, (lo, hi) in cfg["ranges"].items():
        s = pd.to_numeric(sample[col], errors="coerce")
        sam_bad += int((s.notna() & ~s.between(lo, hi)).sum())
    out = {"conflicting_feature_groups": int(len(conflicts)),
           "rows_in_conflicting_groups": int(conflicts["size"].sum()),
           "education_labels_with_multiple_education_num": int((edu > 1).sum()),
           "education_num_outside_1_16": int((~pd.to_numeric(df["education-num"], errors="coerce").between(1, 16)).sum()),
           "negative_capital_values": int(((df["capital-gain"] < 0) | (df["capital-loss"] < 0)).sum()),
           "fnlwgt_not_positive": int((pd.to_numeric(df["fnlwgt"], errors="coerce") <= 0).sum()),
           "impossible_values_in_50_row_sample": sam_bad}
    log.info("LABEL QUALITY %s", out)
    (results / "label_quality.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    results = Path(cfg["paths"]["results"]); results.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(results / "pipeline.log", mode="w")])
    target, positive = cfg["target"], cfg["positive_label"]

    raw = ingest(cfg)

    df = raw.rename(columns={"class": target})            # only a column rename, values untouched
    rep = validate(df, cfg)
    (results / "validation_report.json").write_text(json.dumps(rep, indent=2))
    write_report(rep, results / "validation_report.md")
    log.info("VALIDATE type_violations=%d range_violations=%s missing_cols=%d dup_rows=%d",
             len(rep["type_violations"]), {k: v["count"] for k, v in rep["range_violations"].items()},
             len(rep["missing"]), rep["exact_duplicate_rows"])

    feature_cols = [c for c in df.columns if c != target]
    num_cols = [c for c, t in cfg["schema"].items() if t == "int"]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    # split FIRST (stratified), then everything is learned from train only
    y = to_binary(df[target], positive)
    train, test = train_test_split(df, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"])
    n_before = len(train)
    train = train.drop_duplicates()                       # train only; test untouched
    dupes = n_before - len(train)
    log.info("SPLIT train=%d (dropped %d exact duplicates) test=%d", len(train), dupes, len(test))

    cleaner = Cleaner(num_cols, cat_cols, *cfg["clip_percentiles"]).fit(train)
    train_c, test_c = cleaner.transform(train), cleaner.transform(test)
    joblib.dump(cleaner, results / "preprocessor.joblib")

    base_feats = feature_cols
    new_feats = base_feats + ["capital_net", "hours_per_age"]
    train_f, test_f = add_features(train_c), add_features(test_c)
    all_num = num_cols + ["capital_net", "hours_per_age"]
    metrics = {"without_new_features": baseline(train_f, test_f, base_feats, all_num, cat_cols, target, positive, cfg["seed"]),
               "with_new_features": baseline(train_f, test_f, new_feats, all_num, cat_cols, target, positive, cfg["seed"])}
    log.info("BASELINE %s", metrics)
    (results / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    label_quality(df, feature_cols, cfg, results)

    # processed data + hashes + manifest + data card
    for p in (cfg["paths"]["train"], cfg["paths"]["test"]):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    train_f.to_csv(cfg["paths"]["train"], index=False)
    test_f.to_csv(cfg["paths"]["test"], index=False)
    h_tr, h_te = sha256_file(cfg["paths"]["train"]), sha256_file(cfg["paths"]["test"])
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "sha256": {"processed_train": h_tr, "processed_test": h_te, "raw": sha256_file(cfg["paths"]["raw"])},
                "baseline": metrics, "config": cfg}
    (results / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    card = Path("data_card_template.md").read_text()
    for k, v in {"ROWS": rep["rows"], "COLS": rep["cols"], "TRAIN_ROWS": len(train_f), "DUPES": dupes,
                 "TEST_ROWS": len(test_f), "TRAIN_SHA": h_tr, "TEST_SHA": h_te,
                 "DATE": datetime.date.today().isoformat()}.items():
        card = card.replace("{{" + k + "}}", str(v))
    Path("data_card.md").write_text(card)
    log.info("DONE. train sha256=%s", h_tr)


if __name__ == "__main__":
    main()
