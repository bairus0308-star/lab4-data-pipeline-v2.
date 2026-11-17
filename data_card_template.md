# Data card: UCI Adult (Census Income)

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository "Adult", loaded via `fetch_openml(name='adult', version=2)` |
| Origin | 1994 US Census database, extracted by Barry Becker; donated 1996 |
| License | CC BY 4.0 (check the UCI page when you submit) |
| Size (raw) | {{ROWS}} rows x {{COLS}} columns (14 features + `income`) |
| Split | train {{TRAIN_ROWS}} rows (after removing {{DUPES}} exact duplicates), test {{TEST_ROWS}} rows |
| Processed train SHA256 | `{{TRAIN_SHA}}` |
| Processed test SHA256 | `{{TEST_SHA}}` |
| Generated | {{DATE}} |

## Known biases
- **Old data:** 1994 US population. Incomes and job patterns are not current.
- **Imbalanced groups:** mostly White and mostly male; ~90% of people are from the US.
- **Label bias:** `income >50K` reflects historical pay gaps (e.g. by sex and race); a model can learn them.
- **Class imbalance:** about 24% positive class.
- **Missing values:** `?` appears in workclass, occupation, native-country, and is not random.
- **`fnlwgt`:** a census sampling weight, not a personal trait.
- **Duplicates / conflicts:** see `results/validation_report.md` and `results/label_quality.json`.

## Privacy concerns
- Public, anonymised data (no names), but it has sensitive attributes (race, sex, marital status,
  native country).
- Combinations of age, education, occupation and country could help re-identify someone when joined
  with other data.
- Do not use it to infer or decide things about real individuals.

## Intended use
Teaching and benchmarking tabular pipelines. Not for hiring, lending or any decision about people.
