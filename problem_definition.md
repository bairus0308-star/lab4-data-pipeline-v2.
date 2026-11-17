# Problem definition

**Task.** Predict whether a person earns more than $50K a year from census data
(UCI Adult, OpenML `adult` v2, 48,842 rows).

**Inputs.** The 14 features: age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week, native-country.

**Output.** Binary label `income` (`>50K` = positive class, `<=50K` = negative class).

**Success metrics.** F1 (positive class) and accuracy. F1 matters because only about a
quarter of people earn >50K, so accuracy alone can look good by always guessing `<=50K`.

**Constraints.**
- *No leakage:* the train/test split is made first. Imputation values, clipping limits and
  duplicate removal are learned from the training set only, and the test set is only transformed.
- *Reproducible:* fixed random seed, settings in `config.yaml`, pinned steps in one script,
  SHA256 hashes of the processed data. One command: `python run_pipeline.py`.
- *Honesty about data:* problems are reported (validation report), never silently dropped.

**Why it matters / caution.** The data is from 1994 and includes sensitive attributes (race, sex,
native country). It is used here to learn data engineering, not to make real decisions about people.
