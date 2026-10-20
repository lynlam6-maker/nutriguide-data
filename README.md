# NutriGuide NHANES model training pack

This is the original cleaned model-development cohort reconstructed from the original NHANES August 2021–August 2023 files and preparation code. It is not a clinically validated dataset or a new model.

Files: cleaned_cohort.csv (4,844 rows); train.csv (2,906); validation.csv (969); test.csv (969). The split column records the original assignments. Separate split CSVs preserve the original ordering. Do not concatenate and randomly resplit if comparing with the original experiment.

Inputs: RIDAGEYR (age, years), BMXBMI (BMI, kg/m²), BMXWAIST (waist, cm). Target: target=1 when laboratory LBXGH (HbA1c, %) >=5.7, else 0. It estimates a current measured outcome, not future diabetes or a diagnosis.

Never use LBXGH as a predictor: it defines the label and would leak the answer. SEQN is a public survey participant identifier, not a predictor. WTPH2YR is the phlebotomy survey weight, used for fitting/evaluation, not a predictor. RIAGENDR and RIDRETH3 retain official NHANES sex and race/Hispanic-origin codes for subgroup audits only; consult the official DEMO_L codebook. Do not infer categories from numeric ordering. The split column is never a predictor.

Cohort: adults aged 18+, reported no diagnosed diabetes or borderline status (DIQ010 codes 2 or 3), no reported insulin/diabetes medication use (DIQ050/DIQ070 !=1), no recorded pregnancy (RIDEXPRG !=1), complete required measurements/outcome, WTPH2YR >1 to exclude SAS missing sentinels. Missing pregnancy information does not verify non-pregnancy. Complete-case exclusions can introduce selection bias.

To reproduce: install numpy pandas scikit-learn; run python train_model.py. It writes model.json, evaluation.json and demo_cases.csv. original_model.json and original_evaluation.json preserve the prior experiment. original_prepare.py is the unmodified historical pipeline and requires the original source files and food ZIP; use train_model.py for this CSV pack.

Training uses StandardScaler learned on training data and weighted logistic regression. Threshold is the highest validation threshold meeting weighted sensitivity >=80%, exploratory only. Retain the test set for final evaluation. Because the original test results are already known, repeated tuning against them requires a fresh independent evaluation set. Weights do not incorporate full survey-design variance estimation or prove generalisability.

Known limitations: US survey cohort, no Australian external validation, substantial age disparities, no completed clinical validation. Original weighted test sensitivity ~73.1%, specificity ~58.2%, AUC ~0.710. Public-use data is not an endorsement by CDC/NCHS.

Official sources:
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.htm
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm

This pack covers prediction-model development only; it does not contain USDA food data or laboratory education rules.
