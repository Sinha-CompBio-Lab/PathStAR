
# PathStAR
## Mapping Structural Aging across Human Tissues reveals tissue-specific trajectories, coordinated deterioration and genetic determinants

**Abstract:** Tissue structure, the organization of cells, vasculature and extracellular matrix, determines organ function. Yet how tissue structure changes with aging remains largely unknown. Current aging research primarily focuses on molecular changes, missing this structural dimension. Here, we present PathStAR, Pathology based Structural Aging Rate, the first computational framework that captures when and how tissue structure changes during aging from histopathology images. We applied it to 25,306 postmortem tissues covering 40 tissue types from individuals aged 21–70, connecting structural aging to molecular data, health records and genotype data. Without any training on chronological age, PathStAR captured non-linear functional decline of ovary, undetectable by bulk-molecular profiling. Applying it across 40 tissues, it revealed that structural aging occurs through discrete phases of rapid change (accelerated periods), with tissue-specific trajectories following three patterns: Early Aging Tissues (vascular system with major changes during the 30s), Late Aging Tissues (uterus and vagina with major changes during menopause (50s)) and Biphasic Aging Tissues (digestive, male reproductive tissues, and ovary with two periods of major changes). During these accelerated phases, most tissues exhibited shared aging hallmarks of inflammation and energy production decline, coupled with disruption of pathways governing their specialized functions. Cross-organ analysis revealed coordinated aging within organ systems and an unexpected link between digestive and male reproductive tissues. We next identified 123 germline variants associated with organ-specific accelerated structural aging, including SIRT6 variants linked to accelerated vascular decline. Finally, individuals with systemic autoimmune disease, as well as tissues with classical aging pathologies (atrophy, calcification, fibrosis), showed elevated structural aging scores. We demonstrate that structural aging is measurable from histology scans and provide the first systematic framework for studying it, revealing organ-specific aging processes.

<!-- ![alt text](docs/pics/Title_Photo.png) -->


## Installation
First clone this repo and cd into the directory: 
```
git clone https://github.com/Sinha-CompBio-Lab/PathStAR.git
cd PathStAR
conda env create -f env.yaml
conda activate pathstar
```


### Get Access to data

The data used in this study were downloaded from the **[GTEx Portal](https://www.gtexportal.org/home/)**. Open-access summary resources (e.g., gene expression matrices, eQTL summary stats, sample attributes with limited detail) can be browsed or downloaded directly from the portal.

Controlled (“private”) individual-level GTEx data require an approved dbGaP request for **phs000424** and are accessed via AnVIL/Terra; see the GTEx **[Protected Access](https://www.gtexportal.org/home/protectedDataAccess)** page and Terra’s step-by-step guide.



### 1. Build Trajectory
```bash
python scripts/build_trajectory.py \
  --csv /shares/sinha/anamikay/projects/critical_age_analysis/critical_age_dataset.csv \
  --window 10 \
  --alpha 0.05 \
  --normalize zscore \
  --min-tissue-samples 200 \
  --outdir ./results/age_analysis_basic
  ```

- `--csv` → Path to input CSV with columns: tissue, sex, age, features (stringified list).

- `--window` (optional) → Sliding window size in years for age bins. (default: 10)

- `--alpha` (optional) → p-value threshold for marking “significant” age points (Welch’s t across features). (default: 0.05)

- `--normalize` (optional) → Feature normalization: zscore or raw. (default: zscore)

- `--min-tissue-samples` (optional) → Minimum samples per tissue to include. (default: 200)

- `--outdir` (optional) → Output directory for per-tissue plots and pickled summaries. (default: ./age_analysis_results_combined)

### 2. Smoothen Trajectory

Smoothen the trajectory created using  Spline and Gaussian both. 

```bash
python scripts/trajectory.py \
  --pickle ./results/unprocessed_plot_age_custom/all_results_combined.pkl \
  --outdir ./results/Gaussian_Spline_smoothen_temp
```

- `--pickle → Path to combined simplified results pickle from step 1.`

- `--outdir (optional) → Root output for smoothed curves & summaries. (default: ./Gaussian_Spline_smoothen_temp)`

    - Saves `spline_data/*_comparison.csv`, `*_dense.csv and gaussian_process_data/*.`

### 3. Plot Trajectory

```bash
python scripts/plot_smoothed.py \
  --root ./results/Gaussian_Spline_smoothen_temp \
  --method spline \
  --label combined
```

### 4. Calculate Individual Score

```bash
python scripts/calculate_individual_score.py \
  --csv /shares/sinha/anamikay/projects/critical_age_analysis/critical_age_dataset.csv \
  --min-tissue-samples 200 \
  --min-younger-samples 5 \
  --younger-window-years 10 \
  --outdir ./results/individual_score_all_combined \
  --zscore-by-sex false \
  --sex-matched-younger false
```

- `--csv` → Path to dataset with tissue, sex, age, features.

- `--min-tissue-samples` (optional) → Include tissues with at least this many samples. (default: 200)

- `--min-younger-samples` (optional) → Minimum younger cohort size per target. (default: 5)

- `--younger-window-years` (optional) → Younger window [age - Y, age - 1]. (default: 10)

- `--outdir (optional)` → Output directory per-tissue TSV tables. (default: ./individual_score_all_combined)

- `--zscore-by-sex` (optional) → If true, z-score within sex; if false, pooled. (default: false)

- `--sex-matched-younger` (optional) → If true, younger cohort matches sex. (default: false)


### 5. Calculate Delta Score

Subtract smoothed population curve from each individual’s custom effect size:

```bash
python scripts/calculate_delta_score.py \
  --individual-dir ./results/individual_score_all_combined \
  --spline-dir ./results/Gaussian_Spline_smoothen_temp/spline_data \
  --gaussian-dir ./results/Gaussian_Spline_smoothen_temp/gaussian_process_data \
  --outdir ./results/delta_scores \
  --label combined
```

- `--individual-dir → Directory from step 4 containing *_effect_size_deviation.tsv.`

- `--spline-dir → Spline outputs containing *_spline_comparison.csv.`

- `--gaussian-dir → GP outputs containing *_gaussian_process_comparison.csv.`

- `--outdir (optional) → Where to write per-tissue and combined Δ tables. (default: ./delta_score_temp)`

- `--label (optional) → Filename label to match smoothing outputs (e.g., combined). (default: combined)`

### 6. Cross Tissue Correltaions

Compute ischemia-adjusted cross-tissue correlation matrices and heatmaps.

#### All tissues:
`python scripts/cross_tissue_correlation_all.py \
  --delta ./results/delta_scores/all_tissues_delta_scores.tsv \
  --trischd /shares/sinha/anamikay/projects/path_rem_vs/results/patient_tissue_TRISCHD_matrix.csv \
  --outdir ./results/cross_tissue_correlation/all`

#### Selected 14 tissues:

```bash
python scripts/cross_tissue_correlation_14.py \
  --delta ./results/delta_scores/all_tissues_delta_scores.tsv \
  --trischd /shares/sinha/anamikay/projects/path_rem_vs/results/patient_tissue_TRISCHD_matrix.csv \
  --outdir ./results/cross_tissue_correlation/14
```

- `--delta → Combined Δ-score table from step 5 (all_tissues_delta_scores.tsv).`

- `--trischd → TRISCHD ischemic time matrix (Subject.ID + per-tissue columns).`

- `--outdir (optional) → Output directory for pivots, corr/pval matrices, heatmaps. (default differs per script)`


