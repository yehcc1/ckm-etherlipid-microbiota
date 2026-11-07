# Analysis code — plasma ether-lipid ratio, gut microbiota score and CKM stage (NTCMRC)

Code for the manuscript "A plasma ether-lipid ratio links a gut microbiota score and predicted bacterial fatty-acid synthesis to cardiovascular–kidney–metabolic phenotype". Release 20261107_v66.

## Data
| Layer | Where | Access |
|---|---|---|
| Full-length 16S rRNA gene reads (PacBio HiFi), MIMARKS metadata | NCBI SRA, BioProject PRJNA1542453 | Public |
| Targeted plasma metabolomics (Biocrates AbsoluteIDQ p180) | Zenodo, https://doi.org/10.5281/zenodo.23320219 | Public, CC BY 4.0 |
| Sample-to-accession key for the 825 analysed samples | Supplementary Table S44 | Public |
| CKM stage, clinical covariates, medication | Chang Gung Medical Foundation IRB 202400297A3 | On request, subject to IRB approval |

Participant codes (NTCMRC-S####) are shared between the SRA and Zenodo deposits. No individual-level data are included in this repository.

## Scripts (run in this order; file names carry the analysis version)
1. `X1_pheno_all_*`, `X0_cflags_*`, `stage_v10.py` — metabolite ratio phenotypes (R1–R8) and CKM staging.
2. `X2_core_825_*`, `X2*`/`P1_*`, `xnames.py` — microbiome analysis set (825), genus CLR, pre-specified three-genus score, cross-validated panel (`nested_core_v65.py`, `X15_*`).
3. `X3_*`, `X4_*`, `X11_*`, `X19_*`–`X25_*`, `X27_*`, `X29_*`–`X31_*` — association, robustness, stage-boundary, parallel-mediation and bootstrap analyses.
4. `XD_*`, `XE*`, `XF_*`, `XG_*`, `X28_*`, `P4_*`, `pathmap_v65.py` — PICRUSt2/MetaCyc pathway analyses (NSTI-adjusted).
5. `X32_*`–`X35_*` — linkage to the public deposits and Table S44.
6. `Y0_numbers_merge_v65.py` — collects every reported number into one table; manuscript numbers are generated from it, never typed.
7. `X5*`, `X6_*`, `XB_*`, `XC_*`, `X16_*`, `X17_*`, `Y1_*`–`Y15_*` (figures, SVG); `X8_*`, `X12_*`, `XA_*` (tables).

Input paths are absolute paths of the analysis workspace (`/mnt/user-data/uploads/...`); edit them to your local copies. Some code comments are in Chinese. Fixed random seeds are set in each script that resamples.

Upstream processing (DADA2 1.40.0 PacBio workflow, SILVA 138.2, PICRUSt2) follows the published NTCMRC protocol; the parameters are given in the supplementary methods.

## Requirements
See `requirements.txt` (Python 3.13). Node.js 22 with the `docx` package is needed only for `XA_tables_docx_v65.js`.

## Licence
MIT (see `LICENSE`).
