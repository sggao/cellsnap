This folder contains all the codes that were used in our study.

Note the CellSNAP method we utilized in the scripts was the development version, which has different format and grammar compared to the final package.

The intention of this folder is a record for all the analyses generated in the study and for replication if needed.

File descriptions examples
```
.
├── CODEX_spleen/code # contains all code used for CODEX mouse spleen analysis
│   ├── prep_murine.ipynb # preparation of data
│   ├── cnn_train_murine.py # train SNAP-CNN
│   ├── cnn_get_murine.ipynb # get SNAP-CNN embedding
│   ├── gnn_duo_murine.ipynb # train and get SNAP-GNN-duo embedding
│   ├── cluster_ann_murine.ipynb # cell population annotation of clustering results
│   ├── muse_murine_format.ipynb # preparation of data for muse
│   ├── muse_murine_step1.ipynb # muse run step 1
│   ├── muse_murine_step2.ipynb # muse run step 2
│   ├── muse_murine_doublecheck.ipynb # muse double check to make sure correct
│   ├── spicemix_murine.ipynb # spicemix run
│   ├── benchmark_CH.ipynb # metric: 'calinski_harabasz_score'
│   ├── benchmark_SS.ipynb # metric: 'silhouette_score'
│   ├── benchmark_DB.ipynb # metric: 'davies_bouldin_score'
│   ├── benchmark_Mod.ipynb # metric: 'modularity'
│   └── analysis_plot_murine.Rmd # related plotting and analysis code in R
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── file_in_root.ext
└── README.md

```
