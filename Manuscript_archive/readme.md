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
│
├── CODEX_tonsil/code # contains all code used for CODEX tonsil analysis
│   ....
│   └── files # same file types and naming format as spleen folder
│
├── CODEX_cHL/code # contains all code used for CODEX cHL analysis
│   ....
│   └── files # same file types and naming format as spleen folder
│
├── CoxMx_liver/code # contains all code used for CosMx liver analysis
│   ....
│   └── files # same file types and naming format as spleen folder
│
├── Benchmark_spleen # contains all code used for parameter benchmarking on the CODEX spleen dataset
|   |
│   ├── prep_murine_bench.ipynb # data preparation
|   |
│   ├── cluster_res # benchmark Leiden clustering resolution used to construct neighborhood composition
|   |   ├── cnn_train_res0.2.py # SNAP-CNN train, at res = 0.2
|   |   ├── ... # other resolutions
|   |   ├── get_cnn_res_related.ipynb # SNAP-CNN get embedding
|   |   ├── get_dbGNN_embed_res.ipynb # SNAP-GNN-duo train and get embedding
|   |   ├── benchmark_CH_nbres.ipynb # get metrics
|   |   └── ... # other metrics
|   |
│   ├── NN_sizse # benchmark K number NN to construct neighborhood composition
|   │   ....
│   └── files # same file types and naming format as cluster_res folder
|   |
│   ├── Image_alpha # benchmark image binarization quantile
|   │   ....
│   └── files # same file types and naming format as cluster_res folder
|   |
│   ├── Image_size # benchmark image size
|   │   ....
│   └── files # same file types and naming format as cluster_res folder
|   |
│   └── plot_bench.Rmd # plotting code
|   |
└── Benchmark_duo/code # contains all code used for model structure benchmarking on the CODEX spleen dataset

```
