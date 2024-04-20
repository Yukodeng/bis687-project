## scRNA-seq Data Denoising and Impact on Downstream Analysis

A comparison of approaches to denoise and impute single cell RNA sequencing (scRNA-seq) data to account for technical errors that arise from false "dropouts" due to low RNA capture rates of expressed genes. Our review include several methods that utilizes Deep learning and machine learning methods, as well as statistical assumptions of the gene expression data distribution such as multinomial, poisson, and zero-inflated negative binomial (ZINB). 

Further downstream analysis such as clustering, differentially expressed gene analysis, and classification are carried out on both real-word datasets (RWDs) and simulation data to evluate and compare different methods.

See our [demo](https://github.com/Yukodeng/bis687-project/blob/main/demo.ipynb) for more details.


### Installation

#### pip

The project uses Python 3.8.18 with dependencies that can be installed using the below command line:

```
$ pip install -r requirements.txt
```

### Data

Real-world single-cell RNA datasets are available in the `data/` folder, which include:

- Single-cell and bulk RNA-seq data for definitive endoderm differentiation experiment by Chu et al. (2016), which are available at the Gene Expression Omnibus (GEO) under accession code GSE75748. 

- Multiple selective mouse organ tissues (Qx_Bladder, Qx_Kidney, Qx_LimbMuscle, and Qx_Spleen), which can be downloaded from GEO (GSE109774). 

- More to be included..


### Results

Outputs of denoising algorithms include the main denoised output (representing the recovered UMI counts), as well as some additional matrices, in the following formats:

- `results.h5ad` file contains the raw UMI counts (dimension=`gene x cell`), the denoised output (dimension matching the original input), the lower-dimensional latent representation of the original count matrix (hidden layer size is `32` by default), and other attributes of the gene expression matrix.

You can load the file into python using:

```
import scanpy as sc
adata = sc.read_h5ad("../results.h5ad")
```

- `mean.tsv` is the main output of the method which represents the recovered UMI count matrix. This file has the same dimensions as the input file (except that the zero-expression genes or cells are excluded). It is formatted as a `gene x cell` matrix. 



### Conclusion

TBC