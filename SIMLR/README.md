Single-cell Interpretation via Multi-kernel LeaRning (**SIMLR**) 
================================================================ 

| Branch | Status |
| --- | --- |
| master | [![R-CMD-check-bioc](https://github.com/BatzoglouLabSU/SIMLR/actions/workflows/check-bioc.yml/badge.svg?branch=master)](https://github.com/BatzoglouLabSU/SIMLR/actions/workflows/check-bioc.yml) |
| development | [![R-CMD-check-bioc](https://github.com/BatzoglouLabSU/SIMLR/actions/workflows/check-bioc.yml/badge.svg?branch=development)](https://github.com/BatzoglouLabSU/SIMLR/actions/workflows/check-bioc.yml) |

**OVERVIEW**

In this repository we provide implementations in both R and Matlab of *SIMLR* (https://www.nature.com/articles/nmeth.4207). The main branch of the repository (named SIMLR) provides the code (both R and Matlab) for the method and some example data. We recall that those data are provided purely as examples and should not be used in place of the ones provided in the respective publications. 

Moreover, the tool is also available on Bioconductor at https://www.bioconductor.org/packages/release/bioc/html/SIMLR.html. The branch master of this repository refers to the stable version on Bioconductor and the development branch of this repository refers to the development version on Bioconductor (https://www.bioconductor.org/packages/devel/bioc/html/SIMLR.html). 

The standard implementations of *SIMLR* are provided in the scripts *SIMLR.R* for R and *SIMLR.m* for Matlab. Besides these standard implementations, we also provide *SIMLR_Large_Scale* to handle large scale datasets (scripts *SIMLR_Large_Scale.R* and *SIMLR_Large_Scale.m* for R and Matlab) and *SIMLR_Feature_Ranking* to rank the most important features for the learned similarities (scripts *SIMLR_Feature_Ranking.R* and *SIMLR_Feature_Ranking.m* for R and Matlab). 

Finally, we also provide scripts to estimate the number of clusters from the data as suggested in the original paper in the scripts *SIMLR_Estimate_Number_of_Clusters.R* and *SIMLR_Estimate_Number_of_Clusters.m*. 

**SIMLR**

Single-cell RNA-seq technologies enable high throughput gene expression measurement of individual cells, and allow the discovery of heterogeneity within cell populations.  Measurement of cell-to-cell gene expression similarity is critical for the identification, visualization and analysis of cell populations. However, single-cell data introduce challenges to conventional measures of gene expression similarity because of the high level of noise, outliers and dropouts. We develop a novel similarity-learning framework, *SIMLR* (Single-cell Interpretation via Multi-kernel LeaRning), which learns an appropriate distance metric from the data for dimension reduction, clustering and visualization. *SIMLR* is capable of separating known subpopulations more accurately in single-cell data sets than do existing dimension reduction methods. Additionally, *SIMLR* demonstrates high sensitivity and accuracy on high-throughput peripheral blood mononuclear cells (PBMC) data sets generated by the GemCode single-cell technology from 10x Genomics. 

*SIMLR* offers three main unique advantages over previous methods: (1) it learns a distance metric that best fits the structure of the data via combining multiple kernels. This is important because the diverse statistical characteristics due to large noise and dropout effect of single-cell data produced today do not easily fit specific statistical assumptions made by standard dimension reduction algorithms. The adoption of multiple kernel representations provides a better fit to the true underlying statistical distribution of the specific input scRNA-seq data set; (2) *SIMLR* addresses the challenge of high levels of dropout events that can significantly weaken cell-to-cell similarities even under an appropriate distance metric, by employing graph diffusion, which improves weak similarity measures that are likely to result from noise or dropout events; (3) in contrast to some previous analyses that pre-select gene subsets of known function, *SIMLR* is unsupervised, thus allowing de novo discovery from the data. We empirically demonstrate that *SIMLR* produces more reliable clusters than commonly used linear methods, such as principal component analysis (PCA), and nonlinear methods, such as t-distributed stochastic neighbor embedding (t-SNE), and we use *SIMLR* to provide 2-D and 3-D visualizations that assist with the interpretation of single-cell data derived from several diverse technologies and biological samples. 

Furthermore, here we also provide an implementation of *SIMLR* (see SIMLR large scale) capable of handling large scale datasets. 

**CITATION**

The latest version of the manuscript related to *SIMLR* is published on Nature Methods at https://www.nature.com/articles/nmeth.4207. We also provide a paper describing the software that is published on PROTEOMICS and can be found at http://onlinelibrary.wiley.com/doi/10.1002/pmic.201700232/full. 

When using *SIMLR*, please cite Wang, Bo, et al. "Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning." Nature methods 14.4 (2017): 414. 

The citation of Wang, Bo, et al. "SIMLR: A Tool for Large‐Scale Genomic Analyses by Multi‐Kernel Learning." Proteomics 18.2 (2018) is optional, although appreciated. 

**INSTALLING SIMLR R Bioconductor IMPLEMENTATION**

As mentioned, *SIMLR* is also hosted on Bioconductor at https://bioconductor.org/packages/release/bioc/html/SIMLR.html and can be installed as follow. To install the package directly from Bioconductor, run the following commands directly from R: 

source("https://bioconductor.org/biocLite.R")

biocLite("SIMLR")

Moreover, it is also possible to install the Github version of the tool from R by using the R library devtools. 

library("devtools")

install_github("BatzoglouLabSU/SIMLR", ref = 'master')

library("SIMLR")

or,

library("devtools")

install_github("BatzoglouLabSU/SIMLR", ref = 'development')

library("SIMLR")

The "master" branch hosts the latest stable version of the code which is also available on Bioconductor on the stable repository, while the "development" branch hosts the latest version that is on the devel repository on Bioconductor. 

We describe next the procedure to manually install our software in case one wishes to do so. 

**RUNNING THE R IMPLEMENTATION**

We provide the R demo code to run *SIMLR* on 4 examples in the script *R_main_demo_SIMLR.R*. Furthermore, we provide a large scale implementation of *SIMLR* (see large scale implementation) with 1 example in the script *R_main_demo_SIMLR_Large_Scale.R*. A demo for the estimation of the number of clusters by *SIMLR* is also provided in the script *R_main_demo_SIMLR_Estimate_Number_of_Clusters.R*. 

The R libraries required to run the demos can be installed by running the script *install_R_libraries.R*. We now present a set of requirements to run the examples. 

1) Required R libraries. our tool requires 2 R packages to run, namely the *Matrix* package (see https://cran.r-project.org/web/packages/Matrix/index.html) to handle sparse matrices and the *parallel* package (see https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf) for a parallel implementation of the kernel estimation. 

To run the large scale analysis, it is necessary to install 4 more packages, namely *Rcpp* package (see https://cran.r-project.org/web/packages/Rcpp/index.html), *pracma* package (see https://cran.r-project.org/web/packages/pracma/index.html), *RcppAnnoy* package (see https://cran.rstudio.com/web/packages/RcppAnnoy/index.html) and *RSpectra* package (see https://cran.r-project.org/web/packages/RSpectra/index.html). 

Furthermore, to run the examples, we require the *igraph* package (see http://igraph.org/r/) to compute the normalized mutual informetion metric and the *grDevices* package (see https://stat.ethz.ch/R-manual/R-devel/library/grDevices/html/00Index.html) to color the plots. 

All these packages, can be installed with the R built-in *install.packages* function. 

2) External C code. We make use of an external C program during the computations. The code is located in the R directory in the file *projsplx_R.c*. In order to compite the program, one needs to run on the shell the command *R CMD SHLIB -c projsplx_R.c*. 

An OS X pre-compiled file is also provided. Note: if there are issues in compiling the .c file, try to remove the pre-compiled files (i.e., *projsplx_R.o* and *projsplx_R.so*). 

3) Example datasets. The 5 example datasets are provided in the directory data of the branch. We recall that those data are provided purely as examples and after some pre-processing; they should not be used in place of the ones provided in the respective publications. 

Specifically, the dataset of Test_1_mECS.RData refers to http://www.ncbi.nlm.nih.gov/pubmed/25599176, Test_2_Kolod.RData refers to http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595712/, Test_3_Pollen.RData refers to http://www.ncbi.nlm.nih.gov/pubmed/25086649 and Test_4_Usoskin.RData refers to http://www.ncbi.nlm.nih.gov/pubmed/25420068. 

Moreover, for the large scale example, the dataset of Test_5_Zeisel.RData refers to https://www.ncbi.nlm.nih.gov/pubmed/25700174. 

**RUNNING THE MATLAB IMPLEMENTATION**

We also provide the MATLAB code to run *SIMLR* on the 5 example datasets in the script *Matlab_main_demo_SIMLR.m* and *Matlab_main_demo_SIMLR_Large_Scale.m*. 

Please refer to the directory *MATLAB* and the file README.txt within for further details. 

**DEBUG**

Please feel free to contact us if you have problems running our tool at daniele.ramazzotti1@gmail.com or wangbo.yunze@gmail.com. 