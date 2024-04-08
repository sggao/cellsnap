import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CellSNAP",
    version="0.0.2",
    author="Sheng Gao, Bokai Zhu",
    author_email="gaosheng0321@gmail.com, zhubokai@mit.edu",
    description="A package for enhancing single-cell population delineation by integrating cross-domain information.",
    long_description="Official implementation of Cell Spatial And Neighborhood Pattern (CellSNAP), a computational method that learns a single-cell representation embedding by integrating cross-domain information from tissue samples. Through the analysis of datasets spanning spatial proteomic and spatial transcriptomic modalities, and across different tissue types and disease settings, we demonstrate CellSNAP’s capability to elucidate biologically relevant cell populations that were previously elusive due to the relinquished tissue morphological information from images",
    long_description_content_type="text/markdown",
    url="https://github.com/sggao/CellSNAP",
    packages=setuptools.find_packages(),
    install_requires=['anndata==0.10.6',
                        'leidenalg>=0.9.1',
                        'matplotlib==3.6.2',
                        'numpy==1.23.5',
                        'pandas==2.0.0',
                        'python_igraph==0.11.4',
                        'scanpy==1.10.0',
                        'scikit_image==0.19.3',
                        'scikit_learn==1.2.1',
                        'scipy==1.9.1',
                        'torch==2.0.0',
                        'torch_geometric==2.3.0',
                        'torchvision==0.13.1',
                        'tqdm==4.64.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)