mamba create -n p312 python=3.12 -c conda-forge
mamba activate p312

mamba install pandas numpy
mamba install matplotlib seaborn 
mamba install statsmodels scipy scikit-learn nltk
mamba install jupyter jupyterlab nodejs watermark
mamba install jupyterlab-spellchecker jupyterlab_code_formatter
mamba install pyarrow fastparquet h5py
mamba install lxml bs4 openpyxl xlrd requests_cache
mamba install arrow polars pandasdmx 
mamba install pylint flake8 black nbqa ruff isort
