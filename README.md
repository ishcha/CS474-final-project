# Setup

- Create a conda environment by running the following command:
```conda env create -f environment.yml```
- Activate the conda environment with ```conda activate lnn```

# Training an LNN to represent a dataset
To train an LNN based on m-CNF, use the following command:

```python3 m_cnf_lnn.py <file_name> <m>```

In the above command, the `file_name` is the common location of the dataset csv file and the training loss variation plot. Specifically, the dataset should be present at `data/<file_name>.csv` and the training loss plot will be saved in `figures/<file_name>.png`.

`m` is the number of clauses in the m-CNF form of the LNN. 
