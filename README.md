# IJCAI 2022. FOGS: First-Order Gradient Supervision with Learning-based Graph for Traffic Flow Forecasting

## Data Preparation
download STSGCN_data (https://github.com/Davidham3/STSGCN) and uncompress the file into data directory. The data directory is as follows:

FOGS/data/PEMS03/PEMS03.csv

FOGS/data/PEMS04/PEMS04.csv

## Graph Construction
### 1. temporal correlation graph
```
cd FOGS/node2vec-master/scripts
run graph_preparation.py
```

PEMS03
```
python graph_preparation.py --sensor_ids_filename ../../data/PEMS03/PEMS03.txt --num_of_vertices 358 --distances_filename ../../data/PEMS03/PEMS03.csv --data_filename ../../data/PEMS03/PEMS03.npz --edgelist_filename ../graph/PEMS03.edgelist --filename_T ../graph/PEMS03_graph_T.npz --flow_mean ../../data/PEMS03/PEMS03_flow_count.pkl
```

PEMS04
```
python graph_preparation.py --num_of_vertices 307 --distances_filename ../../data/PEMS04/PEMS04.csv --data_filename ../../data/PEMS04/PEMS04.npz --edgelist_filename ../graph/PEMS04.edgelist --filename_T ../graph/PEMS04_graph_T.npz --flow_mean ../../data/PEMS04/PEMS04_flow_count.pkl
```

PEMS07
```
python graph_preparation.py --num_of_vertices 883 --distances_filename ../../data/PEMS07/PEMS07.csv --data_filename ../../data/PEMS07/PEMS07.npz --edgelist_filename ../graph/PEMS07.edgelist --filename_T ../graph/PEMS07_graph_T.npz --flow_mean ../../data/PEMS07/PEMS07_flow_count.pkl
```

PEMS08
```
python graph_preparation.py --num_of_vertices 170 --distances_filename ../../data/PEMS08/PEMS08.csv --data_filename ../../data/PEMS08/PEMS08.npz --edgelist_filename ../graph/PEMS08.edgelist --filename_T ../graph/PEMS08_graph_T.npz --flow_mean ../../data/PEMS08/PEMS08_flow_count.pkl
```
### 2. spatio-temporal graph
