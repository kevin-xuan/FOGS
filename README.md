# IJCAI 2022. FOGS: First-Order Gradient Supervision with Learning-based Graph for Traffic Flow Forecasting
![image](framework.PNG)
## Data Preparation
download [STSGCN_data](https://github.com/Davidham3/STSGCN) and uncompress the file into data directory. The data directory is as follows:

FOGS/data/PEMS03/PEMS03.csv

FOGS/data/PEMS04/PEMS04.csv
<!-- 再将poi_graph.zip放到根目录的KGE文件夹下解压后得到36个graph.pkl文件，目录如下：

Graph_Flashback/KGE/gowalla_scheme1_transh_loc_temporal_20.pkl -->
## Usage
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
<!-- 
PEMS07 
```
python graph_preparation.py --num_of_vertices 883 --distances_filename ../../data/PEMS07/PEMS07.csv --data_filename ../../data/PEMS07/PEMS07.npz --edgelist_filename ../graph/PEMS07.edgelist --filename_T ../graph/PEMS07_graph_T.npz --flow_mean ../../data/PEMS07/PEMS07_flow_count.pkl
```
PEMS08
```
python graph_preparation.py --num_of_vertices 170 --distances_filename ../../data/PEMS08/PEMS08.csv --data_filename ../../data/PEMS08/PEMS08.npz --edgelist_filename ../graph/PEMS08.edgelist --filename_T ../graph/PEMS08_graph_T.npz --flow_mean ../../data/PEMS08/PEMS08_flow_count.pkl
```
-->
### 2. embedding by random walk 
```
cd FOGS/node2vec-master/src
run main_tra.py
```

PEMS03 
```
python main_tra.py --input ../graph/PEMS03.edgelist --input_T ../graph/PEMS03_graph_T.npz --output ../emb/PEMS03.emb 
```

PEMS04 
```
python main_tra.py --input ../graph/PEMS04.edgelist --input_T ../graph/PEMS04_graph_T.npz --output ../emb/PEMS04.emb 
```
<!-- 
PEMS07 
```
python main_tra.py --input ../graph/PEMS07.edgelist --input_T ../graph/PEMS07_graph_T.npz --output ../emb/PEMS07.emb 
```

PEMS08 
```
python main_tra.py --input ../graph/PEMS08.edgelist --input_T ../graph/PEMS08_graph_T.npz --output ../emb/PEMS08.emb 
```
-->
### 3. spatio-temporal graph
```
cd FOGS/node2vec-master/scripts
run learn_graph.py
```

PEMS03 
```
python learn_graph.py --filename_emb ../emb/PEMS03.emb --output_pkl_filename ../../data/PEMS03 --thresh_cos 10 
```
<!-- 
PEMS04 
```
python learn_graph.py --filename_emb ../emb/PEMS04.emb --output_pkl_filename ../../data/PEMS04 --thresh_cos 10 
```

PEMS07 
```
python learn_graph.py --filename_emb ../emb/PEMS07.emb --output_pkl_filename ../../data/PEMS07 --thresh_cos 10 
```
PEMS08 
```
python learn_graph.py --filename_emb ../emb/PEMS08.emb --output_pkl_filename ../../data/PEMS08 --thresh_cos 10 
```
-->
### 4. data preprocessing
```
cd FOGS/STFGNN/
run generate_datasets.py
```

PEMS03 
```
python generate_datasets.py --output_dir ../data/processed/PEMS03/ --flow_mean ../data/PEMS03/PEMS03_flow_count.pkl --traffic_df_filename ../data/PEMS03/PEMS03.npz
```

### 5. train model
```
cd FOGS/STFGNN/
change DATASET = 'PEMS0X' in line 16 in train.py
run train.py
```
# Citing
If you use FOGS in your research, please cite the following [paper](https://www.ijcai.org/proceedings/2022/0545.pdf):
```
@inproceedings{DBLP:conf/ijcai/RaoWZLS022,
  author    = {Xuan Rao and
               Hao Wang and
               Liang Zhang and
               Jing Li and
               Shuo Shang and
               Peng Han},
  title     = {{FOGS:} First-Order Gradient Supervision with Learning-based Graph
               for Traffic Flow Forecasting},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
               2022},
  year      = {2022}
}
```
