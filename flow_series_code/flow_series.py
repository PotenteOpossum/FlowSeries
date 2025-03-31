import networkx as nx

import numpy as np
import pandas as pd
import tqdm, json, os
from utils_flow_series import find_paths, stable_perturbation, generate_random_weighted_network, perturb_network, plot_weight_vs_moving_avg


class FlowSeries:
    """
    A class for generating, perturbing, and analyzing flow series in networks.

    Attributes:
        n_nodes (int): Number of nodes in the network.
        edge_prob (float): Probability of edge creation.
        multi_edge_prob (float): Probability of creating multiple edges between nodes.
        weight_range (tuple): Range of weights for edges.
        x (int): Source node for perturbation.
        y (int): Target node for perturbation.
        max_path_length (int): Maximum path length for analysis.
        num_intermediaries (str or int): Number of intermediary paths ('random' or specific number).
        data_path (str): Path to save or load network data.
        log (bool): Whether to log operations.

    Methods:
        plot(attr='weight', paths='all', lim=None, save=''):
            Visualizes flow dynamics for specified paths and attributes.
    """
    def __init__(self,
                n_nodes: int = 200,
                edge_prob: float = 0.35,
                multi_edge_prob: float = 0.8,
                weight_range: tuple = (1, 10),
                x: int = None,
                y: int = None,
                max_path_length: int = 2,
                num_intermediaries: str = 'random',
                data_path: str = '',
                log: bool = False
                ):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.multi_edge_prob = multi_edge_prob
        self.weight_range = weight_range
        self.x = x
        self.y = y
        self.max_path_length = max_path_length
        self.num_intermediaries = num_intermediaries
        self.data_path = data_path
        self.log = log

        if self.data_path != '':
            if self.data_path[-1] != '/':
                self.data_path += '/'

            if self.__check_data_path():
                print(f"Checking data in {self.data_path}...")
                self.__load_data()  
        else:
            self.data_path = 'data_sintetic_flows/'
            self.__check_directory()
            self.networks, self.edge_sequences, self.possible_edge_sequences = self.__create_time_series_perturbation()
            self.__save_networks()
            self.df, self.paths = self.__find_intermediaries()

    def __load_data(self):   
        # Load the networks from the CSV files
        self.networks = {}
        self.edge_sequences = []
        self.possible_edge_sequences = []
        self.df = pd.DataFrame()
        self.paths = []
        
        # Read the configuration file
        json_files = [f for f in os.listdir(self.data_path) if f.endswith('.json') and 'report' in f]
        self.config_file = os.path.join(self.data_path, json_files[0])
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            self.dates = config['dates']
            self.start_node = config['start_node']
            self.end_node = config['end_node']
            self.max_path_length = config['max_path_length']
            self.num_intermediaries = config['num_intermediaries']
            self.edge_sequences = config['edge_sequences']
            self.possible_edge_sequences = config['possible_edge_sequences']

        for date in tqdm.tqdm(self.dates):
            edgelist = pd.read_csv(self.data_path + f'edgelist_{date}.csv')
            G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=['weight'], create_using=nx.MultiDiGraph())
            self.networks[date] = G
        
        self.df, self.paths = self.__find_intermediaries()
        

    def __check_directory(self):
        if not os.path.exists(self.data_path):
            print(f"Directory {self.data_path} does not exist. Creating it...")
            os.makedirs(self.data_path)
            return False
    def __check_data_path(self):
        """
        Checks if CSV files with edgelists exist in the specified data path.
        
        Returns:
            bool: True if CSV files exist, False otherwise
        """
        import os
        
        # Check if directory exists
        if not self.__check_directory():
            return False
            
        # Check for CSV files containing 'edgelist' in the name
        csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv') and 'edgelist' in f]
        
        if not csv_files:
            print(f"No edgelist CSV files found in {self.data_path}")
            return False
            
        # First find the JSON configuration file
        json_files = [f for f in os.listdir(self.data_path) if f.endswith('.json') and 'report' in f]
        if not json_files:
            print(f"No JSON configuration files found in {self.data_path}")
            return False
        return True

    def __find_intermediaries(self):
        with open(self.config_file, "r") as file:
            data = json.load(file)
        self.dates = data['dates']

        df = pd.DataFrame()
        print("Finding intermediaries...")
        for date in tqdm.tqdm(self.dates):
            def fill_list(input_list):
                n = 2
                fill = [np.nan] * n
                return [sublist[:n] + fill[len(sublist):] for sublist in input_list]

            def get_edge_data(edge):
                data = G.get_edge_data(edge[0], edge[1])
                return [data['amount'], data['count']]

            edgelist = pd.read_csv(f'/Users/acapozzi/Desktop/ISP/data_sintetic_flows/edgelist_{date}.csv')

            edgelist = edgelist.groupby(['source', 'target'], as_index=False).agg({'weight': ['sum', 'size']})
            edgelist.columns = ['source', 'target', 'amount', 'count']

            G = nx.from_pandas_edgelist(edgelist, source='source', target='target', edge_attr=['amount', 'count'], create_using=nx.DiGraph())

            paths = find_paths(G, data['edge_sequences'][0][0][0], 2)
            # paths = [r for r in paths if len(r)>2]

            paths = [[tuple(el) for el in list(path)] for path in map(nx.utils.pairwise, paths)]
            paths = fill_list(paths)

            paths = [r+get_edge_data(r[0])+get_edge_data(r[1]) if r[1]==r[1] else r+get_edge_data(r[0])+[0,0] for r in paths]
            paths = pd.DataFrame(paths, columns=['path'+str(i) for i in range(1,self.max_path_length+1)]+[f'amount{i+1}' if j % 2 == 0 else f'count{i+1}' for i in range(self.max_path_length) for j in range(2)])
            paths['date'] = pd.to_datetime(date, format='%Y_%m_%d')

            df = pd.concat([df, paths])

        df['path2'] = df['path2'].apply(lambda x: x if pd.notna(x) else tuple())
        df['path'] = df[['path1', 'path2']].apply(lambda x: ", ".join(map(str, x['path1'] + x['path2'])), axis=1)
        t = 'amount'
        df['weight'] = df.apply(lambda row: min([row[t+str(i)] for i in range(1, self.max_path_length)]), axis=1)

        df = df.sort_values(by=["path", "date"])

        # compute moving average of flow weight
        df["moving_avg"] = df[['date', 'path', 'weight']].groupby("path")["weight"].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        # compute max weight of flow weight
        df["max_weight"] = df.groupby("path")["weight"].transform("max")
        # remove first 4 weeks without moving average 
        df = df[~df['moving_avg'].isna()].reset_index(drop=True)
        # compute delta_weight for each path 
        df["delta_weight"] = df[['weight', 'moving_avg', "max_weight", 'path2']].apply(lambda x: (x['weight']-x['moving_avg'])/x["max_weight"] if x['path2']==tuple() else abs(x['weight']-x['moving_avg'])/x["max_weight"], axis=1)

        one_edge = df[df['path2']==tuple()][['path', 'date', 'delta_weight']]

        # Trova l'indice del massimo delta_weight per ogni path
        idx = df[df['path2']!=tuple()].groupby("path")["delta_weight"].idxmax()

        max_delta = df[df['path2']!=tuple()].loc[idx, ["path", "date", "delta_weight"]].reset_index(drop=True)
        max_delta['mapped_path'] = max_delta['path'].apply(lambda x: x.split(', ')[0] + ', ' +x.split(', ')[-1])

        paths = max_delta.merge(one_edge, left_on=['mapped_path', 'date'], right_on=['path', 'date'], how='left', suffixes=('', '_df1')).sort_values('delta_weight_df1', ascending=True)
        return df, paths[paths['delta_weight_df1']<=-0.1]['path'].values
    
    def __create_time_series_perturbation(self):

        G = generate_random_weighted_network(n_nodes=self.n_nodes, edge_prob=self.edge_prob, multi_edge_prob=self.multi_edge_prob, weight_range=self.weight_range)
        networks = {}

        start_date = "2021-12-01"
        weeks = pd.date_range(start=start_date, periods=19, freq="W")

        for week in weeks[:-2]:
            if self.log:
                print(f'stable_perturbation ({week})')
            G = stable_perturbation(G.copy())
            networks[week] = G.copy()
        if self.log:
            print(f'perturb_network ({weeks[-2]})')
        networks[weeks[-2]], edge_sequences, possible_edge_sequences = perturb_network(G.copy(), x=self.x, y=self.y, num_intermediaries=self.num_intermediaries, max_path_length=self.max_path_length, log=self.log)
        x = edge_sequences[0][0][0]
        y = edge_sequences[0][-1][1]

        weeks_strong_perturbation = pd.date_range(start=weeks[-1], periods=7, freq="W")

        for week in weeks_strong_perturbation:
            if self.log:
                print(f'perturb_network ({week})')
            G, _ = perturb_network(networks[weeks[-2]].copy(), x=x, y=y, edge_sequences=edge_sequences, num_intermediaries=self.num_intermediaries, max_path_length=self.max_path_length, log=self.log)
            networks[week] = G.copy()
        
        return networks, edge_sequences, possible_edge_sequences
    
    def __save_networks(self):
        self.dates = list(self.networks.keys())
        self.start_node = self.edge_sequences[0][0][0],
        self.end_node = self.edge_sequences[0][-1][1],
        print("Saving networks...")
        for date in tqdm.tqdm(self.dates):
            edgelist = nx.to_pandas_edgelist(self.networks[date])
            edgelist['date'] = date
            edgelist.to_csv(self.data_path+f'edgelist_{date.strftime("%Y_%m_%d")}.csv', index=False)
        report = {
            "dates": [date.strftime('%Y_%m_%d') for date in self.dates],
            "start_node": self.start_node,
            "end_node": self.end_node,
            "max_path_length": self.max_path_length,
            "num_intermediaries": self.num_intermediaries,
            "edge_sequences": self.edge_sequences,
            "possible_edge_sequences": self.possible_edge_sequences
        }
        name = f"_from_{min(self.dates).strftime('%Y_%m_%d')}_to_{max(self.dates).strftime('%Y_%m_%d')}"
        self.config_file = self.data_path+'report'+name+'.json'
        with open(self.config_file, "w") as json_file:
            json.dump(report, json_file, indent=4)

    def plot(self, attr='weight', paths='all', lim=None, save=''):
        if paths == 'all':
            for path in self.paths:
                plot_weight_vs_moving_avg(self.df[(self.df['path']==path)].set_index('date')[[attr, 'moving_avg']], attr=attr, lim=lim, save=save)
        elif type(paths)==list:
            for path in paths:
                plot_weight_vs_moving_avg(self.df[(self.df['path']==path)].set_index('date')[[attr, 'moving_avg']], attr=attr, lim=lim, save=save)
        elif type(paths)==str:
            plot_weight_vs_moving_avg(self.df[(self.df['path']==paths)].set_index('date')[[attr, 'moving_avg']], attr=attr, lim=lim, save=save)
        else:
            raise ValueError('paths must be string, a list of strings or "all"')

