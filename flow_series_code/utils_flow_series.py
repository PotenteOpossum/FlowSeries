import random
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
import pandas as pd

def generate_random_weighted_network(n_nodes=10, edge_prob=0.15, multi_edge_prob=0.8, weight_range=(1, 10)):
    """
    Generate a random weighted network.
    :param n_nodes: Number of nodes in the network
    :param edge_prob: Probability of edge creation
    :param weight_range: Min and max weight for edges
    :return: NetworkX graph
    """
    # G = nx.erdos_renyi_graph(n_nodes, edge_prob, directed=True)
    G = nx.barabasi_albert_graph(n = n_nodes, m=7)
    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(*weight_range)
    # Convert to MultiDiGraph to allow multiple edges
    multi_G = nx.MultiDiGraph(G)

    # Add additional random edges to simulate a true MultiDiGraph
    for u, v in list(G.edges()):  # Iterate over original edges
        while random.uniform(0, 1) < multi_edge_prob:  # 30% chance to add an extra edge
            multi_G.add_edge(u, v, weight=random.randint(*weight_range))

    # Take only the largest strongly connected component
    largest_scc = max(nx.strongly_connected_components(multi_G), key=len)
    multi_G = multi_G.subgraph(largest_scc).copy()
    return multi_G

def reduce_weight(G, x, y):
    """
    Reduces the weight of all edges between nodes x and y by half.
    
    Args:
        G (nx.MultiDiGraph): Input graph
        x (int): Source node
        y (int): Target node
    
    Returns:
        tuple: (Modified graph, Average of reduced weights)
    """
    reduced_weight = []
    for edge in G[x][y]:
        weight = G[x][y][edge]['weight']
        G[x][y][edge]['weight'] = weight//2
        reduced_weight.append(weight//2)
    return G, sum(reduced_weight)/len(reduced_weight)

def increase_weight(G, x, y, reduced_weight):
    """
    Increases the weight of edges between nodes x and y using a random factor.
    
    Args:
        G (nx.MultiDiGraph): Input graph
        x (int): Source node
        y (int): Target node
        reduced_weight (float): Average weight that was reduced, used in calculation
    
    Returns:
        nx.MultiDiGraph: Modified graph with increased weights
    """
    # tot_weight = [G[x][y][edge]['weight'] for edge in G[x][y]]
    for edge in G[x][y]:
        G[x][y][edge]['weight'] = G[x][y][edge]['weight']+random.uniform(G[x][y][edge]['weight']/2, G[x][y][edge]['weight'])*2+random.uniform(0, reduced_weight)*10
    return G

def stable_perturbation(G):
    """
    Applies a stable perturbation to all edge weights in the graph.
    Perturbs each edge weight by a random factor between 0.8 and 1.2.
    
    Args:
        G (nx.MultiDiGraph): Input graph
    
    Returns:
        nx.MultiDiGraph: Graph with perturbed edge weights
    """
    for edge_group in set(edge_group for edge_group in G.edges()):
        std = np.array([G[edge_group[0]][edge_group[1]][edge]['weight'] for edge in G[edge_group[0]][edge_group[1]]]).std()
        for edge in G[edge_group[0]][edge_group[1]]:
            # new_val = (random.choice([-1,1]) * G[edge_group[0]][edge_group[1]][edge]['weight']) + (random.choice([-1,1])*int(std))
            # G[edge_group[0]][edge_group[1]][edge]['weight'] += random.choice([-1,1])*int(std)
            original_weight = G[edge_group[0]][edge_group[1]][edge]['weight']
            perturbation_range=(0.8, 1.2)
            perturbation_factor = random.uniform(*perturbation_range)
            G[edge_group[0]][edge_group[1]][edge]['weight'] = original_weight * perturbation_factor
    return G

def draw_graph(G, perturbed=False, x=0, y=8):
    """
    Visualizes the graph with labeled weights and highlighted nodes/edges.
    
    Args:
        G (nx.MultiDiGraph): Graph to visualize
        perturbed (bool): Whether the graph is perturbed (unused parameter)
        x (int): First node to highlight in red
        y (int): Second node to highlight in red
    """
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nodes_colors = ['red' if node in [x,y] else 'gray' for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=nodes_colors, node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    red_edges = [(x, y)]  # Highlight the edge (x, y) in red
    edge_colors = ['red' if edge in red_edges else 'gray' for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)
    plt.show()

def find_paths(G,u,n):
    """
    Finds all possible paths of length n starting from node u.
    
    Args:
        G (nx.MultiDiGraph): Input graph
        u (int): Starting node
        n (int): Path length
    
    Returns:
        list: List of paths, where each path is a list of node IDs
    """
    paths = []
    if n==0:
        return [[u]]
    if n==1:
        paths.append([u])
    for neighbor in G.neighbors(u):
        for path in find_paths(G,neighbor,n-1):
            if u not in path:
                paths.append([u]+path)
    return paths

def plot_weight_vs_moving_avg(df, attr='weight', lim=None, save='', ylabel='Normalized value'):
    """
    Plots the normalized 'weight' and 'moving_avg' columns of a DataFrame against the date index.
    Connects the markers with dotted lines and fills the area between the lines,
    both colored based on the relationship between the two values.

    Args:
        df (pd.DataFrame): DataFrame with a date index and 'weight' and 'moving_avg' columns.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    
    if attr == 'weight':
        attr2 = 'moving_avg'
    elif attr == 'delta_weight':
        attr2 = 'moving_avg_delta_weight'
        df['moving_avg_delta_weight'] = df['delta_weight'].rolling(window=4).mean()
    else:
        raise ValueError("attr must be 'weight' or 'delta_weight'.")

    plt.figure(figsize=(12, 6))

    # Normalize the y-axis values
    weight_normalized = (df[attr] - df[attr].min()) / (df[attr].max() - df[attr].min())
    moving_avg_normalized = (df[attr2] - df[attr2].min()) / (df[attr2].max() - df[attr2].min())

    plt.plot(df.index, weight_normalized, marker='^', label=attr.title().replace('_', ' '), c='tab:orange')
    plt.plot(df.index, moving_avg_normalized, marker='o', label='Moving Average', c='tab:blue')

    for date, row in df.iterrows():
        weight_norm = (row[attr] - df[attr].min()) / (df[attr].max() - df[attr].min())
        moving_avg_norm = (row[attr2] - df[attr2].min()) / (df[attr2].max() - df[attr2].min())

        color = 'green' if weight_norm > moving_avg_norm else 'red'
        plt.plot([date, date], [weight_norm, moving_avg_norm], linestyle=':', color=color)

    # Fill area between lines (corrected with interpolate=True)
    plt.fill_between(
        df.index,
        weight_normalized,
        moving_avg_normalized,
        where=weight_normalized >= moving_avg_normalized,
        facecolor='green',
        alpha=0.2,
        interpolate=True,
        # label='Normalized Weight >= Moving Average'
    )
    plt.fill_between(
        df.index,
        weight_normalized,
        moving_avg_normalized,
        where=weight_normalized < moving_avg_normalized,
        facecolor='red',
        alpha=0.2,
        interpolate=True,
        # label='Normalized Weight < Moving Average'
    )

    plt.axvline(pd.to_datetime('2022-03-27'), color='gray', lw=2, zorder=0)#, linestyle='--')

    plt.xlabel('Time', fontsize=19)
    if ylabel is None:
        plt.ylabel('')
    else:
        plt.ylabel(ylabel, fontsize=19)#+attr.title(), fontsize=19) # Adjusted label
    plt.legend(fontsize=19, ncol=1)
    plt.grid(False)

    # Reduce number of ticks and use subscripts
    num_ticks = min(8, len(df.index))
    tick_indices = np.linspace(0, len(df.index) - 1, num_ticks, dtype=int)
    tick_labels = [r't$_{' + str(i+1) + r'}$' for i in tick_indices]
    plt.xticks(df.index[tick_indices], tick_labels, fontsize=19) #rotation=45, 

    plt.yticks(fontsize=13)
    if lim is not None:
        plt.xlim(right=pd.to_datetime(lim))

    plt.tight_layout()
    if save!= '':
        plt.savefig(f"{save}.pdf", dpi=300, format='pdf')
    plt.show()

def perturb_network(G, x=None, y=None, max_path_length=3, num_intermediaries='all', edge_sequences=None, log=False):
    """
    Perturb the network by redirecting weight through an intermediary node.
    - Select a random edge (x, y)
    - Find a node j that is connected to BOTH x and y
    - Reduce weight of (x, y) and distribute it to (x, j) and (j, y)
    """
    if edge_sequences==None:
        if len(G.edges) < 3:
            print("Not enough edges to perturb.")
            return G, []

        if x!=None and y!=None and not G.has_edge(x, y):
            print(f"Not edge from {x} to {y}.")
            return G, []

        if x==None:
            x = random.choice(list(G.nodes()))

        paths = find_paths(G, x, max_path_length)
        paths = [r for r in paths if len(r)>2]
        path_ending_in_neig = [r[-1] for r in paths if r[-1] in G.neighbors(x)]

        if len(path_ending_in_neig)==0:
            if y!=None:
                print(f'No alternative path found from {x} to {y}')
            else:
                y = random.choice([r[-1] for r in paths])
                print(f'Adding an edge from {x} to {y}...')
                G.add_edge(x, y, weight=random.choice(list(G.out_edges(x, data=True)))[2]['weight'])
        else:
            if y==None:
                y = random.choice(path_ending_in_neig)
            else:
                if y not in path_ending_in_neig:
                    print(f'No alternative path found from {x} to {y}')
                    return G, []

        paths = [r for r in paths if r[-1]==y]
        possible_edge_sequences = [list(zip(path[:-1], path[1:])) for path in paths]

        if log:
            print(f"Possible intermediaries for ({x}, {y}): {len(possible_edge_sequences)}")
            print(possible_edge_sequences)

        if num_intermediaries=='random':
            num_intermediaries = random.randint(1, len(possible_edge_sequences))
            edge_sequences = random.sample(possible_edge_sequences, num_intermediaries)
        elif type(num_intermediaries)==int:
            edge_sequences = random.sample(possible_edge_sequences, num_intermediaries)
        else:
            edge_sequences = possible_edge_sequences
        # elif num_intermediaries=='all'
        if log:
            print(f"Chosen intermediaries for ({x}, {y}): {len(edge_sequences)}")
            print(edge_sequences)

    G, reduced_weight = reduce_weight(G.copy(), x, y)
    for path in edge_sequences:
        for edge in path:
            G = increase_weight(G.copy(), edge[0], edge[1], reduced_weight)

    if 'possible_edge_sequences' in locals() and possible_edge_sequences:
        return G, edge_sequences, possible_edge_sequences
    else:
        return G, edge_sequences