# FlowSeries
Repository of the paper titled "FlowSeries: flow analysis on financial networks"


This repository contains the code for the paper "FlowSeries: flow analysis on financial networks" by Arthur Capozzi, Salvatore Vilella, Marco Fornasiero, Dario Moncalvo, Valeria Ricci, Silvia Ronchiadin, and Giancarlo Ruffo.

This software is NOT the code that we used to analyze the real data on the 80 million cross-country wire transfers described in the paper, and all the information connected with that data structure is omitted here. The code we used for that analysis is not available, and as for the related data we are not allowed to share it. However, we provide the code that we used to analyze the synthetic data, which is described in the paper. It is also possible to use this code with other datasets, once edgelists are properly provided.



## Features

- Generate random weighted networks with customizable parameters.
- Simulate perturbations in network flows.
- Analyze paths and intermediaries in the network.
- Compute metrics like moving averages, maximum weights, and delta weights.
- Visualize flow dynamics with customizable plots.
- Highlight suspicious flows.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/flow-series-analysis.git
   cd flow-series-analysis
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating and Analyzing Flow Series

1. Import the `FlowSeries` class:
   ```python
   from flow_series import FlowSeries
   ```

2. Create a `FlowSeries` object:
   ```python
   fs = FlowSeries(data_path='data_sintetic_flows')
   ```

3. Access key attributes:
   - `fs.networks`: The generated networks.
   - `fs.edge_sequences`: Perturbed edge sequences.
   - `fs.df`: DataFrame containing flow metrics.

4. Visualize flow dynamics:
   ```python
   fs.plot(attr='weight', paths='all')
   fs.plot(attr='delta_weight', paths='61, 31, 31, 29', save='output_plot')
   ```

### Testing

Run the Jupyter notebook `test_flow_series.ipynb` to explore the functionality interactively.

## File Structure

- `flow_series.py`: Core class for generating and analyzing flow series.
- `utils_flow_series.py`: Utility functions for network generation, perturbation, and visualization.
- `test_flow_series.ipynb`: Notebook for testing and demonstrating the functionality.

## Example

```python
from flow_series import FlowSeries

# Initialize FlowSeries
fs = FlowSeries(data_path='data_sintetic_flows')

# Analyze and visualize
fs.plot(attr='weight', paths='all')
fs.plot(attr='delta_weight', paths='all')
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.