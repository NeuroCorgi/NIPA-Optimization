# Regularization Parameter Optimization in NIPA using Simulated Annealing

### Overview
This project is created for the bachelor thesis _Regularization Parameter Optimization in NIPA using Simulated Annealing_ by Teun Hoven at the University of Twente. The project aims to model, predict, and evaluate the spread of COVID-19 infections in different regions using the Network Inference-based Prediction Algorithm and optimizing it by using Simulated Annealing. 

### Network Inference-based Prediction Algorithm (NIPA)
The NIPA framework is designed to predict the spread of infectious diseases such as COVID-19 by leveraging network inference techniques and the SIR-model. The algorithm is implemented with multiple optimization techniques (cross-validation and simulated annealing) to determine the best set of parameters for accurate predictions.

### Original Paper
The original concept of NIPA can be found in the paper:
- [Network-inference-based prediction of the COVID-19 epidemic outbreak in the Chinese province Hubei](https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00274-2)

### Project Structure
- **main.py**: The starting point of the program. It handles the command-line arguments and orchestrates the data processing, training, and prediction workflows.
- **data_parser.py**: Contains functions to parse and prepare COVID-19 data from different countries.
- **NIPA.py**: Implements the NIPA model, including training and prediction functionalities.
- **Settings.py**: Manages the configuration settings for the program based on command-line arguments.
- **LASSOCV.py**: Implements Lasso regression with cross-validation for parameter optimization.
- **LASSODSA.py**: Implements Lasso regression with Dual Simulated Annealing for parameter optimization.
- **LASSOGSA.py**: Implements Lasso regression with Generalized Simulated Annealing for parameter optimization.
- **evaluation.py**: Contains functions to evaluate model predictions using various metrics.
- **visualisation.py**: Provides functions to visualize the results of the predictions and evaluations.
- **io.py**: Handles input and output operations, such as saving and loading data.

### Usage

#### Command-Line Arguments
The main entry point is `main.py`, which accepts the following arguments:

- `--country`: Country to use [hubei, mexico]
- `--optimizers`: Optimizers _(Cross-Validation, Generalized Simulated Annealing, Dual Simulated Annealing)_ to compare, separated by a comma [cv, gsa, dsa]
- `--visuals`: Visualizations to show, separated by a comma [all, all_pred, all_eval, heatmap, optimizer_pred, optimizer_eval]
- `--visual_days`: Amount of days to show on the prediction visualization [default is 30]
- `--evaluations`: Evaluations to compare, separated by a comma [mse, mape, smape]
- `--type`: Type of NIPA to use [original].
- `--n_days`: Number of days to iterate the model over [default is all days]
- `--train_days`: Number of days to train on when using Dynamic NIPA
- `--pred_days`: Number of days to predict; can be multiple days separated by a comma
- `--compensate_fluctuations`: Compensate for fluctuations in the data where each weekend has lower reported cases
- `--predict`: Only predict the next days based on the trained model
- `--random`: Randomize the seed for LASSO and simulated annealing used for the NIPA (otherwise seed of 42 is used)

### Examples
To run the project with specific settings, use the command:
```bash
python main.py --country hubei --optimizers cv,dsa --evaluations mse,smape --pred_days 1,2,3,4,5,6
```

### Future Implementations

#### Countries
- **The Netherlands**: Adding the Netherlands and later more countries for better comparisons between optimizers.
- **Extended Data Sources**: Integrating additional data sources for a more comprehensive dataset.

#### Optimizers
- **2D Simulated Annealing**: Integrating the optimization of the curing probability to simulated annealing will, in theory, improve accuracy and performance.
- **Bayesian Optimization**: Implementing Bayesian optimization for better (and faster) regularization parameter optimization.

#### Visualisations & Evaluations
- **Additional Visualisations**: Implementing additional (interactive) visualisations can help with understanding the data.
- **Additional Evaluation Metrics**: Implementing more evaluation metrics can help in analysing the data.

#### Types
- **Static NIPA**: Implementing the NIPA static prior ([from Achterberg et al.](https://linkinghub.elsevier.com/retrieve/pii/S0169207020301552)).
- **Dynamic NIPA**: Implementing the NIPA dynamic prior ([from Achterberg et al.](https://linkinghub.elsevier.com/retrieve/pii/S0169207020301552)).

### Requirements
- Python 3.x
- Required Python packages: `numpy`, `pandas`, `polars`, `scikit-learn`, `matplotlib`, `scipy`

### Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

### Acknowledgements
- Alberto García-Robledo and Zangiabady Mahboobeh for providing guidance and support during the thesis project.
- University of Twente for providing computing power

### Notes
The optimizer `dsa`/`gsa` has only been implemented on the regularization parameter. As of now, the curing probability is still been chosen by considering _candidate values_ and iterating over those and see which one fits best. This the the reason that `dsa` and `gsa` take a long time, because for each curing probability (in the _candidate values_ set), the simulated annealing algorithm is performed to find the best regularization parameter for the curing probability. In a later version, simulated annealing will be used to optimize both the curing probability and the regularization parameter.
