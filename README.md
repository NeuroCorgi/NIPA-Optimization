# Regularization Parameter Optimization in NIPA using Simulated Annealing

### Overview
This project is created for the bachelor thesis "Regularization Parameter Optimization in NIPA using Simulated Annealing" by Teun Hoven (University of Twente). The project aims to model, predict, and evaluate the spread of COVID-19 infections in different regions using the Network Inference-based Prediction Algorithm (NIPA) and various optimization techniques. 

### Network Inference-based Prediction Algorithm (NIPA)
The NIPA framework is designed to predict the spread of infectious diseases such as COVID-19 by leveraging network inference techniques. The algorithm is implemented to utilize multiple optimization techniques to determine the best set of parameters for accurate predictions.

### Original Paper
The original concept of NIPA can be found in the paper:
- [NIPA Paper](https://example-link-to-original-paper.com)

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

- `--country`: Country to use [mexico, hubei, netherlands]
- `--optimizers`: Optimizers to compare, separated by a comma [cv, cv_own, gsa, dsa]
- `--visuals`: Visualizations to show, separated by a comma [all, all_pred, all_eval, heatmap, optimizer_pred, optimizer_eval]
- `--visual_days`: Amount of days to show on the prediction visualization [default is 30]
- `--evaluations`: Evaluations to compare, separated by a comma [mse, mape, smape]
- `--type`: Type of NIPA to use [original, dynamic (NOT IMPLEMENTED)]
- `--n_days`: Number of days to iterate the model over [default is all days]
- `--train_days`: Number of days to train on when using Dynamic NIPA
- `--pred_days`: Number of days to predict; can be multiple days separated by a comma
- `--compensate_fluctuations`: Compensate for fluctuations in the data where each weekend has lower reported cases
- `--predict`: Only predict the next days based on the trained model
- `--random`: Randomize the seed for LASSO and simulated annealing used for the NIPA (otherwise random_state=42 is used)

### Examples
To run the project with specific settings, use the command:
```bash
python main.py --country mexico --optimizers cv,dsa --visuals all_pred,all_eval --evaluations mse,smape --type original --pred_days 3,7,14
```

### Future Implementations
- **Dynamic NIPA**: Implementing the dynamic version of NIPA for more flexible and adaptive predictions.
- **Additional Visualizations**: Including more detailed and interactive visualizations.
- **Enhanced Evaluation Metrics**: Introducing more sophisticated evaluation metrics to assess model performance.
- **Extended Data Sources**: Integrating additional data sources for a more comprehensive dataset.

### Requirements
- Python 3.x
- Required Python packages: `numpy`, `pandas`, `polars`, `scikit-learn`, `matplotlib`, `scipy`

### Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

### Contribution
Contributions are welcome! Please fork the repository and submit pull requests for any improvements or additions.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgements
- University of Twente
- Supervisors and colleagues who provided guidance and support during the thesis project.