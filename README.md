# Stock Trader
![logo](Banner.jpg)

## Description

Stock Trader is a command line stock trader reinforcement learning model designed to operate on historical stock data. This model employs a Greedy Technique for buying and selling stocks, allowing users the flexibility to customize their trading strategy. Currently, it operates on the closing data of three sample stocks: Apple, Motorola, and Starbucks.

## Guidelines

### Customization

This model is not linked to any `API`, providing users the freedom to customize their trading strategies.

### Data Input

The model currently operates on closing stock data. Ensure that the dataset provided aligns with this format.

### Educational Use

This project is intended for educational purposes and experimentation. Please refrain from using it for actual trading.


## Installation

To use this Python-based script, ensure you have the required modules installed.
 - NumPy
 - Pandas
 - TensorFlow
 - Scikit-learn
 - Matplotlib
 - Additionally, some pre-installed modules.

## Usage

### Training the Model

1. Load the data into the Dataset directory.
2. Execute the following command in the command line:
   ```bash
   python RL_trader.py -m train
   ```
3. To view training results, run:
   ```bash
   python plot_rewards.py -m train
   ```

### Testing the Model
1. Run the following command in the command line:
  ```bash
  python RL_trader.py -m test
  ```
2. To view testing results, execute:
  ```bash
  python plot_rewards.py -m test
  ```

## RL_trader.py Script

The `RL_trader.py` script implements a reinforcement learning model using deep Q-learning for stock trading. It includes key components like data loading, replay buffer, neural network model, trading environment, and the agent.

## Contributing
Contributions to this project are welcome. To contribute, follow these guidelines:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request, which will be reviewed and merged accordingly.
4. `This Repository is open to all ideas`

### Guidelines for Contributions
When contributing to the project, please keep the following guidelines in mind:
1. Follow the coding style and conventions used in the project.
2. Write clear and concise commit messages and pull request descriptions.
3. Make sure your code is well-documented and tested.
4. Ensure that your changes do not break any existing functionality.
5. Use meaningful variable and function names.
6. Keep your changes focused and specific to the issue you are addressing.

### Code of Conduct
I expect all contributors to follow our Code of Conduct. Please be respectful and considerate of other contributors and users of the project.

### Reporting Bugs
If you find any bugs or issues in the project, please open an issue in the repository and provide as much detail as possible, including steps to reproduce the issue, error messages, and screenshots if applicable.

## License
This project is currently licensed under [Apache](LICENSE).


## Contact
For any inquiries or suggestions, feel free to reach out.