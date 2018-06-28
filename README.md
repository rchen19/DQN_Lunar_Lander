## Project 2: Lunar Lander

The repo contains code to run experiments required for CS 7642 Summer 2018 Project 2.

* Required packages: Python 3.6.1, Numpy 1.12.1, Pandas 0.20.1, Matplotlib 2.0.2, Gym 0.10.5, Box2D 2.3.2, PyTorch 0.4.0.
* The repo contains these folders:
	* `logs`: contains csv files that are results from each experiments, the file names are in the format of `train_param_<experiment number>` or `test_param_<experiment number>` for training or testing results, where `experiment number` is a four-digit number corresponding to the row numbers in `param_list.csv` under `params` directory. As well as plots generated from these log files
	* `params`: contains `param_list.csv` and `param_list.json`. All hyperparameters used in experiments can be found in these two files, index by `experiment number` described above.
	* `report`: full report in PDF and LaTex format, and accessory files
	* `weights`: saved model and model weights, files named after the `experiment number`.


* The repo contains these files:
	* `README.md`: this file, description of this repo
	* `learner.py`: DQN learner class and necessary facilitating classes, called from `run.py`. DQN learner class includes both `learn()` and `test()` methods, `test()` is called after `learn()` is finished
	* `tester.py`: DQN tester class, used to run additional tests on existing trained model weights.
	* `run.py`: used to pass hyperparameters to and run DQN learner.
	* `plotter.py`: plotting functions that are called by other scripts
	* `plotlog.py`: used to plog a log file under `log` directory by running `python plotlog.py <file name>` in terminal, where `file name` is without extension.
	
	
* Perform experiments by running `python run.py` from terminal, after setting desired hyperparameters in `run.py` script.