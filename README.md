# tft_win_rate

This repository is for predicting TFT results using several machine learning algorithms.

## Data
Use the Riot API to get the game data. This data contains the game ID, generation time, player name, placement, gold left, level, total damage, traits, and units.

## machine learning algorithm

- Neural network(deep learning) : Mean squared Error : 0.126, Accuracy : 0.804, F1 Score : 0.7993

- SVM : Mean squared Error : 0.111, Accuracy : 0.840, F1 Score : 0.838 ( Using Grid Search to find best parameter)
