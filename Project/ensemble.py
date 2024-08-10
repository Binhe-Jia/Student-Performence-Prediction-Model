# TO/DO: complete this file.

# Instructions:
# In this problem, you will be implementing bagging ensemble to improve the 
# stability and accuracy of your base models. Select and train 3 base models 
# with bootstrapping the training set. You may use the same or different 
# base models. Your implementation should be completed in ensemble.py. To 
# predict the correctness, generate 3 predictions by using the base model 
# and average the predicted correctness. Report the final validation and 
# test accuracy. 

# Writeup Instructions:
# Explain the ensemble process you implemented. Do you 
# obtain better performance using the ensemble? Why or why not?

# TO/DO list:
# Objective. Bagging ensemble to improve stability and accuracy of base models.
# 1. Prepare / Bootstrap the training set
# 2. Train the base model: Neural Network (single layer, k=50 nodes wide, lr=0.005, lamb=0.001, epochs=75, model=AutoEncoder)
# 3. Generate individual predictions for the base model, and average the predictions.
# 4. Report the final validation and test accuracy of the ensemble.
# 5. Writeup. Compare the naive averaged predictions with the ensemble predictions.


import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import random
import time
import logging
from typing import Any, Dict, List, Tuple, Union
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch

from utils import (
    load_valid_csv,
    load_train_sparse,
    load_public_test_csv,
)

from neural_network import load_data, AutoEncoder, train, evaluate



def take_sample_train_matrix(train_matrix: np.ndarray) -> np.ndarray:
    # Generate a bootstrapped training set.
    # 1. Draw a sample of size n from the original dataset with replacement.
    #   a. Each bootstrap sample should typically have the same size as the 
    #      original dataset, but some data points may be repeated, while 
    #      others may be omitted.
    # 2. Create a new dataset with the drawn samples.
    # 3. Return the new dataset.
    # TODO seed?
    
    n = train_matrix.shape[0]
    rng = np.random.default_rng()
    idx = rng.choice(n, n, replace=True)
    new_train_matrix = train_matrix[idx]
    
    return new_train_matrix

def run_naive_base_NN(zero_train_matrix, train_matrix, valid_data, test_data):
    # Generate a naive prediction. Torch and numpy already have built-in random seeds.
    
    k = 50
    lr = 0.005
    lamb = 0.001
    epochs = 75
    model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
    
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epochs)
    
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    
    return valid_acc, test_acc

def evaluate_n(models: List[nn.Module], train_data: torch.FloatTensor, valid_data: Dict[str, List]):
    """Evaluate the valid_data on the given models.

    :param models: List of Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    if len(models) == 0:
        raise ValueError("models must not be empty")
    elif len(models) % 2 == 0:
        logging.error("Number of models should be odd for majority voting")
    
    for model in models:
        model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        outputs = []
        for model in models:
            outputs.append(model(inputs))

        guesses = []
        for output in outputs:
            guess = output[0][valid_data["question_id"][i]].item() >= 0.5
            guesses.append(guess)
            
        final_guess = sum(guesses) >= len(models) // 2 + 1  # Majority voting. Assume odd number of models
        if final_guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    base_path = "./data"
    ndarray_train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    
    ndarray_zero_train_matrix = ndarray_train_matrix.copy()
    ndarray_zero_train_matrix[np.isnan(ndarray_train_matrix)] = 0
    
    naive_zero_train_matrix = torch.FloatTensor(ndarray_zero_train_matrix)
    naive_train_matrix = torch.FloatTensor(ndarray_train_matrix)

    
    
    # First predict naive NN's correctness.
    temp_valid_accs = []
    temp_test_accs = []
    for _ in range(3):
        valid_acc, test_acc = run_naive_base_NN(naive_zero_train_matrix, naive_train_matrix, valid_data, test_data)
        temp_valid_accs.append(valid_acc)
        temp_test_accs.append(test_acc)
    naive_valid_acc = np.mean(temp_valid_accs)
    naive_test_acc = np.mean(temp_test_accs)
    
    # Setup bagging ensemble.
    NUM_BAGS = 3
    k = 50
    lr = 0.005
    lamb = 0.001
    epochs = 75
    models = []
    for _ in range(NUM_BAGS):
        # Generate a new bootstrapped training set.
        sampled_ndarray_train_matrix = take_sample_train_matrix(ndarray_train_matrix)
        sampled_ndarray_zero_train_matrix = sampled_ndarray_train_matrix.copy()
        sampled_ndarray_zero_train_matrix[np.isnan(sampled_ndarray_train_matrix)] = 0
        
        bootstrapped_zero_train_matrix = torch.FloatTensor(sampled_ndarray_zero_train_matrix)
        bootstrapped_train_matrix = torch.FloatTensor(sampled_ndarray_train_matrix)
        
        model = AutoEncoder(num_question=ndarray_train_matrix.shape[1], k=k)
        train(model, lr, lamb, bootstrapped_train_matrix, bootstrapped_zero_train_matrix, valid_data, epochs)
        models.append(model)
    
    ensemble_valid_acc = evaluate_n(models, naive_zero_train_matrix, valid_data)    # verify here that it's naive_zero_train_matrix, not the inividual bootstrapped ones?
    ensemble_test_acc = evaluate_n(models, naive_zero_train_matrix, test_data)
    
    print("Naive NN validation accuracy:", naive_valid_acc)
    print("Ensemble validation accuracy:", ensemble_valid_acc)
    print()
    print("Naive NN test accuracy:", naive_test_acc)
    print("Ensemble test accuracy:", ensemble_test_acc)
        
        
        







if __name__ == "__main__":
    # logging.getLogger().addHandler(logging.NullHandler())
    main()




