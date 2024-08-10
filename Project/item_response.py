import random
from typing import Any, Dict, List, Tuple
from matplotlib import pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data: Dict[str, List], theta: Any, beta: Any) -> float:
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TO/DO:                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_likelihood = 0.0
    for i in range(len(data['user_id'])):
        user = data['user_id'][i]
        question = data['question_id'][i]
        correct = data['is_correct'][i]
        the = theta[user]
        bet = beta[question]
        prob = sigmoid(the - bet)
        log_likelihood += correct*np.log(prob) + (1-correct)*np.log(1-prob)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -1 * log_likelihood


def update_theta_beta(data: Dict[str, List], lr: float, theta: Any, beta: Any
                      ) -> Tuple[Any, Any]:
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TO/DO:                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(len(data['user_id'])):
        correct = data['is_correct'][i]
        user = data['user_id'][i]
        question = data['question_id'][i]
        prob = sigmoid(theta[user] - beta[question])
        theta[user] += lr * correct - prob * lr
        beta[question] -= lr * correct - prob * lr
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data: Dict[str, List], valid_data: Dict[str, List], lr: float, iterations: int
        ) -> Tuple[Any, Any, List[float], List[float]]:
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TO/DO: Initialize theta and beta.
    num_users = len(set(train_data['user_id']))
    num_questions = len(set(train_data['question_id']))
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)
    train_nll_lst = []
    valid_nll_lst = []
    valid_acc_lst = []

    for iter in range(iterations):
        train_nll = neg_log_likelihood(data=train_data, theta=theta, beta=beta)
        train_nll_lst.append(train_nll)
        valid_nll = neg_log_likelihood(data=valid_data, theta=theta, beta=beta)
        valid_nll_lst.append(valid_nll)
        valid_acc =           evaluate(data=valid_data, theta=theta, beta=beta)
        valid_acc_lst.append(valid_acc)
        
        print(f"iter: {iter}\ttrain_nll: {train_nll:.12f}\tvalid_nll: {valid_nll:.12f}\tvalid_acc: {valid_acc}")
        
        theta, beta = update_theta_beta(train_data, lr, theta, beta)

    # TO/DO: You may change the return values to achieve what you want.
    return theta, beta, train_nll_lst, valid_nll_lst, valid_acc_lst


def evaluate(data: Dict[str, List], theta: Any, beta: Any) -> float:
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    acc = np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])
    return acc


def main() -> None:
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TO/DO:                                                            #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # TO/DO: We need to tune the number of iterations and learning rate
    HYP_LR = 0.001    # Learning rate is usually 0.001, 0.01, 0.1
    HYP_ITER = 75  # Number of iterations is usually 100, 1000, 10000
    
    theta, beta, train_nll_lst, valid_nll_lst, valid_acc_lst = irt(train_data=train_data, valid_data=valid_data, lr=HYP_LR, iterations=HYP_ITER)
    test_acc = evaluate(test_data, theta, beta)
    print(f'Test Accuracy: {test_acc}')
    
    train_ll_lst = [-x for x in train_nll_lst]
    valid_ll_lst = [-x for x in valid_nll_lst]
    
    # Plotting the log-likelihoods against iterations
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Training Log-Likelihood', color='tab:blue')
    ax1.plot(train_ll_lst, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Log-Likelihood', color='tab:red')
    ax2.plot(valid_ll_lst, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Log-Likelihood against Iterations')
    plt.show()
    
    # Plotting the validation accuracy against iterations
    plt.plot(valid_acc_lst, color='tab:green')
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy against Iterations')
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TO/DO:                                                            #
    # Select three questions j_1, j_2, and j_3. Using the trained θ and #
    # β, plot three curves on the same plot that shows the probability  #
    # of the correct response p(c_ij = 1) as a function of θ given a    #
    # question j.                                                       #
    #####################################################################
    
    random_three_questions = [] # random.sample(train_data['question_id'], 3)
    theta_range = np.linspace(-5, 5, 100)
    # current best
    j1 = 58 # lowest
    j2 = 170 # middle
    j3 = 95 # highest
    plt.plot(theta_range, sigmoid(theta_range - beta[j1]), label=f'Question {j1}', color='blue')
    plt.plot(theta_range, sigmoid(theta_range - beta[j2]), label=f'Question {j2}', color='green')
    plt.plot(theta_range, sigmoid(theta_range - beta[j3]), label=f'Question {j3}', color='red')
    
    for i in random_three_questions:
        prob = sigmoid(theta_range - beta[i])
        plt.plot(theta_range, prob, label = f'Question {i}')
    
    plt.xlabel('Theta')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response against Theta')
    plt.legend()
    # plt.savefig('P vs. Theta.png')
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
