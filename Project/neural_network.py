import numpy as np
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
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path: str = "./data") -> Tuple[
    torch.FloatTensor, torch.FloatTensor, Dict[str, List], Dict[str, List]]:
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0.0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def encode(self, inputs):
        return torch.sigmoid(self.g(inputs))

    def decode(self, encoded):
        return torch.sigmoid(self.h(encoded))

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # NOTE is there a typo here in the given code? It says to use sigmoid 
        # activations for f and g, but it's g and h in the code
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        out = decoded
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model: nn.Module, lr: float, lamb: float, train_data, zero_train_data, valid_data: Dict,
          num_epoch: int) -> None:
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TO/DO: Add a regularizer to the cost function.

    try:
        MODE
    except NameError:
        MODE = None

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    if MODE == "partD":
        training_costs = []
        valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        model.train()

        for user_id in range(num_student):  # Pretty much just batches
            optimizer.zero_grad()

            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            targets = inputs.clone()

            inputs = inputs.clone()

            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            # V1 (starter code)
            # nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            # V2
            nan_mask = torch.isnan(train_data[user_id].unsqueeze(0))

            targets[nan_mask] = output[nan_mask]

            reconstruction_loss = torch.sum((output - targets) ** 2.0)
            if MODE == "tuning_lamb":
                loss = reconstruction_loss + (lamb / 2) * model.get_weight_norm()
            else:
                loss = reconstruction_loss  # Ignore lambda
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        if MODE == "partD":
            training_costs.append(train_loss)
            valid_accs.append(valid_acc)

        logging.info("Epoch: %d\tTraining Cost: %.6f\tValid Acc: %.8f" % (epoch, train_loss, valid_acc))

    if MODE == "partD":
        import matplotlib.pyplot as plt
        # We need twin axes for this plot
        fig, ax1 = plt.subplots()

        # First plot the training cost
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Cost', color='tab:blue')
        ax1.plot(range(num_epoch), training_costs, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Then plot the validation accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Accuracy', color='tab:red')
        ax2.plot(range(num_epoch), valid_accs, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # fig.tight_layout()  # questionable: otherwise the right y-label is slightly clipped
        plt.title("Training Cost and Validation Accuracy over Epochs")
        plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


global MODE


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # What is the average number of questions that are answered by a user?
    avg = len(test_data["is_correct"])

    print(f"Average number of questions answered by a user: {avg}")

    #####################################################################
    # TO/DO:                                                            #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    try:
        MODE = "partD"
    except NameError:
        MODE = None

    logging.info(f"Mode: {MODE}")

    if MODE == "tuning_hyp":
        logging.info("Tuning hyperparameters")
        # Set model and optimization hyperparameters
        HYP_K_VALUES = [10, 50, 100, 200, 500]
        HYP_LEARNING_RATE = 0.005  # Usually 0.001, 0.01, 0.1
        HYP_LAMBDA = 0.0  # Try powers of 10, then move in single digits
        HYP_NUM_EPOCHS = 100  # Usually 10, 100, 1000
        HYP_MODEL = AutoEncoder  # The model class to use

        valid_accs = []
        for HYP_K in HYP_K_VALUES:
            logging.info("Training %s with k=%d, lr=%g, lamb=%g, epochs=%d" % (
                str(HYP_MODEL), HYP_K, HYP_LEARNING_RATE, HYP_LAMBDA, HYP_NUM_EPOCHS))
            time_start = time.time()
            model = HYP_MODEL(num_question=train_matrix.shape[1], k=HYP_K)
            train(model, HYP_LEARNING_RATE, HYP_LAMBDA, train_matrix, zero_train_matrix, valid_data, HYP_NUM_EPOCHS)

            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            time_end = time.time()
            logging.info("k=%d, over %d epochs in %.2f seconds. \tValid acc: %.16f" % (
                HYP_K, HYP_NUM_EPOCHS, time_end - time_start, valid_acc))
            valid_accs.append(valid_acc)

        k_star = HYP_K_VALUES[np.argmax(valid_accs)]
        logging.info("k* = %d with validation accuracy: %.16f" % (k_star, max(valid_accs)))

    elif MODE == "partD":
        logging.info("Training model for Part D.")
        # Begin Part D. We fix the hyperparameters k, LR, and epochs.
        # We use the best k from the tuning phase, which was:
        k = 20
        lr = 0.01
        lamb = 0.0
        epochs = 75
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

        logging.info("After training, program will show a plot of training cost and validation accuracy over epochs.")
        logging.info("Training %s with k=%d, lr=%g, lamb=%g, epochs=%d" % (str(type(model)), k, lr, lamb, epochs))
        time_start = time.time()

        # Enter the train function with MODE = "partD". 
        # Function will generate plot itself, no need to return anything
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epochs)

        test_acc = evaluate(model, zero_train_matrix, test_data)
        time_end = time.time()
        logging.info("k=%d, over %d epochs in %.2f seconds. \tFinal test acc: %.16f" % (
            k, epochs, time_end - time_start, test_acc))
        logging.info("Timing unreliable: included time used to plot, as well as for user to inspect plot.")

    elif MODE == "tuning_lamb":
        logging.info("Tuning lambda")
        # Begin part E. We fix the hyperparameters k, LR, and epochs, activate regularized loss, and tune lambda.
        k = 50
        lr = 0.005
        lambs = [0.0, 0.001, 0.01, 0.1, 1.0]
        epochs = 75
        model_class = AutoEncoder

        for lamb in lambs:
            logging.info(
                "Training %s with k=%d, lr=%g, lamb=%g, epochs=%d" % (str(type(model_class)), k, lr, lamb, epochs))
            time_start = time.time()

            model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
            train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, epochs)

            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            test_acc = evaluate(model, zero_train_matrix, test_data)
            time_end = time.time()

            logging.info("k=%d, lamb=%g, over %d epochs in %.2f seconds. \tValid acc: %.16f, Final test acc: %.16f" % (
                k, lamb, epochs, time_end - time_start, valid_acc, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    # CC logging code. It's too tiring watching the logs in terminal.
    ENABLE_LOGGING = True

    MODE = "partD"  # "tuning_hyp", "partD", "tuning_lamb"


    def logger_setup():
        if not ENABLE_LOGGING:
            logging.getLogger().addHandler(logging.NullHandler())
            return

        import os
        import inspect
        import datetime

        base_stack_depth = len(inspect.stack())

        class IndentFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.datetime.fromtimestamp(record.created)
                if datefmt:
                    s = dt.strftime(datefmt)
                else:
                    t = dt.strftime("%Y-%m-%d %H:%M:%S")
                    s = f"{t}.{int(record.msecs):03d}"
                return s

            def format(self, record):
                current_stack = inspect.stack()
                relevant_stack = [frame for frame in current_stack if
                                  frame.function not in {'wrapper', '<module>', 'logging_setup'}]
                indent_level = len(relevant_stack) - base_stack_depth - 1
                indent = ' ' * indent_level
                record.indent_message = f"{indent}{record.getMessage()}"
                return super().format(record)

        formatter = IndentFormatter('%(asctime)s %(levelname)s\t[%(filename)s:%(lineno)d]\t|%(indent_message)s')
        file_handler = logging.FileHandler(f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, stream_handler],
        )

        logging.info("Logging started")


    logger_setup()

    logging.debug("main()")
    main()

    logging.info("Logging finished")
