import torch
from torch import Tensor

from knn import *
from neural_network import *
from sklearn.neighbors import NearestNeighbors
import numpy as np


def predict(users_embedded: np.array, users: Tensor, zeroed_users: Tensor, encoder: AutoEncoder) -> Tensor:
    """
    This function will complete the user tensor with predictions for question correctness
    """

    out = users.clone()

    for question_index in range(users.shape[1]):
        print(f"Predicting question {question_index} out of {users.shape[1]}")
        # Find the users that have answered the question
        answered_mask = ~torch.isnan(users[:, question_index])
        answered_users_embedded = users_embedded[answered_mask]
        unanswered_users_embedded = users_embedded[~answered_mask]
        answered_users = users[answered_mask]
        knn = NearestNeighbors(n_neighbors=min(5, len(answered_users_embedded)))
        knn.fit(answered_users_embedded)
        distances, indices = knn.kneighbors(unanswered_users_embedded)
        non_zero_unanswered = (~answered_mask).nonzero()
        for i in range(len(indices)):
            user_index = non_zero_unanswered[i].item()
            # We will take the weighted average of the k nearest neighbors
            k_nearest = answered_users[indices[i], question_index]
            weights = 1 / distances[i]
            weights /= weights.sum()
            # k_nearest = k_nearest * torch.from_numpy(weights)
            average = torch.sum(k_nearest, dim=0).item()
            out[user_index, question_index] = average

    return out


if __name__ == '__main__':
    # We will load the data
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    print("Data loaded")

    # We will train the autoencoder
    k_list = [10, 15, 20, 25, 50, 75, 100]
    accuracies = []
    for k in k_list:
        encoder = AutoEncoder(train_matrix.shape[1], 25)
        print("Autoencoder created, training...")
        train(encoder, 0.02, 0.0, train_matrix, zero_train_matrix, valid_data, 50)
        # encoder.load_state_dict(torch.load("encoder.pth"))

        print("Autoencoder trained")

        torch.save(encoder.state_dict(), "encoder.pth")

        # We will encode the users
        embedded_users = encoder.encode(zero_train_matrix).detach().numpy()

        # We will predict the missing values
        predicted = predict(embedded_users, train_matrix, zero_train_matrix, encoder)

        # We will evaluate the model
        acc = sparse_matrix_evaluate(valid_data, predicted)

        print(f"Accuracy: {acc}")
        accuracies.append(acc)

    import matplotlib.pyplot as plt
    plt.plot(k_list, accuracies)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
