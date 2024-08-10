import numpy as np
import torch

from ProjectDefault import neural_network
from neural_network import load_data, AutoEncoder


def predict(users: np.array, user: int, cov: np.array, question_index: int) -> bool:
    # Extract the column corresponding to the question
    question_relations = cov[:, question_index].copy()

    # Mask by the questions the user has answered
    answered_mask = ~np.isnan(users[user])  # This is a boolean mask representing the questions the user has answered

    # Zero out the unanswered questions in question_relations
    question_relations[~answered_mask] = 0

    answers = users[user].copy()
    wrong_answers_mask = answers == 0
    answers[wrong_answers_mask] = -1
    answers[~answered_mask] = 0

    # Dot these two together
    dot_product = np.dot(question_relations, answers)

    return dot_product > 0.0


def eval_cov(users: np.array, cov: np.array, test: dict[str, list]) -> float:
    accuracy = 0

    for i, u in enumerate(test["user_id"]):
        question_id = test["question_id"][i]
        prediction = predict(users, u, cov, question_id)
        target = bool(test["is_correct"][i])
        accuracy += int(prediction == target)

    return accuracy / len(test["user_id"])


if __name__ == '__main__':
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Calculate the covariance matrix of train_matrix
    train_matrix_np = train_matrix.numpy()
    data_imputed = np.where(np.isnan(train_matrix_np), np.nanmean(train_matrix_np, axis=0), train_matrix_np)
    data_centered = data_imputed - np.nanmean(data_imputed, axis=0)
    cov_matrix = np.cov(data_centered, rowvar=False)

    acc = eval_cov(train_matrix_np, cov_matrix, test_data)
    print(acc)

    # Train an autoencoder
    encoder = AutoEncoder(train_matrix.shape[1], 15)
    neural_network.train(encoder, 0.015, 0.0, train_matrix, zero_train_matrix, valid_data, 100)

    # Fill in the matrix with the cov predictions
    out = train_matrix.numpy().copy()

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if np.isnan(out[i, j]):
                out[i, j] = predict(train_matrix_np, i, cov_matrix, j)

    acc = neural_network.evaluate(encoder, torch.from_numpy(out), valid_data)

    print(acc)

