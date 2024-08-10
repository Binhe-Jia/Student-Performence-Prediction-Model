import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from torch.optim.lr_scheduler import StepLR

if __name__ == '__main__':
    # Load the datasets
    train_data = pd.read_csv('data/train_data.csv')
    valid_data = pd.read_csv('data/valid_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    student_metadata = pd.read_csv('data/student_meta.csv')

    # Create student meta by student id
    student_meta_dict = {}
    for idx, user_id in enumerate(student_metadata['user_id']):
        gender = float(student_metadata['gender'][idx])
        birthdate = student_metadata['date_of_birth'][idx]
        premium = float(student_metadata['premium_pupil'][idx])

        if np.isnan(premium):
            premium = 0.0

        # Calculate age
        age = 0
        if isinstance(birthdate, str):
            birthdate = pd.to_datetime(birthdate)
            birth_year = birthdate.year
            age = 2024 - birth_year
            age /= 100  # Normalize

        student_meta_dict[user_id] = torch.tensor([gender, age, premium])

    # Load the sparse training matrix
    train_sparse_matrix = load_npz('data/train_sparse.npz')

    # Convert the sparse matrix to a dense format
    train_dense_matrix = train_sparse_matrix.toarray()

    train_nan_mask = torch.isnan(torch.FloatTensor(train_dense_matrix))

    train_dense_matrix = torch.FloatTensor(train_dense_matrix)
    train_dense_matrix[train_nan_mask] = 0.0

    question_embeddings = np.load("data/clustered_subjects.npy")
    question_embeddings_tensor = torch.tensor(question_embeddings, dtype=torch.float)


    class AutoEncoder(nn.Module):
        def __init__(self, num_questions, k=100):
            super(AutoEncoder, self).__init__()

            # Autoencoder layers
            input_dim = num_questions + 10 + 3  # Original inputs + student embedding + student meta
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, k),
                nn.Sigmoid(),
                nn.Linear(k, k),
                nn.Sigmoid(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(k, k),
                nn.Sigmoid(),
                nn.Linear(k, num_questions),
                nn.Sigmoid(),
            )

        def forward(self, inputs, student_embedding, student_meta):
            combined_inputs = torch.cat([inputs, student_embedding.unsqueeze(0), student_meta.unsqueeze(0)], dim=1)
            encoded = self.encoder(combined_inputs)
            decoded = self.decoder(encoded)
            return decoded

        def regularization_loss(self):
            return torch.sum(self.encoder[0].weight ** 2.0) + torch.sum(self.decoder[0].weight ** 2.0)


    def get_student_embedding(user_answers):
        # Calculate the average embedding of the questions the student has answered correctly
        correct_mask = user_answers == 1
        correct_embeddings = question_embeddings_tensor[correct_mask]
        if correct_embeddings.size(0) > 0:
            return torch.mean(correct_embeddings, dim=0)
        else:
            return torch.zeros(20)

    # Function to evaluate the model
    def evaluate(model, data, valid_data):
        model.eval()
        total = 0
        correct = 0

        for i, u in enumerate(valid_data["user_id"]):
            inputs = data[u].unsqueeze(0)
            output = model(inputs, get_student_embedding(inputs.squeeze()), student_meta_dict[u])
            guess = output[0][valid_data["question_id"][i]].item() >= 0.5
            if guess == valid_data["is_correct"][i]:
                correct += 1
            total += 1
        return correct / float(total)


    # Convert validation and test data to dictionaries
    valid_data_dict = {
        "user_id": valid_data['user_id'].tolist(),
        "question_id": valid_data['question_id'].tolist(),
        "is_correct": valid_data['is_correct'].tolist()
    }

    test_data_dict = {
        "user_id": test_data['user_id'].tolist(),
        "question_id": test_data['question_id'].tolist(),
        "is_correct": test_data['is_correct'].tolist()
    }

    train_data_dict = {
        "user_id": train_data['user_id'].tolist()[0:1000],
        "question_id": train_data['question_id'].tolist()[0:1000],
        "is_correct": train_data['is_correct'].tolist()[0:1000]
    }

    k = 16
    model = AutoEncoder(num_questions=train_dense_matrix.shape[1], k=k)

    num_epoch = 250
    lamb = 0.001

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    accuracies = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for user_id in range(train_dense_matrix.shape[0]):
            inputs = train_dense_matrix[user_id].unsqueeze(0)
            targets = torch.FloatTensor(train_sparse_matrix[user_id].toarray().squeeze())

            # Forward pass
            output = model(inputs, get_student_embedding(inputs.squeeze()), student_meta_dict[user_id])

            # Set targets at NaN positions to the corresponding output values
            nan_mask = torch.isnan(targets)
            targets[nan_mask] = output[0][nan_mask]

            # Compute the loss only for non-NaN values
            loss = torch.sum(((output - targets.unsqueeze(0)) ** 2.0)) + lamb * model.regularization_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0:
            epoch_acc = evaluate(model, train_dense_matrix, valid_data_dict)
            accuracies.append(epoch_acc)
            epoch_acc_test = evaluate(model, train_dense_matrix, test_data_dict)
            test_accuracies.append(epoch_acc_test)
            epoch_acc_train = evaluate(model, train_dense_matrix, train_data_dict)
            train_accuracies.append(epoch_acc_train)
            print(f'Epoch {epoch}, Loss: {train_loss}, Test Accuracy: {epoch_acc_test}, Train Accuracy: {epoch_acc_train}')

    valid_acc = evaluate(model, train_dense_matrix, valid_data_dict)
    test_acc = evaluate(model, train_dense_matrix, test_data_dict)
    train_acc = evaluate(model, train_dense_matrix, train_data_dict)

    print(f'validation accuracy: {valid_acc}')
    print(f'test accuracy: {test_acc}')
    print(f'train accuracy: {train_acc}')

    # plot
    import matplotlib.pyplot as plt

    plt.plot(accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.show()
