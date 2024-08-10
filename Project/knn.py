import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data: Dict[str, List], k: int) -> float:
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))  # CC removed
    return acc


def knn_impute_by_item(matrix, valid_data: Dict[str, List], k: int) -> float:
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TO/DO:                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main() -> None:
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TO/DO:                                                            #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_val = [1, 6, 11, 16, 21, 26]
    stu_acc = []
    ques_acc = []
    for k in k_val:
        student_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        stu_acc.append(student_acc)
        question_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        ques_acc.append(question_acc)
    
    max_char_length = max(len(str(k)) for k in k_val)
    for i in range(len(stu_acc)):
        print(f"Valid acc of user with k = {k_val[i]:{max_char_length}}: {stu_acc[i]}")
    for i in range(len(ques_acc)):
        print(f"Valid acc of item with k = {k_val[i]:{max_char_length}}: {ques_acc[i]}")
    
    plt.plot(np.array(k_val), np.array(stu_acc), label='User')
    plt.plot(np.array(k_val), np.array(ques_acc), label='Item')
    plt.title("Validation Accuracy per k-value")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("Validation Accuracy.jpg")
    plt.show()
    
    k_star_user = k_val[np.argmax(stu_acc)]
    k_star_item = k_val[np.argmax(ques_acc)]
    print("User's best k* is", k_star_user, "with valid acc", max(stu_acc))
    print("Item's best k* is", k_star_item, "with valid acc", max(ques_acc))
    
    student_acc_test = knn_impute_by_user(sparse_matrix, test_data, k_star_user)
    question_acc_test = knn_impute_by_item(sparse_matrix, test_data, k_star_item)
    
    print(f"Test acc of chosen user k* = {k_star_user}: {student_acc_test}")
    print(f"Test acc of chosen item k* = {k_star_item}: {question_acc_test}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
