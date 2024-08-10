import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans


if __name__ == '__main__':
    question_meta = pd.read_csv('data/question_meta.csv')

    question_meta['subject_id'] = question_meta['subject_id'].apply(eval)

    mlb = MultiLabelBinarizer()
    question_subject_matrix = mlb.fit_transform(question_meta['subject_id'])

    num_clusters = 10

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    question_clusters = kmeans.fit_predict(question_subject_matrix)

    # Convert these to one-hot vectors
    question_clusters_oh = np.eye(num_clusters)[question_clusters]

    np.save('data/clustered_questions.npy', question_clusters_oh)
