from typing import Any, Union, Optional, Dict, List
import logging
from functools import wraps

import os
import datetime as dt

import csv
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch_directml


if __name__ == "__main__":
    ENABLE_LOGGING = True
    LOG_FUNCTION_ARGS = False
    DATA_ROOT_DIR = "./data"
    DEVICE = 'directml'


# CC TODO list:
# - Consider imputing missing values in question_meta.csv with kNN or SVD. 
#   This will allow us to use the auxilary columns to help predict.

# CC NOTE list:
# The current focus is on trying to integrate metadata into the model in 
# hope that the model can learn which students are good at which subjects. 
# More specifically, we will be taking advantage of the subjects 
# corresponding to eaech question. I believe this is an appropriate 
# direction to take for the second part of the project.

# Before we go into NN training, maybe we can preproceess question metadata 
# in such a way that we can construct a new training dataset such that:

# AN IDEA:
# We preprocess and build a dictionary that tracks subject skills.
# - Initialize all students to have zero skill stats in all subjects.
# - For each row in the training data:
#   - If the student answered the question correctly:
#     - Somewhere, we add 1 to the student's skill stats for every skill 
#       corresponding to the question.

# Students can be considered stronger in certain subjects if we see 
# they have a higher skill stat in that subject (eg. answered a Math 
# question correctly three times. Math Stat = 3).

# Then, we can use the skill stats as features to predict the correctness 
# for questions that are related to subjects.

# PROBLEM: High cardinality of skill_id. After preprocessing, it would be 
# infeasible for a neural network to use this information, since there are 
# 388 subjects. We need to reduce the dimension, either by an embedding 
# layer into a dense vector, or by clustering the subjects into a smaller 
# number of groups, or by some other method.

# If we use an embedding layer, we can use the skill stats as features.
# If we use clustering, we can use the cluster as a feature.
# - Clustering is a good idea because it can group students who are good at 
#   similar subjects together. This can help the model learn the
#   relationships between subjects and students, instead of the naive
#   approach of clustering students based on which questions they answered. 
# - Clustering can also be used to cluster different subjects together, 
#   which can help the model learn the relationships between subjects.
# - Example: If a student is good at the following subjects:
#   subject_id,name
#           73,Block Graphs and Bar Charts
#           74,Histogram
#           75,Pictogram
#           76,Pie Chart
#   Then we can infer that the student is good at "Data Representation".

# NOTE that I'm currently just running naive MLPs on the data (no clustering)




def log_function_call(func):    # Better than individualized logging statements inside
    global LOG_FUNCTION_ARGS
    try:
        LOG_FUNCTION_ARGS
    except NameError:
        LOG_FUNCTION_ARGS = True
    if not LOG_FUNCTION_ARGS:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.debug("Called %s" % (func.__name__))
            return func(*args, **kwargs)
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.debug("Called %s%s" % (func.__name__, args))
            return func(*args, **kwargs)
        return wrapper

@log_function_call
def csv_to_df(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_ROOT_DIR, filename))

@log_function_call
def preprocess_question_metadata(df_question_metadata: pd.DataFrame) -> pd.DataFrame:
    df_question_metadata = df_question_metadata.sort_values(by=["question_id", "subject_id"])   # fundamentally makes no difference but makes it easier to read.
    df_question_metadata = df_question_metadata.reset_index(drop=True)
    return df_question_metadata

@log_function_call
def preprocess_student_metadata(df_student_metadata: pd.DataFrame) -> pd.DataFrame:
    def calculate_age(dob: str) -> Optional[int]:
        if pd.isnull(dob):
            return None
        dob_datetime = dt.datetime.strptime(dob, '%Y-%m-%d %H:%M:%S.%f').date()
        today = dt.datetime.today()
        return today.year - dob_datetime.year - ((today.month, today.day) < (dob_datetime.month, dob_datetime.day))
    
    df_student_metadata['age'] = df_student_metadata['date_of_birth'].apply(calculate_age).astype('Int64')  # Int64 for int-NaN compatibility. Remember to use pd.isnull().
    df_student_metadata = df_student_metadata.drop(columns=['date_of_birth'])
    
    df_student_metadata['gender'] = df_student_metadata['gender'].replace({0: pd.NA}).astype('Int64')
    
    df_student_metadata['premium_pupil'] = df_student_metadata['premium_pupil'].astype('Int64')
    
    return df_student_metadata

@log_function_call
def preprocess_subject_metadata(df_subject_metadata: pd.DataFrame) -> pd.DataFrame:
    # Currently no preprocessing needed. xdx
    return df_subject_metadata

@log_function_call
def build_skills(df_train_data: pd.DataFrame, df_question_metadata: pd.DataFrame, df_subject_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a custom training dataset that includes metadata.
    """
    unique_user_ids = df_train_data["user_id"].drop_duplicates().sort_values()
    unique_user_ids = unique_user_ids.reset_index(drop=True)
    unique_subject_ids = df_subject_metadata["subject_id"]
    df_skills = pd.DataFrame(0, index=unique_user_ids, columns=unique_subject_ids)
    
    questions_to_subjects = df_question_metadata.set_index("question_id")["subject_id"].apply(eval).to_dict()
    for _, row in df_train_data[df_train_data['is_correct'] == 1].iterrows():
        user_id = row["user_id"]
        question_id = row["question_id"]
        subjects = questions_to_subjects.get(question_id, [])
        df_skills.loc[user_id, subjects] += 1
    # Since every question has at least "Math" as a subject, we can use that column as a measure of how many questions the student has answered.
    
    return df_skills
    


# class MetaTrainDataset(Dataset):
#     def __init__(self, df_metatrain_data: pd.DataFrame):
#         self.df_metatrain_data = df_metatrain_data
#         self.user_ids = df_metatrain_data.index
#         self.subject_ids = df_metatrain_data.columns
    
#     def __len__(self):
#         return len(self.user_ids)
    
#     def __getitem__(self, idx):
#         user_id = self.user_ids[idx]
#         user_data = torch.tensor(self.df_metatrain_data.loc[user_id].values, dtype=torch.float32)
#         return user_data

class MetaTrainDataset(Dataset):
    def __init__(self, df_features: pd.DataFrame, df_labels: pd.Series):
        self.features = torch.tensor(df_features.values, dtype=torch.float32)
        self.labels = torch.tensor(df_labels.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MetaNN(torch.nn.Module):
    def __init__(self, num_subjects: int):  # num_subjects could change on different data or on clustering. For Naive MLP it's num_subjects = 388
        super(MetaNN, self).__init__()
        self.fc1 = torch.nn.Linear(num_subjects, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.sigmoid(self.fc3(x))
        return x




def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, num_epochs: int, device: str = 'cpu'):   # TODO
    """
    Learning rate is implicitly set in param optimizer
    criterion is loss function. In-module is a _Loss class which is a subclass of Module
    """
    df = None # TODO
    model = MetaNN(num_subjects=df.shape[1])
    
    
    for epoch in range(num_epochs): # TOO ad batching
        training_loss = 0.0
        
        model.train()
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            inputs.to(device)
            targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()   # try loss.item()
        training_loss /= len(train_dataloader)
    
    # This is stupid, what have I done I cant make features and labels out of the skillset table????????????????????

def evaluate():
    pass





def main() -> None:
    df_train_data = csv_to_df("train_data.csv")
    df_valid_data = csv_to_df("valid_data.csv")
    df_subject_meta = csv_to_df("subject_meta.csv")
    df_student_meta = csv_to_df("student_meta.csv")
    df_question_meta = csv_to_df("question_meta.csv")
    
    df_subject_meta = preprocess_subject_metadata(df_subject_meta)
    # df_student_meta = preprocess_student_metadata(df_student_meta)
    df_question_meta = preprocess_question_metadata(df_question_meta)
    
    df_skills = build_skills(df_train_data, df_question_meta, df_subject_meta)
    # logging.debug(f"Logging df_metatrain_data.to_string():\n{df_metatrain_data.to_string()}")


    # TODO unfinished
    skills_dataset_train = MetaTrainDataset()
    skills_dataset_valid = MetaTrainDataset()
    skills_dataset_test = MetaTrainDataset()
    train_dataloader = DataLoader(skills_dataset_train, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(skills_dataset_valid, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(skills_dataset_test, batch_size=32, shuffle=False)
    
    model = MetaNN(num_subjects=df_skills.shape[1])
    model.to(torch_directml.device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()  # Since the output is binary

    train(model, optimizer, criterion, train_dataloader, valid_dataloader, 100, device=DEVICE)

    



if __name__ == "__main__":
    
    # CC logging code. It's too tiring watching the logs in terminal.
    def logger_setup():
        global ENABLE_LOGGING
        try:
            ENABLE_LOGGING
        except NameError:
            ENABLE_LOGGING = True
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
                relevant_stack = [frame for frame in current_stack if frame.function not in {'wrapper', '<module>', 'logging_setup'}]
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
    logger_setup()
    logging.info("Logging started")
    
    logging.debug("main()")
    main()

    logging.info("Logging finished")
    
    



