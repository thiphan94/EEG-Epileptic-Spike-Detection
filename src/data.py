import numpy as np
# import torch
# from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def read_dataset():
    datasets_dict = []
    
    file_name = '../data/' 

    train_positive_ex1 = np.fromfile(file_name + 'exp1-train-400pos.bin', dtype=np.float32).reshape([400,5,768])
    train_negative_ex1 = np.fromfile(file_name + 'exp1-train-2000neg.bin', dtype=np.float32).reshape([2000,5,768])
    train_positive_ex2 = np.fromfile(file_name + 'exp2-train-400pos.bin', dtype=np.float32).reshape([400,5,768])
    train_negative_ex2 = np.fromfile(file_name + 'exp2-train-2000neg.bin', dtype=np.float32).reshape([2000,5,768])
    
    # Concatenate data for both experiments and classes
    X = np.concatenate([train_positive_ex1, train_negative_ex1, train_positive_ex2, train_negative_ex2], axis=0)

    # Create target labels: positive (1) and negative (0)
    y_positive_ex1 = np.ones(400)
    y_negative_ex1 = np.zeros(2000)
    y_positive_ex2 = np.ones(400)
    y_negative_ex2 = np.zeros(2000)

    y = np.concatenate([y_positive_ex1, y_negative_ex1, y_positive_ex2, y_negative_ex2])


    datasets_dict.extend([X.copy(), y.copy()])
    return datasets_dict
    
def normalize(values, axis=-1, mu=0, sigma=1, epsilon=1e-6):
    assert len(values.shape) == 3
    
    normalized_values = []
    
    for sample in values:
        local_mu = np.mean(sample, axis=axis)
        local_sigma = np.sqrt(np.var(sample, axis=axis))
        
        # Handle case when local_sigma is close to zero
        normalized_sample = np.where(local_sigma <= epsilon, mu, mu + sigma * (sample - local_mu) / local_sigma)
        normalized_values.append(normalized_sample)
    
    return np.array(normalized_values)

def normalize_data(train_data, test_data=None):
    # Reshape train data to 2D array for scaling
    num_samples_train, num_steps_train, num_features_train = train_data.shape
    train_data_2d = train_data.reshape(num_samples_train * num_steps_train, num_features_train)
    
    # Initialize the MinMaxScaler and fit_transform on train data
    scaler = MinMaxScaler()
    normalized_train_data = scaler.fit_transform(train_data_2d)
    
    # Reshape train data back to the original shape
    normalized_train_data_3d = normalized_train_data.reshape(num_samples_train, num_steps_train, num_features_train)
    
    # If test data is provided, transform using the trained scaler
    if test_data is not None:
        num_samples_test, num_steps_test, num_features_test = test_data.shape
        test_data_2d = test_data.reshape(num_samples_test * num_steps_test, num_features_test)
        normalized_test_data = scaler.transform(test_data_2d)
        normalized_test_data_3d = normalized_test_data.reshape(num_samples_test, num_steps_test, num_features_test)
        return normalized_train_data_3d, normalized_test_data_3d
    
    return normalized_train_data_3d
