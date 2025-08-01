import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model, save_model
from keras.layers import Input, Conv1D, Dropout, Conv1DTranspose

# --- Helper Functions ---
# Function to calculate rolling minimum starting from the current row
def rolling_min(group, size, offset=2):
    result = []
    for i in range(len(group)):
        # Define the rolling window based on the current row
        start_idx = min(i + offset, len(group) - 1)
        end_idx = min(start_idx + size, len(group))
        # Calculate the minimum within the rolling window
        min_value = group.iloc[start_idx:end_idx]["win_obstacle-distance"].min()
        result.append(min_value)
    return result


# Function to calculate rolling average for a pandas Series
def rolling_ave_series(series, size):
    result = []
    for i in range(len(series)):
        # Define the rolling window based on the current row
        end_idx = i
        start_idx = max(i - size, 0)
        # Calculate the mean within the rolling window
        mean_value = series.iloc[start_idx:end_idx].mean()
        result.append(mean_value)
    return result


# Function to calculate rolling minimum up to the current row
def rolling_ave(group, size, col):
    result = []
    for i in range(len(group)):
        # Define the rolling window based on the current row
        end_idx = i
        start_idx = max(i - size, 0)
        # Calculate the minimum within the rolling window
        mean_value = group.iloc[start_idx:end_idx][col].mean()
        result.append(mean_value)
    return result


# Custom sigmoid transformation function
def sigmoid(distance):
    a = 0.5  
    b = 3  
    limit = 15  

    return limit / (1 + np.exp(-a * (distance - b)))


# Custom inverse sigmoid transformation function
def inverse_sigmoid(transformed_value):
    a = 0.5  
    b = 3  
    limit = 15  
    distance = b - np.log(limit / (transformed_value) - 1) / a
    return distance


def handle_rotation(headings, threshold=np.pi):
    for i in range(1, len(headings)):
        diff = headings[i] - headings[i - 1]
        if diff > threshold:
            headings[i] -= 2 * np.pi
        elif diff < -threshold:
            headings[i] += 2 * np.pi
    return headings


class DTModel:
    WINSIZE = 25

    def __init__(self):
        self.model = None

    @classmethod
    def extract_and_process_data(cls, file_path):
        df = pd.read_csv(file_path)
        df["r"] = df["r"].astype(str).apply(lambda x: handle_rotation([float(val) for val in x.split(",")]))
        df["x"] = df["x"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        df["y"] = df["y"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        df["z"] = df["z"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        df["r_zero"] = df["r"].apply(lambda x: [val - np.mean(x) for val in x])
        df["x_zero"] = df["x"].apply(lambda x: [val - np.mean(x) for val in x])
        df["y_zero"] = df["y"].apply(lambda x: [val - np.mean(x) for val in x])
        df["z_zero"] = df["z"].apply(lambda x: [val - np.mean(x) for val in x])
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)

        # FIX 2: Added include_groups=False to silence Pandas FutureWarnings
        df["win_dist_.5"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, offset=1, include_groups=False).explode().tolist()
        df["win_dist_1"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, include_groups=False).explode().tolist()
        df["win_dist_2"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, offset=4, include_groups=False).explode().tolist()
        df["win_dist_3"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, offset=6, include_groups=False).explode().tolist()
        df["win_dist_4"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, offset=8, include_groups=False).explode().tolist()
        df["win_dist_5"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=1, offset=10, include_groups=False).explode().tolist()
        df["win_dist_1_10"] = df.groupby(["log_folder", "log_name"], group_keys=False).apply(rolling_min, size=21, offset=0, include_groups=False).explode().tolist()

        df["win_dist_0"] = df["win_obstacle-distance"]
        df["win_dist_1_2"] = df[["win_dist_1", "win_dist_2"]].min(axis=1)
        df["win_dist_1_3"] = df[["win_dist_1_2", "win_dist_3"]].min(axis=1)
        df["win_dist_1_4"] = df[["win_dist_1_3", "win_dist_4"]].min(axis=1)
        df["win_dist_1_5"] = df[["win_dist_1_4", "win_dist_5"]].min(axis=1)
        df["win_dist_0_10"] = df[["win_dist_0", "win_dist_1_10"]].min(axis=1)
        
        sig_cols = ["win_dist_0", "win_dist_.5", "win_dist_1", "win_dist_2", "win_dist_3", "win_dist_4", "win_dist_5", "win_dist_1_2", "win_dist_1_3", "win_dist_1_4", "win_dist_1_5", "win_dist_1_10", "win_dist_0_10"]
        for col in sig_cols:
            df[f"{col}_sig"] = df[col].apply(sigmoid)

        df = df[df["r"].apply(lambda x: len(x) == cls.WINSIZE)]
        return df


    def build_model(self, n_input_features=1):
        """Builds the autoencoder using the best-tuned architecture."""
        print("--- Building Best-Tuned Autoencoder Model ---")
        model = Sequential()
        model.add(Input(shape=(self.WINSIZE, n_input_features)))

        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(Dropout(rate=0.1))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))

        model.add(Conv1DTranspose(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(Dropout(rate=0.2))
        model.add(Conv1DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"))
        
        model.add(Conv1DTranspose(filters=n_input_features, kernel_size=3, padding="same"))
        
        model.compile(optimizer="adam", loss="mse")
        self.model = model
        return model

    def train(self, df, inputs, epochs=50):
        input_values = [np.array(df[col].to_list()) for col in inputs]
        x_train = np.dstack(input_values)
        self.model.fit(x_train, x_train, epochs=epochs, verbose=1, batch_size=128)

    def save(self, path):
        save_model(self.model, path)
    
    def load(self, path):
        self.model = load_model(path)

    def get_anomaly_scores(self, df, inputs):
        input_values = [np.array(df[col].to_list()) for col in inputs]
        x_test = np.dstack(input_values)
        
        reconstructions = self.model.predict(x_test, verbose=0)
        loss = tf.keras.losses.mae(reconstructions, x_test)
        mean_loss = tf.reduce_mean(loss, axis=1)
        
        df["mean_loss"] = -mean_loss # Store as negative
        
        # FIX: Use a lambda function to handle the grouped Series correctly
        df["mean_loss_4"] = (
            df.groupby(["log_folder", "log_name"])["mean_loss"]
            .apply(lambda group: rolling_ave_series(group, size=4))
            .explode()
            .tolist()
        )
        return df