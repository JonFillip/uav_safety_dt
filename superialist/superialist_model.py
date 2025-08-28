from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D, BatchNormalization
from keras.layers import MaxPooling1D, Dropout, Conv1DTranspose
from keras.models import load_model, save_model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch


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
    a = 0.5  # Controls the steepness of the curve
    b = 3  # Shifts the curve to the right (transition occurs around 3)
    limit = 15  # Upper limit for the transformed values

    return limit / (1 + np.exp(-a * (distance - b)))


# Custom inverse sigmoid transformation function
def inverse_sigmoid(transformed_value):
    a = 0.5  # Controls the steepness of the curve
    b = 3  # Shifts the curve to the right (transition occurs around 3)
    limit = 15  # Upper limit for the transformed values
    # Solve for the initial distance using the inverse of the sigmoid
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


class CNNModel(object):
    WINSIZE = 25
    # WINSIZE = 50

    def __init__(self):
        pass

    @classmethod
    def extract_dataset(cls, file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Split the comma-separated strings in the 'x' column and convert them to arrays of floats
        df["r"] = (
            df["r"]
            .astype(str)
            .apply(lambda x: handle_rotation([float(val) for val in x.split(",")]))
        )
        df["x"] = (
            df["x"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        )
        df["y"] = (
            df["y"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        )
        df["z"] = (
            df["z"].astype(str).apply(lambda x: [float(val) for val in x.split(",")])
        )
        # zero center r
        df["r_zero"] = df["r"].apply(lambda x: [val - sum(x) / len(x) for val in x])
        df["x_zero"] = df["x"].apply(lambda x: [val - sum(x) / len(x) for val in x])
        df["y_zero"] = df["y"].apply(lambda x: [val - sum(x) / len(x) for val in x])
        df["z_zero"] = df["z"].apply(lambda x: [val - sum(x) / len(x) for val in x])

        df = df.dropna()
        # Reset the index to retain the original row order
        df.reset_index(drop=True, inplace=True)

        # minimum distance in specific future windows
        df["win_dist_0"] = df["win_obstacle-distance"]
        # Apply the custom rolling function within each group
        df["win_dist_.5"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, offset=1, include_groups=False)
            .explode()
            .tolist()
        )
        df["win_dist_1"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, include_groups=False)
            .explode()
            .tolist()
        )
        df["win_dist_2"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, offset=4, include_groups=False)
            .explode()
            .tolist()
        )
        df["win_dist_3"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, offset=6, include_groups=False)
            .explode()
            .tolist()
        )
        df["win_dist_4"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, offset=8, include_groups=False)
            .explode()
            .tolist()
        )
        df["win_dist_5"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=1, offset=10, include_groups=False)
            .explode()
            .tolist()
        )

        # minimum distance among n future windows
        df["win_dist_1_2"] = df[["win_dist_1", "win_dist_2"]].min(axis=1)
        df["win_dist_1_3"] = df[["win_dist_1_2", "win_dist_3"]].min(axis=1)
        df["win_dist_1_4"] = df[["win_dist_1_3", "win_dist_4"]].min(axis=1)
        df["win_dist_1_5"] = df[["win_dist_1_4", "win_dist_5"]].min(axis=1)

        df["win_dist_1_10"] = (
            df.groupby(["log_folder", "log_name"])
            .apply(rolling_min, size=21, offset=0)
            .explode()
            .tolist()
        )

        df["win_dist_0_10"] = df[["win_dist_0", "win_dist_1_10"]].min(axis=1)

        # transforming all above distances using sigmoid function
        df["win_dist_0_sig"] = df["win_dist_0"].apply(sigmoid)
        df["win_dist_.5_sig"] = df["win_dist_.5"].apply(sigmoid)
        df["win_dist_1_sig"] = df["win_dist_1"].apply(sigmoid)
        df["win_dist_2_sig"] = df["win_dist_2"].apply(sigmoid)
        df["win_dist_3_sig"] = df["win_dist_3"].apply(sigmoid)
        df["win_dist_4_sig"] = df["win_dist_4"].apply(sigmoid)
        df["win_dist_5_sig"] = df["win_dist_5"].apply(sigmoid)

        df["win_dist_1_2_sig"] = df["win_dist_1_2"].apply(sigmoid)
        df["win_dist_1_3_sig"] = df["win_dist_1_3"].apply(sigmoid)
        df["win_dist_1_4_sig"] = df["win_dist_1_4"].apply(sigmoid)
        df["win_dist_1_5_sig"] = df["win_dist_1_5"].apply(sigmoid)
        df["win_dist_1_10_sig"] = df["win_dist_1_10"].apply(sigmoid)
        df["win_dist_0_10_sig"] = df["win_dist_0_10"].apply(sigmoid)

        # Filter rows with exactly 25 columns in the 'x' value
        df = df[df["r"].apply(lambda x: len(x) == cls.WINSIZE)]
        # self.dataset = df
        return df

    @classmethod
    def split(cls, df, test_size: float = 0.2):
        # Split the data into training and testing sets
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)
        return df_train, df_test

    @classmethod
    def balance(cls, df, label="win_risky"):
        # Find the value counts for each class
        value_counts = df[label].value_counts()

        # Determine the minimum count among the classes
        min_count = min(value_counts)

        # Create a balanced DataFrame
        balanced_df = (
            df.groupby(label)
            .apply(lambda x: x.sample(n=min_count))
            .reset_index(drop=True)
        )

        # Shuffle the balanced DataFrame
        shuffled_balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        return shuffled_balanced_df

    @classmethod
    def get_classifier_model(cls, n_output_nodes=1):
        # define model
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation="relu",
                input_shape=(cls.WINSIZE, 1),
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(n_output_nodes, activation="sigmoid"))
        # model.compile(optimizer='adam', loss='mse')
        # model.compile(loss="mean_squared_error", optimizer="rmsprop")
        ## use the default values for batch_size, stateful

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.summary()
        return model

    # Define your model as a function
    @classmethod
    def get_parameterized_model(cls, hp):
        model = Sequential()
        inputs = 4
        outputs = 1
        # Choose the number of layers
        num_conv_layers = hp.Int("num_conv_layers", min_value=1, max_value=2)

        # Add Conv1D layers
        for i in range(num_conv_layers):
            model.add(
                Conv1D(
                    filters=hp.Int(
                        f"conv_filters_{i}", min_value=16, max_value=64, step=16
                    ),
                    kernel_size=hp.Int(f"conv_kernel_{i}", min_value=2, max_value=7),
                    activation=hp.Choice(
                        f"conv_activation_{i}", values=["relu", "tanh"]
                    ),
                    input_shape=(25, inputs),
                )
            )
            model.add(BatchNormalization())
            model.add(MaxPooling1D(hp.Int("pool_size", min_value=2, max_value=5)))

        model.add(Flatten())
        num_dense_layers = hp.Int("num_dense_layers", min_value=1, max_value=2)
        for i in range(num_dense_layers):
            model.add(
                Dense(
                    hp.Int(f"dense_units_{i}", min_value=16, max_value=128, step=16),
                    activation=hp.Choice(
                        f"dense_activation_{i}", values=["relu", "tanh"]
                    ),
                )
            )
        model.add(Dense(outputs))

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
            ),
            loss="mse",
        )
        return model

    @classmethod
    def search_hyperparameters(
        cls,
        train_df,
        test_df,
        inputs=["r"],
        outputs=["win_risky"],
        trials=10,
        epochs=100,
        project_name=None,
    ):
        # Initialize the tuner (RandomSearch in this case)
        tuner = RandomSearch(
            cls.get_parameterized_model,
            objective="val_loss",
            directory="hp_tuning",
            project_name=f'rand-{datetime.now().strftime("%d-%m-%H-%M-%S")}'
            if project_name is None
            else project_name,
            max_trials=trials,
            max_consecutive_failed_trials=10,
        )
        input_values = []
        for input in inputs:
            input_values.append(np.array(train_df[input].to_list()))
        x_train = np.dstack(input_values)
        y_train = train_df[outputs]
        # y_train = np.array(train_data[out_train])

        input_values2 = []
        for input in inputs:
            input_values2.append(np.array(test_df[input].to_list()))
        x_test = np.dstack(input_values2)
        y_test = test_df[outputs]

        # Search for the best hyperparameters and print results after each trial
        # for trial in range(trials):
        # Perform the search for one trial
        tuner.search(
            x_train,
            y_train,
            epochs=epochs,
            verbose=2,
            validation_split=0.2,
            use_multiprocessing=True,
        )  # Set verbose=0 to suppress output

        # Get the best hyperparameters of the current trial
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Print the hyperparameters for the current trial
        print("Hyperparameters for Trial")
        for key, value in best_hps.values.items():
            print(f"{key}: {value}")
        print()
        # Build the model with the best hyperparameters
        best_model = cls.get_parameterized_model(best_hps)

        # # Train the best model
        # best_model.fit(x_train, y_train, epochs=epochs, verbose=0)  # Set verbose=0 to suppress output

        # # Evaluate the best model on the test set
        # test_loss = best_model.evaluate(x_test, y_test, verbose=0)

        # print(f"Test Loss for Trial {trial + 1}: {test_loss}\n")

        tuner.results_summary()
        print("Hyperparameter tuning complete.")
        return tuner

    @classmethod
    def get_autoencoder_model(cls, n_input_features=1):
        model = Sequential(
            [
                Conv1D(
                    filters=32,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    input_shape=(cls.WINSIZE, n_input_features),
                ),
                Dropout(rate=0.2),
                Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"),
                Conv1DTranspose(
                    filters=16, kernel_size=3, padding="same", activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1DTranspose(
                    filters=32, kernel_size=3, padding="same", activation="relu"
                ),
                Conv1DTranspose(filters=1, kernel_size=3, padding="same"),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        return model

    @classmethod
    def get_regressor_model(cls, n_output_features=1, n_input_features=1):
        # define model
        model = Sequential()
        model.add(
            Conv1D(
                filters=64,
                kernel_size=2,
                activation="relu",
                input_shape=(cls.WINSIZE, n_input_features),
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(n_output_features))
        model.compile(optimizer="adam", loss="mse")
        # model.compile(loss="mean_squared_error", optimizer="rmsprop")
        ## use the default values for batch_size, stateful

        # Compile the model
        # model.compile(
        #     optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        # )
        model.summary()
        return model

    def fit(
        self,
        df,
        model=None,
        inputs=["r_zero"],
        outputs=["win_risky"],
        input_is_list=True,
        output_is_list=False,
        epochs=500,
        initial_epoch=0,
    ):
        if model is None:
            model = self.model
        self.training_data = df
        self.input_cols = inputs

        if input_is_list:
            input_values = []
            for input in inputs:
                input_values.append(np.array(df[input].to_list()))
            x_train = np.dstack(input_values)
        else:
            x_train = df[outputs]

        self.output_cols = outputs
        if output_is_list:
            output_values = []
            for output in outputs:
                output_values.append(np.array(df[output].to_list()))
            y_train = np.dstack(output_values)
        else:
            y_train = df[outputs]

        self.model = model
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=1,
            initial_epoch=initial_epoch,
            callbacks=[early_stop],
        )

    def save(self, path):
        # Save the entire model including architecture, weights, and training configuration
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)

    def predict(
        self,
        df,
        inputs=["r"],
        outputs=["win_risky_pred"],
    ):
        # input_values = np.array(df[inputs].to_list())
        # x_test = np.array(input_values).reshape(
        #     int(input_values.shape[0]), input_values.shape[1], 1
        # )
        input_values = []
        for input in inputs:
            input_values.append(np.array(df[input].to_list()))
        x_test = np.dstack(input_values)
        # Prediction Model
        y_pred = self.model.predict(x_test, verbose=0)

        # Convert the predicted probabilities to class labels (0 or 1)
        res_df = pd.DataFrame(
            y_pred,
            columns=outputs,
            index=df.index,
        )
        self.test_data = df.join(res_df)
        return self.test_data

    def predict_encoder(
        self,
        df,
        inputs=["r"],
        # outputs=["win_risky_pred"],
    ):
        # input_values = np.array(df[inputs].to_list())
        # x_test = np.array(input_values).reshape(
        #     int(input_values.shape[0]), input_values.shape[1], 1
        # )
        input_values = []
        for input in inputs:
            input_values.append(np.array(df[input].to_list()))
        x_test = np.dstack(input_values)
        # Prediction Model
        reconstructions = self.model.predict(x_test, verbose=0)
        loss = tf.keras.losses.mae(reconstructions, x_test)
        mean_loss = tf.reduce_mean(loss, axis=1)
        max_loss = tf.reduce_max(loss, axis=1)
        min_loss = tf.reduce_min(loss, axis=1)
        res_df = pd.DataFrame(
            {"mean_loss": -mean_loss, "min_loss": -min_loss, "max_loss": -max_loss},
            index=df.index,
        )
        self.test_data = df.join(res_df)

        # Compute means over consecutive windows
        self.test_data["mean_loss_3"] = (
            self.test_data.groupby(["log_folder", "log_name"])
            .apply(rolling_ave, size=3, col="mean_loss", include_groups=False)
            .explode()
            .tolist()
        )
        self.test_data["mean_loss_4"] = (
            self.test_data.groupby(["log_folder", "log_name"])
            .apply(rolling_ave, size=4, col="mean_loss", include_groups=False)
            .explode()
            .tolist()
        )
        self.test_data["mean_loss_5"] = (
            self.test_data.groupby(["log_folder", "log_name"])
            .apply(rolling_ave, size=5, col="mean_loss", include_groups=False)
            .explode()
            .tolist()
        )
        self.test_data["mean_loss_6"] = (
            self.test_data.groupby(["log_folder", "log_name"])
            .apply(rolling_ave, size=6, col="mean_loss", include_groups=False)
            .explode()
            .tolist()
        )
        return self.test_data
