from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import MaxPooling1D, Dropout, Conv1DTranspose, Conv1D, BatchNormalization, Flatten, Dense, Input
from keras.models import load_model, save_model, Sequential
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


class CNNModel(object):
    WINSIZE = 25

    def __init__(self):
        pass

    @classmethod
    def extract_dataset(cls, file_path):
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

    @classmethod
    def split(cls, df, test_size: float = 0.2):
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)
        return df_train, df_test

    @classmethod
    def balance(cls, df, label="win_risky"):
        value_counts = df[label].value_counts()
        min_count = min(value_counts)
        balanced_df = df.groupby(label, group_keys=False).apply(lambda x: x.sample(n=min_count)).reset_index(drop=True)
        shuffled_balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return shuffled_balanced_df

    @classmethod
    def get_classifier_model(cls, n_output_nodes=1):
        # FIX 3: Using Input layer to define model shape explicitly
        model = Sequential()
        model.add(Input(shape=(cls.WINSIZE, 1)))
        model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(n_output_nodes, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def get_parameterized_model(cls, hp, num_features): # FIX 1: Add num_features as an argument
        model = Sequential()
        # FIX 2: Use the provided num_features instead of a hyperparameter
        inputs = num_features
        outputs = 1
        
        model.add(Input(shape=(cls.WINSIZE, inputs)))

        num_conv_layers = hp.Int("num_conv_layers", min_value=1, max_value=2)
        for i in range(num_conv_layers):
            model.add(
                Conv1D(
                    filters=hp.Int(f"conv_filters_{i}", min_value=16, max_value=64, step=16),
                    kernel_size=hp.Int(f"conv_kernel_{i}", min_value=2, max_value=7),
                    activation=hp.Choice(f"conv_activation_{i}", values=["relu", "tanh"]),
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
                    activation=hp.Choice(f"dense_activation_{i}", values=["relu", "tanh"]),
                )
            )
        model.add(Dense(outputs, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
            loss="mse",
            metrics=['accuracy']
        )
        return model

    @classmethod
    def get_parameterized_autoencoder_model(cls, hp, n_input_features=1):
        print("\n>>> DEBUG: Running the CORRECT version of get_parameterized_autoencoder_model! <<<\n")
        
        """Builds a tunable autoencoder model."""
        model = Sequential()
        model.add(Input(shape=(cls.WINSIZE, n_input_features)))

        # Tunable Encoder
        model.add(
            Conv1D(
                filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                kernel_size=hp.Int('kernel_1', min_value=3, max_value=7, step=2),
                padding="same",
                activation="relu",
            )
        )
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(
            Conv1D(
                filters=hp.Int('filters_2', min_value=8, max_value=32, step=8),
                kernel_size=hp.Int('kernel_2', min_value=3, max_value=7, step=2),
                padding="same",
                activation="relu"
            )
        )

        # Tunable Decoder
        model.add(
            Conv1DTranspose(
                filters=hp.Int('filters_2', min_value=8, max_value=32, step=8),
                kernel_size=hp.Int('kernel_2', min_value=3, max_value=7, step=2),
                padding="same",
                activation="relu"
            )
        )
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(
            Conv1DTranspose(
                filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                kernel_size=hp.Int('kernel_1', min_value=3, max_value=7, step=2),
                padding="same",
                activation="relu"
            )
        )
        
        model.add(Conv1DTranspose(filters=n_input_features, kernel_size=3, padding="same"))
        
        model.compile(optimizer="adam", loss="mse")
        return model

    @classmethod
    def search_hyperparameters(
        cls,
        train_df,
        objective,
        inputs=["r_zero"],
        outputs=None,
        trials=10,
        epochs=100,
        project_name=None,
    ):
        num_features = len(inputs)
        
        # --- FIX: We now decide which model to build INSIDE this function ---
        if outputs is None: # This is how we know it's the autoencoder
            model_builder = lambda hp: cls.get_parameterized_autoencoder_model(hp, n_input_features=num_features)
        else: # Otherwise, it's the classifier
            model_builder = lambda hp: cls.get_parameterized_model(hp, num_features=num_features)

        tuner = RandomSearch(
            model_builder,
            objective=objective,
            directory="hp_tuning",
            project_name=f'tune-{datetime.now().strftime("%d-%m-%H-%M-%S")}' if project_name is None else project_name,
            max_trials=trials,
            overwrite=True,
        )
        
        input_values = [np.array(train_df[input].to_list()) for input in inputs]
        x_train = np.dstack(input_values)

        if outputs:
            y_train = train_df[outputs]
        else:
            y_train = x_train

        tuner.search(x_train, y_train, epochs=epochs, verbose=2, validation_split=0.2)
        
        print("\n--- Hyperparameter Search Complete ---")
        tuner.results_summary()
        return tuner

    @classmethod
    def get_best_autoencoder_model(cls, n_input_features=1):
        """
        Builds the autoencoder using the best hyperparameters
        found by the KerasTuner search.
        """
        model = Sequential()
        model.add(Input(shape=(cls.WINSIZE, n_input_features)))

        # --- Using Best Hyperparameters ---
        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(Dropout(rate=0.1))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))

        model.add(Conv1DTranspose(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(Dropout(rate=0.2))
        model.add(Conv1DTranspose(filters=64, kernel_size=3, padding="same", activation="relu"))
        
        model.add(Conv1DTranspose(filters=n_input_features, kernel_size=3, padding="same"))
        
        model.compile(optimizer="adam", loss="mse")
        print("--- Best Autoencoder Model Summary ---")
        model.summary()
        return model

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
    def get_best_classifier_model(cls, num_features=4, n_output_nodes=1):
        """
        This model uses the best hyperparameters found by the KerasTuner search.
        From Trial #05 on 2025-06-10.
        """
        model = Sequential()
        model.add(Input(shape=(cls.WINSIZE, num_features)))

        # --- Best Hyperparameters from Trial #05 ---
        # Layer 1
        model.add(Conv1D(filters=48, kernel_size=6, activation="relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization()) # Good practice to add after conv/pooling

        # Layer 2
        model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization())

        # Flatten and Dense Layers
        model.add(Flatten())
        model.add(Dense(112, activation="tanh"))
        model.add(Dense(80, activation="tanh"))
        model.add(Dense(n_output_nodes, activation="sigmoid"))
        # -------------------------------------------

        optimizer = Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        print("--- Best Classifier Model Summary ---")
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
