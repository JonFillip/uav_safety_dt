import os
from typing import List, Union
import pandas as pd
from pyulog import ULog
from aerialist.px4.obstacle import Obstacle
from aerialist.px4.position import Position
from aerialist.px4.trajectory import Trajectory


class LogParser(object):
    SURREALIST_VERSION = 1
    THRESHOLD = 150
    DISTACE_THRESHOLD = 1.5
    LOG_RECORDS = {
        "vehicle_trajectory_waypoint": {
            "waypoints[0].position[0]": "wp_x",
            "waypoints[0].position[1]": "wp_y",
            "waypoints[0].position[2]": "wp_z",
            "waypoints[0].yaw": "wp_r",
        },
        "vehicle_local_position": {
            "x": "pos_x",
            "y": "pos_y",
            "z": "pos_z",
            "heading": "pos_r",
        },
    }

    def get_all(self, df: pd.DataFrame):
        all = []
        for col in df.columns:
            col_list = ",".join(str(v) for v in df[col])
            all.append(col_list)
        return all

    def get_count(self, df: pd.DataFrame):
        return [len(df)]

    def get_obstacle_distance(self, df: pd.DataFrame):
        positions = []
        for idx, row in df.iterrows():
            pos = Position(
                row.iloc[1],
                row.iloc[2],
                -row.iloc[3],
                # row.iloc[3],
                timestamp=row.iloc[0],
            )
            positions.append(pos)
        trj = Trajectory(positions)
        distances = [trj.distance_to_obstacles([obs]) for obs in self.obstacles]
        min_distance = min(distances)
        return [min_distance, min_distance < self.DISTACE_THRESHOLD]

    def get_obstacles(self):
        obstacles = []
        if hasattr(self, "fix_obstacle") and self.fix_obstacle is not None:
            obstacles.append(self.fix_obstacle)
        if self.metadata is not None:
            iteration = int(self.log_folder.split("-")[0].replace("iter", ""))
            iteration_data = self.metadata.loc[
                (self.metadata["iteration"] == iteration)
            ].iloc[0]
            if self.SURREALIST_VERSION >= 1:
                # current Surrealist csv log style
                obstacle = Obstacle(
                    Obstacle.Size(
                        iteration_data["l"], iteration_data["w"], iteration_data["h"]
                    ),
                    Obstacle.Position(
                        iteration_data["x"], iteration_data["y"], 0, iteration_data["r"]
                    ),
                )
            else:
                # old Surrealist csv log style
                obstacle = Obstacle(
                    Obstacle.Size(
                        abs(iteration_data["x2"] - iteration_data["x1"]),
                        abs(iteration_data["y2"] - iteration_data["y1"]),
                        20,
                    ),
                    Obstacle.Position(
                        (iteration_data["x1"] + iteration_data["x2"]) / 2,
                        (iteration_data["y1"] + iteration_data["y2"]) / 2,
                        0,
                        0,
                    ),
                )
            obstacles.append(obstacle)
        return obstacles

    WIN_AGGREGATES = {
        "win_obstacle-distance,win_risky": (
            get_obstacle_distance,
            ["timestamp", "pos_x", "pos_y", "pos_z"],
        ),
        "x,y,z,r": (get_all, ["wp_x", "wp_y", "wp_z", "wp_r"]),
    }

    LOG_AGGREGATES = {
        "obstacle-distance,risky": (
            get_obstacle_distance,
            ["timestamp", "pos_x", "pos_y", "pos_z"],
        ),
    }

    def __init__(self, log_address: str, metadata: Union[pd.DataFrame, str] = None):
        self.log_address = log_address
        self.log_folder = os.path.dirname(log_address).split("/")[-1]
        self.log_name = os.path.basename(log_address)

        self.ulog = ULog(self.log_address, None, True)
        self.ulog_data = self.ulog.data_list
        self.columns = []
        self.dataset = pd.DataFrame()
        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata, skipinitialspace=True)

        self.metadata = metadata
        self.obstacles = self.get_obstacles()

    def get_data(
        self, message: str, columns: List[str], column_names: List[str] = None
    ):
        """
        message: level1 ulog category name
        columns: list of level2 ulog data entry
        column_names: list of new names for the columns
        returns dataFrame containing the timestamp and the value of the l1:l2 data
        """
        cur_dataset = [
            elem
            for elem in self.ulog_data
            if elem.name == message and elem.multi_id == 0
        ][0]
        if "timestamp" not in columns:
            columns.insert(0, "timestamp")
            if column_names is not None:
                column_names.insert(0, "timestamp")
        df = pd.DataFrame({k: cur_dataset.data[k] for k in columns})
        if column_names is not None:
            df.columns = column_names
        return df

    def id_cols(self):
        ids = {"log_folder": self.log_folder, "log_name": self.log_name}
        return pd.Series(data=ids)

    def extract_log_data(self):
        res_matrix = []
        column_names = []
        for agg_col in self.LOG_AGGREGATES.keys():
            column_names.extend(agg_col.split(","))
        for (
            agg_fun,
            columns,
        ) in self.LOG_AGGREGATES.values():
            columns_data = self.raw_data[columns].dropna()
            res_matrix.extend(agg_fun(self, columns_data))
        return pd.DataFrame(columns=column_names, data=[res_matrix])

    def extract_window_data(self, w_length: int = 10000000, w_overlap: int = 5000000):
        window_idx = 0
        dataset = pd.DataFrame()
        start = self.raw_data["timestamp"].min()
        end = self.raw_data["timestamp"].max()
        while start < end:
            w_entry = pd.Series(
                index=[
                    "win_start",
                    "win_end",
                    "win_idx",
                ],
                data=[
                    start,
                    start + w_length,
                    window_idx,
                ],
            )
            empty_window = True
            window_data = self.raw_data[
                (self.raw_data["timestamp"] >= start)
                & (self.raw_data["timestamp"] < start + w_length)
            ]
            for agg_names, (
                agg_fun,
                columns,
            ) in self.WIN_AGGREGATES.items():
                columns_data = window_data[columns].dropna()
                if len(columns_data) > 0:
                    try:
                        aggrs = pd.Series(
                            agg_fun(self, columns_data), agg_names.split(",")
                        )
                        w_entry = pd.concat([w_entry, aggrs])
                        empty_window = False
                    except:
                        pass

            if not empty_window:
                dataset = pd.concat([dataset, w_entry.to_frame().T])

            start += w_length - w_overlap
            window_idx += 1

        return dataset

    def load_raw_data(self):
        dataframes = []
        for k, v in self.LOG_RECORDS.items():
            k_df = self.get_data(k, list(v.keys()), list(v.values()))
            dataframes.append(k_df)
        self.raw_data = pd.concat(dataframes)

    def extract_dataset(self, w_length: int = 10000000, w_overlap: int = 5000000):
        self.load_raw_data()
        log_data = self.extract_log_data()
        log_data = pd.concat([self.id_cols(), log_data.iloc[0]])

        if w_length > 0:
            windows_data = self.extract_window_data(w_length, w_overlap)
            for i in range(len(log_data)):
                windows_data.insert(i, log_data.index[i], log_data[i])
            self.dataset = windows_data
        else:
            self.dataset = pd.DataFrame([log_data])
