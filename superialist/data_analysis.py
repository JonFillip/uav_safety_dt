import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from aerialist.px4.trajectory import Trajectory
from aerialist.px4.obstacle import Obstacle

Trajectory.PLOT_R = True
Trajectory.REMOVE_OFFSET = False


def stats(
    data: pd.DataFrame,
    flag="flag",
    flag_threshold=None,
    label="nondeterministic",
    label_threshold=None,
    levels=["log"],  # "window", "segment"
):
    agg_dict = dict()
    if label_threshold is None:
        agg_dict[label] = "sum"
    else:
        agg_dict[label] = "min"
    if flag_threshold is None:
        agg_dict[flag] = "sum"
    else:
        agg_dict[flag] = "min"

    if "window" in levels:
        print(f"Flag:{flag}, Label:{label}, Threshold:{label_threshold}")
        print("\n")
        print("Window Level:")
        window = conf_matrix(data, flag, flag_threshold, label, label_threshold)

    if "segment" in levels:
        print("Segment Level:")
        if label_threshold is None:
            above_threshold = data[label]
        else:
            above_threshold = data[label] > label_threshold

        mask = above_threshold != above_threshold.shift(
            fill_value=above_threshold.iloc[0]
        )
        segment_id = mask.cumsum()
        seg_aggr = data.groupby(
            ["log_folder", "log_name", segment_id], as_index=False
        ).agg(agg_dict)
        segment = conf_matrix(seg_aggr, flag, flag_threshold, label, label_threshold)

    if "log" in levels:
        print("Log Level:")
        log_aggr = data.groupby(["log_folder", "log_name"], as_index=False).agg(
            agg_dict
        )
        log = conf_matrix(log_aggr, flag, flag_threshold, label, label_threshold)

    if "test" in levels:
        print("Test Level:")
        flag_aggr = data.groupby(["log_folder", "log_name"], as_index=False).agg(
            {flag: agg_dict[flag]}
        )
        label_aggr = data.groupby("log_folder", as_index=False).agg(
            {label: agg_dict[label]}
        )
        test_aggr = pd.merge(flag_aggr, label_aggr, on="log_folder")
        test = conf_matrix(test_aggr, flag, flag_threshold, label, label_threshold)
    return log


def conf_matrix(
    data: pd.DataFrame,
    flag="flag",
    flag_threshold=None,
    label="nondeterministic",
    label_threshold=None,
):
    if label_threshold is not None and flag_threshold is None:
        flag_threshold = label_threshold
    if label_threshold is None:
        pos_label = f"{label}>0"
        neg_label = f"{label}==0"
    else:
        pos_label = f"{label}<={label_threshold}"
        neg_label = f"{label}>{label_threshold}"

    if flag_threshold is None:
        pos_flag = f"{flag}>0"
        neg_flag = f"{flag}==0"
    else:
        pos_flag = f"{flag}<={flag_threshold}"
        neg_flag = f"{flag}>{flag_threshold}"

    TNs = data.query(f"{neg_label} and {neg_flag}")
    TPs = data.query(f"{pos_label} and {pos_flag}")
    FNs = data.query(f"{pos_label} and {neg_flag}")
    FPs = data.query(f"{neg_label} and {pos_flag}")
    TN = len(TNs)
    TP = len(TPs)
    FN = len(FNs)
    FP = len(FPs)

    print(f"\nFlag/GT\tTrue\tFalse\nTrue\t{TP}\t{FP}\nFalse\t{FN}\t{TN}\n")
    try:
        print(f"precision:\t{(TP)/(TP+FP)}")
    except:
        print(f"precision:\t0.0")
    try:
        print(f"recall:\t\t{(TP)/(TP+FN)}")
    except:
        print(f"recall:\t\t0.0")
    try:
        print(f"accuracy:\t{(TP+TN)/(TP+TN+FP+FN)}")
    except:
        print(f"accuracy:\t0.0")
    try:
        print(f"f1score:\t{(2*TP)/(2*TP+FP+FN)}")
    except:
        print(f"f1score:\t0.0")
    print("\n")
    return {"TP": TPs, "FP": FPs, "TN": TNs, "FN": FNs}


def get_conf_test_summary(conf):
    fp = pd.DataFrame(conf["FP"]["log_folder"].value_counts())
    fp.columns = ["FP"]
    tp = pd.DataFrame(conf["TP"]["log_folder"].value_counts())
    tp.columns = ["TP"]
    fn = pd.DataFrame(conf["FN"]["log_folder"].value_counts())
    fn.columns = ["FN"]
    tn = pd.DataFrame(conf["TN"]["log_folder"].value_counts())
    tn.columns = ["TN"]
    sum = pd.concat([fp, tp, fn, tn], axis=1).fillna(0)
    return sum


def get_probability(count, total):
    prob = count / total
    # calculate 95% confidence interval with 56 successes in 100 trials
    wilson = proportion_confint(count=count, nobs=total, method="wilson")
    return prob, wilson


def plot_test_sumary(test_summary):
    # Create a stacked bar chart
    ax = test_summary.plot(kind="bar", stacked=True, figsize=(5, 3))

    # Customizing the plot
    ax.set_xlabel("Test", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()

    plt.xticks(rotation=45)
    plt.show()


def get_anomaly_threshold(nominal_data, fp_rate):
    (a, loc, scale) = gamma.fit(nominal_data)

    # Compute anomaly threshold th for a given false positive rate
    th = gamma.ppf(1 - fp_rate, a, scale=scale, loc=loc)

    # plot data r and fitted distribution x (between 1st and 99th percentiles)
    x = np.linspace(
        gamma.ppf(0.01, a, scale=scale, loc=loc),
        gamma.ppf(0.99, a, scale=scale, loc=loc),
        100,
    )
    y = gamma.pdf(x, a, scale=scale, loc=loc)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([x[0], x[-1]])
    ax.legend(loc="best", frameon=False)
    ax.plot(x, y, "r-", lw=1, alpha=0.6, label="gamma pdf")
    ax.hist(nominal_data, density=True, bins="auto", histtype="stepfilled", alpha=0.2)
    plt.show()

    return th


def plot_for_labeling(
    df, log_folder, plot_folder, min_dist, max_dist, version=1, fix_obstacle=None
):
    df["distance"] = df["obstacle-distance"]
    logs = df.query(f"distance>{min_dist} and distance<={max_dist}")[
        ["log_folder", "log_name", "distance"]
    ].drop_duplicates()
    logs.to_csv(plot_folder + "data.csv")
    metadata = pd.read_csv(log_folder + "log.csv", skipinitialspace=True)
    Trajectory.DIR = plot_folder
    for idx, row in logs.iterrows():
        log_add = log_folder + row["log_folder"] + "/" + row["log_name"]
        print(row["log_folder"] + "/" + row["log_name"])
        iteration = int(row["log_folder"].split("-")[0].replace("iter", ""))
        obstacles = [get_obstacle(metadata, iteration, version)]
        if fix_obstacle:
            obstacles.append(Obstacle.from_coordinates(fix_obstacle))
        trj = Trajectory.extract_from_log(log_add)
        trj.handle_rotation()
        trj.plot(obstacles=obstacles, filename=row["log_name"])


def plot(
    subdata: pd.DataFrame,
    data: pd.DataFrame,
    base_fld: str,
    flag="flag",
    threshold=None,
    version=1,
    fix_obstacle=None,
    plot_waypoints=False,
    plot_flag=False,
    save=False,
):
    metadata = pd.read_csv(base_fld + "log.csv", skipinitialspace=True)
    for idx, row in subdata.iterrows():
        log_add = base_fld + row["log_folder"] + "/" + row["log_name"]
        print(row["log_folder"] + "/" + row["log_name"])
        log_rows = data.query(f"log_name == \"{log_add.split('/')[-1]}\"")
        if threshold is None:
            flags = log_rows.query(f"{flag}>0")
            min_flag = log_rows[flag].min()
        else:
            flags = log_rows.query(f"{flag}<={threshold}")
            min_flag = round(log_rows[flag].min(), 3)
        changes = list((flags["win_start"] + flags["win_end"]) / 2)
        distance = round(log_rows["obstacle-distance"].iloc[0], 3)
        # distance_changes = rows["obstacle-distance-changes"].iloc[0]
        # print(f"{len(changes)} flags\t{distance} m\t{distance_changes} changes")
        print(f"{len(changes)} flags (>={min_flag})\t{distance} m")
        iteration = int(row["log_folder"].split("-")[0].replace("iter", ""))
        obstacles = [get_obstacle(metadata, iteration, version)]
        if fix_obstacle:
            obstacles.append(Obstacle.from_coordinates(fix_obstacle))
        trj = Trajectory.extract_from_log(log_add)
        trj.handle_rotation()
        trj.plot(save=save, highlights=changes, obstacles=obstacles)
        if plot_waypoints:
            waypoints = Trajectory.extract_waypoints(log_add)
            waypoints.plot(save=save, highlights=changes, obstacles=obstacles)
        if plot_flag:
            flags_data = log_rows[flag].tolist()
            timestamps = ((log_rows["win_start"] + log_rows["win_end"]) / 2).tolist()
            # Plotting
            fig, ax = plt.subplots()

            # Plot the flags_data over time
            ax.plot(timestamps, flags_data, color="blue", label=flag)

            # Set y-axis scale for flags_data (adjust these values accordingly)
            ax.set_ylim(
                data[flag].min() - (data[flag].max() - data[flag].min()) * 0.1,
                data[flag].max() + (data[flag].max() - data[flag].min()) * 0.1,
            )  # or use set_yticks

            # Labeling and show the plot
            ax.set_xlabel("Time")
            ax.legend()
            plt.show()


def get_obstacle(metadata, iteration, version=1):
    iteration_data = metadata.loc[(metadata["iteration"] == iteration)].iloc[0]
    if version >= 1:
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
    return obstacle


def label_dataset(dataset_path, labels_path, threshold=1.5, minimum_range=0.1):
    ds = pd.read_csv(dataset_path)
    ds["unsafe"] = ds["win_obstacle-distance"].apply(
        lambda x: True if x <= threshold else False
    )
    ds["uncertain"] = ds["win_obstacle-distance"].apply(
        lambda x: True if x <= threshold else False
    )
    if labels_path is not None:
        ls = pd.read_csv(labels_path)
        for idx, row in ls.iterrows():
            if bool(row["unsafe"]) > 0:
                min_dist = ds[ds["log_name"] == row["filename"][0:-4]][
                    "win_obstacle-distance"
                ].min()
                ds.loc[
                    (ds["log_name"] == row["filename"][0:-4])
                    & (ds["win_obstacle-distance"] <= min_dist + minimum_range),
                    "unsafe",
                ] = True
            else:
                ds.loc[(ds["log_name"] == row["filename"][0:-4]), "unsafe"] = False
            if bool(row["uncertain"]) > 0:
                min_dist = ds[ds["log_name"] == row["filename"][0:-4]][
                    "win_obstacle-distance"
                ].min()
                ds.loc[
                    (ds["log_name"] == row["filename"][0:-4])
                    & (ds["win_obstacle-distance"] <= min_dist + minimum_range),
                    "uncertain",
                ] = True
            else:
                ds.loc[(ds["log_name"] == row["filename"][0:-4]), "uncertain"] = False
    ds.to_csv(dataset_path, index=False)
