import datetime
import model as autoencoder
import data_analysis
import matplotlib.pyplot as plt


### updating manul labels
# data_analysis.label_dataset("../datasets/supervisor_dataset_test.csv","../datasets/supervisor_test_labels.csv")
# data_analysis.label_dataset("../datasets/surrealist_dataset.csv","../datasets/surrealist_labels.csv")

### certainty-safety conf matrix
model = autoencoder.CNNModel()
test1_data = model.extract_dataset("../datasets/supervisor_dataset_test.csv")
test1_conf = data_analysis.stats(
    test1_data,
    "unsafe",
    None,
    "uncertain",
    None,
    levels=["log"],
)

test2_data = model.extract_dataset("../datasets/surrealist_dataset.csv")
test2_conf = data_analysis.stats(
    test2_data,
    "unsafe",
    None,
    "uncertain",
    None,
    levels=["log"],
)

### confidence interval
# data_analysis.get_probability(count=TP, total=TP+FP)
# data_analysis.get_probability(count=TP, total=TP+FN)


### model training
model = autoencoder.CNNModel()
train_data = model.extract_dataset("../datasets/train_dataset.csv")
nominal_data = train_data[train_data["win_dist_0_10"] > 3]
model.fit(
    # train_data,
    nominal_data,
    model=model.get_autoencoder_model(1),
    inputs=["r_zero"],
    outputs=["r_zero"],
    output_is_list=True,
    epochs=500,
)
model.save(f'models/encoder-{datetime.now().strftime("%d-%m-%H-%M-%S")}.keras')

### histogram for training loss
pred0_data = model.predict_encoder(nominal_data, inputs=["r_zero"])
plt.figure(figsize=(5, 3))  # Set the size of the figure
ax = (pred0_data["mean_loss_4"] * -1).hist(
    bins=10, grid=False, log=True
)  # Set grid to False

ax.set_xlabel("Reconstruction Loss", fontsize=12)  # Set x-axis label
ax.set_ylabel("#", fontsize=12)  # Set y-axis label
ax.axvline(x=0.3, color="red", linestyle="dashed", linewidth=2)
plt.show()  # Display the histogram

### model loading
model = autoencoder.CNNModel()
model.load("models/autoencoder_1.keras")

test1_data = model.extract_dataset("../datasets/test_dataset.csv")
pred1_data = model.predict_encoder(test1_data, inputs=["r_zero"])
test1_certain = data_analysis.stats(
    pred1_data,
    "mean_loss_4",
    -0.3,
    "uncertain",
    None,
    levels=["log"],
)
test1_safe = data_analysis.stats(
    pred1_data,
    "mean_loss_4",
    -0.3,
    "unsafe",
    None,
    levels=["log"],
)
## plot the flights
# data_analysis.plot(
#     test1_certain["FP"],
#     # test1_safe["FP"],
#     pred1_data,
#     "/media/sajad/Extreme Pro/datasets/supervisor/left-seed-1/",
#     "mean_loss_4",
#     -0.3,
#     version=1,
#     fix_obstacle=[7.625, 5.0625, 20, -6.375, 17.53125, 0, 0],
#     # save=True,
# )


test2_data = model.extract_dataset("../datasets/test2_dataset.csv")
pred2_data = model.predict_encoder(test2_data, inputs=["r_zero"])
test2_certain = data_analysis.stats(
    pred2_data,
    "mean_loss_4",
    -0.3,
    "uncertain",
    None,
    levels=["log"],
)
test2_safe = data_analysis.stats(
    pred2_data,
    "mean_loss_4",
    -0.3,
    "unsafe",
    None,
    levels=["log"],
)
### ploting the flights
# data_analysis.plot(
#     test2_certain["FP"],
#     # test2_safe["FP"],
#     pred2_data,
#     "/media/sajad/Extreme Pro/datasets/rq2/r2/",
#     "mean_loss_4",
#     -0.3,
#     version=0,
# fix_obstacle =[7.625, 5.0625, 20,-6.1875, 17.53125, 0, 0],
# )
