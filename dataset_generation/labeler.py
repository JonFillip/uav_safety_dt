import os
import csv
from PIL import Image, ImageTk
import tkinter as tk


# Function to load labels from an existing CSV file
def load_existing_labels(csv_file):
    existing_labels = {}
    if os.path.exists(csv_file):
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                filename = row["filename"]
                safety = row["unsafe"]
                certainty = row["uncertain"]
                existing_labels[filename] = {"unsafe": safety, "uncertain": certainty}
    return existing_labels


def save_labels_to_csv(csv_file, labels):
    fieldnames = ["filename", "unsafe", "uncertain"]

    # Load existing labels
    existing_labels = load_existing_labels(csv_file)

    # Check if the file already exists
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write labels for each image
        for filename, values in labels.items():
            if filename in existing_labels:
                # Update existing row
                existing_labels[filename].update(
                    {"unsafe": values["unsafe"], "uncertain": values["uncertain"]}
                )
            else:
                # Add new row
                writer.writerow(
                    {
                        "filename": filename,
                        "unsafe": values["unsafe"],
                        "uncertain": values["uncertain"],
                    }
                )
                existing_labels[filename] = {
                    "unsafe": values["unsafe"],
                    "uncertain": values["uncertain"],
                }

    return existing_labels  # Return the updated labels dictionary


# Function to manually label images using a GUI
def label_images_gui(image_folder, csv_file, skip_labeled=False):
    existing_labels = load_existing_labels(csv_file)

    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    # Check if there are images in the folder
    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Image Labeling")

    # Get the dimensions of the first image
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = Image.open(first_image_path)
    initial_width, initial_height = first_image.size

    # Set the initial size of the window
    root.geometry(
        f"{initial_width}x{initial_height + 100}"
    )  # Increased height for the button

    # Create labels and entry widgets for displaying the image and collecting input
    label_image = tk.Label(root)
    label_filename = tk.Label(root, text="", font=("Helvetica", 12))
    label_safety = tk.Label(root, text="Is Unsafe", font=("Helvetica", 10))
    label_certainty = tk.Label(root, text="Is Uncertain", font=("Helvetica", 10))
    entry_safety = tk.Entry(root, width=10)
    entry_certainty = tk.Entry(root, width=10)

    def next_image(event=None):
        nonlocal image_index
        # if skip_labeled:
        #     while image_files[image_index] in existing_labels:
        #         image_index += 1
        if image_index < len(image_files) - 1:
            labels = {"unsafe": entry_safety.get(), "uncertain": entry_certainty.get()}
            existing_labels = save_labels_to_csv(
                csv_file, {image_files[image_index]: labels}
            )
            image_index += 1
            show_image(existing_labels)

    def show_image(existing_labels=None):
        if existing_labels is None:
            existing_labels = load_existing_labels(csv_file)

        filename = image_files[image_index]
        image_path = os.path.join(image_folder, filename)

        # Display the image using PIL and Tkinter
        img = Image.open(image_path)

        # Resize the image to fit the window
        img.thumbnail((initial_width, initial_height))

        img_tk = ImageTk.PhotoImage(img)

        # Update the label to show the current image
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Update the filename label
        label_filename.config(text=filename)
        label_filename.pack()

        # Update entry widgets
        entry_safety.delete(0, tk.END)
        entry_certainty.delete(0, tk.END)
        entry_safety.insert(tk.END, existing_labels.get(filename, {}).get("unsafe", ""))
        entry_certainty.insert(
            tk.END, existing_labels.get(filename, {}).get("uncertain", "")
        )

        # Set focus on the first text box
        entry_safety.focus_set()

    # Initialize image index
    image_index = 0

    # Show the first image
    show_image(existing_labels)

    # Pack the widgets
    label_image.pack()
    label_safety.pack(side=tk.LEFT, padx=10)
    entry_safety.pack(side=tk.LEFT, padx=10)
    label_certainty.pack(side=tk.LEFT, padx=10)
    entry_certainty.pack(side=tk.LEFT, padx=10)

    # Create a button to move to the next image
    next_button = tk.Button(root, text="Next", command=next_image)
    next_button.pack(side=tk.BOTTOM, pady=20)  # Increased padding for the button

    # Bind the "Enter" key press event to move to the next image
    root.bind("<Return>", next_image)
    root.bind("<KP_Enter>", next_image)

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    # Example usage
    image_folder = "../datasets/labeled_flights/test1"
    csv_file = image_folder + "/labels.csv"

    label_images_gui(image_folder, csv_file, True)
