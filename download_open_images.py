from openimages.download import download_dataset

classes = [
    "Car",
    "Bus",
    "Truck",
    "Boat",
    "Person",
    "Clock",
    "Bicycle",
    "Motorcycle",
    "Traffic light",
    "Sheep"
]

download_dataset(
    dest_dir="dataset",
    classes=classes,
    annotation_format="pascal",
    limit=300
)
