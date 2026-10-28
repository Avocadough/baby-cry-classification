"""List WAV paths and labels using helpers adapted from the notebook.

The split-folder helper uses one class mapping from train for all splits. This
change applies only to this module; the original notebook remains unchanged and
its evaluation and data-leakage limitations remain.
"""

import os


def load_data(data_dir):
    """List train/val/test paths using class indices defined by train folders.

    Missing split directories return empty lists. A class directory in val or
    test that is absent from train raises ValueError, even when it is empty.
    """
    def get_classes(split_dir):
        if not os.path.exists(split_dir):
            return []
        return sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )

    class_names = get_classes(os.path.join(data_dir, "train"))
    label_map = {name: index for index, name in enumerate(class_names)}

    def get_files_and_labels(split):
        split_dir = os.path.join(data_dir, split)
        files, labels = [], []
        for class_name in get_classes(split_dir):
            if class_name not in label_map:
                raise ValueError(
                    f"Unknown class {class_name!r} in {split} split; not found in train"
                )
            class_dir = os.path.join(split_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".wav"):
                    files.append(os.path.join(class_dir, file))
                    labels.append(label_map[class_name])
        return files, labels

    train = get_files_and_labels("train")
    val = get_files_and_labels("val")
    test = get_files_and_labels("test")
    return train, val, test, class_names


def load_full_data(data_dir):
    """
    โหลดข้อมูลแบบโฟลเดอร์รวม:
    data_dir/
        class1/
        class2/
        ...
    คืนค่า: X_all (list เส้นทางไฟล์), y_encoded_all (list label int), class_names
    """
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    X_all, y_all = [], []

    label_map = {cls: i for i, cls in enumerate(class_names)}
    for cls in class_names:
        class_dir = os.path.join(data_dir, cls)
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                X_all.append(os.path.join(class_dir, f))
                y_all.append(label_map[cls])

    return X_all, y_all, class_names
