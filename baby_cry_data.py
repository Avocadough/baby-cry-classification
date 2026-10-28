"""List WAV paths and labels using the notebook's Wav2Vec2 folder logic.

The split-folder loader indexes each split independently. A missing class folder
can change its numeric labels; this module does not repair that behavior.
"""

import os


def load_data(data_dir):
    """
    โหลดข้อมูลแบบโครงสร้าง:
    data_dir/
        train/
            class1/
            class2/
        val/
            ...
        test/
            ...
    """
    def get_files_and_labels(split_dir):
        files, labels, classes = [], [], []
        if not os.path.exists(split_dir):
            return files, labels, classes
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        for i, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.wav'):
                    files.append(os.path.join(class_dir, file))
                    labels.append(i)
        return files, labels, classes

    train_files, train_labels, class_names = get_files_and_labels(os.path.join(data_dir, 'train'))
    val_files, val_labels, _ = get_files_and_labels(os.path.join(data_dir, 'val'))
    test_files, test_labels, _ = get_files_and_labels(os.path.join(data_dir, 'test'))

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), class_names


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
