import tempfile
import unittest
from pathlib import Path

import baby_cry_data


class DataLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def create_file(self, relative_path):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_single_folder_uses_sorted_class_names_and_wav_files(self):
        hungry = self.create_file("hungry/hungry.wav")
        tired = self.create_file("tired/tired.wav")
        self.create_file("hungry/notes.txt")
        self.create_file("tired/ignored.WAV")

        paths, labels, classes = baby_cry_data.load_full_data(str(self.root))

        self.assertEqual(classes, ["hungry", "tired"])
        self.assertEqual(dict(zip(paths, labels)), {hungry: 0, tired: 1})

    def test_split_folders_reindex_classes_independently(self):
        train_a = self.create_file("train/a/a.wav")
        train_b = self.create_file("train/b/b.wav")
        val_b = self.create_file("val/b/b.wav")
        test_a = self.create_file("test/a/a.wav")
        test_b = self.create_file("test/b/b.wav")

        train, val, test, classes = baby_cry_data.load_data(str(self.root))

        self.assertEqual(classes, ["a", "b"])
        self.assertEqual(dict(zip(*train)), {train_a: 0, train_b: 1})
        self.assertEqual(dict(zip(*val)), {val_b: 0})
        self.assertEqual(dict(zip(*test)), {test_a: 0, test_b: 1})

    def test_missing_split_folders_return_empty_lists(self):
        train, val, test, classes = baby_cry_data.load_data(str(self.root))
        self.assertEqual((train, val, test, classes), (([], []), ([], []), ([], []), []))

    def test_missing_single_folder_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            baby_cry_data.load_full_data(str(self.root / "missing"))


if __name__ == "__main__":
    unittest.main()
