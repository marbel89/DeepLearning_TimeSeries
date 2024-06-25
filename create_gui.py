import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PyQt5.QtWidgets import QApplication, QWidget, \
    QVBoxLayout, QLabel, QPushButton, \
    QFileDialog, QCheckBox, QComboBox, QMessageBox


class MlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train and evaluate DL model")
        self.layout = QVBoxLayout()

        self.label = QLabel("No file selected")
        self.layout.addWidget(self.label)

        self.button = QPushButton("Upload File")
        self.button.clicked.connect(self.select_file)
        self.layout.addWidget(self.button)

        self.submit_button = QPushButton("Submit")
        # self.submit_button.clicked.connect()  # add functionality)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def select_file(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, "Select File")[0]

        if file_path:
            self.label. \
                setText(f"File selected: {file_path}")
            self.process_file(file_path)

    def process_file(self, file_path):
        data = pd.read_csv(file_path)
        self.set_label_selection(data.columns)
        self.set_feature_checkboxes(data.columns)

    @staticmethod
    def show_message_box(title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MlApp()
    window.show()
    sys.exit(app.exec_())
