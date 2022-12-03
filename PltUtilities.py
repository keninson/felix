import matplotlib.pyplot as plt
import torch


class PltUtilities:
    @staticmethod
    def load_data(train_data, train_labels, test_data, test_labels, predictions=None):
        plt.figure(figsize=(10, 7))
        plt.scatter(train_data, train_labels, c="b", s=4, label="Данные для тренировки")
        plt.scatter(test_data, test_labels, c="g", s=4, label="Данные для тестирования")
        if predictions is not None:
            plt.scatter(
                test_data, predictions, c="r", s=4, label="Предсказанные данные"
            )
        plt.legend(prop={"size": 14})

    @staticmethod
    def create_plots(
        initial_data: torch.Tensor,
        initial_label: torch.Tensor = None,
        initial_data_title: str = "Initial Plot",
        predicted_label: torch.Tensor = None,
        predicted_data_title: str = "Predicted Plot",
    ) -> None:
        plt.figure(figsize=(10, 7))
        if initial_label is not None:
            initial_plot = plt.plot(
                initial_data, initial_label, c="g", label=initial_data_title
            )
        if predicted_label is not None:
            predicted_plot = plt.plot(
                initial_data, predicted_label, c="r", label=predicted_data_title
            )

        plt.legend(prop={"size": 14})

    @staticmethod
    def show(title: str = "New Data"):
        plt.title = title
        plt.show()
