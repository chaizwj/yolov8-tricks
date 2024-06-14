import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics_and_loss(experiment_names, metrics_info, loss_info, metrics_subplot_layout, loss_subplot_layout,
                          metrics_figure_size=(15, 10), loss_figure_size=(15, 10), base_directory='../runs/detect/'):
    # Plot metrics
    plt.figure(figsize=metrics_figure_size)
    for i, (metric_name, title) in enumerate(metrics_info):
        plt.subplot(*metrics_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            plt.plot(data[column_name], label=name)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    metrics_filename = 'metrics_curves.png'
    plt.savefig(metrics_filename)
    plt.show()

    # Plot loss
    plt.figure(figsize=loss_figure_size)
    for i, (loss_name, title) in enumerate(loss_info):
        plt.subplot(*loss_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == loss_name][0]
            plt.plot(data[column_name], label=name)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    loss_filename = 'loss_curves.png'
    plt.savefig(loss_filename)
    plt.show()
    return metrics_filename, loss_filename


# Metrics to plot
metrics_info = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP50 at IoU=0.5'),
    ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
]

# Loss to plot
loss_info = [
    ('train/box_loss', 'Training Box Loss'),
    ('train/cls_loss', 'Training Classification Loss'),
    ('train/dfl_loss', 'Training DFL Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/cls_loss', 'Validation Classification Loss'),
    ('val/dfl_loss', 'Validation DFL Loss')
]



if __name__ == '__main__':
    # Plot the metrics and loss from multiple experiments
    metrics_filename, loss_filename = plot_metrics_and_loss(
        experiment_names=['YOLOv8s', 'Improved YOLOv8s'],
        metrics_info=metrics_info,
        loss_info=loss_info,
        metrics_subplot_layout=(2, 2),
        loss_subplot_layout=(2, 3)
    )
