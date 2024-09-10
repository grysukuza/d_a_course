import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ###############################
#
# Used in the task_types notebook
#
# ###############################

def generate_home_sale_data():
    np.random.seed(42)

    # Create a discrete feature representing number of bedrooms (1 to 3)
    bedrooms = np.random.randint(1, 4, size=100)

    # Create a continuous feature representing 1000s of square feet (0.5 to 2)
    square_feet = 0.5 + 1.5 * np.random.rand(100)

    # Create a linear relationship for labels representing sale price in millions
    labels = (
        0.4
        + 0.8 * bedrooms
        + 0.5 * square_feet
        + np.random.normal(scale=0.1, size=100)
    )

    # Introduce some outliers
    outliers_indices = np.random.choice(
        np.arange(100), size=5, replace=False
    )
    labels[outliers_indices] = 0.415  # Set some specific outliers

    # Stack the features into a single array
    input_features = np.column_stack((bedrooms, square_feet))

    # Split the data into training and testing sets
    input_features_train, input_features_test, labels_train, labels_test = (
        train_test_split(
            input_features, labels, test_size=0.2, random_state=42
        )
    )

    return (
        input_features_train,
        input_features_test,
        labels_train,
        labels_test,
    )


def generate_cancer_data():
    np.random.seed(42)

    # Create a discrete feature representing tumor size category (1 to 3)
    tumor_size_category = np.random.randint(1, 4, size=100)

    # Create a continuous feature representing cell density (1.0 to 5.0)
    cell_density = 1.0 + 4.0 * np.random.rand(100)

    # Create binary labels representing benign (0) or malignant (1)
    labels = (
        0.3 * tumor_size_category
        + 0.6 * cell_density
        + np.random.normal(scale=0.1, size=100)
        > 2.5
    ).astype(int)

    # Introduce some noise by flipping labels for some instances
    noise_indices = np.random.choice(np.arange(100), size=5, replace=False)
    labels[noise_indices] = (
        1 - labels[noise_indices]
    )  # Flip some labels to introduce noise

    # Stack the features into a single array
    input_features = np.column_stack((tumor_size_category, cell_density))

    # Split the data into training and testing sets
    input_features_train, input_features_test, labels_train, labels_test = (
        train_test_split(
            input_features, labels, test_size=0.2, random_state=42
        )
    )

    return (
        input_features_train,
        input_features_test,
        labels_train,
        labels_test,
    )


def generate_colony_data():
    return make_blobs(
        n_samples=100, centers=3, n_features=2, random_state=42
    )


def plot_colony_data(X, labels=None):
    if labels is None:
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(X[:, 0], X[:, 1])
        plt.title("Bacteria")
        plt.show()
    else:
        plt.scatter(X[labels == 0, 0], X[labels == 0, 1], c="red")
        plt.scatter(X[labels == 1, 0], X[labels == 1, 1], c="blue")
        plt.scatter(X[labels == 2, 0], X[labels == 2, 1], c="green")
        plt.title("Bacteria (Clustered)")
        plt.show()


def generate_credit_card_transactions(
    normal_size=500, anomaly_size=20, test_size=0.2, random_state=42
):
    """
    Generates synthetic credit card transaction data and splits it into training and test sets.

    Parameters:
    - normal_size (int): Number of normal transactions.
    - anomaly_size (int): Number of anomalous transactions (fraud).
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (np.ndarray): Training set.
    - X_test (np.ndarray): Test set.
    """
    np.random.seed(random_state)

    # Normal transactions
    amounts_normal = np.random.normal(
        loc=50, scale=15, size=normal_size
    )  # Average transaction amount
    latitudes_normal = np.random.uniform(
        35.0, 45.0, size=normal_size
    )  # Random US latitude range
    longitudes_normal = np.random.uniform(
        -120.0, -70.0, size=normal_size
    )  # Random US longitude range
    times_normal = np.random.uniform(
        0, 24, size=normal_size
    )  # Time of day in hours

    # Anomalous transactions (fraud)
    amounts_anomalous = np.random.normal(
        loc=500, scale=100, size=anomaly_size
    )  # Higher transaction amounts
    latitudes_anomalous = np.random.uniform(
        25.0, 35.0, size=anomaly_size
    )  # Suspicious latitudes
    longitudes_anomalous = np.random.uniform(
        -100.0, -80.0, size=anomaly_size
    )  # Suspicious longitudes
    times_anomalous = np.random.uniform(
        0, 24, size=anomaly_size
    )  # Random times

    # Concatenate normal and anomalous data
    data_normal = np.vstack(
        (amounts_normal, latitudes_normal, longitudes_normal, times_normal)
    ).T
    data_anomalous = np.vstack(
        (
            amounts_anomalous,
            latitudes_anomalous,
            longitudes_anomalous,
            times_anomalous,
        )
    ).T
    data = np.concatenate([data_normal, data_anomalous], axis=0)

    # Split the data into training and test sets
    X_train, X_test = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return X_train, X_test


def plot_transactions(
    data, predictions=None, title="Transaction Data", show_anomalies=False
):
    """
    Plots the transaction data, optionally highlighting anomalies.

    Parameters:
    - data (np.ndarray): The transaction data to plot.
    - predictions (np.ndarray, optional): Anomaly predictions to highlight anomalies.
    - title (str): Title of the plot.
    - show_anomalies (bool): Whether to highlight anomalies.
    """
    plt.figure(figsize=(12, 8))
    cmap = cm.get_cmap("coolwarm")

    if show_anomalies and predictions is not None:
        normal_data = data[predictions == 1]
        anomalous_data = data[predictions == -1]

        plt.scatter(
            normal_data[:, 2],
            normal_data[:, 1],
            s=normal_data[:, 0] / 2,  # Scale size by amount
            c=normal_data[:, 3],
            cmap=cmap,
            label="Normal Transactions",
            alpha=0.6,
            edgecolor="k",
        )
        plt.scatter(
            anomalous_data[:, 2],
            anomalous_data[:, 1],
            s=anomalous_data[:, 0] / 2,  # Scale size by amount
            c=anomalous_data[:, 3],
            cmap=cmap,
            label="Anomalous Transactions",
            alpha=0.9,
            edgecolor="r",
            linewidths=1.5,
        )
    else:
        plt.scatter(
            data[:, 2],
            data[:, 1],
            s=data[:, 0] / 2,  # Scale size by amount
            c=data[:, 3],
            cmap=cmap,
            label="Transactions",
            alpha=0.6,
            edgecolor="k",
        )

    plt.colorbar(label="Time of Day (hours)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.show()


# ###############################
#
# Used in the data_splits notebook
#
# ###############################

def generate_data():
    # Simulate a dataset with ten classes with limited imbalance
    np.random.seed(42)
    data = []
    labels = []
    class_sizes = [
        100,
        90,
        80,
        70,
        60,
        50,
        50,
        50,
        50,
        50,
    ]  # Class sizes with less imbalance
    for i, size in enumerate(class_sizes):
        data.append(np.random.normal(loc=i, scale=0.5, size=(size, 2)))
        labels.extend([i] * size)
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels


def plot_class_distribution(labels, title="Class Distribution"):
    plt.figure(figsize=(8, 4))
    plt.hist(
        labels,
        bins=np.arange(-0.5, len(np.unique(labels)) + 0.5, 1),
        rwidth=0.8,
    )
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(np.arange(0, len(np.unique(labels)), 1))
    plt.show()


def plot_data_split(
    y_train=None, y_val=None, y_test=None, title="Data Splits"
):
    if y_train is not None:
        plt.figure(figsize=(8, 4))
        plt.hist(
            y_train,
            bins=np.arange(-0.5, len(np.unique(y_train)) + 0.5, 1),
            rwidth=0.8,
            color="b",
            alpha=0.7,
        )
        plt.title(f"{title} - Training Set")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(np.arange(0, len(np.unique(y_train)), 1))
        plt.show()

    if y_val is not None:
        plt.figure(figsize=(8, 4))
        plt.hist(
            y_val,
            bins=np.arange(-0.5, len(np.unique(y_val)) + 0.5, 1),
            rwidth=0.8,
            color="g",
            alpha=0.7,
        )
        plt.title(f"{title} - Validation Set")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(np.arange(0, len(np.unique(y_val)), 1))
        plt.show()

    if y_test is not None:
        plt.figure(figsize=(8, 4))
        plt.hist(
            y_test,
            bins=np.arange(-0.5, len(np.unique(y_test)) + 0.5, 1),
            rwidth=0.8,
            color="r",
            alpha=0.7,
        )
        plt.title(f"{title} - Testing Set")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(np.arange(0, len(np.unique(y_test)), 1))
        plt.show()


# ###############################
#
# Used in the training_epochs notebook
#
# ###############################


def plot_predictions(
    X_train, y_train, X_val, y_val, predictions, description
):
    plt.scatter(X_train, y_train, color="blue", label="Training data")
    plt.scatter(X_val, y_val, color="red", label="Validation data")
    plt.plot(X_val, predictions, color="green", label="Prediction Line")
    plt.title(f"Model Prediction - {description}")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()


def plot_cost_over_epochs(training_loss, validation_loss, total_epochs):
    plt.plot(
        range(1, total_epochs + 1), training_loss, label="Training Loss"
    )
    plt.plot(
        range(1, total_epochs + 1), validation_loss, label="Validation Loss"
    )
    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.show()
