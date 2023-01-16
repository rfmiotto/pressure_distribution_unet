import torch
import numpy as np
from numpy.typing import NDArray


def _get_class_of_each_sample(
    class_sample_counts: list, selected_indices: list = None
) -> NDArray:
    """
    Given a list with the number of samples that each class has, it returns a
    list of samples with their respective class.
    Example:
    `get_class_of_each_sample([2, 3, 1])` returns: `[0, 0, 1, 1, 1, 2]`

    Parameters:
        class_sample_counts (list): List with the number of samples that each class has.
        selected_indices (list): List with indices of the samples to be selected (default=None).

    Returns:
        class_of_each_sample (list): list of samples with their respective class.
    """
    class_of_each_sample = []
    for i, cls in enumerate(class_sample_counts):
        for _ in range(cls):
            class_of_each_sample.append(i)

    class_of_each_sample = np.array(class_of_each_sample)

    if not selected_indices:
        return class_of_each_sample

    return class_of_each_sample[selected_indices]


def _get_class_weights(class_of_each_sample: NDArray) -> torch.Tensor:
    number_of_classes = class_of_each_sample.max() + 1

    counter = np.bincount(class_of_each_sample)

    class_sample_counts = []
    for value in range(number_of_classes):
        class_sample_counts.append(counter[value])

    class_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)

    return class_weights


def handle_imbalanced_data(
    class_sample_counts_from_entire_dataset: list, subset_indices: list = None
) -> torch.Tensor:
    class_of_each_sample = _get_class_of_each_sample(
        class_sample_counts_from_entire_dataset, subset_indices
    )

    class_weights = _get_class_weights(class_of_each_sample)

    sample_weights = class_weights[class_of_each_sample]

    return sample_weights


def _test():
    class_sample_counts = [1, 3, 2]

    sample_classes = _get_class_of_each_sample(class_sample_counts)
    expected_output = np.array([0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(sample_classes, expected_output)

    class_wights = _get_class_weights(sample_classes)
    expected_output = np.array([1, 1 / 3, 1 / 2], dtype=np.float32)
    np.testing.assert_array_equal(class_wights, expected_output)

    sample_weights = handle_imbalanced_data(class_sample_counts)
    expected_output = np.array([1, 1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2], dtype=np.float32)
    expected_output = torch.from_numpy(expected_output)
    np.testing.assert_array_equal(sample_weights.numpy(), expected_output.numpy())


if __name__ == "__main__":
    _test()
