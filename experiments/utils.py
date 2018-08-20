import numpy as np


def generate_synthetic_arithmetic_dataset(arithmetic_op, min_value, max_value, sample_size, set_size, boundaries = None):
    """
    generates a dataset of integers for the synthetics arithmetic task

    :param arithmetic_op: the type of operation to perform on the sum of the two sub sections can be either :
    ["add" , "subtract", "multiply", "divide", "root", "square"]
    :param min_value: the minimum possible value of the generated integers
    :param max_value: the maximum possible value of the generated integers
    :param sample_size: the number of integers per sample
    :param set_size: the number of samples in the dataset
    :param boundaries: [Optional] an iterable of 4 integer indices in the following format :
    [start of 1st section, end of 1st section, start of 2nd section, end of 2nd section]
    if None, the boundaries are randomly generated.
    :return: the training dataset input, the training true outputs, the boundaries of the sub sections used
    """
    scaled_input_values = np.random.randint(min_value, max_value, (set_size, sample_size))

    if boundaries is None or len(boundaries) != 4:
        boundaries = [np.random.randint(sample_size) for i in range(4)]
        boundaries[1] = np.random.randint(boundaries[0], sample_size)
        boundaries[3] = np.random.randint(boundaries[2], sample_size)

    a = np.array([np.sum(sample[boundaries[0]:boundaries[1]]) for sample in scaled_input_values])
    b = np.array([np.sum(sample[boundaries[2]:boundaries[3]]) for sample in scaled_input_values])
    true_outputs = None
    if "add" in str.lower(arithmetic_op):
        true_outputs = a + b
    elif "sub" in str.lower(arithmetic_op):
        true_outputs = a - b
    elif "mult" in str.lower(arithmetic_op):
        true_outputs = a * b
    elif "div" in str.lower(arithmetic_op):
        true_outputs = a / b
    elif "square" == str.lower(arithmetic_op):
        true_outputs = a * a
    elif "root" in str.lower(arithmetic_op):
        true_outputs = np.sqrt(a)

    return scaled_input_values, true_outputs, boundaries
