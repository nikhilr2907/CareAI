import csv
import torch

def load_states(file_path='data/mock_states.csv'):
    """
    Loads mock state data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        A tuple containing:
        - header (list): The header row of the CSV file.
        - states (list of lists): A list of states, where each state is a list of floats.
    """
    states = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            states.append([float(x) for x in row])
    return header, states

def load_states_as_tensors(file_path='data/mock_states.csv'):
    """
    Loads mock state data from a CSV file and converts it to PyTorch tensors.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        A list of PyTorch tensors, where each tensor represents a state.
    """
    _, states = load_states(file_path)
    return [torch.tensor(state, dtype=torch.float32) for state in states]

if __name__ == '__main__':
    header, states = load_states()
    print("Header:")
    print(header)
    print("\nNumber of states loaded:", len(states))
    if states:
        print("First state:", states[0])

    tensors = load_states_as_tensors()
    print("\nNumber of tensors loaded:", len(tensors))
    if tensors:
        print("First tensor:", tensors[0])
