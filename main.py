from src.dataset import get_data
from src.model import NeuralNetwork


def main():
    # Load the data
    train_loader, test_loader = get_data(batch_size=32)
    y, X = 
    
    # Initialize the model
    model = NeuralNetwork(num_layers=3, neurons=[2,3,5,2])
    fit = model.fit(y, X, optimizer="SGD", epochs=10, verbose=True)



if __name__ == "__main__":
    main()