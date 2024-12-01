# MNIST Classification

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification. The model achieves >99.4% accuracy on the validation set with less than 20,000 parameters.

## Model Architecture
- Uses 3x3 and 1x1 convolutions
- Batch Normalization layers
- Dropout layers
- MaxPooling layers
- Fully connected layer for final classification

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Other dependencies listed in requirements.txt


2. Create and activate virtual environment:
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. Ensure your virtual environment is activated:
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

2. Train the model:
```bash
python src/train.py
```

The model will automatically download the MNIST dataset and start training. Training will stop when either:
- The model reaches 99.4% validation accuracy
- 15 epochs are completed

The best model weights will be saved as 'best_model.pth'.

## Model Checks

The repository includes GitHub Actions that automatically verify:
1. The model has less than 20,000 parameters
2. The model includes batch normalization layers
3. The model includes dropout layers
4. The model includes fully connected layers

These checks run automatically on every push and pull request.

To run tests locally:
```bash
# Make sure your virtual environment is activated
pytest tests/
```

## Results

The model achieves:
- >99.4% validation accuracy
- Less than 20,000 parameters
- Convergence within 15 epochs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deactivating Virtual Environment

When you're done working with the project:
```bash
deactivate
```