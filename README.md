# MNIST Classification

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification. The model achieves >99.4% accuracy on the validation set with less than 20,000 parameters.

## Model Architecture
- Uses 3x3 and 1x1 convolutions
- Batch Normalization layers
- Dropout layers (0.1)
- MaxPooling layers
- Fully connected layer for final classification
- Total parameters: < 20,000

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```

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

## Training Results

The model achieves excellent performance on the MNIST dataset:

### Training Progress
| Epoch | Training Loss | Training Accuracy | Validation Accuracy |
|-------|--------------|-------------------|-------------------|
| 1     | 1.1541       | 66.08%           | 93.95%           |
| 5     | 0.0642       | 97.95%           | 97.52%           |
| 10    | 0.0463       | 98.54%           | 98.84%           |
| 15    | 0.0250       | 99.21%           | 99.16%           |
| 20    | 0.0139       | 99.56%           | 99.40%           |

### Key Achievements
- **Final Validation Accuracy**: 99.40%
- **Best Training Accuracy**: 99.56%
- **Convergence**: Model shows stable training with consistent improvement
- **Early Performance**: >98% validation accuracy by epoch 7
- **Final Loss**: 0.0139
- **Model Size**: has 14314 parameters

### Training Characteristics
- Rapid initial convergence (93.95% validation accuracy in first epoch)
- Steady improvement in both training and validation metrics
- No signs of overfitting (training and validation accuracies remain close)
- Achieved target accuracy of >99.4% within 20 epochs

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deactivating Virtual Environment

When you're done working with the project:
```bash
deactivate
```