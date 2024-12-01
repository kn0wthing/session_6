import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.model import MNISTNet
from src.utils import count_parameters, check_batchnorm, check_dropout, check_linear

def test_parameter_count():
    model = MNISTNet()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_batchnorm_presence():
    model = MNISTNet()
    assert check_batchnorm(model), "Model should contain batch normalization layers"

def test_dropout_presence():
    model = MNISTNet()
    assert check_dropout(model), "Model should contain dropout layers"

def test_linear_presence():
    model = MNISTNet()
    assert check_linear(model), "Model should contain fully connected layers" 