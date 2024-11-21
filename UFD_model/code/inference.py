# inference.py

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from io import BytesIO
import json
import base64
import logging
import subprocess
import sys
import torch.nn.functional as F

# Install required packages if they're missing
required_packages = ['ftfy==6.1.1', 'regex']
for package in required_packages:
    try:
        __import__(package.split('==')[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Import your model definition
from models import get_model

# Mean and std for normalization
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

# Add device handling function
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    return device

def model_fn(model_dir):
    logger.info("Loading the model.")
    try:
        arch = 'CLIP:ViT-L/14'
        logger.info(f"Model architecture: {arch}")
        
        device = get_device()
        model = get_model(arch).to(device)
        model_path = os.path.join(model_dir, 'model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Debug logging for model structure and features
        logger.debug("Model structure:")
        logger.debug(str(model))
        logger.debug("CLIP feature dimension: 768")
        
        # Load only the linear classification layer
        if 'fc.weight' in state_dict:
            model.fc.load_state_dict({
                'weight': state_dict['fc.weight'],
                'bias': state_dict['fc.bias']
            })
            logger.info("Successfully loaded linear classification layer")
        else:
            raise ValueError("State dict must contain fc.weight and fc.bias")
        
        model.eval()
        logger.info(f"Model moved to {device}")
            
        return model
    except Exception as e:
        logger.exception("Exception in model_fn")
        raise

def input_fn(request_body, content_type):
    """
    Deserialize and preprocess the input data.
    """
    if content_type == 'application/json':
        # Parse the JSON data
        input_json = json.loads(request_body)
        # Extract the base64-encoded image data
        image_data = input_json['image_data']
        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)
        # Open the image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Apply the same transformations as during training
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN['clip'], std=STD['clip']),
        ])
        img = transform(img)
        # Add a batch dimension
        img = img.unsqueeze(0)
        return img
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    
def predict_fn(input_data, model):
    """
    Make a prediction using the model and preprocessed data.
    """
    logger.info(f"Input tensor shape: {input_data.shape}")
    
    device = next(model.parameters()).device
    with torch.no_grad():
        # Move input to same device as model
        input_data = input_data.to(device)
        
        # Get image features from CLIP model
        features = model.model.encode_image(input_data)
        
        # L2 normalize features as mentioned in paper
        features = F.normalize(features, p=2, dim=1)
        logger.debug(f"CLIP features shape: {features.shape}")
        logger.debug(f"Features norm: {torch.norm(features, dim=1)}")
        
        # Pass through the linear classification head
        output = model.fc(features)
        logger.debug(f"Raw logits: {output.flatten().tolist()}")
        
        # Apply temperature scaling (optional, as mentioned in paper)
        temperature = 1.5
        output = output / temperature
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(output)
        logger.debug(f"Probabilities: {probabilities.flatten().tolist()}")
        
        # Calculate confidence metric as suggested in paper
        confidence = abs(probabilities - 0.5) * 2
        logger.debug(f"Prediction confidence: {confidence.flatten().tolist()}")
        
        return {
            'logits': output.flatten().tolist(),
            'probabilities': probabilities.flatten().tolist(),
            'confidence': confidence.flatten().tolist()
        }

def output_fn(prediction, content_type):
    """
    Serialize the predictions into the desired response content type.
    """
    if content_type == 'application/json':
        logit = prediction['logits'][0]
        probability = prediction['probabilities'][0]
        confidence = prediction['confidence'][0]
        result = {
            'logit': float(logit),
            'probability': float(probability),
            'confidence': float(confidence),
            'is_fake': bool(probability > 0.5)
        }
        logger.info(f"Final prediction: {result}")
        return json.dumps(result), content_type
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
