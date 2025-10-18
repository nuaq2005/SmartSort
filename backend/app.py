from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64
from torchvision import transforms

app = Flask(__name__)
CORS(app)

# CO2 factors from your code
co2_factors = {
    "plastic_soda_bottles": 2.5,
    "aerosol_cans": 6.5,
    "steel_food_cans": 2.9,
    "disposable_plastic_cutlery": 3.2,
    "cardboard_boxes": 0.8,
    "glass_beverage_bottles": 1.2,
    "plastic_cup_lids": 2.8,
    "plastic_straws": 2.0,
    "plastic_shopping_bags": 3.0,
    "styrofoam_cups": 6.0,
    "cardboard_packaging": 0.9,
    "glass_food_jars": 1.0,
    "styrofoam_food_containers": 6.5,
    "eggshells": 0.1,
    "aluminum_food_cans": 9.0,
    "coffee_grounds": 0.3,
    "plastic_food_containers": 2.4,
    "food_waste": 0.4,
    "magazines": 0.7,
    "shoes": 4.5,
    "clothing": 3.0,
    "aluminum_soda_cans": 8.8,
    "plastic_detergent_bottles": 2.6,
    "newspaper": 0.6,
    "tea_bags": 0.2,
    "office_paper": 0.8,
    "plastic_water_bottles": 2.5,
    "paper_cups": 1.0,
    "glass_cosmetic_containers": 1.1,
    "plastic_trash_bags": 2.8
}

# Class names (must match your dataset.classes order)
class_names = list(co2_factors.keys())

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Get the directory where app.py is located
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'waste_classifier_full.pth')
    
    print(f"Current directory: {current_dir}")
    print(f"Looking for model at: {model_path}")
    print(f"Files in current directory: {os.listdir(current_dir)}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Same transform as training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def calculate_co2(predicted_class, weight):
    """Calculate CO2 saved by recycling"""
    factor = co2_factors.get(predicted_class.lower(), 0)
    co2_saved = weight * factor
    return co2_saved

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_class = class_names[predicted_idx]
        
        # Calculate CO2 savings if weight provided
        weight = data.get('weight', 1.0)  # Default 1 kg
        co2_saved = calculate_co2(predicted_class, weight)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {
                'class': class_names[idx.item()],
                'confidence': prob.item()
            }
            for idx, prob in zip(top3_idx, top3_prob)
        ]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'co2_factor': co2_factors[predicted_class],
            'co2_saved_kg': co2_saved,
            'weight_kg': weight,
            'top3_predictions': top3_predictions,
            'all_probabilities': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Return all available classes"""
    return jsonify({
        'classes': class_names,
        'count': len(class_names)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')