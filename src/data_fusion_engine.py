"""
EcoGuardian AI - Data Fusion Engine
Combines sensor model and vision model outputs for risk assessment and alerts
"""

import joblib
import numpy as np
import os
import cv2
from computer_vision_module import MicroplasticVisionSystem

class EcoGuardianFusionEngine:
    """Fuses predictions from environmental sensor and computer vision models."""

    def __init__(self,
                 sensor_model_path="models/sensor_model.joblib",
                 vision_model_path="models/vision_model.keras"):
        self.sensor_model = self._load_sensor_model(sensor_model_path)
        self.vision_system = MicroplasticVisionSystem(vision_model_path)
        # Order must match training features in main_analysis.py
        self.sensor_features = [
            'temp_c', 'ph', 'turbidity_ntu', 'dissolved_o2_mg_l', 'flow_rate_cfs'
        ]

    def _load_sensor_model(self, path):
        """Load the trained Random Forest model."""
        if os.path.exists(path):
            print(f"🧠 Loading sensor model from {path}")
            return joblib.load(path)
        print(f"❌ Sensor model not found at {path}")
        return None

    def get_fused_assessment(self, sensor_data, image_path):
        """
        Perform a multi-modal assessment.
        
        Args:
            sensor_data (dict): Environmental parameters
            image_path (str): Path to water sample image

        Returns:
            dict: Contains individual probabilities, fused risk level, and recommendations
        """
        if self.sensor_model is None or self.vision_system.model is None:
            return {"error": "Models not loaded. Ensure both models are trained."}

        # Sensor model prediction
        try:
            # Build input array in correct feature order
            sensor_input = [sensor_data[feat] for feat in self.sensor_features]
            sensor_prob = float(self.sensor_model.predict_proba([sensor_input])[0][1])
        except Exception as e:
            return {"error": f"Sensor prediction error: {e}"}

        # Vision model prediction
        vision_pred = self.vision_system.predict_from_image(image_path)
        if 'error' in vision_pred:
            return vision_pred
        vision_prob = vision_pred['confidence']

        # Weighted fusion
        fused_prob = 0.6 * sensor_prob + 0.4 * vision_prob

        # Determine risk level and recommendation
        if fused_prob > 0.75:
            risk_level = "CRITICAL"
            recommendation = "Immediate investigation required. Issue public advisory."
        elif fused_prob > 0.5:
            risk_level = "HIGH"
            recommendation = "Increase monitoring frequency and check upstream sources."
        elif fused_prob > 0.25:
            risk_level = "MODERATE"
            recommendation = "Conditions warrant routine monitoring."
        else:
            risk_level = "LOW"
            recommendation = "Conditions normal. Continue standard monitoring."

        return {
            "sensor_probability": sensor_prob,
            "vision_probability": vision_prob,
            "fused_probability": fused_prob,
            "risk_level": risk_level,
            "estimated_particle_count": vision_pred.get('estimated_particle_count', 0),
            "recommendation": recommendation
        }

if __name__ == '__main__':
    print("🔄 Running Multi-Modal Fusion Engine Demo...")
    fusion = EcoGuardianFusionEngine()
    
    # Sample environmental data (simulate a high-risk scenario)
    sensor_data = {
        'temp_c': 20.0,
        'ph': 8.0,
        'turbidity_ntu': 120.0,
        'dissolved_o2_mg_l': 4.0,
        'flow_rate_cfs': 0.5
    }
    
    # Ensure test image exists
    test_image_path = "outputs/test_vision_sample.png"
    if not os.path.exists(test_image_path):
        vis = MicroplasticVisionSystem()
        img, _ = vis.generate_synthetic_image(particle_count=10)
        os.makedirs('outputs', exist_ok=True)
        cv2.imwrite(test_image_path, img)
    
    # Get assessment
    result = fusion.get_fused_assessment(sensor_data, test_image_path)
    print("\n--- Fusion Assessment Report ---")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    print("--------------------------------")
