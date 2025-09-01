"""
EcoGuardian AI - Computer Vision Module
CNN-based microplastic detection and synthetic image generation
"""
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
import os

class MicroplasticVisionSystem:
    """Computer vision system for microplastic detection."""
    
    def __init__(self, model_path="models/vision_model.keras"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Load existing model or None if not found."""
        if os.path.exists(self.model_path):
            print(f"🧠 Loading vision model from {self.model_path}")
            return load_model(self.model_path)
        return None

    def generate_synthetic_image(self, width=128, height=128, particle_count=0):
        """
        Generate a synthetic water sample image with randomly placed particles.
        Returns:
            image: RGB NumPy array
            label: 1 if particles>0 else 0
        """
        # Blue-green background
        image = np.full((height, width, 3), (200, 110, 30), dtype=np.uint8)
        noise = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
        image = cv2.add(image, noise)

        # Draw particles
        for _ in range(particle_count):
            x, y = np.random.randint(10, width-10), np.random.randint(10, height-10)
            r = np.random.randint(1, 4)
            color = (
                np.random.randint(200, 256),
                np.random.randint(200, 256),
                np.random.randint(200, 256)
            )
            cv2.circle(image, (x, y), r, color, -1)
        
        image = cv2.GaussianBlur(image, (3, 3), 0)
        label = 1 if particle_count > 0 else 0
        return image, label

    def create_and_train_model(self, num_samples=2000, epochs=10):
        """
        Generates synthetic dataset and trains a CNN model.
        Saves model to self.model_path.
        """
        print("🔬 Generating synthetic data for vision model...")
        X, y = [], []
        for i in range(num_samples):
            count = np.random.randint(0, 15) if i % 2 == 0 else 0
            img, lbl = self.generate_synthetic_image(particle_count=count)
            X.append(img)
            y.append(lbl)
        
        X = np.array(X) / 255.0
        y = np.array(y)
        print(f"❇️  Training on {len(X)} samples...")

        # Build CNN
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,3)),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, validation_split=0.2, batch_size=32, verbose=1)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        print(f"✅ Vision model saved: {self.model_path}")
        self.model = model

    def predict_from_image(self, image_path):
        """
        Predict microplastic presence and estimate particle count from an image.
        Returns a dict with keys: has_microplastics, confidence, estimated_particle_count.
        """
        if self.model is None:
            return {"error":"Vision model not loaded."}
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (128,128))
        batch = np.expand_dims(img_resized/255.0,axis=0)
        prob = float(self.model.predict(batch, verbose=0)[0,0])
        has = prob>0.5

        # Estimate count via thresholding
        cnt=0
        if has:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            _,th = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = len([c for c in contours if cv2.contourArea(c)>2])

        return {
            "has_microplastics":bool(has),
            "confidence":prob,
            "estimated_particle_count":cnt
        }

if __name__=='__main__':
    print("👁️ Computer Vision Module")
    vis = MicroplasticVisionSystem()
    if vis.model is None:
        vis.create_and_train_model(num_samples=1000, epochs=5)
    os.makedirs('outputs',exist_ok=True)
    img,_=vis.generate_synthetic_image(particle_count=8)
    path='outputs/test_vision_sample.png'
    cv2.imwrite(path,img)
    print("📸 Saved test image:",path)
    print(vis.predict_from_image(path))
