import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.api.models import load_model
from PIL import Image
import numpy as np
from class_labels import classs

model = load_model('myModel.h5', compile=False)
print("✅ Model loaded successfully!")

test_folder = './DataSets/TEST'
if not os.path.exists(test_folder):
    print(f"❌ Test folder not found: {test_folder}")
    exit()

def load_and_process(image_path):
    img = Image.open(image_path).resize((30, 30)).convert('RGB')
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

count = 0
for root, dirs, files in os.walk(test_folder):
    for filename in files:
        image_path = os.path.join(root, filename)
        try:
            image = load_and_process(image_path)
            result = model.predict(image, verbose=0)[0]
            predicted_class = np.argmax(result)
            sign = classs[predicted_class]
            print(f"{filename} → {sign}")
            count += 1
        except Exception as e:
            print(f"⚠️ Skipped {filename} — {e}")

print(f"\n✅ Testing Completed! Processed {count} images.")
