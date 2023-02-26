from flask import Flask, render_template
from PIL import Image
import io

def predict_flower_class(model, img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_batch)[0]
    class_idx = np.argmax(pred)
    class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    return class_names[class_idx]

app = Flask(__name__)

# Define the input shape and number of classes
input_shape = (224, 224, 3)
num_classes = 5

# Load the trained model
model = tf.keras.models.load_model("flower_classifier.h5")

# Define the list of flower class names
class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Define the route for the home page
@app.route("/")
def home():
    # Load all the images in the dataset and their predicted flower classes
    images = []
    for class_name in class_names:
        img_dir = f"flower_photos/{class_name}"
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            flower_class = predict_flower_class(model, img_path)
            images.append((img_byte_arr, flower_class))
    return render_template("index.html", images=images)

# Run the application
if __name__ == "__main__":
    app.run()
