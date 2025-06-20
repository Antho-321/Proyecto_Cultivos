# Import the necessary libraries.
import keras_cv
import tensorflow as tf

# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# The backbone you want to use.
# --- FIX IS HERE ---
# Instantiate the full model from a known preset.
# The UNet will automatically use it as a backbone.
print("Loading backbone from preset 'efficientnetv2_b0_imagenet'...")
BACKBONE = keras_cv.models.EfficientNetV1B0.from_preset(
    "efficientnetv2_b0_imagenet"
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")

model = keras_cv.models.segmentation.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
)

# ... (The rest of your code remains the same)
# Step 3: Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_pred=False)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Resumen de la Arquitectura del Modelo ---")
# No need to call model.build(), model.summary() will do it.
model.summary()