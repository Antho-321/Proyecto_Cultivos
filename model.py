# Import the necessary libraries.
import keras_cv
import tensorflow as tf

# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# --- CORRECTED BACKBONE DEFINITION ---
# Step 1: Create the backbone using a KerasCV preset.
# This ensures full compatibility with the Keras 3 API.
BACKBONE = keras_cv.models.EfficientNetV2B0Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")
model = keras_cv.models.segmentation.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
)

# Step 3: Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Use from_logits=False if your model's last layer has a softmax activation.
# KerasCV's UNet does, so set from_logits=False.
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) 
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_true=True)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False) # jit_compile can sometimes cause issues, disable if you see unrelated errors

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Model Architecture Summary ---")
model.summary()