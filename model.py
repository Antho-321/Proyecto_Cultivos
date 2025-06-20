# Import the necessary libraries.
import keras_cv
import tensorflow as tf

# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# --- CORRECTED BACKBONE AND MODEL DEFINITION ---
# Step 1: Create the backbone using a KerasCV preset.
BACKBONE = keras_cv.models.EfficientNetV2B0Backbone.from_preset(
    "efficientnetv2_b0_imagenet"
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
# The UNet model is directly available under keras_cv.models
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")
model = keras_cv.models.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
)

# Step 3: Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# KerasCV's UNet output layer does not have a softmax activation by default.
# Therefore, it's recommended to use from_logits=True with SparseCategoricalCrossentropy.
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_true=True)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Model Architecture Summary ---")
model.summary()