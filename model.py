# Import the necessary libraries.
import keras_cv
import tensorflow as tf

# You can uncomment the following lines to see all available backbone presets
# print("Available backbone presets:")
# print(keras_cv.models.Backbone.presets.keys())

# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# The backbone you want to use.
# --- FIX: Use the correct preset name with an underscore ---
preset_name = "efficientnet_v2_b0_imagenet"
print(f"Loading backbone from preset '{preset_name}'...")

# Use Backbone.from_preset to load the model
BACKBONE = keras_cv.models.Backbone.from_preset(
    preset_name,
    input_shape=INPUT_SHAPE
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")

model = keras_cv.models.segmentation.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
)

# Step 3: Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Use sparse categorical crossentropy if your masks are integer-encoded
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Adjust the metric accordingly if you use sparse labels
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_true=True, sparse_y_pred=False)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Resumen de la Arquitectura del Modelo ---")
model.summary()