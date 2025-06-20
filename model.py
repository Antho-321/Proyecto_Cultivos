# Import the necessary libraries.
# These imports should now work correctly after the restart.
import keras_cv
import tensorflow as tf

# --- DIAGNOSTIC STEP: Check for presets again after reinstallation ---
print("Available KerasCV backbone presets:")
print(list(keras_cv.models.Backbone.presets.keys()))


# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# Define the preset name
preset_name = 'efficientnetv2_b0'  # Use a valid preset name

# Load the backbone from the preset
BACKBONE = keras_cv.models.Backbone.from_preset(
    preset_name,
    include_rescaling=True,  # Set to True if you want KerasCV to handle input rescaling
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
# The UNet will automatically handle the input shape.
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")

model = keras_cv.models.segmentation.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
)

# Step 3: Compile the model.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_true=True, sparse_y_pred=False)]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=True)
model.build((None, *INPUT_SHAPE)) # Explicitly build the model

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Resumen de la Arquitectura del Modelo ---")
model.summary()