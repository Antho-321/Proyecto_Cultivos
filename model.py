# Import the necessary libraries.
import keras_cv
import tensorflow as tf

# Define the parameters for the model.
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 3  # 3 classes: Background, Kidney, and Tumor.

# The backbone you want to use. KerasCV has many pre-built backbones.
# For EfficientNetB0, the name is 'EfficientNetV1B0'.
BACKBONE = keras_cv.models.EfficientNetV1B0Backbone.from_preset(
    "efficientnetv1_b0_imagenet" # Using pre-trained weights from ImageNet
)

# Step 2: Create the segmentation model using KerasCV's U-Net.
# KerasCV's UNet is highly configurable.
print(f"Constructing the model U-Net with backbone: {BACKBONE.name}...")

model = keras_cv.models.segmentation.UNet(
    backbone=BACKBONE,
    num_classes=N_CLASSES,
    # The final activation is handled by the loss function (from_logits=True)
)

# Step 3: Compile the model.
# Using Adam optimizer and a suitable loss for segmentation.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# For multi-class segmentation, CategoricalCrossentropy is a standard loss.
# Note: Set from_logits=True because the model does not have a final activation layer.
# This is numerically more stable and is the modern best practice.
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# We can use the IoU metric from TensorFlow's metrics.
# MeanIoU requires class IDs as integers.
metrics = [tf.keras.metrics.MeanIoU(num_classes=N_CLASSES, sparse_y_pred=False)]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Step 4: Visualize a summary of the model's architecture.
print("\n--- Resumen de la Arquitectura del Modelo ---")
# Build the model by passing a dummy input shape
model.build((*INPUT_SHAPE,))
model.summary()