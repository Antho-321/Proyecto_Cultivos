import segmentation_models as sm

# Crear una U-Net con un backbone EfficientNetB0 pre-entrenado en ImageNet
model = sm.Unet(
    'efficientnetb0',
    encoder_weights='imagenet',
    input_shape=(256, 256, 3),
    classes=1,
    activation='sigmoid'
)

model.summary()