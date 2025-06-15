import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =================================================================================
# 1. MÓDULO DE ATENCIÓN (Reconstrucción inspirada en CBAM)
# Esta es la innovación clave aplicada a las características de bajo nivel.
# =================================================================================
class ChannelAttention(nn.Module):
    """Módulo de Atención de Canal para enfocar qué características son importantes."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Módulo de Atención Espacial para enfocar dónde están las características importantes."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    """Módulo de Atención completo que combina Canal y Espacial."""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x) # Aplica atención de canal
        x = x * self.spatial_attention(x) # Aplica atención espacial
        return x

# =================================================================================
# 2. MÓDULO ASPP (Atrous Spatial Pyramid Pooling)
# Componente estándar de DeepLabV3+ para capturar contexto multi-escala.
# =================================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# =================================================================================
# 3. ARQUITECTURA PRINCIPAL: CloudDeepLabV3+
# =================================================================================
class CloudDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, atrous_rates=(12, 24, 36)):
        super(CloudDeepLabV3Plus, self).__init__()
        
        # --- Backbone: EfficientNetV2-S Ligero ---
        # Usamos timm para cargar el backbone pre-entrenado en ImageNet-21k.
        # 'features_only=True' nos devuelve las características de cada etapa.
        # 'out_indices=(1, 3)' selecciona las salidas de la etapa 1 (stride 4, low-level)
        # y la etapa 3 (stride 16, high-level). Esto implementa la idea de "eliminar la última etapa".
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s_in21k',
            pretrained=True,
            features_only=True,
            out_indices=(1, 3)
        )
        
        # Obtenemos las dimensiones de los canales de salida del backbone
        backbone_channels = self.backbone.feature_info.channels()
        low_level_channels = backbone_channels[0]
        high_level_channels = backbone_channels[1]
        
        # --- Módulo ASPP ---
        self.aspp = ASPP(in_channels=high_level_channels, atrous_rates=atrous_rates)
        
        # --- Módulo de Atención para Características de Bajo Nivel ---
        self.attention = AttentionModule(in_channels=low_level_channels)

        # --- Decoder ---
        # Proyección de 1x1 para las características de bajo nivel después de la atención
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Capas de convolución para fusionar características y refinar la segmentación
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --- Clasificador Final ---
        # Convierte las características refinadas en el mapa de segmentación final
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # 1. Obtener características del backbone
        features = self.backbone(x)
        low_level_features = features[0]   # Stride 4
        high_level_features = features[1]  # Stride 16

        # 2. Aplicar atención a las características de bajo nivel
        low_level_features_refined = self.attention(low_level_features)
        low_level_features_refined = self.low_level_conv(low_level_features_refined)
        
        # 3. Procesar características de alto nivel con ASPP
        aspp_output = self.aspp(high_level_features)
        
        # 4. Sobredimensionar la salida del ASPP
        aspp_output_upsampled = F.interpolate(
            aspp_output, 
            size=low_level_features_refined.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        # 5. Concatenar y decodificar
        concatenated_features = torch.cat([aspp_output_upsampled, low_level_features_refined], dim=1)
        decoded_features = self.decoder_conv(concatenated_features)

        # 6. Generar el mapa de segmentación final
        final_logits = self.classifier(decoded_features)
        
        # 7. Sobredimensionar a tamaño de entrada original
        final_logits_upsampled = F.interpolate(
            final_logits, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_logits_upsampled

# =================================================================================
# 4. SCRIPT DE PRUEBA
# Para verificar que la arquitectura está correctamente ensamblada.
# =================================================================================
if __name__ == '__main__':
    # Parámetros del modelo
    NUM_CLASSES = 1  # Segmentación binaria (nube vs no-nube)
    INPUT_SIZE = (256, 256)
    
    # Crear una instancia del modelo
    model = CloudDeepLabV3Plus(num_classes=NUM_CLASSES)
    model.eval() # Poner en modo de evaluación
    
    # Crear un tensor de entrada de prueba (batch_size=1, canales=3, altura=256, ancho=256)
    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    
    # Realizar una pasada hacia adelante (forward pass)
    with torch.no_grad():
        output = model(dummy_input)
    
    # Imprimir información para verificación
    print("Arquitectura CloudDeepLabV3+ instanciada correctamente.")
    print(f"Tamaño de la entrada: {dummy_input.shape}")
    print(f"Tamaño de la salida: {output.shape}")
    
    # Verificar que el tamaño de salida coincide con el de entrada (y canales de clase)
    assert output.shape == (1, NUM_CLASSES, *INPUT_SIZE)
    print("¡La forma de la salida es correcta!")

    # Contar parámetros para tener una idea de la complejidad
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parámetros entrenables: {total_params / 1e6:.2f} M")