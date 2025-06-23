import re
import matplotlib.pyplot as plt

def calcular_f1_score(texto_log: str) -> list:
    """
    Calcula el F1-Score (Dice score) por clase para el conjunto de validación de cada época.

    Args:
        texto_log: Un string que contiene el log del entrenamiento del modelo.

    Returns:
        Una lista de listas, donde cada sublista contiene los F1-Scores (Dice)
        por clase para el conjunto de validación de una época.
    """
    # Expresión regular para encontrar las líneas de "Dice por clase" de validación
    # y capturar la lista de valores numéricos que le siguen.
    # Se utiliza re.DOTALL para que '.' pueda coincidir también con saltos de línea.
    patron = re.compile(r"Calculando métricas de validación\.\.\.\n.*?Dice por clase: \[(.*?)\]", re.DOTALL)

    # Buscar todas las coincidencias en el texto
    coincidencias = patron.findall(texto_log)

    # Procesar cada coincidencia para convertir los strings en listas de floats
    f1_scores_por_epoca = []
    for grupo_valores in coincidencias:
        # Limpiar espacios y dividir por espacios para obtener los valores individuales
        valores_str = grupo_valores.strip().split()
        # Convertir cada valor a float y guardarlo en una lista
        scores = [float(valor) for valor in valores_str]
        f1_scores_por_epoca.append(scores)

    return f1_scores_por_epoca

def generar_grafico_f1_scores(f1_scores_por_epoca: list):
    """
    Genera un gráfico de líneas que muestra la evolución del F1-Score por clase.

    Args:
        f1_scores_por_epoca: Una lista de listas con los F1-Scores por época.
    """
    if not f1_scores_por_epoca:
        print("No se encontraron datos para graficar.")
        return

    # plt.figure(figsize=(10, 6))

    # Transponer la lista de listas para tener una lista por clase a través de las épocas
    scores_por_clase = list(zip(*f1_scores_por_epoca))
    epocas = range(1, len(f1_scores_por_epoca) + 1)

    for i, scores_clase in enumerate(scores_por_clase):
        plt.plot(epocas, scores_clase, marker='o', linestyle='-', label=f'Clase {i}')

    plt.title('Evolución del F1-Score (Dice) por Clase en Validación')
    plt.xlabel('Época')
    plt.ylabel('F1-Score (Dice)')
    plt.legend()
    plt.grid(True)
    plt.xticks(epocas)
    plt.tight_layout()
    
    # Guardar la figura en un archivo
    plt.savefig('f1_scores_evolucion.png')
    print("Gráfico guardado como 'f1_scores_evolucion.png'")

# Texto del log proporcionado por el usuario
texto_log = """
Using device: cuda

--- Distribución de clases en ENTRENAMIENTO (post-aug) ---
Analizando distribución: 100% 210/210 [00:05<00:00, 37.11it/s]
Número total de píxeles analizados: 110100480
Distribución final de píxeles por clase:
  Clase 0: 76.1269%  (83816081 píxeles)
  Clase 1: 3.2099%  (3534073 píxeles)
  Clase 2: 4.9413%  (5440430 píxeles)
  Clase 3: 2.8203%  (3105193 píxeles)
  Clase 4: 1.3238%  (1457514 píxeles)
  Clase 5: 11.5778%  (12747189 píxeles)

Pesos de importancia por clase (para CrossEntropy/Focal):
  Clase 0: 0.0457
  Clase 1: 1.0846
  Clase 2: 0.7046
  Clase 3: 1.2344
  Clase 4: 2.6299
  Clase 5: 0.3007
--------------------------------------------------------
model.safetensors: 100% 193M/193M [00:01<00:00, 156MB/s]
Unexpected keys (bn2.bias, bn2.num_batches_tracked, bn2.running_mean, bn2.running_var, bn2.weight, conv_head.weight) found while loading pretrained weights. This may be expected if model is being adapted.
Compiling the model... (this may take a minute)

--- Epoch 1/200 ---
  0% 0/210 [00:00<?, ?it/s]W0622 15:14:20.503000 1791 torch/_inductor/utils.py:1137] [0/0] Not enough SMs to use max_autotune_gemm mode
100% 210/210 [03:45<00:00,  1.07s/it, loss=1.2]
Calculando métricas de entrenamiento...
IoU por clase : [0.9044211  0.03763814 0.38215318 0.01682564 0.01809993 0.66628456]
Dice por clase: [0.9498121  0.0725458  0.5529824  0.03309445 0.0355563  0.7997248 ]
mIoU macro = 0.3376 | Dice macro = 0.4073
Calculando métricas de validación...
IoU por clase : [0.90835625 0.04363872 0.30428535 0.02579277 0.02281508 0.64009076]
Dice por clase: [0.95197767 0.08362802 0.4665932  0.05028846 0.04461232 0.7805553 ]
mIoU macro = 0.3242 | Dice macro = 0.3963
🔹 Nuevo mejor mIoU: 0.3242 | Dice: 0.3963  →  guardando modelo…

--- Epoch 2/200 ---
100% 210/210 [00:24<00:00,  8.71it/s, loss=0.904]
Calculando métricas de entrenamiento...
IoU por clase : [0.9228911  0.00463803 0.55498403 0.06702058 0.03703254 0.5973621 ]
Dice por clase: [0.9598995  0.00923324 0.7138132  0.1256219  0.07142022 0.7479358 ]
mIoU macro = 0.3640 | Dice macro = 0.4380
Calculando métricas de validación...
IoU por clase : [0.91819084 0.00528035 0.50569814 0.08167081 0.04056352 0.58091813]
Dice por clase: [0.9573509  0.01050522 0.6717125  0.1510086  0.07796453 0.73491234]
mIoU macro = 0.3554 | Dice macro = 0.4339
🔹 Nuevo mejor mIoU: 0.3554 | Dice: 0.4339  →  guardando modelo…

--- Epoch 3/200 ---
100% 210/210 [00:22<00:00,  9.26it/s, loss=0.756]
Calculando métricas de entrenamiento...
IoU por clase : [0.92757946 0.00354895 0.6638793  0.24362999 0.02483097 0.5723732 ]
Dice por clase: [0.96242934 0.0070728  0.7979897  0.39180464 0.04845867 0.72803736]
mIoU macro = 0.4060 | Dice macro = 0.4893
Calculando métricas de validación...
IoU por clase : [0.91972524 0.00440688 0.6126923  0.26029727 0.02385705 0.5507094 ]
Dice por clase: [0.95818424 0.00877509 0.7598378  0.41307282 0.04660231 0.7102677 ]
mIoU macro = 0.3953 | Dice macro = 0.4828
🔹 Nuevo mejor mIoU: 0.3953 | Dice: 0.4828  →  guardando modelo…

--- Epoch 4/200 ---
100% 210/210 [00:23<00:00,  8.84it/s, loss=0.613]
Calculando métricas de entrenamiento...
IoU por clase : [0.9413268  0.00633595 0.71250653 0.4700515  0.06394978 0.6800401 ]
Dice por clase: [0.9697767  0.01259212 0.83212125 0.6395034  0.12021203 0.8095522 ]
mIoU macro = 0.4790 | Dice macro = 0.5640
Calculando métricas de validación...
IoU por clase : [0.9356607  0.00676211 0.64912015 0.55639046 0.05390648 0.6963321 ]
Dice por clase: [0.9667611  0.01343338 0.7872321  0.7149754  0.10229842 0.8209856 ]
mIoU macro = 0.4830 | Dice macro = 0.5676
🔹 Nuevo mejor mIoU: 0.4830 | Dice: 0.5676  →  guardando modelo…

--- Epoch 5/200 ---
100% 210/210 [00:23<00:00,  9.09it/s, loss=0.44]
Calculando métricas de entrenamiento...
IoU por clase : [0.9496804  0.22302599 0.7090146  0.5613419  0.03063393 0.75876856]
Dice por clase: [0.97419083 0.3647118  0.829735   0.7190505  0.05944678 0.86284065]
mIoU macro = 0.5387 | Dice macro = 0.6350
Calculando métricas de validación...
IoU por clase : [0.9426476  0.23501904 0.6626945  0.6340102  0.0555581  0.7739143 ]
Dice por clase: [0.9704772  0.38059178 0.7971332  0.7760174  0.10526772 0.87254983]
mIoU macro = 0.5506 | Dice macro = 0.6503
🔹 Nuevo mejor mIoU: 0.5506 | Dice: 0.6503  →  guardando modelo…

--- Epoch 6/200 ---
100% 210/210 [00:24<00:00,  8.67it/s, loss=0.342]
Calculando métricas de entrenamiento...
IoU por clase : [0.9537116  0.5520334  0.7522068  0.57337874 0.0094696  0.850432  ]
Dice por clase: [0.97630745 0.71136796 0.8585822  0.72885025 0.01876154 0.9191713 ]
mIoU macro = 0.6152 | Dice macro = 0.7022
Calculando métricas de validación...
IoU por clase : [0.94600385 0.5028963  0.70192295 0.61901414 0.01889287 0.85096616]
Dice por clase: [0.97225285 0.66923624 0.82485867 0.7646803  0.03708509 0.9194832 ]
mIoU macro = 0.6066 | Dice macro = 0.6979
🔹 Nuevo mejor mIoU: 0.6066 | Dice: 0.6979  →  guardando modelo…

--- Epoch 7/200 ---
100% 210/210 [00:22<00:00,  9.17it/s, loss=0.279]
Calculando métricas de entrenamiento...
IoU por clase : [0.95451605 0.69650364 0.7750937  0.57579976 0.01586296 0.8994759 ]
Dice por clase: [0.9767288  0.8211048  0.8732989  0.73080325 0.03123052 0.94707793]
mIoU macro = 0.6529 | Dice macro = 0.7300
Calculando métricas de validación...
IoU por clase : [0.94756013 0.638574   0.7299365  0.6380216  0.02374763 0.8919072 ]
Dice por clase: [0.9730741  0.7794265  0.8438882  0.7790149  0.04639352 0.9428657 ]
mIoU macro = 0.6450 | Dice macro = 0.7274
🔹 Nuevo mejor mIoU: 0.6450 | Dice: 0.7274  →  guardando modelo…

--- Epoch 8/200 ---
100% 210/210 [00:23<00:00,  8.84it/s, loss=0.239]
Calculando métricas de entrenamiento...
IoU por clase : [0.9541348  0.7311868  0.78775126 0.5483694  0.0039851  0.9119264 ]
Dice por clase: [0.97652924 0.84472317 0.88127613 0.70831853 0.00793856 0.95393467]
mIoU macro = 0.6562 | Dice macro = 0.7288
Calculando métricas de validación...
IoU por clase : [0.9474766  0.6509949  0.76153964 0.59882355 0.00469006 0.89239556]
Dice por clase: [0.97303003 0.78860927 0.86462957 0.74908024 0.00933633 0.9431385 ]
mIoU macro = 0.6427 | Dice macro = 0.7213

--- Epoch 9/200 ---
100% 210/210 [00:20<00:00, 10.05it/s, loss=0.312]
Calculando métricas de entrenamiento...
IoU por clase : [0.95487076 0.7556747  0.75259715 0.63947266 0.0837197  0.9091709 ]
Dice por clase: [0.9769144  0.8608368  0.8588364  0.7800956  0.15450434 0.9524248 ]
mIoU macro = 0.6826 | Dice macro = 0.7639
Calculando métricas de validación...
IoU por clase : [0.9470675  0.68884176 0.70977175 0.66141516 0.03701003 0.893567  ]
Dice por clase: [0.97281426 0.8157564  0.83025324 0.79620695 0.07137834 0.94379234]
mIoU macro = 0.6563 | Dice macro = 0.7384
🔹 Nuevo mejor mIoU: 0.6563 | Dice: 0.7384  →  guardando modelo…

--- Epoch 10/200 ---
100% 210/210 [00:23<00:00,  9.06it/s, loss=0.224]
Calculando métricas de entrenamiento...
IoU por clase : [0.9541046  0.7797485  0.7397001  0.6330872  0.11526293 0.91840786]
Dice por clase: [0.97651327 0.87624574 0.85037655 0.77532566 0.2067009  0.9574688 ]
mIoU macro = 0.6901 | Dice macro = 0.7738
Calculando métricas de validación...
IoU por clase : [0.9467557  0.7163473  0.6674291  0.66824317 0.05535304 0.908585  ]
Dice por clase: [0.97264975 0.8347347  0.8005487  0.801134   0.10489957 0.95210326]
mIoU macro = 0.6605 | Dice macro = 0.7443
🔹 Nuevo mejor mIoU: 0.6605 | Dice: 0.7443  →  guardando modelo…

--- Epoch 11/200 ---
100% 210/210 [00:23<00:00,  8.85it/s, loss=0.171]
Calculando métricas de entrenamiento...
IoU por clase : [0.9580491  0.78752    0.782701   0.6985529  0.33811754 0.9240642 ]
Dice por clase: [0.9785751 0.8811314 0.8781069 0.8225271 0.505363  0.9605337]
mIoU macro = 0.7482 | Dice macro = 0.8377
Calculando métricas de validación...
IoU por clase : [0.95058846 0.73016125 0.7165444  0.69984186 0.22022396 0.906986  ]
Dice por clase: [0.9746684  0.84403837 0.83486843 0.8234199  0.36095664 0.9512246 ]
mIoU macro = 0.7041 | Dice macro = 0.7982
🔹 Nuevo mejor mIoU: 0.7041 | Dice: 0.7982  →  guardando modelo…

--- Epoch 12/200 ---
100% 210/210 [00:23<00:00,  8.86it/s, loss=0.295]
Calculando métricas de entrenamiento...
IoU por clase : [0.9578844  0.7880174  0.79756945 0.6597059  0.32971564 0.9242744 ]
Dice por clase: [0.9784892  0.88144267 0.88738656 0.79496723 0.49591902 0.9606472 ]
mIoU macro = 0.7429 | Dice macro = 0.8331
Calculando métricas de validación...
IoU por clase : [0.95073074 0.7295711  0.73087543 0.6683151  0.20084466 0.9069266 ]
Dice por clase: [0.9747432  0.84364396 0.8445154  0.8011857  0.33450565 0.9511919 ]
mIoU macro = 0.6979 | Dice macro = 0.7916

--- Epoch 13/200 ---
100% 210/210 [00:22<00:00,  9.46it/s, loss=0.139]
Calculando métricas de entrenamiento...
IoU por clase : [0.9584317  0.7937812  0.81314194 0.7024566  0.48012203 0.91963935]
Dice por clase: [0.9787747  0.8850368  0.89694244 0.825227   0.6487601  0.9581376 ]
mIoU macro = 0.7779 | Dice macro = 0.8655
Calculando métricas de validación...
IoU por clase : [0.9510151  0.7406807  0.77271706 0.68138754 0.38885882 0.90488315]
Dice por clase: [0.9748926 0.8510242 0.8717884 0.8105062 0.5599688 0.9500668]
mIoU macro = 0.7399 | Dice macro = 0.8364
🔹 Nuevo mejor mIoU: 0.7399 | Dice: 0.8364  →  guardando modelo…

--- Epoch 14/200 ---
100% 210/210 [00:22<00:00,  9.16it/s, loss=0.135]
Calculando métricas de entrenamiento...
IoU por clase : [0.95925194 0.7954365  0.795465   0.7376938  0.5054167  0.9226399 ]
Dice por clase: [0.9792022  0.88606477 0.8860824  0.8490492  0.67146415 0.9597636 ]
mIoU macro = 0.7860 | Dice macro = 0.8719
Calculando métricas de validación...
IoU por clase : [0.95102215 0.74632245 0.74780595 0.69588846 0.41284975 0.9110339 ]
Dice por clase: [0.9748964  0.85473615 0.85570824 0.82067716 0.58442134 0.9534461 ]
mIoU macro = 0.7442 | Dice macro = 0.8406
🔹 Nuevo mejor mIoU: 0.7442 | Dice: 0.8406  →  guardando modelo…

--- Epoch 15/200 ---
100% 210/210 [00:23<00:00,  8.86it/s, loss=0.143]
Calculando métricas de entrenamiento...
IoU por clase : [0.958857   0.7900385  0.82816726 0.66160196 0.39599404 0.9265664 ]
Dice por clase: [0.97899646 0.8827056  0.9060082  0.79634225 0.5673291  0.96188366]
mIoU macro = 0.7602 | Dice macro = 0.8489
Calculando métricas de validación...
IoU por clase : [0.9495023  0.725751   0.7945019  0.6510509  0.33552948 0.9094975 ]
Dice por clase: [0.97409713 0.84108424 0.8854846  0.78865033 0.5024666  0.95260406]
mIoU macro = 0.7276 | Dice macro = 0.8241

--- Epoch 16/200 ---
100% 210/210 [00:20<00:00, 10.03it/s, loss=0.196]
Calculando métricas de entrenamiento...
IoU por clase : [0.95914465 0.8006543  0.82377005 0.71119785 0.56847376 0.9263066 ]
Dice por clase: [0.9791463  0.88929266 0.9033705  0.8312281  0.72487503 0.96174365]
mIoU macro = 0.7983 | Dice macro = 0.8816
Calculando métricas de validación...
IoU por clase : [0.9510386  0.74850506 0.7812029  0.6831331  0.4940779  0.9113517 ]
Dice por clase: [0.97490495 0.8561657  0.8771633  0.81173986 0.6613817  0.9536201 ]
mIoU macro = 0.7616 | Dice macro = 0.8558
🔹 Nuevo mejor mIoU: 0.7616 | Dice: 0.8558  →  guardando modelo…

--- Epoch 17/200 ---
100% 210/210 [00:23<00:00,  8.95it/s, loss=0.176]
Calculando métricas de entrenamiento...
IoU por clase : [0.9593869  0.8059072  0.83352166 0.7390398  0.6230426  0.9277822 ]
Dice por clase: [0.97927254 0.89252335 0.90920293 0.84994006 0.76774645 0.9625384 ]
mIoU macro = 0.8148 | Dice macro = 0.8935
Calculando métricas de validación...
IoU por clase : [0.9504833  0.7526392  0.7814573  0.6938097  0.52754486 0.9113247 ]
Dice por clase: [0.97461313 0.8588638  0.8773236  0.81922984 0.6907095  0.9536053 ]
mIoU macro = 0.7695 | Dice macro = 0.8624
🔹 Nuevo mejor mIoU: 0.7695 | Dice: 0.8624  →  guardando modelo…

--- Epoch 18/200 ---
100% 210/210 [00:23<00:00,  9.07it/s, loss=0.108]
Calculando métricas de entrenamiento...
IoU por clase : [0.9604433  0.81185883 0.828202   0.7508481  0.6254549  0.9301145 ]
Dice por clase: [0.9798226  0.89616126 0.906029   0.8576965  0.7695752  0.9637921 ]
mIoU macro = 0.8178 | Dice macro = 0.8955
Calculando métricas de validación...
IoU por clase : [0.95190686 0.75488997 0.77687335 0.69968563 0.5252701  0.9123826 ]
Dice por clase: [0.975361  0.8603274 0.8744274 0.8233118 0.6887568 0.9541842]
mIoU macro = 0.7702 | Dice macro = 0.8627
🔹 Nuevo mejor mIoU: 0.7702 | Dice: 0.8627  →  guardando modelo…

--- Epoch 19/200 ---
100% 210/210 [00:24<00:00,  8.71it/s, loss=0.158]
Calculando métricas de entrenamiento...
IoU por clase : [0.96111757 0.8125807  0.7919892  0.74755466 0.5469731  0.9314951 ]
Dice por clase: [0.98017335 0.89660084 0.88392186 0.8555437  0.70715266 0.96453273]
mIoU macro = 0.7986 | Dice macro = 0.8813
Calculando métricas de validación...
IoU por clase : [0.9522356  0.75210464 0.76531655 0.7096178  0.5103758  0.90992475]
Dice por clase: [0.9755334  0.8585157  0.8670587  0.83014786 0.67582625 0.9528383 ]
mIoU macro = 0.7666 | Dice macro = 0.8600

--- Epoch 20/200 ---
100% 210/210 [00:21<00:00,  9.78it/s, loss=0.112]
Calculando métricas de entrenamiento...
IoU por clase : [0.9608037  0.8095085  0.8433831  0.7459161  0.65650177 0.9295727 ]
Dice por clase: [0.9800101  0.89472747 0.9150383  0.8544696  0.7926364  0.96350104]
mIoU macro = 0.8243 | Dice macro = 0.9001
Calculando métricas de validación...
IoU por clase : [0.9526138 0.75261   0.7955184 0.6916609 0.534817  0.9106288]
Dice por clase: [0.9757319  0.8588448  0.88611555 0.81772995 0.69691306 0.9532242 ]
mIoU macro = 0.7730 | Dice macro = 0.8648
🔹 Nuevo mejor mIoU: 0.7730 | Dice: 0.8648  →  guardando modelo…

--- Epoch 21/200 ---
100% 210/210 [00:24<00:00,  8.68it/s, loss=0.185]
Calculando métricas de entrenamiento...
IoU por clase : [0.96161306 0.8218174  0.8472547  0.7342127  0.65381515 0.9313031 ]
Dice por clase: [0.9804309  0.90219516 0.91731226 0.84673893 0.790675   0.9644298 ]
mIoU macro = 0.8250 | Dice macro = 0.9003
Calculando métricas de validación...
IoU por clase : [0.9535831  0.76467127 0.80469835 0.704416   0.5314078  0.91183275]
Dice por clase: [0.97624016 0.86664444 0.89178157 0.82657754 0.6940121  0.95388335]
mIoU macro = 0.7784 | Dice macro = 0.8682
🔹 Nuevo mejor mIoU: 0.7784 | Dice: 0.8682  →  guardando modelo…

--- Epoch 22/200 ---
100% 210/210 [00:23<00:00,  8.87it/s, loss=0.11]
Calculando métricas de entrenamiento...
IoU por clase : [0.96121347 0.8244009  0.8460952  0.7596585  0.70154864 0.9335085 ]
Dice por clase: [0.98022324 0.9037497  0.91663224 0.86341584 0.82460016 0.965611  ]
mIoU macro = 0.8377 | Dice macro = 0.9090
Calculando métricas de validación...
IoU por clase : [0.9521281  0.767603   0.790786   0.7148312  0.61036086 0.91540545]
Dice por clase: [0.97547704 0.8685242  0.883172   0.8337044  0.75804234 0.9558346 ]
mIoU macro = 0.7919 | Dice macro = 0.8791
🔹 Nuevo mejor mIoU: 0.7919 | Dice: 0.8791  →  guardando modelo…

--- Epoch 23/200 ---
100% 210/210 [00:23<00:00,  8.87it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.9612377 0.8113221 0.7992329 0.7474036 0.6033789 0.9328401]
Dice por clase: [0.9802358  0.89583415 0.88841516 0.8554447  0.7526342  0.96525323]
mIoU macro = 0.8092 | Dice macro = 0.8896
Calculando métricas de validación...
IoU por clase : [0.9527519  0.751054   0.75731224 0.6730507  0.4862194  0.90956324]
Dice por clase: [0.97580427 0.85783076 0.86189836 0.80457896 0.65430367 0.95264006]
mIoU macro = 0.7550 | Dice macro = 0.8512

--- Epoch 24/200 ---
100% 210/210 [00:23<00:00,  8.93it/s, loss=0.084]
Calculando métricas de entrenamiento...
IoU por clase : [0.96245813 0.8276423  0.85196304 0.7662448  0.70798427 0.9340092 ]
Dice por clase: [0.98087    0.905694   0.9200648  0.86765414 0.829029   0.9658787 ]
mIoU macro = 0.8417 | Dice macro = 0.9115
Calculando métricas de validación...
IoU por clase : [0.9532727  0.768015   0.79941225 0.7126557  0.6002248  0.9148349 ]
Dice por clase: [0.9760774  0.8687879  0.88852596 0.832223   0.7501756  0.95552355]
mIoU macro = 0.7914 | Dice macro = 0.8786

--- Epoch 25/200 ---
100% 210/210 [00:22<00:00,  9.28it/s, loss=0.134]
Calculando métricas de entrenamiento...
IoU por clase : [0.96210444 0.830056   0.8441331  0.75395566 0.67669123 0.9346688 ]
Dice por clase: [0.9806862  0.9071373  0.91547954 0.8597203  0.8071745  0.9662313 ]
mIoU macro = 0.8336 | Dice macro = 0.9061
Calculando métricas de validación...
IoU por clase : [0.95230407 0.7745175  0.7811074  0.67678946 0.5117121  0.9164455 ]
Dice por clase: [0.97556937 0.8729331  0.8771031  0.8072444  0.6769967  0.9564013 ]
mIoU macro = 0.7688 | Dice macro = 0.8610

--- Epoch 26/200 ---
100% 210/210 [00:22<00:00,  9.32it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.96288806 0.8219848  0.8533483  0.75926834 0.7088897  0.9343246 ]
Dice por clase: [0.98109317 0.902296   0.92087203 0.86316377 0.8296495  0.9660474 ]
mIoU macro = 0.8401 | Dice macro = 0.9105
Calculando métricas de validación...
IoU por clase : [0.9536363  0.7606159  0.8041151  0.7028609  0.5624382  0.91392034]
Dice por clase: [0.976268   0.8640339  0.8914233  0.8255059  0.71994936 0.9550244 ]
mIoU macro = 0.7829 | Dice macro = 0.8720

--- Epoch 27/200 ---
100% 210/210 [00:23<00:00,  8.96it/s, loss=0.0851]
Calculando métricas de entrenamiento...
IoU por clase : [0.9625334  0.83171743 0.8486904  0.7686388  0.7172355  0.93489116]
Dice por clase: [0.98090905 0.90812856 0.9181531  0.8691869  0.8353374  0.9663501 ]
mIoU macro = 0.8440 | Dice macro = 0.9130
Calculando métricas de validación...
IoU por clase : [0.95296437 0.7713409  0.79204315 0.7181011  0.6083045  0.91468084]
Dice por clase: [0.97591585 0.8709119  0.8839555  0.83592415 0.75645435 0.9554395 ]
mIoU macro = 0.7929 | Dice macro = 0.8798
🔹 Nuevo mejor mIoU: 0.7929 | Dice: 0.8798  →  guardando modelo…

--- Epoch 28/200 ---
100% 210/210 [00:23<00:00,  8.94it/s, loss=0.164]
Calculando métricas de entrenamiento...
IoU por clase : [0.9629909  0.8302312  0.85850763 0.77256787 0.7383004  0.9342638 ]
Dice por clase: [0.98114663 0.9072419  0.92386776 0.8716934  0.8494509  0.9660149 ]
mIoU macro = 0.8495 | Dice macro = 0.9166
Calculando métricas de validación...
IoU por clase : [0.9533052  0.7666972  0.8035729  0.7119782  0.60077447 0.9157639 ]
Dice por clase: [0.9760945  0.8679441  0.89109004 0.8317608  0.75060475 0.95603   ]
mIoU macro = 0.7920 | Dice macro = 0.8789

--- Epoch 29/200 ---
100% 210/210 [00:21<00:00,  9.80it/s, loss=0.157]
Calculando métricas de entrenamiento...
IoU por clase : [0.9636454  0.8295555  0.8595057  0.77553177 0.7508432  0.93497753]
Dice por clase: [0.98148614 0.9068383  0.9244454  0.87357694 0.85769325 0.96639633]
mIoU macro = 0.8523 | Dice macro = 0.9184
Calculando métricas de validación...
IoU por clase : [0.95431566 0.76692367 0.8067149  0.719276   0.63015413 0.91499555]
Dice por clase: [0.97662383 0.8680892  0.8930185  0.83671963 0.7731222  0.95561117]
mIoU macro = 0.7987 | Dice macro = 0.8839
🔹 Nuevo mejor mIoU: 0.7987 | Dice: 0.8839  →  guardando modelo…

--- Epoch 30/200 ---
100% 210/210 [00:23<00:00,  8.80it/s, loss=0.085]
Calculando métricas de entrenamiento...
IoU por clase : [0.9637299  0.8354651  0.86293787 0.78066826 0.75195324 0.9347389 ]
Dice por clase: [0.98153    0.9103579  0.9264269  0.87682617 0.85841703 0.9662688 ]
mIoU macro = 0.8549 | Dice macro = 0.9200
Calculando métricas de validación...
IoU por clase : [0.95411015 0.7677811  0.81184185 0.7223768  0.63503146 0.91607237]
Dice por clase: [0.9765163  0.86863816 0.8961509  0.8388139  0.776782   0.9561981 ]
mIoU macro = 0.8012 | Dice macro = 0.8855
🔹 Nuevo mejor mIoU: 0.8012 | Dice: 0.8855  →  guardando modelo…

--- Epoch 31/200 ---
100% 210/210 [00:23<00:00,  9.11it/s, loss=0.0545]
Calculando métricas de entrenamiento...
IoU por clase : [0.9645042 0.8399325 0.865696  0.7862034 0.7711062 0.9362821]
Dice por clase: [0.9819314  0.9130036  0.928014   0.88030666 0.8707622  0.9670927 ]
mIoU macro = 0.8606 | Dice macro = 0.9235
Calculando métricas de validación...
IoU por clase : [0.9553646  0.7903226  0.80813897 0.72259504 0.6315782  0.918914  ]
Dice por clase: [0.9771728 0.8828829 0.8938903 0.838961  0.774193  0.9577438]
mIoU macro = 0.8045 | Dice macro = 0.8875
🔹 Nuevo mejor mIoU: 0.8045 | Dice: 0.8875  →  guardando modelo…

--- Epoch 32/200 ---
100% 210/210 [00:24<00:00,  8.54it/s, loss=0.103]
Calculando métricas de entrenamiento...
IoU por clase : [0.9627946  0.8327269  0.8547679  0.7776171  0.72696084 0.9360658 ]
Dice por clase: [0.9810447  0.9087299  0.921698   0.8748983  0.8418961  0.96697724]
mIoU macro = 0.8485 | Dice macro = 0.9159
Calculando métricas de validación...
IoU por clase : [0.95300084 0.76924187 0.7952359  0.7209688  0.6106471  0.9164282 ]
Dice por clase: [0.97593486 0.86957234 0.88594025 0.83786386 0.75826305 0.9563919 ]
mIoU macro = 0.7943 | Dice macro = 0.8807

--- Epoch 33/200 ---
100% 210/210 [00:21<00:00,  9.77it/s, loss=0.0952]
Calculando métricas de entrenamiento...
IoU por clase : [0.96440345 0.83722854 0.8646086  0.7834843  0.7532406  0.9363182 ]
Dice por clase: [0.98187923 0.9114038  0.92738885 0.8785996  0.85925525 0.96711195]
mIoU macro = 0.8565 | Dice macro = 0.9209
Calculando métricas de validación...
IoU por clase : [0.95514727 0.77500397 0.8143563  0.7202929  0.61683834 0.91869795]
Dice por clase: [0.9770592  0.87324196 0.8976807  0.8374073  0.76301795 0.95762646]
mIoU macro = 0.8001 | Dice macro = 0.8843

--- Epoch 34/200 ---
100% 210/210 [00:21<00:00,  9.59it/s, loss=0.0822]
Calculando métricas de entrenamiento...
IoU por clase : [0.96477604 0.84345853 0.86857206 0.78022444 0.7602612  0.93869966]
Dice por clase: [0.9820723  0.9150827  0.92966396 0.87654614 0.863805   0.9683807 ]
mIoU macro = 0.8593 | Dice macro = 0.9226
Calculando métricas de validación...
IoU por clase : [0.954765  0.7820792 0.8058939 0.7049754 0.59294   0.9198124]
Dice por clase: [0.9768591  0.87771547 0.89251524 0.82696253 0.7444599  0.9582315 ]
mIoU macro = 0.7934 | Dice macro = 0.8795

--- Epoch 35/200 ---
100% 210/210 [00:21<00:00,  9.55it/s, loss=0.0884]
Calculando métricas de entrenamiento...
IoU por clase : [0.9646397  0.8441693  0.86662364 0.78037655 0.76271635 0.93611836]
Dice por clase: [0.98200166 0.9155009  0.9285467  0.8766421  0.8653875  0.9670054 ]
mIoU macro = 0.8591 | Dice macro = 0.9225
Calculando métricas de validación...
IoU por clase : [0.95478    0.7698554  0.81436694 0.7125483  0.65122885 0.9117073 ]
Dice por clase: [0.97686696 0.8699642  0.89768714 0.83214974 0.78878087 0.9538147 ]
mIoU macro = 0.8024 | Dice macro = 0.8865

--- Epoch 36/200 ---
100% 210/210 [00:21<00:00,  9.95it/s, loss=0.267]
Calculando métricas de entrenamiento...
IoU por clase : [0.9649436  0.84329027 0.86928606 0.78545296 0.7685193  0.93798393]
Dice por clase: [0.9821591 0.9149837 0.9300728 0.8798361 0.8691104 0.9679997]
mIoU macro = 0.8616 | Dice macro = 0.9240
Calculando métricas de validación...
IoU por clase : [0.95460385 0.7804802  0.8025845  0.7148305  0.6195426  0.91651076]
Dice por clase: [0.97677475 0.87670755 0.890482   0.833704   0.76508343 0.9564369 ]
mIoU macro = 0.7981 | Dice macro = 0.8832

--- Epoch 37/200 ---
100% 210/210 [00:21<00:00,  9.80it/s, loss=0.0946]
Calculando métricas de entrenamiento...
IoU por clase : [0.9641968  0.84190696 0.86546516 0.77980006 0.75395477 0.9362427 ]
Dice por clase: [0.98177207 0.91416883 0.92788136 0.8762783  0.85971975 0.96707165]
mIoU macro = 0.8569 | Dice macro = 0.9211
Calculando métricas de validación...
IoU por clase : [0.9547159  0.78212833 0.7995219  0.7206523  0.62826914 0.9186202 ]
Dice por clase: [0.9768334  0.87774634 0.8885937  0.83765006 0.7717019  0.95758426]
mIoU macro = 0.8007 | Dice macro = 0.8850

--- Epoch 38/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.0718]
Calculando métricas de entrenamiento...
IoU por clase : [0.96496415 0.8465499  0.8719588  0.77739984 0.77284545 0.93832225]
Dice por clase: [0.98216975 0.916899   0.9316004  0.87476075 0.8718701  0.9681798 ]
mIoU macro = 0.8620 | Dice macro = 0.9242
Calculando métricas de validación...
IoU por clase : [0.9547024  0.7798259  0.8120925  0.71020615 0.6464347  0.91748816]
Dice por clase: [0.9768263 0.8762946 0.8963036 0.8305504 0.785254  0.9569688]
mIoU macro = 0.8035 | Dice macro = 0.8870

--- Epoch 39/200 ---
100% 210/210 [00:20<00:00, 10.10it/s, loss=0.0781]
Calculando métricas de entrenamiento...
IoU por clase : [0.9648648  0.84271234 0.8730463  0.78778267 0.7795781  0.93845177]
Dice por clase: [0.98211825 0.9146434  0.9322207  0.8812958  0.8761381  0.9682488 ]
mIoU macro = 0.8644 | Dice macro = 0.9258
Calculando métricas de validación...
IoU por clase : [0.9551661  0.78217185 0.8197783  0.7245603  0.63816494 0.9185737 ]
Dice por clase: [0.977069   0.87777376 0.90096503 0.8402841  0.7791217  0.95755893]
mIoU macro = 0.8064 | Dice macro = 0.8888
🔹 Nuevo mejor mIoU: 0.8064 | Dice: 0.8888  →  guardando modelo…

--- Epoch 40/200 ---
100% 210/210 [00:23<00:00,  8.96it/s, loss=0.0912]
Calculando métricas de entrenamiento...
IoU por clase : [0.96543384 0.84802943 0.87395465 0.7931604  0.78513944 0.93988365]
Dice por clase: [0.982413   0.91776615 0.9327383  0.88465077 0.8796393  0.9690103 ]
mIoU macro = 0.8676 | Dice macro = 0.9277
Calculando métricas de validación...
IoU por clase : [0.9554931  0.78708917 0.8176593  0.72741586 0.64338034 0.91884565]
Dice por clase: [0.9772401  0.88086164 0.89968383 0.8422012  0.78299624 0.9577067 ]
mIoU macro = 0.8083 | Dice macro = 0.8901
🔹 Nuevo mejor mIoU: 0.8083 | Dice: 0.8901  →  guardando modelo…

--- Epoch 41/200 ---
100% 210/210 [00:23<00:00,  8.89it/s, loss=0.0728]
Calculando métricas de entrenamiento...
IoU por clase : [0.96579474 0.84418213 0.87535185 0.7940129  0.7849218  0.93954915]
Dice por clase: [0.9825998  0.9155084  0.93353343 0.88518083 0.8795028  0.96883255]
mIoU macro = 0.8673 | Dice macro = 0.9275
Calculando métricas de validación...
IoU por clase : [0.9554538  0.78674614 0.82295555 0.71955293 0.6427449  0.9191465 ]
Dice por clase: [0.9772195  0.8806468  0.90288055 0.83690697 0.7825255  0.95787007]
mIoU macro = 0.8078 | Dice macro = 0.8897

--- Epoch 42/200 ---
100% 210/210 [00:22<00:00,  9.37it/s, loss=0.0853]
Calculando métricas de entrenamiento...
IoU por clase : [0.96561456 0.845511   0.8697412  0.79503006 0.7805931  0.9391545 ]
Dice por clase: [0.9825065  0.9162893  0.93033326 0.8858125  0.8767787  0.9686227 ]
mIoU macro = 0.8659 | Dice macro = 0.9267
Calculando métricas de validación...
IoU por clase : [0.95560354 0.78546053 0.8291973  0.7237836  0.6331916  0.91914666]
Dice por clase: [0.9772978  0.8798408  0.90662426 0.83976156 0.7754039  0.9578701 ]
mIoU macro = 0.8077 | Dice macro = 0.8895

--- Epoch 43/200 ---
100% 210/210 [00:22<00:00,  9.19it/s, loss=0.0953]
Calculando métricas de entrenamiento...
IoU por clase : [0.965988   0.84819746 0.8764208  0.7912061  0.79175425 0.9405814 ]
Dice por clase: [0.98269975 0.91786456 0.934141   0.88343394 0.8837755  0.969381  ]
mIoU macro = 0.8690 | Dice macro = 0.9285
Calculando métricas de validación...
IoU por clase : [0.95587295 0.78423727 0.81784123 0.7239833  0.6397465  0.9206099 ]
Dice por clase: [0.97743875 0.87907284 0.8997939  0.83989596 0.78029925 0.9586641 ]
mIoU macro = 0.8070 | Dice macro = 0.8892

--- Epoch 44/200 ---
100% 210/210 [00:21<00:00,  9.75it/s, loss=0.073]
Calculando métricas de entrenamiento...
IoU por clase : [0.96567893 0.8497985  0.8799678  0.79836255 0.7834994  0.9377401 ]
Dice por clase: [0.98253983 0.9188011  0.936152   0.88787717 0.8786091  0.96786976]
mIoU macro = 0.8692 | Dice macro = 0.9286
Calculando métricas de validación...
IoU por clase : [0.9554933  0.7848384  0.8341532  0.7374999  0.64664996 0.91810226]
Dice por clase: [0.9772401  0.8794504  0.90957856 0.84892076 0.7854128  0.95730275]
mIoU macro = 0.8128 | Dice macro = 0.8930
🔹 Nuevo mejor mIoU: 0.8128 | Dice: 0.8930  →  guardando modelo…

--- Epoch 45/200 ---
100% 210/210 [00:23<00:00,  8.80it/s, loss=0.111]
Calculando métricas de entrenamiento...
IoU por clase : [0.9667304  0.8451077  0.88281524 0.8008151  0.8014538  0.9391353 ]
Dice por clase: [0.98308384 0.9160524  0.9377609  0.8893918  0.8897856  0.96861243]
mIoU macro = 0.8727 | Dice macro = 0.9308
Calculando métricas de validación...
IoU por clase : [0.95643383 0.77179503 0.82170886 0.7335291  0.65117836 0.91633624]
Dice por clase: [0.9777318  0.8712013  0.9021297  0.84628415 0.78874385 0.9563418 ]
mIoU macro = 0.8085 | Dice macro = 0.8904

--- Epoch 46/200 ---
100% 210/210 [00:20<00:00, 10.05it/s, loss=0.0903]
Calculando métricas de entrenamiento...
IoU por clase : [0.96647996 0.851648   0.8752356  0.7910192  0.77883345 0.9375667 ]
Dice por clase: [0.9829543  0.9198811  0.9334674  0.8833174  0.87566763 0.9677775 ]
mIoU macro = 0.8668 | Dice macro = 0.9272
Calculando métricas de validación...
IoU por clase : [0.95576334 0.78834563 0.80524516 0.70305026 0.58967537 0.91803753]
Dice por clase: [0.9773814  0.88164794 0.89211726 0.8256365  0.7418815  0.9572675 ]
mIoU macro = 0.7934 | Dice macro = 0.8793

--- Epoch 47/200 ---
100% 210/210 [00:21<00:00,  9.95it/s, loss=0.0721]
Calculando métricas de entrenamiento...
IoU por clase : [0.9666791  0.8530011  0.8832124  0.8002702  0.79522574 0.94107574]
Dice por clase: [0.9830572  0.92066985 0.9379849  0.88905567 0.88593394 0.9696435 ]
mIoU macro = 0.8732 | Dice macro = 0.9311
Calculando métricas de validación...
IoU por clase : [0.9559896  0.79094744 0.8104711  0.73125136 0.59415233 0.9217861 ]
Dice por clase: [0.9774996  0.88327265 0.8953152  0.84476626 0.74541473 0.9593015 ]
mIoU macro = 0.8008 | Dice macro = 0.8843

--- Epoch 48/200 ---
100% 210/210 [00:21<00:00,  9.66it/s, loss=0.0902]
Calculando métricas de entrenamiento...
IoU por clase : [0.96527725 0.8424426  0.88016874 0.79813635 0.79328096 0.93541014]
Dice por clase: [0.98233193 0.9144845  0.9362657  0.8877373  0.8847258  0.9666273 ]
mIoU macro = 0.8691 | Dice macro = 0.9287
Calculando métricas de validación...
IoU por clase : [0.9546395  0.7798091  0.8176184  0.72795403 0.6160504  0.9123029 ]
Dice por clase: [0.97679347 0.876284   0.89965904 0.8425618  0.7624149  0.9541406 ]
mIoU macro = 0.8014 | Dice macro = 0.8853

--- Epoch 49/200 ---
100% 210/210 [00:21<00:00,  9.94it/s, loss=0.127]
Calculando métricas de entrenamiento...
IoU por clase : [0.9657874  0.85101295 0.86917496 0.7991064  0.77192736 0.9397485 ]
Dice por clase: [0.982596   0.91951054 0.9300092  0.888337   0.87128556 0.96893847]
mIoU macro = 0.8661 | Dice macro = 0.9268
Calculando métricas de validación...
IoU por clase : [0.9554763  0.7845432  0.82236    0.73867947 0.65001523 0.91891253]
Dice por clase: [0.9772312  0.879265   0.90252197 0.8497017  0.78789    0.957743  ]
mIoU macro = 0.8117 | Dice macro = 0.8924

--- Epoch 50/200 ---
100% 210/210 [00:21<00:00,  9.85it/s, loss=0.0734]
Calculando métricas de entrenamiento...
IoU por clase : [0.9669052  0.85225964 0.8759946  0.79648983 0.78580034 0.9409897 ]
Dice por clase: [0.98317415 0.9202378  0.93389887 0.88671786 0.88005394 0.9695978 ]
mIoU macro = 0.8697 | Dice macro = 0.9289
Calculando métricas de validación...
IoU por clase : [0.95586777 0.7760212  0.80276704 0.7145923  0.59839654 0.91714257]
Dice por clase: [0.97743595 0.8738873  0.8905943  0.8335419  0.74874604 0.9567808 ]
mIoU macro = 0.7941 | Dice macro = 0.8802

--- Epoch 51/200 ---
100% 210/210 [00:22<00:00,  9.43it/s, loss=0.0737]
Calculando métricas de entrenamiento...
IoU por clase : [0.9665862 0.8511261 0.8819463 0.7995803 0.7979909 0.9390283]
Dice por clase: [0.9830092  0.9195765  0.9372704  0.88862973 0.88764733 0.9685555 ]
mIoU macro = 0.8727 | Dice macro = 0.9308
Calculando métricas de validación...
IoU por clase : [0.9556229 0.787466  0.8291662 0.7343039 0.6647396 0.9160264]
Dice por clase: [0.9773079  0.8810976  0.9066056  0.84679955 0.79861087 0.95617306]
mIoU macro = 0.8146 | Dice macro = 0.8944
🔹 Nuevo mejor mIoU: 0.8146 | Dice: 0.8944  →  guardando modelo…

--- Epoch 52/200 ---
100% 210/210 [00:22<00:00,  9.22it/s, loss=0.0672]
Calculando métricas de entrenamiento...
IoU por clase : [0.96708447 0.85410076 0.8760912  0.80316263 0.7967181  0.94148785]
Dice por clase: [0.9832669  0.92131    0.93395376 0.8908377  0.88685936 0.9698622 ]
mIoU macro = 0.8731 | Dice macro = 0.9310
Calculando métricas de validación...
IoU por clase : [0.9562752 0.7911329 0.8232346 0.7278606 0.6381526 0.9196899]
Dice por clase: [0.9776489  0.8833883  0.90304846 0.8424992  0.7791125  0.95816505]
mIoU macro = 0.8094 | Dice macro = 0.8906

--- Epoch 53/200 ---
100% 210/210 [00:21<00:00,  9.66it/s, loss=0.0672]
Calculando métricas de entrenamiento...
IoU por clase : [0.96696967 0.85222334 0.88230985 0.8051922  0.8116815  0.94154495]
Dice por clase: [0.9832075  0.92021656 0.9374757  0.8920847  0.8960532  0.96989256]
mIoU macro = 0.8767 | Dice macro = 0.9332
Calculando métricas de validación...
IoU por clase : [0.95664775 0.7886163  0.8259091  0.7412814  0.6320697  0.9202741 ]
Dice por clase: [0.9778436  0.88181716 0.9046552  0.8514206  0.7745622  0.958482  ]
mIoU macro = 0.8108 | Dice macro = 0.8915

--- Epoch 54/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.116]
Calculando métricas de entrenamiento...
IoU por clase : [0.96761334 0.857109   0.8849589  0.80632746 0.8128016  0.94274455]
Dice por clase: [0.9835401  0.92305726 0.9389689  0.892781   0.8967353  0.97052854]
mIoU macro = 0.8786 | Dice macro = 0.9343
Calculando métricas de validación...
IoU por clase : [0.95681334 0.78975844 0.8330696  0.7277438  0.6432073  0.9206832 ]
Dice por clase: [0.97793007 0.88253075 0.90893394 0.84242094 0.7828681  0.9587039 ]
mIoU macro = 0.8119 | Dice macro = 0.8922

--- Epoch 55/200 ---
100% 210/210 [00:20<00:00, 10.09it/s, loss=0.0609]
Calculando métricas de entrenamiento...
IoU por clase : [0.96739906 0.85926753 0.8871187  0.8076876  0.8090947  0.9419025 ]
Dice por clase: [0.98342943 0.9243076  0.9401832  0.8936141  0.8944747  0.97008216]
mIoU macro = 0.8787 | Dice macro = 0.9343
Calculando métricas de validación...
IoU por clase : [0.9565085  0.77971387 0.8251317  0.73373616 0.63057816 0.9184498 ]
Dice por clase: [0.9777708  0.87622386 0.9041887  0.84642196 0.7734412  0.95749164]
mIoU macro = 0.8074 | Dice macro = 0.8893

--- Epoch 56/200 ---
100% 210/210 [00:21<00:00,  9.77it/s, loss=0.0697]
Calculando métricas de entrenamiento...
IoU por clase : [0.9658304  0.8531011  0.8783523  0.8028828  0.79893637 0.94083685]
Dice por clase: [0.9826182  0.920728   0.935237   0.89066553 0.88823193 0.9695167 ]
mIoU macro = 0.8733 | Dice macro = 0.9312
Calculando métricas de validación...
IoU por clase : [0.9552632  0.78226566 0.8194281  0.7294825  0.63092667 0.91695815]
Dice por clase: [0.9771198  0.87783283 0.90075344 0.8435847  0.7737033  0.9566804 ]
mIoU macro = 0.8057 | Dice macro = 0.8883

--- Epoch 57/200 ---
100% 210/210 [00:21<00:00,  9.61it/s, loss=0.0595]
Calculando métricas de entrenamiento...
IoU por clase : [0.96677256 0.8532756  0.8776073  0.80716646 0.80541426 0.9410436 ]
Dice por clase: [0.9831056 0.9208297 0.9348145 0.8932951 0.892221  0.9696264]
mIoU macro = 0.8752 | Dice macro = 0.9323
Calculando métricas de validación...
IoU por clase : [0.95640856 0.7858159  0.80198914 0.73013836 0.5942228  0.91888034]
Dice por clase: [0.9777186 0.8800637 0.8901154 0.8440231 0.7454702 0.9577255]
mIoU macro = 0.7979 | Dice macro = 0.8825

--- Epoch 58/200 ---
100% 210/210 [00:21<00:00,  9.79it/s, loss=0.0677]
Calculando métricas de entrenamiento...
IoU por clase : [0.9654889  0.84625494 0.8705478  0.784486   0.79284483 0.93825257]
Dice por clase: [0.9824415  0.916726   0.9307945  0.87922907 0.8844545  0.96814275]
mIoU macro = 0.8663 | Dice macro = 0.9270
Calculando métricas de validación...
IoU por clase : [0.95520544 0.7878964  0.8045141  0.7159401  0.592175   0.91706276]
Dice por clase: [0.9770896  0.88136697 0.89166844 0.8344582  0.74385667 0.95673734]
mIoU macro = 0.7955 | Dice macro = 0.8809

--- Epoch 59/200 ---
100% 210/210 [00:21<00:00,  9.63it/s, loss=0.0616]
Calculando métricas de entrenamiento...
IoU por clase : [0.9674949 0.8588513 0.8849667 0.8027605 0.8184841 0.9418914]
Dice por clase: [0.98347896 0.9240667  0.9389733  0.89059025 0.90018284 0.97007626]
mIoU macro = 0.8791 | Dice macro = 0.9346
Calculando métricas de validación...
IoU por clase : [0.9565703  0.7889726  0.83293307 0.73419327 0.64854354 0.9185489 ]
Dice por clase: [0.97780323 0.8820399  0.9088527  0.846726   0.7868079  0.95754546]
mIoU macro = 0.8133 | Dice macro = 0.8933

--- Epoch 60/200 ---
100% 210/210 [00:21<00:00,  9.71it/s, loss=0.0559]
Calculando métricas de entrenamiento...
IoU por clase : [0.9674282  0.85895205 0.8882933  0.79915273 0.81699324 0.9426942 ]
Dice por clase: [0.9834445  0.924125   0.9408425  0.8883656  0.8992804  0.97050184]
mIoU macro = 0.8789 | Dice macro = 0.9344
Calculando métricas de validación...
IoU por clase : [0.9563039 0.7850645 0.8376573 0.7344706 0.6554    0.9176656]
Dice por clase: [0.97766393 0.8795923  0.91165775 0.8469104  0.79183275 0.9570653 ]
mIoU macro = 0.8144 | Dice macro = 0.8941

--- Epoch 61/200 ---
100% 210/210 [00:20<00:00, 10.12it/s, loss=0.0659]
Calculando métricas de entrenamiento...
IoU por clase : [0.9667721  0.8518145  0.8730427  0.7986955  0.7990298  0.94086444]
Dice por clase: [0.9831053  0.9199782  0.9322187  0.88808304 0.8882897  0.96953136]
mIoU macro = 0.8717 | Dice macro = 0.9302
Calculando métricas de validación...
IoU por clase : [0.9564797  0.77356017 0.81725514 0.7374095  0.60393846 0.91489923]
Dice por clase: [0.9777558  0.8723247  0.8994391  0.84886086 0.7530694  0.9555586 ]
mIoU macro = 0.8006 | Dice macro = 0.8845

--- Epoch 62/200 ---
100% 210/210 [00:22<00:00,  9.26it/s, loss=0.0708]
Calculando métricas de entrenamiento...
IoU por clase : [0.9665892  0.8569071  0.8835266  0.781044   0.80626816 0.94108826]
Dice por clase: [0.9830108  0.9229402  0.9381621  0.8770631  0.8927447  0.96965015]
mIoU macro = 0.8726 | Dice macro = 0.9306
Calculando métricas de validación...
IoU por clase : [0.9557627  0.78143835 0.8317129  0.7149825  0.658676   0.917696  ]
Dice por clase: [0.97738105 0.8773117  0.9081258  0.83380735 0.794219   0.95708185]
mIoU macro = 0.8100 | Dice macro = 0.8913

--- Epoch 63/200 ---
100% 210/210 [00:21<00:00,  9.58it/s, loss=0.0666]
Calculando métricas de entrenamiento...
IoU por clase : [0.9678154  0.85815614 0.88452166 0.8037153  0.80592066 0.9424252 ]
Dice por clase: [0.9836445  0.9236642  0.9387227  0.89117754 0.89253163 0.97035927]
mIoU macro = 0.8771 | Dice macro = 0.9333
Calculando métricas de validación...
IoU por clase : [0.9563449 0.789209  0.8095538 0.7345176 0.5939969 0.915482 ]
Dice por clase: [0.97768533 0.8821876  0.8947552  0.84694165 0.74529237 0.95587635]
mIoU macro = 0.7999 | Dice macro = 0.8838

--- Epoch 64/200 ---
100% 210/210 [00:20<00:00, 10.12it/s, loss=0.0668]
Calculando métricas de entrenamiento...
IoU por clase : [0.96828306 0.8621695  0.8866598  0.81225604 0.8165521  0.94382024]
Dice por clase: [0.98388594 0.9259839  0.9399255  0.89640313 0.8990131  0.9710983 ]
mIoU macro = 0.8816 | Dice macro = 0.9361
Calculando métricas de validación...
IoU por clase : [0.9571117  0.78634083 0.8325835  0.7415553  0.66558796 0.9207666 ]
Dice por clase: [0.97808594 0.88039285 0.90864456 0.8516012  0.7992228  0.95874906]
mIoU macro = 0.8173 | Dice macro = 0.8961
🔹 Nuevo mejor mIoU: 0.8173 | Dice: 0.8961  →  guardando modelo…

--- Epoch 65/200 ---
100% 210/210 [00:24<00:00,  8.71it/s, loss=0.0681]
Calculando métricas de entrenamiento...
IoU por clase : [0.96785176 0.86201715 0.89080805 0.80751276 0.8241828  0.94266635]
Dice por clase: [0.9836633  0.92589605 0.94225115 0.8935071  0.90361863 0.9704871 ]
mIoU macro = 0.8825 | Dice macro = 0.9366
Calculando métricas de validación...
IoU por clase : [0.95710886 0.79159474 0.84030116 0.7427793  0.68852514 0.9207718 ]
Dice por clase: [0.9780844  0.8836761  0.91322136 0.85240775 0.81553435 0.9587519 ]
mIoU macro = 0.8235 | Dice macro = 0.9003
🔹 Nuevo mejor mIoU: 0.8235 | Dice: 0.9003  →  guardando modelo…

--- Epoch 66/200 ---
100% 210/210 [00:23<00:00,  8.87it/s, loss=0.0472]
Calculando métricas de entrenamiento...
IoU por clase : [0.967243   0.84799206 0.89043754 0.8100416  0.80948895 0.9428822 ]
Dice por clase: [0.98334885 0.9177443  0.94204384 0.895053   0.89471555 0.97060144]
mIoU macro = 0.8780 | Dice macro = 0.9339
Calculando métricas de validación...
IoU por clase : [0.9561332  0.776781   0.8337992  0.73799026 0.65424997 0.91844475]
Dice por clase: [0.97757477 0.8743689  0.90936804 0.84924555 0.79099286 0.9574889 ]
mIoU macro = 0.8129 | Dice macro = 0.8932

--- Epoch 67/200 ---
100% 210/210 [00:22<00:00,  9.28it/s, loss=0.0637]
Calculando métricas de entrenamiento...
IoU por clase : [0.96916115 0.8644168  0.8941285  0.81446475 0.82321066 0.9445119 ]
Dice por clase: [0.9843391  0.92727846 0.94410545 0.89774656 0.90303403 0.9714643 ]
mIoU macro = 0.8850 | Dice macro = 0.9380
Calculando métricas de validación...
IoU por clase : [0.95782137 0.7970068  0.83411837 0.7437668  0.65742964 0.9214918 ]
Dice por clase: [0.9784564  0.8870382  0.9095578  0.8530576  0.79331225 0.959142  ]
mIoU macro = 0.8186 | Dice macro = 0.8968

--- Epoch 68/200 ---
100% 210/210 [00:22<00:00,  9.41it/s, loss=0.0659]
Calculando métricas de entrenamiento...
IoU por clase : [0.96831596 0.86289823 0.8920141  0.81254923 0.8228061  0.9435596 ]
Dice por clase: [0.983903   0.92640406 0.94292545 0.8965817  0.9027906  0.97096026]
mIoU macro = 0.8837 | Dice macro = 0.9373
Calculando métricas de validación...
IoU por clase : [0.95749557 0.8007312  0.83649707 0.73804414 0.66309756 0.92182875]
Dice por clase: [0.9782863  0.88934004 0.9109702  0.84928125 0.79742473 0.95932454]
mIoU macro = 0.8196 | Dice macro = 0.8974

--- Epoch 69/200 ---
100% 210/210 [00:21<00:00,  9.60it/s, loss=0.0604]
Calculando métricas de entrenamiento...
IoU por clase : [0.9688421  0.8647767  0.8936774  0.81559104 0.82875127 0.9427756 ]
Dice por clase: [0.98417443 0.9274855  0.9438539  0.89843035 0.9063576  0.97054505]
mIoU macro = 0.8857 | Dice macro = 0.9385
Calculando métricas de validación...
IoU por clase : [0.95765007 0.795962   0.8323082  0.7449909  0.66788554 0.91920364]
Dice por clase: [0.9783669 0.8863907 0.9084806 0.8538622 0.8008769 0.9579011]
mIoU macro = 0.8197 | Dice macro = 0.8976

--- Epoch 70/200 ---
100% 210/210 [00:22<00:00,  9.31it/s, loss=0.073]
Calculando métricas de entrenamiento...
IoU por clase : [0.9693749  0.8664591  0.8917547  0.81299525 0.8328554  0.94455016]
Dice por clase: [0.9844494  0.92845225 0.94278044 0.89685315 0.90880644 0.9714844 ]
mIoU macro = 0.8863 | Dice macro = 0.9388
Calculando métricas de validación...
IoU por clase : [0.9579559  0.7948673  0.83642    0.7340217  0.6704329  0.92198366]
Dice por clase: [0.9785266  0.8857115  0.91092455 0.8466119  0.8027056  0.9594084 ]
mIoU macro = 0.8193 | Dice macro = 0.8973

--- Epoch 71/200 ---
100% 210/210 [00:21<00:00,  9.89it/s, loss=0.0716]
Calculando métricas de entrenamiento...
IoU por clase : [0.96955705 0.86482406 0.89497685 0.81554383 0.8310422  0.9438578 ]
Dice por clase: [0.9845433  0.92751276 0.9445782  0.8984017  0.9077259  0.9711182 ]
mIoU macro = 0.8866 | Dice macro = 0.9390
Calculando métricas de validación...
IoU por clase : [0.95753527 0.7926188  0.8396918  0.7370319  0.67796695 0.92014545]
Dice por clase: [0.97830707 0.8843138  0.9128614  0.84861064 0.8080814  0.95841223]
mIoU macro = 0.8208 | Dice macro = 0.8984

--- Epoch 72/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.077]
Calculando métricas de entrenamiento...
IoU por clase : [0.9694513 0.8552233 0.8969685 0.8173768 0.8297412 0.9413429]
Dice por clase: [0.9844887  0.9219626  0.9456862  0.89951277 0.9069492  0.9697853 ]
mIoU macro = 0.8850 | Dice macro = 0.9381
Calculando métricas de validación...
IoU por clase : [0.95799553 0.7559866  0.8400932  0.74582267 0.6700775  0.9115921 ]
Dice por clase: [0.9785473  0.8610391  0.9130985  0.85440826 0.8024508  0.9537517 ]
mIoU macro = 0.8136 | Dice macro = 0.8939

--- Epoch 73/200 ---
100% 210/210 [00:22<00:00,  9.29it/s, loss=0.101]
Calculando métricas de entrenamiento...
IoU por clase : [0.96890044 0.864744   0.88892955 0.8152762  0.8255718  0.9440098 ]
Dice por clase: [0.9842046  0.9274667  0.9411993  0.8982393  0.90445286 0.9711986 ]
mIoU macro = 0.8846 | Dice macro = 0.9378
Calculando métricas de validación...
IoU por clase : [0.9577789  0.79498553 0.8280851  0.7359497  0.6466778  0.9220444 ]
Dice por clase: [0.9784341  0.8857849  0.90595907 0.8478929  0.7854333  0.9594413 ]
mIoU macro = 0.8143 | Dice macro = 0.8938

--- Epoch 74/200 ---
100% 210/210 [00:21<00:00,  9.98it/s, loss=0.0494]
Calculando métricas de entrenamiento...
IoU por clase : [0.9685917  0.86317474 0.89351106 0.81455064 0.8297175  0.94354385]
Dice por clase: [0.98404527 0.9265634  0.9437611  0.8977987  0.9069351  0.970952  ]
mIoU macro = 0.8855 | Dice macro = 0.9383
Calculando métricas de validación...
IoU por clase : [0.9569581  0.78587586 0.8298619  0.7387608  0.6484635  0.92032236]
Dice por clase: [0.97800565 0.8801013  0.90702134 0.8497556  0.786749   0.9585082 ]
mIoU macro = 0.8134 | Dice macro = 0.8934

--- Epoch 75/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0678]
Calculando métricas de entrenamiento...
IoU por clase : [0.9695648  0.8624429  0.89273983 0.8194107  0.830779   0.9444663 ]
Dice por clase: [0.98454726 0.92614156 0.9433307  0.90074295 0.9075689  0.97144014]
mIoU macro = 0.8866 | Dice macro = 0.9390
Calculando métricas de validación...
IoU por clase : [0.9579524  0.7981597  0.83423465 0.74382174 0.6643544  0.92149943]
Dice por clase: [0.9785246  0.88775176 0.90962696 0.85309374 0.79833287 0.9591462 ]
mIoU macro = 0.8200 | Dice macro = 0.8977

--- Epoch 76/200 ---
100% 210/210 [00:22<00:00,  9.44it/s, loss=0.0688]
Calculando métricas de entrenamiento...
IoU por clase : [0.9677498 0.8593704 0.8843013 0.8115956 0.8033742 0.9416393]
Dice por clase: [0.98361063 0.9243671  0.93859863 0.89600086 0.8909678  0.9699426 ]
mIoU macro = 0.8780 | Dice macro = 0.9339
Calculando métricas de validación...
IoU por clase : [0.9561894  0.79087156 0.81840646 0.713963   0.59574604 0.91818744]
Dice por clase: [0.9776041  0.8832253  0.9001359  0.8331136  0.74666774 0.95734906]
mIoU macro = 0.7989 | Dice macro = 0.8830

--- Epoch 77/200 ---
100% 210/210 [00:20<00:00, 10.05it/s, loss=0.0384]
Calculando métricas de entrenamiento...
IoU por clase : [0.9689389  0.8618668  0.88976514 0.81402856 0.8256861  0.9445112 ]
Dice por clase: [0.9842245  0.92580926 0.94166744 0.8974815  0.9045214  0.9714638 ]
mIoU macro = 0.8841 | Dice macro = 0.9375
Calculando métricas de validación...
IoU por clase : [0.95741594 0.78108245 0.8357961  0.73102105 0.6783834  0.91948247]
Dice por clase: [0.9782447  0.87708735 0.9105544  0.84461254 0.80837715 0.95805246]
mIoU macro = 0.8172 | Dice macro = 0.8962

--- Epoch 78/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0624]
Calculando métricas de entrenamiento...
IoU por clase : [0.9689803 0.8661388 0.8947109 0.8151002 0.8280216 0.9445317]
Dice por clase: [0.9842458  0.9282684  0.94443    0.89813244 0.90592104 0.97147477]
mIoU macro = 0.8862 | Dice macro = 0.9387
Calculando métricas de validación...
IoU por clase : [0.957323   0.7911559  0.8317893  0.73366636 0.6683275  0.9204904 ]
Dice por clase: [0.97819626 0.8834026  0.90817136 0.84637547 0.8011946  0.9585993 ]
mIoU macro = 0.8171 | Dice macro = 0.8960

--- Epoch 79/200 ---
100% 210/210 [00:22<00:00,  9.52it/s, loss=0.0762]
Calculando métricas de entrenamiento...
IoU por clase : [0.969573   0.86955565 0.8882837  0.81942457 0.83305323 0.94553304]
Dice por clase: [0.9845514  0.9302271  0.94083714 0.90075135 0.9089242  0.9720041 ]
mIoU macro = 0.8876 | Dice macro = 0.9395
Calculando métricas de validación...
IoU por clase : [0.95797956 0.7909903  0.8221875  0.73379034 0.6603294  0.92065364]
Dice por clase: [0.9785388  0.88329935 0.9024181  0.84645796 0.79541975 0.95868784]
mIoU macro = 0.8143 | Dice macro = 0.8941

--- Epoch 80/200 ---
100% 210/210 [00:20<00:00, 10.08it/s, loss=0.0665]
Calculando métricas de entrenamiento...
IoU por clase : [0.9690729  0.86600584 0.88853836 0.8150822  0.8310872  0.9445757 ]
Dice por clase: [0.9842936  0.928192   0.94097996 0.89812154 0.9077527  0.9714981 ]
mIoU macro = 0.8857 | Dice macro = 0.9385
Calculando métricas de validación...
IoU por clase : [0.9579848  0.7889751  0.8394962  0.73910207 0.677      0.91868734]
Dice por clase: [0.9785416  0.88204145 0.9127458  0.84998125 0.80739415 0.9576207 ]
mIoU macro = 0.8202 | Dice macro = 0.8981

--- Epoch 81/200 ---
100% 210/210 [00:21<00:00,  9.59it/s, loss=0.0592]
Calculando métricas de entrenamiento...
IoU por clase : [0.9692505  0.8625372  0.8950959  0.81250757 0.8265148  0.9423619 ]
Dice por clase: [0.9843852  0.9261959  0.9446444  0.8965563  0.90501845 0.97032577]
mIoU macro = 0.8847 | Dice macro = 0.9379
Calculando métricas de validación...
IoU por clase : [0.95748824 0.77298    0.8371155  0.7369548  0.6521141  0.9126183 ]
Dice por clase: [0.97828245 0.8719557  0.91133684 0.84855956 0.78942984 0.95431304]
mIoU macro = 0.8115 | Dice macro = 0.8923

--- Epoch 82/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0579]
Calculando métricas de entrenamiento...
IoU por clase : [0.970354   0.86966056 0.8981832  0.82274836 0.83877593 0.94590724]
Dice por clase: [0.984954   0.9302871  0.94636095 0.90275586 0.9123199  0.97220176]
mIoU macro = 0.8909 | Dice macro = 0.9415
Calculando métricas de validación...
IoU por clase : [0.95833254 0.79461074 0.8376179  0.74646705 0.6746765  0.92053777]
Dice por clase: [0.97872293 0.8855522  0.91163445 0.8548309  0.8057395  0.958625  ]
mIoU macro = 0.8220 | Dice macro = 0.8992

--- Epoch 83/200 ---
100% 210/210 [00:20<00:00, 10.03it/s, loss=0.058]
Calculando métricas de entrenamiento...
IoU por clase : [0.9697364 0.8679864 0.8894465 0.8213451 0.8297985 0.9457298]
Dice por clase: [0.98463565 0.9293284  0.941489   0.9019104  0.9069835  0.97210807]
mIoU macro = 0.8873 | Dice macro = 0.9394
Calculando métricas de validación...
IoU por clase : [0.9578282  0.79475725 0.8269224  0.73404753 0.66165334 0.9205654 ]
Dice por clase: [0.9784599  0.8856431  0.90526277 0.8466291  0.7963795  0.95864   ]
mIoU macro = 0.8160 | Dice macro = 0.8952

--- Epoch 84/200 ---
100% 210/210 [00:21<00:00,  9.55it/s, loss=0.0528]
Calculando métricas de entrenamiento...
IoU por clase : [0.97002435 0.86697793 0.8949438  0.8235868  0.83078504 0.94606805]
Dice por clase: [0.9847841  0.9287501  0.9445597  0.9032603  0.90757245 0.9722867 ]
mIoU macro = 0.8887 | Dice macro = 0.9402
Calculando métricas de validación...
IoU por clase : [0.9583745  0.7934432  0.8335812  0.7367305  0.6757472  0.91989666]
Dice por clase: [0.97874486 0.88482666 0.9092384  0.8484109  0.8065026  0.9582773 ]
mIoU macro = 0.8196 | Dice macro = 0.8977

--- Epoch 85/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.0812]
Calculando métricas de entrenamiento...
IoU por clase : [0.9697963  0.87035125 0.8890152  0.81382316 0.8343961  0.9455195 ]
Dice por clase: [0.9846666 0.9306821 0.9412473 0.8973567 0.9097229 0.9719969]
mIoU macro = 0.8872 | Dice macro = 0.9393
Calculando métricas de validación...
IoU por clase : [0.95836425 0.79687345 0.81644773 0.73288715 0.6549199  0.92024654]
Dice por clase: [0.9787395  0.88695556 0.89894986 0.8458567  0.79148227 0.95846707]
mIoU macro = 0.8133 | Dice macro = 0.8934

--- Epoch 86/200 ---
100% 210/210 [00:20<00:00, 10.05it/s, loss=0.0726]
Calculando métricas de entrenamiento...
IoU por clase : [0.9698858  0.86673915 0.89306855 0.8156082  0.8311235  0.94541246]
Dice por clase: [0.9847127 0.928613  0.9435142 0.8984408 0.9077744 0.9719404]
mIoU macro = 0.8870 | Dice macro = 0.9392
Calculando métricas de validación...
IoU por clase : [0.9578442  0.7933395  0.82984984 0.73196065 0.6356807  0.9207607 ]
Dice por clase: [0.97846824 0.88476217 0.90701413 0.84523934 0.77726746 0.9587459 ]
mIoU macro = 0.8116 | Dice macro = 0.8919

--- Epoch 87/200 ---
100% 210/210 [00:22<00:00,  9.49it/s, loss=0.0601]
Calculando métricas de entrenamiento...
IoU por clase : [0.9696239  0.87038666 0.89664304 0.817852   0.83858204 0.9467017 ]
Dice por clase: [0.98457766 0.9307024  0.9455053  0.8998004  0.9122052  0.97262126]
mIoU macro = 0.8900 | Dice macro = 0.9409
Calculando métricas de validación...
IoU por clase : [0.9576057  0.7935426  0.8416631  0.7394295  0.66184074 0.91967434]
Dice por clase: [0.9783438  0.8848885  0.91402507 0.8501977  0.7965152  0.9581566 ]
mIoU macro = 0.8190 | Dice macro = 0.8970

--- Epoch 88/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0732]
Calculando métricas de entrenamiento...
IoU por clase : [0.9693867  0.8658935  0.83488786 0.82415766 0.6736773  0.94452804]
Dice por clase: [0.9844554  0.92812747 0.91001517 0.90360355 0.80502653 0.97147286]
mIoU macro = 0.8521 | Dice macro = 0.9171
Calculando métricas de validación...
IoU por clase : [0.95765394 0.7919819  0.7752015  0.74096614 0.55606467 0.91943485]
Dice por clase: [0.978369   0.8839173  0.8733673  0.85121256 0.71470636 0.9580266 ]
mIoU macro = 0.7902 | Dice macro = 0.8766

--- Epoch 89/200 ---
100% 210/210 [00:21<00:00,  9.90it/s, loss=0.0835]
Calculando métricas de entrenamiento...
IoU por clase : [0.96830523 0.8416451  0.8852425  0.81627655 0.7964286  0.9412333 ]
Dice por clase: [0.9838974  0.91401446 0.9391285  0.8988461  0.88667995 0.9697271 ]
mIoU macro = 0.8749 | Dice macro = 0.9320
Calculando métricas de validación...
IoU por clase : [0.95595753 0.7648859  0.8222578  0.7352775  0.619109   0.91512024]
Dice por clase: [0.97748286 0.8667823  0.90246046 0.84744656 0.7647527  0.9556792 ]
mIoU macro = 0.8021 | Dice macro = 0.8858

--- Epoch 90/200 ---
100% 210/210 [00:22<00:00,  9.32it/s, loss=0.0905]
Calculando métricas de entrenamiento...
IoU por clase : [0.97004604 0.8693164  0.8965772  0.8233924  0.8349423  0.9459196 ]
Dice por clase: [0.98479533 0.9300902  0.94546866 0.9031434  0.9100475  0.97220826]
mIoU macro = 0.8900 | Dice macro = 0.9410
Calculando métricas de validación...
IoU por clase : [0.9579192  0.79452634 0.83557725 0.74623525 0.67159903 0.91966695]
Dice por clase: [0.97850734 0.8854998  0.9104245  0.8546789  0.8035408  0.9581526 ]
mIoU macro = 0.8209 | Dice macro = 0.8985

--- Epoch 91/200 ---
100% 210/210 [00:22<00:00,  9.46it/s, loss=0.0535]
Calculando métricas de entrenamiento...
IoU por clase : [0.9700802  0.87152314 0.8978055  0.8237561  0.839725   0.946444  ]
Dice por clase: [0.9848129  0.9313517  0.94615126 0.90336215 0.912881   0.9724852 ]
mIoU macro = 0.8916 | Dice macro = 0.9418
Calculando métricas de validación...
IoU por clase : [0.9577955  0.79578495 0.8447383  0.73995394 0.6773166  0.92134273]
Dice por clase: [0.97844285 0.8862809  0.9158354  0.8505443  0.8076193  0.9590613 ]
mIoU macro = 0.8228 | Dice macro = 0.8996

--- Epoch 92/200 ---
100% 210/210 [00:21<00:00,  9.99it/s, loss=0.0663]
Calculando métricas de entrenamiento...
IoU por clase : [0.9705498  0.8728093  0.90093946 0.82530767 0.84258384 0.9452446 ]
Dice por clase: [0.9850549  0.93208563 0.9478886  0.9042943  0.9145677  0.97185165]
mIoU macro = 0.8929 | Dice macro = 0.9426
Calculando métricas de validación...
IoU por clase : [0.95825577 0.8003743  0.8441879  0.73843473 0.66438293 0.92165214]
Dice por clase: [0.97868294 0.8891199  0.91551185 0.8495398  0.7983535  0.9592289 ]
mIoU macro = 0.8212 | Dice macro = 0.8984

--- Epoch 93/200 ---
100% 210/210 [00:22<00:00,  9.42it/s, loss=0.0524]
Calculando métricas de entrenamiento...
IoU por clase : [0.97068155 0.87191844 0.9006421  0.8259804  0.8414298  0.94714093]
Dice por clase: [0.9851227  0.9315774  0.94772404 0.904698   0.9138875  0.972853  ]
mIoU macro = 0.8930 | Dice macro = 0.9426
Calculando métricas de validación...
IoU por clase : [0.9583117  0.80069965 0.8399544  0.7483034  0.665316   0.9218226 ]
Dice por clase: [0.97871214 0.8893206  0.91301656 0.8560338  0.7990267  0.95932126]
mIoU macro = 0.8224 | Dice macro = 0.8992

--- Epoch 94/200 ---
100% 210/210 [00:21<00:00,  9.59it/s, loss=0.0556]
Calculando métricas de entrenamiento...
IoU por clase : [0.97112215 0.8740301  0.90232813 0.82967585 0.8466605  0.94666755]
Dice por clase: [0.9853495  0.9327813  0.9486567  0.90691024 0.9169639  0.9726032 ]
mIoU macro = 0.8951 | Dice macro = 0.9439
Calculando métricas de validación...
IoU por clase : [0.95827305 0.8045268  0.84743744 0.74441016 0.6769323  0.9205341 ]
Dice por clase: [0.978692   0.8916762  0.9174194  0.85348064 0.807346   0.958623  ]
mIoU macro = 0.8254 | Dice macro = 0.9012
🔹 Nuevo mejor mIoU: 0.8254 | Dice: 0.9012  →  guardando modelo…

--- Epoch 95/200 ---
100% 210/210 [00:23<00:00,  8.87it/s, loss=0.048]
Calculando métricas de entrenamiento...
IoU por clase : [0.9713011  0.87553895 0.90382606 0.8298192  0.8474817  0.94758207]
Dice por clase: [0.9854416  0.9336398  0.9494839  0.90699583 0.9174453  0.97308564]
mIoU macro = 0.8959 | Dice macro = 0.9443
Calculando métricas de validación...
IoU por clase : [0.9587675 0.7992552 0.8510449 0.7453237 0.6867497 0.9229865]
Dice por clase: [0.9789498 0.8884289 0.9195292 0.8540808 0.8142876 0.9599511]
mIoU macro = 0.8274 | Dice macro = 0.9025
🔹 Nuevo mejor mIoU: 0.8274 | Dice: 0.9025  →  guardando modelo…

--- Epoch 96/200 ---
100% 210/210 [00:23<00:00,  8.86it/s, loss=0.054]
Calculando métricas de entrenamiento...
IoU por clase : [0.97044945 0.8719737  0.90203804 0.8258134  0.8472347  0.94708097]
Dice por clase: [0.9850031  0.9316089  0.94849634 0.9045978  0.9173005  0.97282135]
mIoU macro = 0.8941 | Dice macro = 0.9433
Calculando métricas de validación...
IoU por clase : [0.95829475 0.8020048  0.83892393 0.7559413  0.6741626  0.9224746 ]
Dice por clase: [0.9787033  0.89012504 0.91240746 0.8610097  0.8053729  0.9596742 ]
mIoU macro = 0.8253 | Dice macro = 0.9012

--- Epoch 97/200 ---
100% 210/210 [00:20<00:00, 10.04it/s, loss=0.0574]
Calculando métricas de entrenamiento...
IoU por clase : [0.97141814 0.8743679  0.90032876 0.83077    0.8491715  0.9454911 ]
Dice por clase: [0.9855019  0.9329736  0.94755054 0.9075635  0.91843456 0.9719819 ]
mIoU macro = 0.8953 | Dice macro = 0.9440
Calculando métricas de validación...
IoU por clase : [0.9587788 0.7994548 0.8487036 0.7426203 0.6936375 0.9212684]
Dice por clase: [0.9789557  0.88855225 0.91816086 0.852303   0.81910974 0.95902103]
mIoU macro = 0.8274 | Dice macro = 0.9027
🔹 Nuevo mejor mIoU: 0.8274 | Dice: 0.9027  →  guardando modelo…

--- Epoch 98/200 ---
100% 210/210 [00:23<00:00,  8.78it/s, loss=0.0472]
Calculando métricas de entrenamiento...
IoU por clase : [0.97128606 0.87621224 0.9002679  0.8279151  0.84812456 0.94764835]
Dice por clase: [0.98543394 0.93402255 0.9475168  0.90585726 0.9178219  0.97312057]
mIoU macro = 0.8952 | Dice macro = 0.9440
Calculando métricas de validación...
IoU por clase : [0.9587195  0.79862106 0.83741325 0.75735784 0.6801826  0.9211076 ]
Dice por clase: [0.9789248  0.888037   0.91151327 0.86192787 0.80965316 0.9589339 ]
mIoU macro = 0.8256 | Dice macro = 0.9015

--- Epoch 99/200 ---
100% 210/210 [00:21<00:00,  9.98it/s, loss=0.0467]
Calculando métricas de entrenamiento...
IoU por clase : [0.9714015  0.87364084 0.90412825 0.8311841  0.84859353 0.94725126]
Dice por clase: [0.9854933  0.93255955 0.9496506  0.9078105  0.9180964  0.9729112 ]
mIoU macro = 0.8960 | Dice macro = 0.9444
Calculando métricas de validación...
IoU por clase : [0.9587132  0.8030973  0.8449373  0.75185674 0.69314843 0.9228819 ]
Dice por clase: [0.9789215  0.89079756 0.9159523  0.85835415 0.8187686  0.95989454]
mIoU macro = 0.8291 | Dice macro = 0.9038
🔹 Nuevo mejor mIoU: 0.8291 | Dice: 0.9038  →  guardando modelo…

--- Epoch 100/200 ---
100% 210/210 [00:24<00:00,  8.74it/s, loss=0.0648]
Calculando métricas de entrenamiento...
IoU por clase : [0.9707649 0.8750683 0.898919  0.8275728 0.8362379 0.9469586]
Dice por clase: [0.9851656  0.9333722  0.9467692  0.90565234 0.91081655 0.9727568 ]
mIoU macro = 0.8926 | Dice macro = 0.9424
Calculando métricas de validación...
IoU por clase : [0.9586806  0.79797417 0.843172   0.7444118  0.67590016 0.9217364 ]
Dice por clase: [0.97890455 0.88763696 0.9149141  0.85348177 0.8066115  0.95927453]
mIoU macro = 0.8236 | Dice macro = 0.9001

--- Epoch 101/200 ---
100% 210/210 [00:22<00:00,  9.45it/s, loss=0.0467]
Calculando métricas de entrenamiento...
IoU por clase : [0.97053254 0.86696196 0.8976354  0.82295704 0.8331184  0.94704545]
Dice por clase: [0.9850459  0.92874086 0.9460568  0.90288144 0.90896297 0.9728026 ]
mIoU macro = 0.8897 | Dice macro = 0.9407
Calculando métricas de validación...
IoU por clase : [0.9579184  0.7865105  0.8268775  0.72841716 0.6712987  0.9200294 ]
Dice por clase: [0.978507   0.8804992  0.9052358  0.8428719  0.80332583 0.9583493 ]
mIoU macro = 0.8152 | Dice macro = 0.8948

--- Epoch 102/200 ---
100% 210/210 [00:22<00:00,  9.14it/s, loss=0.0817]
Calculando métricas de entrenamiento...
IoU por clase : [0.97147864 0.8753638  0.9028828  0.82951057 0.8381227  0.94775647]
Dice por clase: [0.985533   0.9335403  0.94896317 0.9068115  0.91193336 0.9731776 ]
mIoU macro = 0.8942 | Dice macro = 0.9433
Calculando métricas de validación...
IoU por clase : [0.9587936  0.793031   0.8422385  0.75635993 0.70976985 0.92125666]
Dice por clase: [0.9789634  0.8845703  0.9143642  0.8612813  0.83025193 0.95901465]
mIoU macro = 0.8302 | Dice macro = 0.9047
🔹 Nuevo mejor mIoU: 0.8302 | Dice: 0.9047  →  guardando modelo…

--- Epoch 103/200 ---
100% 210/210 [00:24<00:00,  8.66it/s, loss=0.0576]
Calculando métricas de entrenamiento...
IoU por clase : [0.9710846  0.87554663 0.9028922  0.82388943 0.8441784  0.9471064 ]
Dice por clase: [0.9853302  0.93364424 0.9489683  0.9034423  0.9155062  0.9728347 ]
mIoU macro = 0.8941 | Dice macro = 0.9433
Calculando métricas de validación...
IoU por clase : [0.95859414 0.79647404 0.8361631  0.7417966  0.6534335  0.9210815 ]
Dice por clase: [0.9788594  0.8867081  0.91077214 0.8517603  0.79039586 0.95891976]
mIoU macro = 0.8179 | Dice macro = 0.8962

--- Epoch 104/200 ---
100% 210/210 [00:21<00:00,  9.97it/s, loss=0.0532]
Calculando métricas de entrenamiento...
IoU por clase : [0.97165513 0.87695843 0.9038771  0.83445466 0.8494317  0.9471377 ]
Dice por clase: [0.98562384 0.9344463  0.949512   0.90975773 0.91858673 0.9728513 ]
mIoU macro = 0.8973 | Dice macro = 0.9451
Calculando métricas de validación...
IoU por clase : [0.95926327 0.80045605 0.8559908  0.7577137  0.7201933  0.92246646]
Dice por clase: [0.9792081  0.88917035 0.92240846 0.86215824 0.83734    0.95966977]
mIoU macro = 0.8360 | Dice macro = 0.9083
🔹 Nuevo mejor mIoU: 0.8360 | Dice: 0.9083  →  guardando modelo…

--- Epoch 105/200 ---
100% 210/210 [00:24<00:00,  8.65it/s, loss=0.0518]
Calculando métricas de entrenamiento...
IoU por clase : [0.97135407 0.87412983 0.90347975 0.81849474 0.8428947  0.94573075]
Dice por clase: [0.98546886 0.9328381  0.9492928  0.9001893  0.9147508  0.97210854]
mIoU macro = 0.8927 | Dice macro = 0.9424
Calculando métricas de validación...
IoU por clase : [0.95879316 0.79796165 0.8465556  0.73746204 0.6890869  0.9197218 ]
Dice por clase: [0.97896314 0.8876292  0.91690236 0.8488957  0.81592834 0.9581824 ]
mIoU macro = 0.8249 | Dice macro = 0.9011

--- Epoch 106/200 ---
100% 210/210 [00:21<00:00,  9.80it/s, loss=0.0494]
Calculando métricas de entrenamiento...
IoU por clase : [0.9718258  0.87905407 0.9056268  0.83555347 0.85293573 0.94795847]
Dice por clase: [0.9857116  0.9356347  0.9504765  0.9104104  0.92063177 0.973284  ]
mIoU macro = 0.8988 | Dice macro = 0.9460
Calculando métricas de validación...
IoU por clase : [0.9591053  0.798117   0.84537095 0.75731105 0.6900967  0.9223083 ]
Dice por clase: [0.9791258 0.8877253 0.9162071 0.8618976 0.8166357 0.9595842]
mIoU macro = 0.8287 | Dice macro = 0.9035

--- Epoch 107/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0414]
Calculando métricas de entrenamiento...
IoU por clase : [0.9722282  0.87916833 0.90512    0.83455604 0.85086405 0.94896215]
Dice por clase: [0.9859186  0.9356994  0.9501974  0.909818   0.91942364 0.9738128 ]
mIoU macro = 0.8985 | Dice macro = 0.9458
Calculando métricas de validación...
IoU por clase : [0.9596382  0.80058736 0.8529483  0.75372106 0.69147867 0.9229523 ]
Dice por clase: [0.9794034  0.88925135 0.9206391  0.85956776 0.81760263 0.9599326 ]
mIoU macro = 0.8302 | Dice macro = 0.9044

--- Epoch 108/200 ---
100% 210/210 [00:22<00:00,  9.46it/s, loss=0.0507]
Calculando métricas de entrenamiento...
IoU por clase : [0.9721193  0.8798086  0.9054223  0.83014876 0.8503653  0.94909793]
Dice por clase: [0.9858625  0.9360619  0.9503639  0.90719265 0.91913235 0.9738843 ]
mIoU macro = 0.8978 | Dice macro = 0.9454
Calculando métricas de validación...
IoU por clase : [0.95915836 0.7970141  0.8512832  0.74399614 0.6892009  0.9235443 ]
Dice por clase: [0.9791535  0.8870427  0.91966826 0.8532085  0.81600815 0.9602527 ]
mIoU macro = 0.8274 | Dice macro = 0.9026

--- Epoch 109/200 ---
100% 210/210 [00:21<00:00,  9.91it/s, loss=0.065]
Calculando métricas de entrenamiento...
IoU por clase : [0.97168046 0.87407225 0.903027   0.82844263 0.84702647 0.9481208 ]
Dice por clase: [0.98563683 0.9328053  0.94904274 0.906173   0.91717845 0.9733696 ]
mIoU macro = 0.8954 | Dice macro = 0.9440
Calculando métricas de validación...
IoU por clase : [0.9591176  0.79800403 0.8323521  0.7434527  0.66829723 0.9226141 ]
Dice por clase: [0.9791323  0.88765544 0.90850675 0.85285103 0.80117285 0.95974964]
mIoU macro = 0.8206 | Dice macro = 0.8982

--- Epoch 110/200 ---
100% 210/210 [00:22<00:00,  9.47it/s, loss=0.0773]
Calculando métricas de entrenamiento...
IoU por clase : [0.97135663 0.8738902  0.9019349  0.82674557 0.8363537  0.9469359 ]
Dice por clase: [0.9854703  0.93270165 0.9484393  0.9051568  0.9108852  0.9727449 ]
mIoU macro = 0.8929 | Dice macro = 0.9426
Calculando métricas de validación...
IoU por clase : [0.9587159  0.79791874 0.85149825 0.7389986  0.6854751  0.92277646]
Dice por clase: [0.9789229 0.8876026 0.9197937 0.8499128 0.813391  0.9598375]
mIoU macro = 0.8259 | Dice macro = 0.9016

--- Epoch 111/200 ---
100% 210/210 [00:22<00:00,  9.33it/s, loss=0.0621]
Calculando métricas de entrenamiento...
IoU por clase : [0.9709241 0.8716324 0.9019797 0.8246002 0.8467851 0.9438513]
Dice por clase: [0.98524755 0.9314141  0.94846404 0.90386945 0.917037   0.9711147 ]
mIoU macro = 0.8933 | Dice macro = 0.9429
Calculando métricas de validación...
IoU por clase : [0.95862293 0.7919303  0.837677   0.73555535 0.67642945 0.91951746]
Dice por clase: [0.9788743  0.88388515 0.91166943 0.8476311  0.80698824 0.9580715 ]
mIoU macro = 0.8200 | Dice macro = 0.8979

--- Epoch 112/200 ---
100% 210/210 [00:21<00:00,  9.95it/s, loss=0.0629]
Calculando métricas de entrenamiento...
IoU por clase : [0.97133315 0.8684515  0.90291214 0.83122635 0.8479525  0.9487059 ]
Dice por clase: [0.98545814 0.9295949  0.9489793  0.9078357  0.9177211  0.9736779 ]
mIoU macro = 0.8951 | Dice macro = 0.9439
Calculando métricas de validación...
IoU por clase : [0.95879835 0.78875935 0.846015   0.751685   0.6958725  0.92233205]
Dice por clase: [0.9789659  0.8819066  0.91658515 0.8582422  0.8206661  0.959597  ]
mIoU macro = 0.8272 | Dice macro = 0.9027

--- Epoch 113/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0531]
Calculando métricas de entrenamiento...
IoU por clase : [0.9723629  0.8788123  0.9069283  0.8359588  0.85351396 0.94897276]
Dice por clase: [0.9859878  0.9354977  0.95119286 0.9106509  0.9209685  0.97381836]
mIoU macro = 0.8994 | Dice macro = 0.9464
Calculando métricas de validación...
IoU por clase : [0.9594332  0.79828733 0.85307413 0.74832225 0.7052485  0.9229163 ]
Dice por clase: [0.9792967  0.8878307  0.92071235 0.85604614 0.8271504  0.95991313]
mIoU macro = 0.8312 | Dice macro = 0.9052

--- Epoch 114/200 ---
100% 210/210 [00:22<00:00,  9.46it/s, loss=0.0628]
Calculando métricas de entrenamiento...
IoU por clase : [0.9720802  0.87986875 0.906883   0.8385682  0.8522486  0.9473356 ]
Dice por clase: [0.98584247 0.93609595 0.95116794 0.912197   0.92023134 0.9729557 ]
mIoU macro = 0.8995 | Dice macro = 0.9464
Calculando métricas de validación...
IoU por clase : [0.959383   0.80098706 0.8615386  0.7450207  0.69951713 0.9221078 ]
Dice por clase: [0.9792705  0.8894978  0.9256199  0.8538818  0.82319516 0.95947564]
mIoU macro = 0.8314 | Dice macro = 0.9052

--- Epoch 115/200 ---
100% 210/210 [00:21<00:00,  9.97it/s, loss=0.048]
Calculando métricas de entrenamiento...
IoU por clase : [0.97237366 0.87963945 0.9093385  0.8378952  0.85132134 0.949448  ]
Dice por clase: [0.9859933  0.93596613 0.9525168  0.91179866 0.91969055 0.9740686 ]
mIoU macro = 0.9000 | Dice macro = 0.9467
Calculando métricas de validación...
IoU por clase : [0.95938545 0.79834414 0.8597339  0.75080305 0.7038087  0.9230175 ]
Dice por clase: [0.9792718  0.8878658  0.9245773  0.8576671  0.82615936 0.95996785]
mIoU macro = 0.8325 | Dice macro = 0.9059

--- Epoch 116/200 ---
100% 210/210 [00:22<00:00,  9.41it/s, loss=0.0442]
Calculando métricas de entrenamiento...
IoU por clase : [0.9728172  0.8808595  0.90952086 0.8390012  0.8563017  0.94918877]
Dice por clase: [0.9862213  0.93665636 0.9526168  0.9124531  0.92258894 0.97393215]
mIoU macro = 0.9013 | Dice macro = 0.9474
Calculando métricas de validación...
IoU por clase : [0.9597994  0.7983503  0.85596704 0.7479487  0.68127084 0.923568  ]
Dice por clase: [0.97948736 0.8878696  0.92239463 0.8558017  0.8104237  0.9602655 ]
mIoU macro = 0.8278 | Dice macro = 0.9027

--- Epoch 117/200 ---
100% 210/210 [00:22<00:00,  9.44it/s, loss=0.0509]
Calculando métricas de entrenamiento...
IoU por clase : [0.9721228  0.8764663  0.90480554 0.83506954 0.7390197  0.92889816]
Dice por clase: [0.98586434 0.93416685 0.95002407 0.91012305 0.84992677 0.96313864]
mIoU macro = 0.8761 | Dice macro = 0.9322
Calculando métricas de validación...
IoU por clase : [0.9589543  0.7997311  0.84332997 0.7452464  0.5095771  0.879473  ]
Dice por clase: [0.9790472  0.8887229  0.91500705 0.85403    0.6751256  0.9358719 ]
mIoU macro = 0.7894 | Dice macro = 0.8746

--- Epoch 118/200 ---
100% 210/210 [00:21<00:00,  9.95it/s, loss=0.0457]
Calculando métricas de entrenamiento...
IoU por clase : [0.97198725 0.8784192  0.9040086  0.8302871  0.8482652  0.94856423]
Dice por clase: [0.9857947  0.93527496 0.9495846  0.90727526 0.9179042  0.9736032 ]
mIoU macro = 0.8969 | Dice macro = 0.9449
Calculando métricas de validación...
IoU por clase : [0.95934016 0.7989252  0.84852874 0.7468495  0.68910384 0.92247415]
Dice por clase: [0.97924817 0.8882251  0.91805845 0.8550817  0.81594014 0.9596739 ]
mIoU macro = 0.8275 | Dice macro = 0.9027

--- Epoch 119/200 ---
100% 210/210 [00:21<00:00,  9.61it/s, loss=0.055]
Calculando métricas de entrenamiento...
IoU por clase : [0.9726289  0.8780083  0.908224   0.84037286 0.8561664  0.9494459 ]
Dice por clase: [0.9861246  0.93504196 0.951905   0.9132637  0.9225104  0.97406745]
mIoU macro = 0.9008 | Dice macro = 0.9472
Calculando métricas de validación...
IoU por clase : [0.9597756  0.7952129  0.85556877 0.7606605  0.70796573 0.92209   ]
Dice por clase: [0.9794751  0.885926   0.92216337 0.86406267 0.8290163  0.959466  ]
mIoU macro = 0.8335 | Dice macro = 0.9067

--- Epoch 120/200 ---
100% 210/210 [00:22<00:00,  9.40it/s, loss=0.0458]
Calculando métricas de entrenamiento...
IoU por clase : [0.97171396 0.8801454  0.90688086 0.8313702  0.85220885 0.9489575 ]
Dice por clase: [0.9856541  0.9362525  0.95116675 0.9079215  0.92020816 0.9738104 ]
mIoU macro = 0.8985 | Dice macro = 0.9458
Calculando métricas de validación...
IoU por clase : [0.9586913 0.8028925 0.8435833 0.7537967 0.664463  0.9219944]
Dice por clase: [0.97891    0.89067155 0.9151562  0.85961694 0.79841125 0.95941424]
mIoU macro = 0.8242 | Dice macro = 0.9004

--- Epoch 121/200 ---
100% 210/210 [00:21<00:00,  9.98it/s, loss=0.0492]
Calculando métricas de entrenamiento...
IoU por clase : [0.9720436  0.8165142  0.9046966  0.8380463  0.85115325 0.9321222 ]
Dice por clase: [0.9858237  0.89899015 0.949964   0.9118881  0.91959244 0.96486884]
mIoU macro = 0.8858 | Dice macro = 0.9385
Calculando métricas de validación...
IoU por clase : [0.95939946 0.7226381  0.8371959  0.76242185 0.6788923  0.9032506 ]
Dice por clase: [0.9792791  0.83899003 0.91138446 0.8651979  0.80873835 0.9491662 ]
mIoU macro = 0.8106 | Dice macro = 0.8921

--- Epoch 122/200 ---
100% 210/210 [00:22<00:00,  9.50it/s, loss=0.0447]
Calculando métricas de entrenamiento...
IoU por clase : [0.97293043 0.8825617  0.90979475 0.8338467  0.85721594 0.95039237]
Dice por clase: [0.98627955 0.9376178  0.952767   0.9093963  0.9231193  0.9745653 ]
mIoU macro = 0.9011 | Dice macro = 0.9473
Calculando métricas de validación...
IoU por clase : [0.95992094 0.8035966  0.8493727  0.7522701  0.6943944  0.9227976 ]
Dice por clase: [0.97955066 0.8911046  0.9185522  0.85862345 0.8196373  0.95984894]
mIoU macro = 0.8304 | Dice macro = 0.9046

--- Epoch 123/200 ---
100% 210/210 [00:22<00:00,  9.35it/s, loss=0.0395]
Calculando métricas de entrenamiento...
IoU por clase : [0.97334534 0.88231546 0.9114406  0.84290594 0.8576512  0.9493151 ]
Dice por clase: [0.9864927  0.93747884 0.9536688  0.91475743 0.9233716  0.9739986 ]
mIoU macro = 0.9028 | Dice macro = 0.9483
Calculando métricas de validación...
IoU por clase : [0.9602203  0.8033457  0.85093665 0.766482   0.7061284  0.92214334]
Dice por clase: [0.9797066  0.89095026 0.919466   0.8678062  0.8277553  0.9594949 ]
mIoU macro = 0.8349 | Dice macro = 0.9075

--- Epoch 124/200 ---
100% 210/210 [00:21<00:00,  9.96it/s, loss=0.0673]
Calculando métricas de entrenamiento...
IoU por clase : [0.9733034  0.8837047  0.9109185  0.8382045  0.8593327  0.95063055]
Dice por clase: [0.9864711  0.93826246 0.95338285 0.91198176 0.92434525 0.9746905 ]
mIoU macro = 0.9027 | Dice macro = 0.9482
Calculando métricas de validación...
IoU por clase : [0.9605196  0.80548006 0.8471116  0.76001525 0.6916151  0.9226489 ]
Dice por clase: [0.9798623  0.8922614  0.9172284  0.8636462  0.81769794 0.9597685 ]
mIoU macro = 0.8312 | Dice macro = 0.9051

--- Epoch 125/200 ---
100% 210/210 [00:22<00:00,  9.47it/s, loss=0.0403]
Calculando métricas de entrenamiento...
IoU por clase : [0.9725607  0.86320436 0.90798795 0.8291447  0.85068554 0.9471341 ]
Dice por clase: [0.98608947 0.9265804  0.9517754  0.9065928  0.9193194  0.9728494 ]
mIoU macro = 0.8951 | Dice macro = 0.9439
Calculando métricas de validación...
IoU por clase : [0.9593456  0.7678282  0.84887975 0.7535807  0.70961416 0.9170914 ]
Dice por clase: [0.9792511  0.8686684  0.9182639  0.8594765  0.8301454  0.95675296]
mIoU macro = 0.8261 | Dice macro = 0.9021

--- Epoch 126/200 ---
100% 210/210 [00:22<00:00,  9.47it/s, loss=0.0745]
Calculando métricas de entrenamiento...
IoU por clase : [0.9722035  0.878734   0.90453565 0.83115786 0.84839195 0.9486633 ]
Dice por clase: [0.9859058  0.93545336 0.9498753  0.9077949  0.9179784  0.97365546]
mIoU macro = 0.8973 | Dice macro = 0.9451
Calculando métricas de validación...
IoU por clase : [0.9595104  0.7964816  0.85108805 0.75233424 0.6824426  0.9200396 ]
Dice por clase: [0.9793369  0.8867128  0.91955435 0.8586652  0.8112522  0.95835483]
mIoU macro = 0.8270 | Dice macro = 0.9023

--- Epoch 127/200 ---
100% 210/210 [00:21<00:00,  9.94it/s, loss=0.0471]
Calculando métricas de entrenamiento...
IoU por clase : [0.97287893 0.8804504  0.91068536 0.83353865 0.85502225 0.9498218 ]
Dice por clase: [0.9862531  0.93642503 0.9532552  0.90921307 0.9218458  0.9742653 ]
mIoU macro = 0.9004 | Dice macro = 0.9469
Calculando métricas de validación...
IoU por clase : [0.959707   0.80140686 0.85141814 0.74866146 0.6888121  0.9230609 ]
Dice por clase: [0.9794393  0.8897567  0.919747   0.85626805 0.81573564 0.95999134]
mIoU macro = 0.8288 | Dice macro = 0.9035

--- Epoch 128/200 ---
100% 210/210 [00:22<00:00,  9.41it/s, loss=0.0447]
Calculando métricas de entrenamiento...
IoU por clase : [0.9735155  0.8799615  0.9108399  0.84177494 0.8606281  0.95036715]
Dice por clase: [0.9865801 0.9361484 0.9533398 0.914091  0.9250942 0.9745521]
mIoU macro = 0.9028 | Dice macro = 0.9483
Calculando métricas de validación...
IoU por clase : [0.96024    0.7953322  0.842593   0.75280017 0.6785276  0.9239645 ]
Dice por clase: [0.9797167  0.88600004 0.91457313 0.8589686  0.8084795  0.9604798 ]
mIoU macro = 0.8256 | Dice macro = 0.9014

--- Epoch 129/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.043]
Calculando métricas de entrenamiento...
IoU por clase : [0.973124  0.8811145 0.9108417 0.8446353 0.8622056 0.9498305]
Dice por clase: [0.98637897 0.9368005  0.9533408  0.9157749  0.92600477 0.97426975]
mIoU macro = 0.9036 | Dice macro = 0.9488
Calculando métricas de validación...
IoU por clase : [0.9601431  0.80508685 0.8509907  0.75626796 0.67675257 0.92326313]
Dice por clase: [0.9796664  0.89202005 0.91949755 0.8612216  0.8072182  0.9601007 ]
mIoU macro = 0.8288 | Dice macro = 0.9033

--- Epoch 130/200 ---
100% 210/210 [00:21<00:00,  9.90it/s, loss=0.0445]
Calculando métricas de entrenamiento...
IoU por clase : [0.9734394  0.88413864 0.91303605 0.84444624 0.86484873 0.94989103]
Dice por clase: [0.9865409  0.93850696 0.9545414  0.9156637  0.92752695 0.97430164]
mIoU macro = 0.9050 | Dice macro = 0.9495
Calculando métricas de validación...
IoU por clase : [0.9602336  0.8004885  0.86095613 0.7538865  0.697785   0.9241302 ]
Dice por clase: [0.9797134  0.8891903  0.9252836  0.8596754  0.82199454 0.96056926]
mIoU macro = 0.8329 | Dice macro = 0.9061

--- Epoch 131/200 ---
100% 210/210 [00:22<00:00,  9.35it/s, loss=0.0615]
Calculando métricas de entrenamiento...
IoU por clase : [0.97211283 0.8819533  0.89778614 0.8309906  0.84709615 0.94964117]
Dice por clase: [0.9858593  0.9372744  0.94614047 0.9076951  0.91721934 0.9741702 ]
mIoU macro = 0.8966 | Dice macro = 0.9447
Calculando métricas de validación...
IoU por clase : [0.958961  0.7996724 0.8319531 0.741992  0.6737648 0.9223118]
Dice por clase: [0.97905064 0.88868666 0.908269   0.8518891  0.805089   0.959586  ]
mIoU macro = 0.8214 | Dice macro = 0.8988

--- Epoch 132/200 ---
100% 210/210 [00:22<00:00,  9.36it/s, loss=0.0564]
Calculando métricas de entrenamiento...
IoU por clase : [0.9721102  0.87871635 0.9038761  0.8304268  0.8485166  0.9488339 ]
Dice por clase: [0.9858579  0.93544334 0.94951147 0.90735865 0.91805136 0.9737453 ]
mIoU macro = 0.8971 | Dice macro = 0.9450
Calculando métricas de validación...
IoU por clase : [0.9592656  0.79319507 0.8288971  0.73105556 0.6646222  0.92138684]
Dice por clase: [0.9792094  0.8846724  0.9064448  0.8446356  0.79852617 0.9590852 ]
mIoU macro = 0.8164 | Dice macro = 0.8954

--- Epoch 133/200 ---
100% 210/210 [00:21<00:00,  9.83it/s, loss=0.0557]
Calculando métricas de entrenamiento...
IoU por clase : [0.9732755  0.882885   0.9090219  0.84279335 0.85614324 0.9502632 ]
Dice por clase: [0.98645675 0.9378002  0.9523431  0.9146911  0.922497   0.9744974 ]
mIoU macro = 0.9024 | Dice macro = 0.9480
Calculando métricas de validación...
IoU por clase : [0.96022546 0.8005182  0.8454572  0.7548357  0.6905114  0.9230481 ]
Dice por clase: [0.9797092  0.8892087  0.91625774 0.8602922  0.81692606 0.9599844 ]
mIoU macro = 0.8291 | Dice macro = 0.9037

--- Epoch 134/200 ---
100% 210/210 [00:22<00:00,  9.29it/s, loss=0.0463]
Calculando métricas de entrenamiento...
IoU por clase : [0.9736467  0.88609236 0.91228974 0.8449519  0.86303294 0.95115125]
Dice por clase: [0.98664737 0.93960655 0.9541334  0.9159609  0.92648166 0.97496414]
mIoU macro = 0.9052 | Dice macro = 0.9496
Calculando métricas de validación...
IoU por clase : [0.9604154  0.7961123  0.8429234  0.75451577 0.6805092  0.9241049 ]
Dice por clase: [0.9798081  0.88648385 0.9147677  0.86008435 0.80988455 0.9605557 ]
mIoU macro = 0.8264 | Dice macro = 0.9019

--- Epoch 135/200 ---
100% 210/210 [00:22<00:00,  9.39it/s, loss=0.052]
Calculando métricas de entrenamiento...
IoU por clase : [0.97361726 0.88510877 0.91212076 0.8427186  0.8635635  0.9505596 ]
Dice por clase: [0.98663235 0.93905324 0.95404094 0.9146471  0.92678726 0.97465324]
mIoU macro = 0.9046 | Dice macro = 0.9493
Calculando métricas de validación...
IoU por clase : [0.96047103 0.79781526 0.8572337  0.7572211  0.71120805 0.9244201 ]
Dice por clase: [0.979837   0.8875387  0.9231296  0.8618393  0.83123505 0.9607259 ]
mIoU macro = 0.8347 | Dice macro = 0.9074

--- Epoch 136/200 ---
100% 210/210 [00:20<00:00, 10.01it/s, loss=0.0486]
Calculando métricas de entrenamiento...
IoU por clase : [0.9735622 0.8827892 0.9111651 0.8447071 0.8629232 0.950362 ]
Dice por clase: [0.986604   0.93774617 0.9535179  0.915817   0.9264184  0.9745493 ]
mIoU macro = 0.9043 | Dice macro = 0.9491
Calculando métricas de validación...
IoU por clase : [0.9601789  0.797734   0.8586022  0.7514338  0.69194794 0.92306197]
Dice por clase: [0.9796849  0.88748837 0.92392254 0.8580785  0.8179305  0.9599919 ]
mIoU macro = 0.8305 | Dice macro = 0.9045

--- Epoch 137/200 ---
100% 210/210 [00:22<00:00,  9.48it/s, loss=0.0566]
Calculando métricas de entrenamiento...
IoU por clase : [0.9737444  0.8830995  0.9134707  0.84479856 0.861825   0.94999814]
Dice por clase: [0.98669755 0.9379212  0.95477885 0.9158708  0.9257852  0.974358  ]
mIoU macro = 0.9045 | Dice macro = 0.9492
Calculando métricas de validación...
IoU por clase : [0.9603114  0.80087596 0.85098195 0.75774896 0.6889087  0.92407215]
Dice por clase: [0.979754   0.88942933 0.9194924  0.8621811  0.81580335 0.9605379 ]
mIoU macro = 0.8305 | Dice macro = 0.9045

--- Epoch 138/200 ---
100% 210/210 [00:22<00:00,  9.47it/s, loss=0.061]
Calculando métricas de entrenamiento...
IoU por clase : [0.97384375 0.88553965 0.9138444  0.84384817 0.86066    0.95041126]
Dice por clase: [0.9867486  0.9392957  0.95498294 0.915312   0.9251126  0.9745752 ]
mIoU macro = 0.9047 | Dice macro = 0.9493
Calculando métricas de validación...
IoU por clase : [0.9607044  0.80498356 0.8582358  0.75896454 0.7152261  0.924346  ]
Dice por clase: [0.9799585  0.8919567  0.9237103  0.86296743 0.833973   0.9606859 ]
mIoU macro = 0.8371 | Dice macro = 0.9089
🔹 Nuevo mejor mIoU: 0.8371 | Dice: 0.9089  →  guardando modelo…

--- Epoch 139/200 ---
100% 210/210 [00:23<00:00,  9.07it/s, loss=0.0778]
Calculando métricas de entrenamiento...
IoU por clase : [0.97300124 0.88310075 0.9080305  0.84393257 0.85091186 0.9486867 ]
Dice por clase: [0.98631597 0.93792194 0.95179874 0.91536164 0.91945153 0.9736678 ]
mIoU macro = 0.9013 | Dice macro = 0.9474
Calculando métricas de validación...
IoU por clase : [0.9601051 0.8057705 0.8518219 0.7611009 0.7235875 0.9232527]
Dice por clase: [0.97964656 0.89243954 0.91998255 0.86434674 0.8396296  0.96009505]
mIoU macro = 0.8376 | Dice macro = 0.9094
🔹 Nuevo mejor mIoU: 0.8376 | Dice: 0.9094  →  guardando modelo…

--- Epoch 140/200 ---
100% 210/210 [00:25<00:00,  8.16it/s, loss=0.0562]
Calculando métricas de entrenamiento...
IoU por clase : [0.9735338  0.8848773  0.91060036 0.84644186 0.86337054 0.9500051 ]
Dice por clase: [0.9865894  0.938923   0.9532086  0.91683567 0.9266762  0.97436166]
mIoU macro = 0.9048 | Dice macro = 0.9494
Calculando métricas de validación...
IoU por clase : [0.9603143  0.8037702  0.85427254 0.75166535 0.6988849  0.9230403 ]
Dice por clase: [0.97975546 0.8912113  0.9214099  0.8582294  0.82275724 0.9599802 ]
mIoU macro = 0.8320 | Dice macro = 0.9056

--- Epoch 141/200 ---
100% 210/210 [00:22<00:00,  9.54it/s, loss=0.0694]
Calculando métricas de entrenamiento...
IoU por clase : [0.9737289  0.8833038  0.91134316 0.84527713 0.8611685  0.94956267]
Dice por clase: [0.98668957 0.93803644 0.9536154  0.916152   0.9254063  0.97412884]
mIoU macro = 0.9041 | Dice macro = 0.9490
Calculando métricas de validación...
IoU por clase : [0.9601176  0.79962534 0.85937154 0.7501024  0.71044004 0.9231102 ]
Dice por clase: [0.97965306 0.88865757 0.9243677  0.85720974 0.8307103  0.960018  ]
mIoU macro = 0.8338 | Dice macro = 0.9068

--- Epoch 142/200 ---
100% 210/210 [00:23<00:00,  9.10it/s, loss=0.0717]
Calculando métricas de entrenamiento...
IoU por clase : [0.97339374 0.88646144 0.9114699  0.8430639  0.86275595 0.95090497]
Dice por clase: [0.9865174  0.93981403 0.9536848  0.9148504  0.92632204 0.97483474]
mIoU macro = 0.9047 | Dice macro = 0.9493
Calculando métricas de validación...
IoU por clase : [0.9602739  0.794554   0.85938966 0.7501022  0.6965527  0.92426807]
Dice por clase: [0.9797345  0.88551694 0.9243782  0.8572096  0.8211389  0.96064377]
mIoU macro = 0.8309 | Dice macro = 0.9048

--- Epoch 143/200 ---
100% 210/210 [00:22<00:00,  9.54it/s, loss=0.0505]
Calculando métricas de entrenamiento...
IoU por clase : [0.9744249  0.8884073  0.9157927  0.8487036  0.8659351  0.95170766]
Dice por clase: [0.9870468  0.94090647 0.9560457  0.91816086 0.92815137 0.9752564 ]
mIoU macro = 0.9075 | Dice macro = 0.9509
Calculando métricas de validación...
IoU por clase : [0.9607191  0.8041182  0.8660594  0.75013226 0.70521563 0.9242534 ]
Dice por clase: [0.9799661  0.8914252  0.9282228  0.85722923 0.8271278  0.96063584]
mIoU macro = 0.8351 | Dice macro = 0.9074

--- Epoch 144/200 ---
100% 210/210 [00:22<00:00,  9.24it/s, loss=0.0424]
Calculando métricas de entrenamiento...
IoU por clase : [0.9741853 0.888535  0.9125591 0.8483753 0.8704759 0.9519446]
Dice por clase: [0.9869239 0.9409781 0.9542807 0.9179687 0.9307534 0.9753807]
mIoU macro = 0.9077 | Dice macro = 0.9510
Calculando métricas de validación...
IoU por clase : [0.960689   0.8044859  0.8593537  0.7528674  0.70270604 0.9246371 ]
Dice por clase: [0.9799504  0.8916511  0.9243575  0.85901237 0.82539916 0.960843  ]
mIoU macro = 0.8341 | Dice macro = 0.9069

--- Epoch 145/200 ---
100% 210/210 [00:22<00:00,  9.37it/s, loss=0.063]
Calculando métricas de entrenamiento...
IoU por clase : [0.97418064 0.8888578  0.91506064 0.847424   0.8702447  0.9520544 ]
Dice por clase: [0.9869215  0.94115907 0.95564663 0.91741145 0.9306212  0.97543836]
mIoU macro = 0.9080 | Dice macro = 0.9512
Calculando métricas de validación...
IoU por clase : [0.96037316 0.8050866  0.8599429  0.75701874 0.71959215 0.92437345]
Dice por clase: [0.9797861 0.8920199 0.9246982 0.8617082 0.8369335 0.9607007]
mIoU macro = 0.8377 | Dice macro = 0.9093
🔹 Nuevo mejor mIoU: 0.8377 | Dice: 0.9093  →  guardando modelo…

--- Epoch 146/200 ---
100% 210/210 [00:23<00:00,  8.96it/s, loss=0.0522]
Calculando métricas de entrenamiento...
IoU por clase : [0.9738466  0.88682973 0.9155959  0.8458116  0.8677989  0.9509115 ]
Dice por clase: [0.98675    0.9400209  0.95593846 0.9164658  0.9292209  0.9748382 ]
mIoU macro = 0.9068 | Dice macro = 0.9505
Calculando métricas de validación...
IoU por clase : [0.9602309  0.80177695 0.8598408  0.75806445 0.71691316 0.92447895]
Dice por clase: [0.979712   0.88998467 0.92463917 0.8623853  0.8351187  0.9607577 ]
mIoU macro = 0.8369 | Dice macro = 0.9088

--- Epoch 147/200 ---
100% 210/210 [00:22<00:00,  9.27it/s, loss=0.0429]
Calculando métricas de entrenamiento...
IoU por clase : [0.97450143 0.88849455 0.916375   0.8459094  0.8696049  0.95216775]
Dice por clase: [0.98708606 0.9409554  0.9563629  0.9165232  0.93025523 0.97549784]
mIoU macro = 0.9078 | Dice macro = 0.9511
Calculando métricas de validación...
IoU por clase : [0.9607004  0.8027905  0.861816   0.75566417 0.71570796 0.9245227 ]
Dice por clase: [0.9799563  0.8906088  0.92578    0.86083    0.83430046 0.9607813 ]
mIoU macro = 0.8369 | Dice macro = 0.9087

--- Epoch 148/200 ---
100% 210/210 [00:21<00:00,  9.67it/s, loss=0.0714]
Calculando métricas de entrenamiento...
IoU por clase : [0.9743072  0.8883639  0.91571057 0.84858066 0.87063795 0.9512803 ]
Dice por clase: [0.9869864  0.9408821  0.956001   0.91808885 0.93084604 0.975032  ]
mIoU macro = 0.9081 | Dice macro = 0.9513
Calculando métricas de validación...
IoU por clase : [0.9606915  0.80532163 0.8589437  0.75628585 0.7132711  0.923481  ]
Dice por clase: [0.97995174 0.8921642  0.9241202  0.86123323 0.8326424  0.9602185 ]
mIoU macro = 0.8363 | Dice macro = 0.9084

--- Epoch 149/200 ---
100% 210/210 [00:21<00:00,  9.95it/s, loss=0.096]
Calculando métricas de entrenamiento...
IoU por clase : [0.97127044 0.87518847 0.7201585  0.8276408  0.4894259  0.9466953 ]
Dice por clase: [0.9854259  0.93344057 0.83731645 0.90569305 0.65720075 0.97261786]
mIoU macro = 0.8051 | Dice macro = 0.8819
Calculando métricas de validación...
IoU por clase : [0.9580059  0.7929932  0.6317284  0.7468448  0.41449445 0.91778344]
Dice por clase: [0.97855264 0.8845468  0.7743058  0.85507864 0.58606726 0.95712936]
mIoU macro = 0.7436 | Dice macro = 0.8393

--- Epoch 150/200 ---
100% 210/210 [00:22<00:00,  9.43it/s, loss=0.0472]
Calculando métricas de entrenamiento...
IoU por clase : [0.97382534 0.8837457  0.9119175  0.8458023  0.8635723  0.9508702 ]
Dice por clase: [0.9867391  0.9382855  0.9539298  0.91646034 0.9267924  0.9748165 ]
mIoU macro = 0.9050 | Dice macro = 0.9495
Calculando métricas de validación...
IoU por clase : [0.9603227  0.79670763 0.85334146 0.75527185 0.70951635 0.9230258 ]
Dice por clase: [0.9797598  0.88685286 0.92086804 0.8605754  0.8300785  0.9599723 ]
mIoU macro = 0.8330 | Dice macro = 0.9064

--- Epoch 151/200 ---
100% 210/210 [00:22<00:00,  9.41it/s, loss=0.056]
Calculando métricas de entrenamiento...
IoU por clase : [0.972086   0.88396615 0.9020313  0.8315072  0.850285   0.9498999 ]
Dice por clase: [0.9858454  0.9384098  0.9484926  0.9080032  0.91908544 0.97430634]
mIoU macro = 0.8983 | Dice macro = 0.9457
Calculando métricas de validación...
IoU por clase : [0.9590469  0.79874116 0.84743655 0.74894047 0.7018897  0.9206262 ]
Dice por clase: [0.97909546 0.8881113  0.91741884 0.8564505  0.82483566 0.958673  ]
mIoU macro = 0.8294 | Dice macro = 0.9041

--- Epoch 152/200 ---
100% 210/210 [00:21<00:00,  9.58it/s, loss=0.0631]
Calculando métricas de entrenamiento...
IoU por clase : [0.9732665  0.885452   0.9047119  0.8445909  0.85896564 0.950773  ]
Dice por clase: [0.9864522  0.93924636 0.94997245 0.9157487  0.9241329  0.9747654 ]
mIoU macro = 0.9030 | Dice macro = 0.9484
Calculando métricas de validación...
IoU por clase : [0.96020806 0.80224806 0.84903765 0.7496662  0.70191723 0.9215423 ]
Dice por clase: [0.97970015 0.8902749  0.91835624 0.85692483 0.82485473 0.9591694 ]
mIoU macro = 0.8308 | Dice macro = 0.9049

--- Epoch 153/200 ---
100% 210/210 [00:23<00:00,  9.11it/s, loss=0.0483]
Calculando métricas de entrenamiento...
IoU por clase : [0.97387105 0.887201   0.91054267 0.846257   0.86320156 0.9512335 ]
Dice por clase: [0.9867625  0.9402295  0.953177   0.9167272  0.9265788  0.97500736]
mIoU macro = 0.9054 | Dice macro = 0.9497
Calculando métricas de validación...
IoU por clase : [0.9601112  0.7937699  0.85202026 0.7511078  0.6978793  0.9215967 ]
Dice por clase: [0.9796498  0.8850298  0.9200982  0.85786587 0.82206    0.9591989 ]
mIoU macro = 0.8294 | Dice macro = 0.9040

--- Epoch 154/200 ---
100% 210/210 [00:21<00:00,  9.69it/s, loss=0.0578]
Calculando métricas de entrenamiento...
IoU por clase : [0.9742352  0.88912696 0.91237336 0.8490112  0.8651607  0.9521186 ]
Dice por clase: [0.9869495  0.9413099  0.9541791  0.91834074 0.92770636 0.9754721 ]
mIoU macro = 0.9070 | Dice macro = 0.9507
Calculando métricas de validación...
IoU por clase : [0.96038914 0.8049303  0.8524338  0.75247884 0.7061906  0.9228193 ]
Dice por clase: [0.9797944  0.89192396 0.9203393  0.8587594  0.827798   0.9598607 ]
mIoU macro = 0.8332 | Dice macro = 0.9064

--- Epoch 155/200 ---
100% 210/210 [00:22<00:00,  9.35it/s, loss=0.0492]
Calculando métricas de entrenamiento...
IoU por clase : [0.9744248 0.8895124 0.9151726 0.8510648 0.869941  0.9519796]
Dice por clase: [0.9870468  0.9415259  0.95570767 0.9195408  0.9304475  0.97539914]
mIoU macro = 0.9087 | Dice macro = 0.9516
Calculando métricas de validación...
IoU por clase : [0.9606216  0.8049539  0.843998   0.75085336 0.68025255 0.92266357]
Dice por clase: [0.9799154  0.89193845 0.91540015 0.8576999  0.80970275 0.9597764 ]
mIoU macro = 0.8272 | Dice macro = 0.9024

--- Epoch 156/200 ---
100% 210/210 [00:23<00:00,  9.12it/s, loss=0.0409]
Calculando métricas de entrenamiento...
IoU por clase : [0.97442913 0.8886517  0.9144117  0.8500548  0.86728996 0.9517554 ]
Dice por clase: [0.987049   0.9410435  0.95529264 0.9189509  0.92892903 0.9752815 ]
mIoU macro = 0.9078 | Dice macro = 0.9511
Calculando métricas de validación...
IoU por clase : [0.96062315 0.8029892  0.8557451  0.75426394 0.69785637 0.9240192 ]
Dice por clase: [0.9799162  0.890731   0.92226577 0.8599207  0.8220441  0.96050936]
mIoU macro = 0.8326 | Dice macro = 0.9059

--- Epoch 157/200 ---
100% 210/210 [00:21<00:00,  9.82it/s, loss=0.0358]
Calculando métricas de entrenamiento...
IoU por clase : [0.9746198  0.8894114  0.915064   0.8423437  0.8711299  0.95280397]
Dice por clase: [0.9871468  0.94146925 0.9556485  0.9144262  0.9311271  0.9758317 ]
mIoU macro = 0.9076 | Dice macro = 0.9509
Calculando métricas de validación...
IoU por clase : [0.9605786  0.80460167 0.8564154  0.7463125  0.69611365 0.925417  ]
Dice por clase: [0.979893   0.8917222  0.9226549  0.8547296  0.82083374 0.96126395]
mIoU macro = 0.8316 | Dice macro = 0.9052

--- Epoch 158/200 ---
100% 210/210 [00:22<00:00,  9.25it/s, loss=0.0559]
Calculando métricas de entrenamiento...
IoU por clase : [0.9743316  0.88977766 0.91632557 0.84985274 0.8670163  0.95114744]
Dice por clase: [0.986999   0.9416745  0.956336   0.91883284 0.9287721  0.9749621 ]
mIoU macro = 0.9081 | Dice macro = 0.9513
Calculando métricas de validación...
IoU por clase : [0.96033514 0.80397606 0.85852    0.7519574  0.7099305  0.9228913 ]
Dice por clase: [0.9797663 0.8913378 0.9238749 0.8584197 0.8303618 0.9598996]
mIoU macro = 0.8346 | Dice macro = 0.9073

--- Epoch 159/200 ---
100% 210/210 [00:23<00:00,  9.10it/s, loss=0.058]
Calculando métricas de entrenamiento...
IoU por clase : [0.97366726 0.8841229  0.9105615  0.84772265 0.8602989  0.951915  ]
Dice por clase: [0.986658   0.93849814 0.95318735 0.9175865  0.9249039  0.9753652 ]
mIoU macro = 0.9047 | Dice macro = 0.9494
Calculando métricas de validación...
IoU por clase : [0.9597624  0.80197406 0.85596544 0.7432995  0.700746   0.9228834 ]
Dice por clase: [0.9794681  0.89010614 0.92239374 0.8527502  0.8240454  0.9598953 ]
mIoU macro = 0.8308 | Dice macro = 0.9048

--- Epoch 160/200 ---
100% 210/210 [00:21<00:00,  9.69it/s, loss=0.0592]
Calculando métricas de entrenamiento...
IoU por clase : [0.9737499  0.8882975  0.9081124  0.8486271  0.86319584 0.9503527 ]
Dice por clase: [0.9867004  0.9408449  0.95184374 0.91811603 0.92657554 0.97454447]
mIoU macro = 0.9054 | Dice macro = 0.9498
Calculando métricas de validación...
IoU por clase : [0.96014154 0.80657595 0.84528273 0.75848603 0.70413464 0.92333686]
Dice por clase: [0.97966546 0.89293337 0.9161553  0.86265796 0.8263838  0.9601405 ]
mIoU macro = 0.8330 | Dice macro = 0.9063

--- Epoch 161/200 ---
100% 210/210 [00:22<00:00,  9.31it/s, loss=0.0476]
Calculando métricas de entrenamiento...
IoU por clase : [0.97455364 0.8898028  0.9151117  0.8495934  0.8699634  0.95245916]
Dice por clase: [0.9871129  0.94168854 0.9556745  0.91868126 0.93046033 0.9756508 ]
mIoU macro = 0.9086 | Dice macro = 0.9515
Calculando métricas de validación...
IoU por clase : [0.9607337  0.810863   0.8626638  0.75670266 0.69659644 0.9243564 ]
Dice por clase: [0.97997373 0.89555424 0.92626894 0.8615034  0.8211693  0.96069145]
mIoU macro = 0.8353 | Dice macro = 0.9075

--- Epoch 162/200 ---
100% 210/210 [00:22<00:00,  9.32it/s, loss=0.0473]
Calculando métricas de entrenamiento...
IoU por clase : [0.9745855  0.8908891  0.91613567 0.850484   0.8676202  0.95131534]
Dice por clase: [0.98712915 0.9422965  0.9562326  0.9192017  0.92911845 0.97505033]
mIoU macro = 0.9085 | Dice macro = 0.9515
Calculando métricas de validación...
IoU por clase : [0.9607561  0.8074688  0.86456835 0.75563973 0.72014093 0.9239213 ]
Dice por clase: [0.97998536 0.8934802  0.9273657  0.8608141  0.8373046  0.96045643]
mIoU macro = 0.8387 | Dice macro = 0.9099
🔹 Nuevo mejor mIoU: 0.8387 | Dice: 0.9099  →  guardando modelo…

--- Epoch 163/200 ---
100% 210/210 [00:24<00:00,  8.63it/s, loss=0.0501]
Calculando métricas de entrenamiento...
IoU por clase : [0.9749146  0.8899788  0.91738135 0.8535645  0.87447464 0.9525449 ]
Dice por clase: [0.987298   0.94178706 0.95691067 0.92099786 0.93303436 0.9756958 ]
mIoU macro = 0.9105 | Dice macro = 0.9526
Calculando métricas de validación...
IoU por clase : [0.9611143  0.81064767 0.8668337  0.75768626 0.7237169  0.92582977]
Dice por clase: [0.9801716  0.8954229  0.9286673  0.8621405  0.8397167  0.96148664]
mIoU macro = 0.8410 | Dice macro = 0.9113
🔹 Nuevo mejor mIoU: 0.8410 | Dice: 0.9113  →  guardando modelo…

--- Epoch 164/200 ---
100% 210/210 [00:24<00:00,  8.48it/s, loss=0.0642]
Calculando métricas de entrenamiento...
IoU por clase : [0.97502303 0.8917178  0.9181201  0.8527737  0.8747362  0.9525618 ]
Dice por clase: [0.98735356 0.9427599  0.9573124  0.92053735 0.93318325 0.97570467]
mIoU macro = 0.9108 | Dice macro = 0.9528
Calculando métricas de validación...
IoU por clase : [0.96091694 0.80473983 0.85830045 0.753276   0.7096548  0.9252809 ]
Dice por clase: [0.980069   0.891807   0.9237478  0.85927826 0.8301732  0.9611906 ]
mIoU macro = 0.8354 | Dice macro = 0.9077

--- Epoch 165/200 ---
100% 210/210 [00:21<00:00,  9.60it/s, loss=0.0437]
Calculando métricas de entrenamiento...
IoU por clase : [0.97494346 0.89224553 0.9144583  0.85202384 0.8670505  0.95251226]
Dice por clase: [0.9873128  0.9430547  0.95531803 0.9201003  0.9287917  0.9756787 ]
mIoU macro = 0.9089 | Dice macro = 0.9517
Calculando métricas de validación...
IoU por clase : [0.9610661  0.80701244 0.8446222  0.7521877  0.6970748  0.9252579 ]
Dice por clase: [0.9801466  0.89320076 0.91576713 0.8585698  0.82150155 0.9611781 ]
mIoU macro = 0.8312 | Dice macro = 0.9051

--- Epoch 166/200 ---
100% 210/210 [00:22<00:00,  9.30it/s, loss=0.0475]
Calculando métricas de entrenamiento...
IoU por clase : [0.9738368  0.8894914  0.91646785 0.8503991  0.8643499  0.95017827]
Dice por clase: [0.986745   0.9415141  0.95641345 0.9191521  0.92724    0.9744528 ]
mIoU macro = 0.9075 | Dice macro = 0.9509
Calculando métricas de validación...
IoU por clase : [0.9601535  0.80986804 0.85250354 0.75574344 0.7232702  0.9241405 ]
Dice por clase: [0.9796717  0.89494705 0.92037994 0.8608814  0.8394159  0.96057487]
mIoU macro = 0.8376 | Dice macro = 0.9093

--- Epoch 167/200 ---
100% 210/210 [00:22<00:00,  9.53it/s, loss=0.0421]
Calculando métricas de entrenamiento...
IoU por clase : [0.97516334 0.89263225 0.91750294 0.85332865 0.8699824  0.9528077 ]
Dice por clase: [0.98742557 0.9432707  0.95697683 0.9208606  0.93047124 0.9758336 ]
mIoU macro = 0.9102 | Dice macro = 0.9525
Calculando métricas de validación...
IoU por clase : [0.9612901  0.8024526  0.85391116 0.7561228  0.69951457 0.92513436]
Dice por clase: [0.980263  0.8904008 0.9211997 0.8611275 0.8231934 0.9611115]
mIoU macro = 0.8331 | Dice macro = 0.9062

--- Epoch 168/200 ---
100% 210/210 [00:21<00:00,  9.57it/s, loss=0.0454]
Calculando métricas de entrenamiento...
IoU por clase : [0.97486216 0.8930212  0.91533655 0.8531464  0.8723956  0.9523831 ]
Dice por clase: [0.9872711  0.9434878  0.9557971  0.92075443 0.93184966 0.97561085]
mIoU macro = 0.9102 | Dice macro = 0.9525
Calculando métricas de validación...
IoU por clase : [0.96085644 0.8079419  0.8462287  0.75614464 0.7028545  0.9235545 ]
Dice por clase: [0.9800375  0.89376974 0.9167106  0.8611417  0.82550156 0.9602582 ]
mIoU macro = 0.8329 | Dice macro = 0.9062

--- Epoch 169/200 ---
100% 210/210 [00:22<00:00,  9.19it/s, loss=0.0718]
Calculando métricas de entrenamiento...
IoU por clase : [0.97443527 0.8905793  0.9163744  0.8450469  0.873388   0.9521868 ]
Dice por clase: [0.98705214 0.9421232  0.9563626  0.9160167  0.9324155  0.9755079 ]
mIoU macro = 0.9087 | Dice macro = 0.9516
Calculando métricas de validación...
IoU por clase : [0.9604088 0.8023294 0.8514618 0.7602572 0.7227658 0.9244646]
Dice por clase: [0.97980464 0.89032495 0.91977245 0.8638024  0.8390761  0.9607499 ]
mIoU macro = 0.8369 | Dice macro = 0.9089
"""

# --- Bloque de ejecución ---
if __name__ == "__main__":
    # 1. Extraer los datos del texto
    f1_scores = calcular_f1_score(texto_log)
    
    # Imprimir los scores extraídos para verificación
    for i, scores in enumerate(f1_scores, 1):
        print(f"Época {i} (Validación): {scores}")

    # 2. Generar y guardar el gráfico
    generar_grafico_f1_scores(f1_scores)