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
Analizando distribución: 100% 210/210 [00:05<00:00, 36.25it/s]
Número total de píxeles analizados: 110100480
Distribución final de píxeles por clase:
  Clase 0: 76.1759%  (83870000 píxeles)
  Clase 1: 3.2003%  (3523539 píxeles)
  Clase 2: 4.9403%  (5439327 píxeles)
  Clase 3: 2.7969%  (3079394 píxeles)
  Clase 4: 1.3227%  (1456308 píxeles)
  Clase 5: 11.5639%  (12731912 píxeles)

Pesos de importancia por clase (para CrossEntropy/Focal):
  Clase 0: 0.0456
  Clase 1: 1.0849
  Clase 2: 0.7028
  Clase 3: 1.2414
  Clase 4: 2.6250
  Clase 5: 0.3003
--------------------------------------------------------
model.safetensors: 100% 193M/193M [00:00<00:00, 384MB/s]
Unexpected keys (bn2.bias, bn2.num_batches_tracked, bn2.running_mean, bn2.running_var, bn2.weight, conv_head.weight) found while loading pretrained weights. This may be expected if model is being adapted.

--- Epoch 1/200 ---
  0% 0/210 [00:00<?, ?it/s]/content/Proyecto_Cultivos/train_test3.py:142: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
100% 210/210 [00:28<00:00,  7.45it/s, loss=1.13]
Calculando métricas de entrenamiento...
IoU por clase : [8.92925239e-01 1.19732470e-01 4.03084740e-01 6.66637309e-03
 4.17391304e-04 6.88480533e-01]
Dice por clase: [9.43434237e-01 2.13859066e-01 5.74569345e-01 1.32444537e-02
 8.34434323e-04 8.15503075e-01]
mIoU macro = 0.3519 | Dice macro = 0.4269
Calculando métricas de validación...
IoU por clase : [8.82717254e-01 1.06891424e-01 3.13497047e-01 4.57497428e-03
 7.32419302e-04 6.76755056e-01]
Dice por clase: [0.9377056  0.19313805 0.47734717 0.00910828 0.00146377 0.80721994]
mIoU macro = 0.3309 | Dice macro = 0.4043
🔹 Nuevo mejor mIoU: 0.3309 | Dice: 0.4043  →  guardando modelo…

--- Epoch 2/200 ---
100% 210/210 [00:28<00:00,  7.47it/s, loss=0.855]
Calculando métricas de entrenamiento...
IoU por clase : [9.08365612e-01 2.81864763e-01 5.05002283e-01 9.49234623e-04
 6.86690563e-15 7.96838141e-01]
Dice por clase: [9.51982792e-01 4.39773010e-01 6.71098361e-01 1.89666886e-03
 6.86690563e-15 8.86933690e-01]
mIoU macro = 0.4155 | Dice macro = 0.4919
Calculando métricas de validación...
IoU por clase : [8.95541561e-01 2.06020355e-01 4.19592806e-01 1.25132649e-03
 5.36060816e-06 7.61811261e-01]
Dice por clase: [9.44892562e-01 3.41653197e-01 5.91145298e-01 2.49952525e-03
 1.07211588e-05 8.64804622e-01]
mIoU macro = 0.3807 | Dice macro = 0.4575
🔹 Nuevo mejor mIoU: 0.3807 | Dice: 0.4575  →  guardando modelo…

--- Epoch 3/200 ---
100% 210/210 [00:28<00:00,  7.34it/s, loss=0.635]
Calculando métricas de entrenamiento...
IoU por clase : [9.37060819e-01 3.98071573e-01 6.53529238e-01 7.71767617e-03
 2.29536033e-04 8.45426831e-01]
Dice por clase: [9.67507896e-01 5.69458074e-01 7.90465899e-01 1.53171396e-02
 4.58966717e-04 9.16239882e-01]
mIoU macro = 0.4737 | Dice macro = 0.5432
Calculando métricas de validación...
IoU por clase : [9.30683545e-01 2.93293192e-01 6.04001901e-01 7.26340595e-03
 9.11376662e-05 8.21401132e-01]
Dice por clase: [9.64097454e-01 4.53560251e-01 7.53118685e-01 1.44220586e-02
 1.82258722e-04 9.01944243e-01]
mIoU macro = 0.4428 | Dice macro = 0.5146
🔹 Nuevo mejor mIoU: 0.4428 | Dice: 0.5146  →  guardando modelo…

--- Epoch 4/200 ---
100% 210/210 [00:28<00:00,  7.35it/s, loss=0.401]
Calculando métricas de entrenamiento...
IoU por clase : [0.9454641  0.50093928 0.68538018 0.06763279 0.02996044 0.8998927 ]
Dice por clase: [0.97196767 0.66750106 0.81332412 0.12669673 0.05817785 0.94730897]
mIoU macro = 0.5215 | Dice macro = 0.5975
Calculando métricas de validación...
IoU por clase : [0.9385497  0.35562739 0.66071322 0.0431176  0.00904239 0.88340339]
Dice por clase: [0.96830089 0.52466835 0.79569815 0.08267065 0.01792271 0.9380926 ]
mIoU macro = 0.4817 | Dice macro = 0.5546
🔹 Nuevo mejor mIoU: 0.4817 | Dice: 0.5546  →  guardando modelo…

--- Epoch 5/200 ---
100% 210/210 [00:27<00:00,  7.50it/s, loss=0.353]
Calculando métricas de entrenamiento...
IoU por clase : [9.47307768e-01 6.93778459e-01 6.80852218e-01 4.68770753e-01
 1.26504475e-04 8.94481196e-01]
Dice por clase: [9.72940984e-01 8.19208032e-01 8.10127399e-01 6.38317112e-01
 2.52976947e-04 9.44302005e-01]
mIoU macro = 0.6142 | Dice macro = 0.6975
Calculando métricas de validación...
IoU por clase : [9.40071758e-01 5.79922733e-01 6.64146745e-01 4.83067185e-01
 6.96834229e-05 8.72977383e-01]
Dice por clase: [9.69110296e-01 7.34115309e-01 7.98182909e-01 6.51443427e-01
 1.39357135e-04 9.32181447e-01]
mIoU macro = 0.5900 | Dice macro = 0.6809
🔹 Nuevo mejor mIoU: 0.5900 | Dice: 0.6809  →  guardando modelo…

--- Epoch 6/200 ---
100% 210/210 [00:28<00:00,  7.48it/s, loss=0.353]
Calculando métricas de entrenamiento...
IoU por clase : [0.95283465 0.77478595 0.77544291 0.54279006 0.00528823 0.91440035]
Dice por clase: [0.97584775 0.87310354 0.87352053 0.70364734 0.01052082 0.95528644]
mIoU macro = 0.6609 | Dice macro = 0.7320
Calculando métricas de validación...
IoU por clase : [9.45164653e-01 7.23111471e-01 7.66839511e-01 5.90689778e-01
 8.89727400e-04 8.99745429e-01]
Dice por clase: [0.97180941 0.83930899 0.86803528 0.74268382 0.00177787 0.94722737]
mIoU macro = 0.6544 | Dice macro = 0.7285
🔹 Nuevo mejor mIoU: 0.6544 | Dice: 0.7285  →  guardando modelo…

--- Epoch 7/200 ---
100% 210/210 [00:27<00:00,  7.66it/s, loss=0.428]
Calculando métricas de entrenamiento...
IoU por clase : [9.53246950e-01 7.85897370e-01 7.83249360e-01 5.42249948e-01
 1.59012805e-04 9.13147563e-01]
Dice por clase: [9.76063933e-01 8.80114819e-01 8.78451862e-01 7.03193343e-01
 3.17975047e-04 9.54602332e-01]
mIoU macro = 0.6630 | Dice macro = 0.7321
Calculando métricas de validación...
IoU por clase : [9.45187332e-01 7.44250629e-01 7.65295664e-01 5.89485072e-01
 3.21664076e-05 9.02522416e-01]
Dice por clase: [9.71821394e-01 8.53375790e-01 8.67045311e-01 7.41730869e-01
 6.43307459e-05 9.48764029e-01]
mIoU macro = 0.6578 | Dice macro = 0.7305
🔹 Nuevo mejor mIoU: 0.6578 | Dice: 0.7305  →  guardando modelo…

--- Epoch 8/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.206]
Calculando métricas de entrenamiento...
IoU por clase : [0.94968289 0.78744011 0.7524326  0.58571326 0.05116163 0.90939295]
Dice por clase: [0.97419215 0.88108139 0.85872929 0.73873792 0.09734303 0.95254667]
mIoU macro = 0.6726 | Dice macro = 0.7504
Calculando métricas de validación...
IoU por clase : [0.9420445  0.73972166 0.70324758 0.62241365 0.0096933  0.89708363]
Dice por clase: [0.97015748 0.85039082 0.82577259 0.76726875 0.01920048 0.94575022]
mIoU macro = 0.6524 | Dice macro = 0.7298

--- Epoch 9/200 ---
100% 210/210 [00:25<00:00,  8.21it/s, loss=0.188]
Calculando métricas de entrenamiento...
IoU por clase : [0.95597835 0.79338006 0.79915993 0.59881757 0.08754651 0.92558532]
Dice por clase: [0.9774938  0.88478742 0.88837009 0.74907554 0.16099819 0.96135477]
mIoU macro = 0.6934 | Dice macro = 0.7703
Calculando métricas de validación...
IoU por clase : [0.94792517 0.73068237 0.7530515  0.62954513 0.04187373 0.90466719]
Dice por clase: [0.97326651 0.84438645 0.8591322  0.77266363 0.08038158 0.94994779]
mIoU macro = 0.6680 | Dice macro = 0.7466
🔹 Nuevo mejor mIoU: 0.6680 | Dice: 0.7466  →  guardando modelo…

--- Epoch 10/200 ---
100% 210/210 [00:27<00:00,  7.51it/s, loss=0.145]
Calculando métricas de entrenamiento...
IoU por clase : [0.9568714  0.80072412 0.78910295 0.65936178 0.24048924 0.9271909 ]
Dice por clase: [0.97796043 0.8893357  0.88212134 0.79471732 0.38773289 0.96222009]
mIoU macro = 0.7290 | Dice macro = 0.8157
Calculando métricas de validación...
IoU por clase : [0.94898666 0.75317306 0.76085688 0.65334536 0.1443221  0.90995622]
Dice por clase: [0.97382571 0.85921131 0.86418935 0.79033138 0.25224034 0.95285558]
mIoU macro = 0.6951 | Dice macro = 0.7821
🔹 Nuevo mejor mIoU: 0.6951 | Dice: 0.7821  →  guardando modelo…

--- Epoch 11/200 ---
100% 210/210 [00:27<00:00,  7.64it/s, loss=0.176]
Calculando métricas de entrenamiento...
IoU por clase : [0.95744864 0.7951084  0.80588381 0.6895817  0.44147037 0.92717551]
Dice por clase: [0.97826183 0.88586115 0.89250904 0.81627506 0.61252784 0.9622118 ]
mIoU macro = 0.7694 | Dice macro = 0.8579
Calculando métricas de validación...
IoU por clase : [0.94915281 0.74415627 0.78403941 0.65561598 0.29309905 0.90812247]
Dice por clase: [0.97391318 0.85331376 0.87894853 0.7919904  0.45332807 0.95184925]
mIoU macro = 0.7224 | Dice macro = 0.8172
🔹 Nuevo mejor mIoU: 0.7224 | Dice: 0.8172  →  guardando modelo…

--- Epoch 12/200 ---
100% 210/210 [00:27<00:00,  7.62it/s, loss=0.138]
Calculando métricas de entrenamiento...
IoU por clase : [0.95723636 0.79511618 0.81247741 0.71322994 0.47133202 0.92744974]
Dice por clase: [0.97815101 0.88586598 0.89653797 0.83261438 0.6406875  0.96235945]
mIoU macro = 0.7795 | Dice macro = 0.8660
Calculando métricas de validación...
IoU por clase : [0.94889145 0.74268355 0.78164164 0.67553603 0.30688814 0.90742771]
Dice por clase: [0.97377558 0.8523447  0.8774398  0.80635214 0.46964715 0.95146747]
mIoU macro = 0.7272 | Dice macro = 0.8218
🔹 Nuevo mejor mIoU: 0.7272 | Dice: 0.8218  →  guardando modelo…

--- Epoch 13/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.16]
Calculando métricas de entrenamiento...
IoU por clase : [0.95780868 0.78817256 0.81820518 0.71797678 0.49447741 0.92796304]
Dice por clase: [0.97844972 0.88153971 0.90001413 0.83583991 0.66173956 0.96263571]
mIoU macro = 0.7841 | Dice macro = 0.8700
Calculando métricas de validación...
IoU por clase : [0.95033643 0.74011384 0.79418762 0.68069774 0.33403114 0.9082906 ]
Dice por clase: [0.9745359  0.85064991 0.88528938 0.81001804 0.50078462 0.95194159]
mIoU macro = 0.7346 | Dice macro = 0.8289
🔹 Nuevo mejor mIoU: 0.7346 | Dice: 0.8289  →  guardando modelo…

--- Epoch 14/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.164]
Calculando métricas de entrenamiento...
IoU por clase : [0.9590581  0.81514842 0.82592655 0.72033169 0.54627729 0.93026028]
Dice por clase: [0.97910123 0.89816173 0.90466569 0.8374335  0.70657093 0.9638703 ]
mIoU macro = 0.7995 | Dice macro = 0.8816
Calculando métricas de validación...
IoU por clase : [0.95074457 0.75602784 0.77999709 0.68780882 0.42741109 0.91136077]
Dice por clase: [0.97475044 0.86106589 0.87640266 0.81503167 0.59886195 0.95362506]
mIoU macro = 0.7522 | Dice macro = 0.8466
🔹 Nuevo mejor mIoU: 0.7522 | Dice: 0.8466  →  guardando modelo…

--- Epoch 15/200 ---
100% 210/210 [00:27<00:00,  7.61it/s, loss=0.134]
Calculando métricas de entrenamiento...
IoU por clase : [0.95934527 0.81672473 0.81565572 0.73373742 0.55783057 0.93011631]
Dice por clase: [0.97925086 0.89911776 0.89846958 0.84642278 0.71616333 0.96379301]
mIoU macro = 0.8022 | Dice macro = 0.8839
Calculando métricas de validación...
IoU por clase : [0.95123103 0.76081976 0.77040793 0.68006854 0.41229571 0.91367032]
Dice por clase: [0.97500605 0.8641654  0.87031685 0.80957237 0.58386598 0.9548879 ]
mIoU macro = 0.7481 | Dice macro = 0.8430

--- Epoch 16/200 ---
100% 210/210 [00:26<00:00,  8.02it/s, loss=0.151]
Calculando métricas de entrenamiento...
IoU por clase : [0.95957189 0.81963704 0.83579712 0.71399153 0.55499701 0.93170755]
Dice por clase: [0.97936891 0.9008797  0.91055499 0.83313309 0.71382389 0.96464659]
mIoU macro = 0.8026 | Dice macro = 0.8837
Calculando métricas de validación...
IoU por clase : [0.95130057 0.76045684 0.80443085 0.69124591 0.43720937 0.91044034]
Dice por clase: [0.97504258 0.86393125 0.89161726 0.81743986 0.6084143  0.95312094]
mIoU macro = 0.7592 | Dice macro = 0.8516
🔹 Nuevo mejor mIoU: 0.7592 | Dice: 0.8516  →  guardando modelo…

--- Epoch 17/200 ---
100% 210/210 [00:27<00:00,  7.53it/s, loss=0.167]
Calculando métricas de entrenamiento...
IoU por clase : [0.95866109 0.81934453 0.8246455  0.7378406  0.59556472 0.92796536]
Dice por clase: [0.9788943  0.90070299 0.90389667 0.84914646 0.74652531 0.96263696]
mIoU macro = 0.8107 | Dice macro = 0.8903
Calculando métricas de validación...
IoU por clase : [0.95021502 0.76776098 0.77853984 0.69241462 0.47061066 0.91224902]
Dice por clase: [0.97447206 0.86862533 0.87548204 0.81825648 0.64002074 0.95411111]
mIoU macro = 0.7620 | Dice macro = 0.8552
🔹 Nuevo mejor mIoU: 0.7620 | Dice: 0.8552  →  guardando modelo…

--- Epoch 18/200 ---
100% 210/210 [00:27<00:00,  7.63it/s, loss=0.112]
Calculando métricas de entrenamiento...
IoU por clase : [0.96002799 0.81701143 0.82213627 0.7539778  0.64553234 0.92884707]
Dice por clase: [0.97960641 0.89929146 0.90238725 0.85973471 0.78458785 0.96311116]
mIoU macro = 0.8213 | Dice macro = 0.8981
Calculando métricas de validación...
IoU por clase : [0.95231865 0.77127959 0.79283757 0.70429014 0.48326178 0.91343931]
Dice por clase: [0.97557707 0.87087278 0.88444997 0.82649089 0.65162035 0.95476173]
mIoU macro = 0.7696 | Dice macro = 0.8606
🔹 Nuevo mejor mIoU: 0.7696 | Dice: 0.8606  →  guardando modelo…

--- Epoch 19/200 ---
100% 210/210 [00:27<00:00,  7.52it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.9606242  0.82930418 0.84389041 0.74153566 0.65753375 0.93349887]
Dice por clase: [0.9799167  0.90668812 0.91533684 0.85158826 0.79338807 0.96560581]
mIoU macro = 0.8277 | Dice macro = 0.9021
Calculando métricas de validación...
IoU por clase : [0.95208818 0.76438561 0.79695943 0.72296742 0.51031166 0.91307484]
Dice por clase: [0.97545612 0.86646094 0.88700882 0.83921195 0.67577001 0.95456259]
mIoU macro = 0.7766 | Dice macro = 0.8664
🔹 Nuevo mejor mIoU: 0.7766 | Dice: 0.8664  →  guardando modelo…

--- Epoch 20/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.106]
Calculando métricas de entrenamiento...
IoU por clase : [0.95940391 0.81219338 0.8301914  0.75239842 0.65810847 0.93235904]
Dice por clase: [0.97928141 0.89636502 0.90721812 0.85870703 0.79380629 0.96499566]
mIoU macro = 0.8241 | Dice macro = 0.9001
Calculando métricas de validación...
IoU por clase : [0.95105334 0.75951871 0.78480235 0.70533916 0.53749994 0.91364843]
Dice por clase: [0.9749127  0.86332553 0.87942774 0.82721276 0.69918694 0.95487595]
mIoU macro = 0.7753 | Dice macro = 0.8665

--- Epoch 21/200 ---
100% 210/210 [00:25<00:00,  8.33it/s, loss=0.0946]
Calculando métricas de entrenamiento...
IoU por clase : [0.96107862 0.82154272 0.84401111 0.75746487 0.65742049 0.92900649]
Dice por clase: [0.98015307 0.90202959 0.91540784 0.86199717 0.79330561 0.96319686]
mIoU macro = 0.8284 | Dice macro = 0.9027
Calculando métricas de validación...
IoU por clase : [0.95210161 0.77063963 0.79379006 0.69398527 0.48159624 0.91121208]
Dice por clase: [0.97546317 0.87046468 0.88504232 0.81935219 0.65010457 0.95354366]
mIoU macro = 0.7672 | Dice macro = 0.8590

--- Epoch 22/200 ---
100% 210/210 [00:25<00:00,  8.22it/s, loss=0.091]
Calculando métricas de entrenamiento...
IoU por clase : [0.96024043 0.82721395 0.82256385 0.74522534 0.6345421  0.92784714]
Dice por clase: [0.97971699 0.90543743 0.90264476 0.85401618 0.77641573 0.96257335]
mIoU macro = 0.8196 | Dice macro = 0.8968
Calculando métricas de validación...
IoU por clase : [0.95188107 0.77430765 0.76955329 0.69355062 0.47760699 0.91224552]
Dice por clase: [0.97534741 0.87279977 0.86977125 0.81904917 0.64646012 0.9541092 ]
mIoU macro = 0.7632 | Dice macro = 0.8563

--- Epoch 23/200 ---
100% 210/210 [00:26<00:00,  8.00it/s, loss=0.0871]
Calculando métricas de entrenamiento...
IoU por clase : [0.96224688 0.83319854 0.85179137 0.7576637  0.7062165  0.93551397]
Dice por clase: [0.98076026 0.90901069 0.91996473 0.8621259  0.82781581 0.96668274]
mIoU macro = 0.8411 | Dice macro = 0.9111
Calculando métricas de validación...
IoU por clase : [0.95336713 0.76834091 0.80975641 0.69491067 0.54051553 0.91272889]
Dice por clase: [0.97612693 0.86899636 0.8948789  0.81999681 0.70173331 0.95437351]
mIoU macro = 0.7799 | Dice macro = 0.8694
🔹 Nuevo mejor mIoU: 0.7799 | Dice: 0.8694  →  guardando modelo…

--- Epoch 24/200 ---
100% 210/210 [00:27<00:00,  7.51it/s, loss=0.126]
Calculando métricas de entrenamiento...
IoU por clase : [0.96147684 0.82578528 0.82136584 0.75042608 0.63442196 0.93409449]
Dice por clase: [0.98036012 0.90458094 0.90192296 0.85742105 0.77632579 0.96592436]
mIoU macro = 0.8213 | Dice macro = 0.8978
Calculando métricas de validación...
IoU por clase : [0.95183074 0.76810902 0.7505951  0.68158776 0.48733718 0.91449384]
Dice por clase: [0.97532098 0.86884803 0.85753136 0.81064786 0.65531499 0.95533746]
mIoU macro = 0.7590 | Dice macro = 0.8538

--- Epoch 25/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.13]
Calculando métricas de entrenamiento...
IoU por clase : [0.96219968 0.8350247  0.84019534 0.76099581 0.72201041 0.93525887]
Dice por clase: [0.98073574 0.91009642 0.91315886 0.86427896 0.83856684 0.96654653]
mIoU macro = 0.8426 | Dice macro = 0.9122
Calculando métricas de validación...
IoU por clase : [0.95230356 0.77353432 0.78994376 0.67303407 0.52248428 0.91673255]
Dice por clase: [0.97556915 0.87230826 0.88264646 0.80456708 0.68635754 0.9565576 ]
mIoU macro = 0.7713 | Dice macro = 0.8630

--- Epoch 26/200 ---
100% 210/210 [00:25<00:00,  8.19it/s, loss=0.0947]
Calculando métricas de entrenamiento...
IoU por clase : [0.96297759 0.83395768 0.85093087 0.76324973 0.70967623 0.93507729]
Dice por clase: [0.98113967 0.90946229 0.91946262 0.86573072 0.83018786 0.96644955]
mIoU macro = 0.8426 | Dice macro = 0.9121
Calculando métricas de validación...
IoU por clase : [0.95336187 0.76949448 0.79856379 0.67612562 0.51891837 0.91239125]
Dice por clase: [0.97612417 0.86973369 0.88800163 0.80677202 0.68327355 0.9541889 ]
mIoU macro = 0.7715 | Dice macro = 0.8630

--- Epoch 27/200 ---
100% 210/210 [00:26<00:00,  8.06it/s, loss=0.103]
Calculando métricas de entrenamiento...
IoU por clase : [0.9619061  0.83112139 0.83811114 0.7739509  0.7033671  0.93508245]
Dice por clase: [0.98058322 0.90777312 0.91192651 0.87257308 0.82585498 0.96645231]
mIoU macro = 0.8406 | Dice macro = 0.9109
Calculando métricas de validación...
IoU por clase : [0.95220745 0.76291367 0.77008443 0.69313475 0.49715309 0.91074626]
Dice por clase: [0.97551871 0.8655145  0.87011039 0.81875911 0.66413127 0.95328854]
mIoU macro = 0.7644 | Dice macro = 0.8579

--- Epoch 28/200 ---
100% 210/210 [00:26<00:00,  8.06it/s, loss=0.0782]
Calculando métricas de entrenamiento...
IoU por clase : [0.96106329 0.83509228 0.85647028 0.7491979  0.71854036 0.93477789]
Dice por clase: [0.9801451  0.91013655 0.92268677 0.8566188  0.83622169 0.96628961]
mIoU macro = 0.8425 | Dice macro = 0.9120
Calculando métricas de validación...
IoU por clase : [0.95215218 0.77547532 0.78986883 0.69459599 0.54975482 0.9127761 ]
Dice por clase: [0.97548971 0.87354108 0.88259968 0.81977768 0.70947328 0.95439932]
mIoU macro = 0.7791 | Dice macro = 0.8692

--- Epoch 29/200 ---
100% 210/210 [00:25<00:00,  8.22it/s, loss=0.0682]
Calculando métricas de entrenamiento...
IoU por clase : [0.96330322 0.83338494 0.86070145 0.7740229  0.7470787  0.93606267]
Dice por clase: [0.98130866 0.90912161 0.92513654 0.87261884 0.85523188 0.96697559]
mIoU macro = 0.8524 | Dice macro = 0.9184
Calculando métricas de validación...
IoU por clase : [0.95287861 0.77618021 0.79791023 0.69846621 0.55228217 0.91594028]
Dice por clase: [0.9758708  0.87398813 0.88759741 0.82246701 0.71157446 0.95612613]
mIoU macro = 0.7823 | Dice macro = 0.8713
🔹 Nuevo mejor mIoU: 0.7823 | Dice: 0.8713  →  guardando modelo…

--- Epoch 30/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.0873]
Calculando métricas de entrenamiento...
IoU por clase : [0.9643485  0.83719929 0.86671472 0.77523476 0.74910096 0.93762037]
Dice por clase: [0.98185072 0.91138647 0.92859901 0.87338844 0.85655543 0.96780606]
mIoU macro = 0.8550 | Dice macro = 0.9199
Calculando métricas de validación...
IoU por clase : [0.95411088 0.77246581 0.81137132 0.69311136 0.56577057 0.91482617]
Dice por clase: [0.97651663 0.87162844 0.89586415 0.81874279 0.72267365 0.95551876]
mIoU macro = 0.7853 | Dice macro = 0.8735
🔹 Nuevo mejor mIoU: 0.7853 | Dice: 0.8735  →  guardando modelo…

--- Epoch 31/200 ---
100% 210/210 [00:27<00:00,  7.55it/s, loss=0.0752]
Calculando métricas de entrenamiento...
IoU por clase : [0.96384661 0.84313186 0.85640022 0.77324894 0.74338364 0.93852407]
Dice por clase: [0.98159052 0.91489044 0.92264611 0.87212677 0.85280557 0.96828725]
mIoU macro = 0.8531 | Dice macro = 0.9187
Calculando métricas de validación...
IoU por clase : [0.95401207 0.77695825 0.81153062 0.68151004 0.54869653 0.91615859]
Dice por clase: [0.97646487 0.87448115 0.89596125 0.81059289 0.70859141 0.95624506]
mIoU macro = 0.7815 | Dice macro = 0.8704

--- Epoch 32/200 ---
100% 210/210 [00:24<00:00,  8.44it/s, loss=0.0896]
Calculando métricas de entrenamiento...
IoU por clase : [0.96446562 0.84220586 0.86848374 0.77479026 0.75803046 0.93855892]
Dice por clase: [0.98191143 0.914345   0.92961338 0.87310628 0.86236329 0.96830579]
mIoU macro = 0.8578 | Dice macro = 0.9216
Calculando métricas de validación...
IoU por clase : [0.9538745  0.77282205 0.81192614 0.69014704 0.56357394 0.9181508 ]
Dice por clase: [0.9763928  0.87185519 0.89620225 0.81667101 0.72087917 0.95732911]
mIoU macro = 0.7851 | Dice macro = 0.8732

--- Epoch 33/200 ---
100% 210/210 [00:25<00:00,  8.33it/s, loss=0.0824]
Calculando métricas de entrenamiento...
IoU por clase : [0.96185001 0.8011175  0.8640344  0.77797684 0.74511956 0.93598858]
Dice por clase: [0.98055407 0.88957828 0.92705843 0.87512596 0.85394672 0.96693605]
mIoU macro = 0.8477 | Dice macro = 0.9155
Calculando métricas de validación...
IoU por clase : [0.95168878 0.73091465 0.80938575 0.70288739 0.57626143 0.91508871]
Dice por clase: [0.97524645 0.84454152 0.89465251 0.82552422 0.73117494 0.95566195]
mIoU macro = 0.7810 | Dice macro = 0.8711

--- Epoch 34/200 ---
100% 210/210 [00:25<00:00,  8.23it/s, loss=0.0932]
Calculando métricas de entrenamiento...
IoU por clase : [0.96459859 0.84642922 0.86772701 0.77983252 0.75965415 0.93796605]
Dice por clase: [0.98198033 0.91682823 0.9291797  0.87629877 0.86341302 0.96799018]
mIoU macro = 0.8594 | Dice macro = 0.9226
Calculando métricas de validación...
IoU por clase : [0.95481152 0.78422837 0.80585371 0.68112715 0.51927659 0.9169384 ]
Dice por clase: [0.97688346 0.87906726 0.89249058 0.81032199 0.683584   0.95666965]
mIoU macro = 0.7770 | Dice macro = 0.8665

--- Epoch 35/200 ---
100% 210/210 [00:25<00:00,  8.09it/s, loss=0.0863]
Calculando métricas de entrenamiento...
IoU por clase : [0.96398508 0.84699236 0.86676795 0.76778263 0.77524691 0.93889327]
Dice por clase: [0.98166233 0.91715849 0.92862956 0.86863918 0.87339615 0.96848371]
mIoU macro = 0.8599 | Dice macro = 0.9230
Calculando métricas de validación...
IoU por clase : [0.95392262 0.78821809 0.82038587 0.6812775  0.59493025 0.91954101]
Dice por clase: [0.97641801 0.88156819 0.90133183 0.81042838 0.74602667 0.95808426]
mIoU macro = 0.7930 | Dice macro = 0.8790
🔹 Nuevo mejor mIoU: 0.7930 | Dice: 0.8790  →  guardando modelo…

--- Epoch 36/200 ---
100% 210/210 [00:27<00:00,  7.59it/s, loss=0.0602]
Calculando métricas de entrenamiento...
IoU por clase : [0.96527918 0.84692476 0.87187148 0.79074702 0.77642044 0.93941269]
Dice por clase: [0.98233288 0.91711885 0.93155058 0.88314766 0.8741404  0.96875997]
mIoU macro = 0.8651 | Dice macro = 0.9262
Calculando métricas de validación...
IoU por clase : [0.95506464 0.78514027 0.81804587 0.71175727 0.57794693 0.91982725]
Dice por clase: [0.97701592 0.87963986 0.89991774 0.83161005 0.73253025 0.9582396 ]
mIoU macro = 0.7946 | Dice macro = 0.8798
🔹 Nuevo mejor mIoU: 0.7946 | Dice: 0.8798  →  guardando modelo…

--- Epoch 37/200 ---
100% 210/210 [00:27<00:00,  7.58it/s, loss=0.0619]
Calculando métricas de entrenamiento...
IoU por clase : [0.96536851 0.84866401 0.87249121 0.78637272 0.77394702 0.93879139]
Dice por clase: [0.98237914 0.91813764 0.9319042  0.88041281 0.87257061 0.9684295 ]
mIoU macro = 0.8643 | Dice macro = 0.9256
Calculando métricas de validación...
IoU por clase : [0.95499557 0.79103393 0.81870949 0.70535192 0.59128419 0.92082069]
Dice por clase: [0.97697978 0.88332657 0.90031915 0.82722154 0.74315348 0.95877839]
mIoU macro = 0.7970 | Dice macro = 0.8816
🔹 Nuevo mejor mIoU: 0.7970 | Dice: 0.8816  →  guardando modelo…

--- Epoch 38/200 ---
100% 210/210 [00:28<00:00,  7.39it/s, loss=0.175]
Calculando métricas de entrenamiento...
IoU por clase : [0.96306315 0.7997442  0.84103265 0.74682395 0.67814943 0.93456914]
Dice por clase: [0.98118408 0.88873096 0.91365316 0.85506493 0.80821102 0.96617807]
mIoU macro = 0.8272 | Dice macro = 0.9022
Calculando métricas de validación...
IoU por clase : [0.95301605 0.72125483 0.78751753 0.66779684 0.55104682 0.91073778]
Dice por clase: [0.97594288 0.838057   0.88112986 0.80081317 0.71054827 0.9532839 ]
mIoU macro = 0.7652 | Dice macro = 0.8600

--- Epoch 39/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.0583]
Calculando métricas de entrenamiento...
IoU por clase : [0.96529531 0.84848617 0.86962304 0.78658672 0.77368758 0.93913658]
Dice por clase: [0.98234123 0.91803356 0.93026565 0.88054692 0.8724057  0.96861313]
mIoU macro = 0.8638 | Dice macro = 0.9254
Calculando métricas de validación...
IoU por clase : [0.95436452 0.78313666 0.80100269 0.69844662 0.56548739 0.9195189 ]
Dice por clase: [0.97664945 0.87838098 0.88950749 0.82245342 0.7224426  0.95807225]
mIoU macro = 0.7870 | Dice macro = 0.8746

--- Epoch 40/200 ---
100% 210/210 [00:26<00:00,  7.95it/s, loss=0.0759]
Calculando métricas de entrenamiento...
IoU por clase : [0.96562053 0.85023831 0.87289166 0.77591751 0.7865907  0.93994046]
Dice por clase: [0.98250961 0.91905816 0.93213257 0.87382157 0.88054942 0.96904053]
mIoU macro = 0.8652 | Dice macro = 0.9262
Calculando métricas de validación...
IoU por clase : [0.95471567 0.78998896 0.8161132  0.6852694  0.60676241 0.92122236]
Dice por clase: [0.97683329 0.88267467 0.89874706 0.81324612 0.7552609  0.95899608]
mIoU macro = 0.7957 | Dice macro = 0.8810

--- Epoch 41/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.079]
Calculando métricas de entrenamiento...
IoU por clase : [0.96524408 0.85165178 0.87447979 0.79757472 0.78347303 0.93712374]
Dice por clase: [0.98231471 0.91988331 0.93303731 0.88738978 0.87859252 0.96754144]
mIoU macro = 0.8683 | Dice macro = 0.9281
Calculando métricas de validación...
IoU por clase : [0.95438156 0.78892004 0.82461033 0.71076277 0.60982158 0.918528  ]
Dice por clase: [0.97665838 0.88200705 0.90387555 0.83093083 0.7576263  0.95753411]
mIoU macro = 0.8012 | Dice macro = 0.8848
🔹 Nuevo mejor mIoU: 0.8012 | Dice: 0.8848  →  guardando modelo…

--- Epoch 42/200 ---
100% 210/210 [00:27<00:00,  7.54it/s, loss=0.0622]
Calculando métricas de entrenamiento...
IoU por clase : [0.96568663 0.83150284 0.87597996 0.79389798 0.79001252 0.93405166]
Dice por clase: [0.98254382 0.9080006  0.93389053 0.8851094  0.88268938 0.96590146]
mIoU macro = 0.8652 | Dice macro = 0.9264
Calculando métricas de validación...
IoU por clase : [0.9547562  0.77498437 0.82061476 0.70326256 0.59494362 0.91441955]
Dice por clase: [0.97685451 0.87322951 0.90146996 0.82578292 0.74603718 0.95529692]
mIoU macro = 0.7938 | Dice macro = 0.8798

--- Epoch 43/200 ---
100% 210/210 [00:26<00:00,  7.94it/s, loss=0.0716]
Calculando métricas de entrenamiento...
IoU por clase : [0.96619876 0.85359367 0.87857898 0.79247051 0.78574873 0.94114934]
Dice por clase: [0.98280884 0.92101487 0.9353655  0.88422153 0.8800216  0.96968257]
mIoU macro = 0.8696 | Dice macro = 0.9289
Calculando métricas de validación...
IoU por clase : [0.95478591 0.79269251 0.82746032 0.70074104 0.60852108 0.92208164]
Dice por clase: [0.97687006 0.88435971 0.90558499 0.82404202 0.75662183 0.95946147]
mIoU macro = 0.8010 | Dice macro = 0.8845

--- Epoch 44/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.07]
Calculando métricas de entrenamiento...
IoU por clase : [0.96625806 0.85451117 0.87066425 0.79118478 0.80266482 0.94158998]
Dice por clase: [0.98283952 0.92154869 0.93086106 0.88342061 0.89053141 0.9699164 ]
mIoU macro = 0.8711 | Dice macro = 0.9299
Calculando métricas de validación...
IoU por clase : [0.95482543 0.78241527 0.82242848 0.69319957 0.60858507 0.92103574]
Dice por clase: [0.97689074 0.87792703 0.90256324 0.81880433 0.75667129 0.95889496]
mIoU macro = 0.7971 | Dice macro = 0.8820

--- Epoch 45/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0694]
Calculando métricas de entrenamiento...
IoU por clase : [0.96676279 0.85559892 0.87769622 0.8000233  0.79626523 0.94145247]
Dice por clase: [0.98310055 0.92218088 0.93486498 0.88890327 0.88657868 0.96984344]
mIoU macro = 0.8730 | Dice macro = 0.9309
Calculando métricas de validación...
IoU por clase : [0.95589197 0.79292222 0.82179837 0.71492795 0.61434445 0.92118576]
Dice por clase: [0.97744864 0.88450264 0.90218367 0.83377025 0.76110702 0.95897625]
mIoU macro = 0.8035 | Dice macro = 0.8863
🔹 Nuevo mejor mIoU: 0.8035 | Dice: 0.8863  →  guardando modelo…

--- Epoch 46/200 ---
100% 210/210 [00:28<00:00,  7.43it/s, loss=0.0673]
Calculando métricas de entrenamiento...
IoU por clase : [0.96553546 0.85090418 0.87257185 0.79377442 0.78341038 0.9412947 ]
Dice por clase: [0.98246557 0.91944703 0.9319502  0.8850326  0.87855313 0.96975972]
mIoU macro = 0.8679 | Dice macro = 0.9279
Calculando métricas de validación...
IoU por clase : [0.95486436 0.78764329 0.81155939 0.70495303 0.58235641 0.92083885]
Dice por clase: [0.97691111 0.88120856 0.89597879 0.82694715 0.73606225 0.95878824]
mIoU macro = 0.7937 | Dice macro = 0.8793

--- Epoch 47/200 ---
100% 210/210 [00:26<00:00,  8.07it/s, loss=0.0606]
Calculando métricas de entrenamiento...
IoU por clase : [0.96647381 0.85561399 0.8790063  0.79149781 0.79693977 0.94186788]
Dice por clase: [0.98295111 0.92218963 0.93560761 0.88361571 0.88699664 0.97006381]
mIoU macro = 0.8719 | Dice macro = 0.9302
Calculando métricas de validación...
IoU por clase : [0.955114   0.79172076 0.8125474  0.69742608 0.62316307 0.92185779]
Dice por clase: [0.97704175 0.88375463 0.89658058 0.82174545 0.76783791 0.95934028]
mIoU macro = 0.8003 | Dice macro = 0.8844

--- Epoch 48/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0535]
Calculando métricas de entrenamiento...
IoU por clase : [0.9665523  0.85268776 0.87735001 0.79816753 0.80029861 0.94251273]
Dice por clase: [0.98299171 0.92048728 0.93466855 0.88775658 0.88907318 0.97040572]
mIoU macro = 0.8729 | Dice macro = 0.9309
Calculando métricas de validación...
IoU por clase : [0.95542888 0.78114243 0.8130916  0.70004625 0.64041842 0.91987767]
Dice por clase: [0.97720647 0.87712517 0.89691177 0.82356142 0.78079887 0.95826696]
mIoU macro = 0.8017 | Dice macro = 0.8856

--- Epoch 49/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0482]
Calculando métricas de entrenamiento...
IoU por clase : [0.96715835 0.85713362 0.88214472 0.80611679 0.80196302 0.94222542]
Dice por clase: [0.98330503 0.92307156 0.93738246 0.8926519  0.89009931 0.97025341]
mIoU macro = 0.8761 | Dice macro = 0.9328
Calculando métricas de validación...
IoU por clase : [0.95586071 0.78943906 0.81690524 0.71031178 0.62763684 0.92212045]
Dice por clase: [0.97743229 0.88233132 0.89922713 0.83062256 0.77122467 0.95948248]
mIoU macro = 0.8037 | Dice macro = 0.8867
🔹 Nuevo mejor mIoU: 0.8037 | Dice: 0.8867  →  guardando modelo…

--- Epoch 50/200 ---
100% 210/210 [00:29<00:00,  7.11it/s, loss=0.108]
Calculando métricas de entrenamiento...
IoU por clase : [0.96562673 0.83549809 0.87127949 0.79565411 0.79205496 0.93978259]
Dice por clase: [0.98251282 0.91037751 0.93121257 0.88619975 0.8839628  0.96895662]
mIoU macro = 0.8666 | Dice macro = 0.9272
Calculando métricas de validación...
IoU por clase : [0.95487672 0.77447899 0.8104298  0.70386021 0.62823073 0.91963179]
Dice por clase: [0.97691758 0.8729086  0.89528995 0.82619479 0.77167286 0.95813353]
mIoU macro = 0.7986 | Dice macro = 0.8835

--- Epoch 51/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.064]
Calculando métricas de entrenamiento...
IoU por clase : [0.96682083 0.85772924 0.87055925 0.7899799  0.78972921 0.94186037]
Dice por clase: [0.98313056 0.92341685 0.93080104 0.88266902 0.88251251 0.97005983]
mIoU macro = 0.8694 | Dice macro = 0.9288
Calculando métricas de validación...
IoU por clase : [0.95566062 0.78844942 0.79512568 0.69463331 0.59600554 0.91984074]
Dice por clase: [0.97732767 0.88171285 0.88587188 0.81980367 0.74687151 0.95824692]
mIoU macro = 0.7916 | Dice macro = 0.8783

--- Epoch 52/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0716]
Calculando métricas de entrenamiento...
IoU por clase : [0.96638389 0.85276876 0.84000511 0.78556246 0.72398858 0.94170737]
Dice por clase: [0.9829046  0.92053448 0.9130465  0.87990476 0.83989951 0.96997867]
mIoU macro = 0.8517 | Dice macro = 0.9177
Calculando métricas de validación...
IoU por clase : [0.9554198  0.77713896 0.72027619 0.69209325 0.50978282 0.91836624]
Dice por clase: [0.97720173 0.8745956  0.83739599 0.81803204 0.67530616 0.95744621]
mIoU macro = 0.7622 | Dice macro = 0.8567

--- Epoch 53/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0583]
Calculando métricas de entrenamiento...
IoU por clase : [0.9675091  0.85898076 0.8793408  0.80493266 0.80230579 0.94224611]
Dice por clase: [0.98348627 0.92414164 0.93579706 0.89192542 0.89031039 0.97026438]
mIoU macro = 0.8759 | Dice macro = 0.9327
Calculando métricas de validación...
IoU por clase : [0.95619051 0.79896041 0.81612259 0.7127389  0.62594107 0.92213168]
Dice por clase: [0.97760469 0.8882468  0.89875276 0.83227969 0.76994312 0.95948856]
mIoU macro = 0.8053 | Dice macro = 0.8877
🔹 Nuevo mejor mIoU: 0.8053 | Dice: 0.8877  →  guardando modelo…

--- Epoch 54/200 ---
100% 210/210 [00:28<00:00,  7.29it/s, loss=0.0503]
Calculando métricas de entrenamiento...
IoU por clase : [0.96752481 0.85930936 0.88306855 0.80500712 0.81145894 0.94328289]
Dice por clase: [0.98349439 0.92433177 0.93790377 0.89197113 0.89591756 0.97081377]
mIoU macro = 0.8783 | Dice macro = 0.9341
Calculando métricas de validación...
IoU por clase : [0.95614662 0.78826595 0.82215385 0.70292276 0.61420854 0.92168965]
Dice por clase: [0.97758175 0.88159812 0.90239784 0.82554861 0.76100271 0.95924922]
mIoU macro = 0.8009 | Dice macro = 0.8846

--- Epoch 55/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0534]
Calculando métricas de entrenamiento...
IoU por clase : [0.96718271 0.85957104 0.88483249 0.78113402 0.80526938 0.94252701]
Dice por clase: [0.98331762 0.92448314 0.93889775 0.87711987 0.8921321  0.97041329]
mIoU macro = 0.8734 | Dice macro = 0.9311
Calculando métricas de validación...
IoU por clase : [0.95501945 0.78846925 0.81470248 0.68963316 0.62128818 0.92003652]
Dice por clase: [0.97699228 0.88172525 0.89789096 0.81631111 0.76641301 0.95835315]
mIoU macro = 0.7982 | Dice macro = 0.8829

--- Epoch 56/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.04]
Calculando métricas de entrenamiento...
IoU por clase : [0.96804206 0.86071894 0.88665766 0.80683923 0.8140944  0.94331237]
Dice por clase: [0.98376156 0.92514664 0.93992427 0.89309465 0.89752154 0.97082938]
mIoU macro = 0.8799 | Dice macro = 0.9350
Calculando métricas de validación...
IoU por clase : [0.95648732 0.78380861 0.82699943 0.71663297 0.66201564 0.91979862]
Dice por clase: [0.9777598  0.87880348 0.90530891 0.83492859 0.79664189 0.95822406]
mIoU macro = 0.8110 | Dice macro = 0.8919
🔹 Nuevo mejor mIoU: 0.8110 | Dice: 0.8919  →  guardando modelo…

--- Epoch 57/200 ---
100% 210/210 [00:28<00:00,  7.46it/s, loss=0.0628]
Calculando métricas de entrenamiento...
IoU por clase : [0.9671119  0.85619154 0.87920306 0.81128275 0.81611479 0.94364018]
Dice por clase: [0.98328102 0.92252499 0.93571906 0.89581017 0.89874803 0.97100296]
mIoU macro = 0.8789 | Dice macro = 0.9345
Calculando métricas de validación...
IoU por clase : [0.95590815 0.78460573 0.8238187  0.72981848 0.68064712 0.92003832]
Dice por clase: [0.97745709 0.87930428 0.90339978 0.84380932 0.80998219 0.95835412]
mIoU macro = 0.8158 | Dice macro = 0.8954
🔹 Nuevo mejor mIoU: 0.8158 | Dice: 0.8954  →  guardando modelo…

--- Epoch 58/200 ---
100% 210/210 [00:28<00:00,  7.44it/s, loss=0.0844]
Calculando métricas de entrenamiento...
IoU por clase : [0.968478   0.85883678 0.88947936 0.81106203 0.81218743 0.94377838]
Dice por clase: [0.98398661 0.9240583  0.94150736 0.89567559 0.8963614  0.97107611]
mIoU macro = 0.8806 | Dice macro = 0.9354
Calculando métricas de validación...
IoU por clase : [0.95670718 0.78745386 0.83523898 0.7149282  0.64983959 0.92164904]
Dice por clase: [0.97787466 0.88109    0.91022367 0.83377042 0.78776094 0.95922723]
mIoU macro = 0.8110 | Dice macro = 0.8917

--- Epoch 59/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0852]
Calculando métricas de entrenamiento...
IoU por clase : [0.96848169 0.85833482 0.89007138 0.80650185 0.82340329 0.94444757]
Dice por clase: [0.98398852 0.92376768 0.9418389  0.89288793 0.90314994 0.97143022]
mIoU macro = 0.8819 | Dice macro = 0.9362
Calculando métricas de validación...
IoU por clase : [0.95641016 0.78200818 0.83858501 0.71350177 0.64473958 0.92046331]
Dice por clase: [0.97771948 0.8776707  0.91220695 0.83279957 0.78400202 0.95858463]
mIoU macro = 0.8093 | Dice macro = 0.8905

--- Epoch 60/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0557]
Calculando métricas de entrenamiento...
IoU por clase : [0.96800819 0.84726942 0.8869133  0.81392423 0.8226884  0.94440585]
Dice por clase: [0.98374407 0.9173209  0.94006789 0.89741812 0.90271974 0.97140816]
mIoU macro = 0.8805 | Dice macro = 0.9354
Calculando métricas de validación...
IoU por clase : [0.95638312 0.77537188 0.83264327 0.72066271 0.65571884 0.92220271]
Dice por clase: [0.97770535 0.87347546 0.90868014 0.83765715 0.79206545 0.95952701]
mIoU macro = 0.8105 | Dice macro = 0.8915

--- Epoch 61/200 ---
100% 210/210 [00:27<00:00,  7.67it/s, loss=0.0829]
Calculando métricas de entrenamiento...
IoU por clase : [0.96803082 0.85962812 0.88587479 0.80692069 0.8031496  0.94445358]
Dice por clase: [0.98375575 0.92451616 0.93948421 0.89314456 0.89082969 0.9714334 ]
mIoU macro = 0.8780 | Dice macro = 0.9339
Calculando métricas de validación...
IoU por clase : [0.95637858 0.78655155 0.83937271 0.71808412 0.65317634 0.9229534 ]
Dice por clase: [0.97770298 0.88052489 0.91267279 0.83591265 0.7902077  0.9599332 ]
mIoU macro = 0.8128 | Dice macro = 0.8928

--- Epoch 62/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0693]
Calculando métricas de entrenamiento...
IoU por clase : [0.96781378 0.8443206  0.87928239 0.80654733 0.80011931 0.93830463]
Dice por clase: [0.98364367 0.91558984 0.93576398 0.8929158  0.88896253 0.96817045]
mIoU macro = 0.8727 | Dice macro = 0.9308
Calculando métricas de validación...
IoU por clase : [0.95588431 0.76150136 0.8070746  0.71284164 0.61548314 0.91135499]
Dice por clase: [0.97744463 0.86460491 0.89323883 0.83234973 0.76198027 0.9536219 ]
mIoU macro = 0.7940 | Dice macro = 0.8805

--- Epoch 63/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0632]
Calculando métricas de entrenamiento...
IoU por clase : [0.9681922  0.85848497 0.88863403 0.8018202  0.81677821 0.94348387]
Dice por clase: [0.98383908 0.92385463 0.94103359 0.89001134 0.89915016 0.97092019]
mIoU macro = 0.8796 | Dice macro = 0.9348
Calculando métricas de validación...
IoU por clase : [0.95606299 0.79597365 0.83298248 0.71072548 0.64688188 0.92109453]
Dice por clase: [0.97753804 0.88639792 0.9088821  0.83090536 0.78558382 0.95892681]
mIoU macro = 0.8106 | Dice macro = 0.8914

--- Epoch 64/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0895]
Calculando métricas de entrenamiento...
IoU por clase : [0.96704375 0.84324668 0.88658207 0.80831473 0.80705928 0.93104621]
Dice por clase: [0.9832458  0.91495804 0.93988179 0.89399784 0.89322945 0.96429201]
mIoU macro = 0.8739 | Dice macro = 0.9316
Calculando métricas de validación...
IoU por clase : [0.95520022 0.7730142  0.82089006 0.7184307  0.6416233  0.90703237]
Dice por clase: [0.97708686 0.87197745 0.90163605 0.83614742 0.7816937  0.9512501 ]
mIoU macro = 0.8027 | Dice macro = 0.8866

--- Epoch 65/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0605]
Calculando métricas de entrenamiento...
IoU por clase : [0.96910805 0.86456759 0.89234437 0.80826191 0.82123833 0.94402673]
Dice por clase: [0.9843117  0.92736524 0.94310992 0.89396553 0.90184609 0.97120756]
mIoU macro = 0.8833 | Dice macro = 0.9370
Calculando métricas de validación...
IoU por clase : [0.95675114 0.79422974 0.82367988 0.70313211 0.63736524 0.92104499]
Dice por clase: [0.97789762 0.88531555 0.9033163  0.82569298 0.77852543 0.95889997]
mIoU macro = 0.8060 | Dice macro = 0.8883

--- Epoch 66/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0758]
Calculando métricas de entrenamiento...
IoU por clase : [0.96763045 0.85563673 0.88262777 0.80977337 0.79240094 0.93996535]
Dice por clase: [0.98354897 0.92220284 0.9376551  0.89488925 0.88417823 0.96905375]
mIoU macro = 0.8747 | Dice macro = 0.9319
Calculando métricas de validación...
IoU por clase : [0.95528432 0.78872761 0.80888007 0.71211271 0.58802007 0.91844866]
Dice por clase: [0.97713085 0.88188677 0.8943435  0.8318526  0.74057007 0.957491  ]
mIoU macro = 0.7952 | Dice macro = 0.8805

--- Epoch 67/200 ---
100% 210/210 [00:26<00:00,  8.05it/s, loss=0.0635]
Calculando métricas de entrenamiento...
IoU por clase : [0.96926896 0.86073576 0.89173672 0.81862205 0.8295089  0.94469554]
Dice por clase: [0.9843947  0.92515636 0.94277043 0.90026627 0.90681045 0.97156138]
mIoU macro = 0.8858 | Dice macro = 0.9385
Calculando métricas de validación...
IoU por clase : [0.95683707 0.7883903  0.83882161 0.72712419 0.62912555 0.92050582]
Dice por clase: [0.9779425  0.88167588 0.91234691 0.84200568 0.77234754 0.95860769]
mIoU macro = 0.8101 | Dice macro = 0.8908

--- Epoch 68/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0771]
Calculando métricas de entrenamiento...
IoU por clase : [0.96950937 0.86820265 0.89444295 0.81765718 0.8342817  0.94497666]
Dice por clase: [0.98451867 0.92945233 0.94428069 0.8996825  0.90965493 0.97171003]
mIoU macro = 0.8882 | Dice macro = 0.9399
Calculando métricas de validación...
IoU por clase : [0.95730544 0.79981677 0.8429976  0.72944994 0.64138734 0.92470298]
Dice por clase: [0.97818707 0.88877577 0.91481139 0.84356294 0.78151856 0.96087863]
mIoU macro = 0.8159 | Dice macro = 0.8946
🔹 Nuevo mejor mIoU: 0.8159 | Dice: 0.8946  →  guardando modelo…

--- Epoch 69/200 ---
100% 210/210 [00:28<00:00,  7.30it/s, loss=0.0618]
Calculando métricas de entrenamiento...
IoU por clase : [0.96923147 0.86634852 0.89444943 0.81700496 0.82708096 0.94273097]
Dice por clase: [0.98437536 0.92838879 0.9442843  0.89928754 0.90535776 0.97052138]
mIoU macro = 0.8861 | Dice macro = 0.9387
Calculando métricas de validación...
IoU por clase : [0.95678253 0.79787307 0.83751321 0.72242481 0.64916466 0.92065557]
Dice por clase: [0.97791402 0.88757441 0.91157245 0.83884626 0.78726482 0.95868888]
mIoU macro = 0.8141 | Dice macro = 0.8936

--- Epoch 70/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0619]
Calculando métricas de entrenamiento...
IoU por clase : [0.9693828  0.86662742 0.89263622 0.81993561 0.8280494  0.94393801]
Dice por clase: [0.98445341 0.9285489  0.94327289 0.90106002 0.90593766 0.97116061]
mIoU macro = 0.8868 | Dice macro = 0.9391
Calculando métricas de validación...
IoU por clase : [0.95759    0.79468563 0.83122586 0.71960479 0.6302127  0.92363762]
Dice por clase: [0.9783356  0.8855987  0.90783543 0.83694206 0.77316623 0.96030313]
mIoU macro = 0.8095 | Dice macro = 0.8904

--- Epoch 71/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.0635]
Calculando métricas de entrenamiento...
IoU por clase : [0.96823115 0.86615854 0.88635357 0.81532724 0.82463704 0.94026851]
Dice por clase: [0.98385919 0.9282797  0.93975338 0.89827026 0.90389159 0.96921484]
mIoU macro = 0.8835 | Dice macro = 0.9372
Calculando métricas de validación...
IoU por clase : [0.95638686 0.79831368 0.8228015  0.71894908 0.60874328 0.91682562]
Dice por clase: [0.9777073  0.88784698 0.90278783 0.8364984  0.75679356 0.95660827]
mIoU macro = 0.8037 | Dice macro = 0.8864

--- Epoch 72/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0858]
Calculando métricas de entrenamiento...
IoU por clase : [0.96961485 0.86914423 0.89100727 0.81931423 0.82972327 0.94568704]
Dice por clase: [0.98457305 0.92999162 0.94236261 0.90068468 0.90693854 0.97208546]
mIoU macro = 0.8874 | Dice macro = 0.9394
Calculando métricas de validación...
IoU por clase : [0.9576342  0.80459828 0.83597255 0.72408513 0.65247884 0.92413385]
Dice por clase: [0.97835868 0.8917201  0.9106591  0.83996447 0.78969706 0.96057127]
mIoU macro = 0.8165 | Dice macro = 0.8952
🔹 Nuevo mejor mIoU: 0.8165 | Dice: 0.8952  →  guardando modelo…

--- Epoch 73/200 ---
100% 210/210 [00:28<00:00,  7.26it/s, loss=0.0482]
Calculando métricas de entrenamiento...
IoU por clase : [0.96869566 0.86230248 0.87720027 0.81789642 0.79369058 0.94442352]
Dice por clase: [0.98409894 0.92606061 0.93458358 0.89982731 0.88498048 0.97141751]
mIoU macro = 0.8774 | Dice macro = 0.9335
Calculando métricas de validación...
IoU por clase : [0.957267   0.78008959 0.7970995  0.71463244 0.55233285 0.91484729]
Dice por clase: [0.978167   0.87646105 0.88709557 0.83356925 0.71161652 0.95553029]
mIoU macro = 0.7860 | Dice macro = 0.8737

--- Epoch 74/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.0828]
Calculando métricas de entrenamiento...
IoU por clase : [0.96833891 0.85971481 0.88529317 0.81176614 0.8147622  0.9409423 ]
Dice por clase: [0.98391482 0.92456629 0.93915703 0.89610477 0.89792723 0.96957267]
mIoU macro = 0.8801 | Dice macro = 0.9352
Calculando métricas de validación...
IoU por clase : [0.95681228 0.78732064 0.81266092 0.71587479 0.61654367 0.90953423]
Dice por clase: [0.97792956 0.8810066  0.89664968 0.83441378 0.76279247 0.95262417]
mIoU macro = 0.7998 | Dice macro = 0.8842

--- Epoch 75/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0688]
Calculando métricas de entrenamiento...
IoU por clase : [0.96676188 0.86121375 0.86519574 0.76018293 0.70474427 0.93758671]
Dice por clase: [0.98310008 0.92543239 0.92772648 0.86375446 0.82680351 0.96778813]
mIoU macro = 0.8493 | Dice macro = 0.9158
Calculando métricas de validación...
IoU por clase : [0.95576738 0.79424301 0.79334937 0.67051279 0.51215452 0.91647348]
Dice por clase: [0.9773835  0.88532379 0.88476833 0.80276283 0.67738384 0.95641655]
mIoU macro = 0.7738 | Dice macro = 0.8640

--- Epoch 76/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0697]
Calculando métricas de entrenamiento...
IoU por clase : [0.96878742 0.86521682 0.8881824  0.80395368 0.82275213 0.94520254]
Dice por clase: [0.98414629 0.9277386  0.9407803  0.89132408 0.90275811 0.97182943]
mIoU macro = 0.8823 | Dice macro = 0.9364
Calculando métricas de validación...
IoU por clase : [0.95709352 0.79864221 0.83481865 0.71801531 0.62427666 0.92083804]
Dice por clase: [0.97807643 0.88805012 0.90997402 0.83586602 0.76868267 0.9587878 ]
mIoU macro = 0.8089 | Dice macro = 0.8899

--- Epoch 77/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0763]
Calculando métricas de entrenamiento...
IoU por clase : [0.96974723 0.86931528 0.89223679 0.82113931 0.82966771 0.94564494]
Dice por clase: [0.98464129 0.93008952 0.94304983 0.90178638 0.90690534 0.97206322]
mIoU macro = 0.8880 | Dice macro = 0.9398
Calculando métricas de validación...
IoU por clase : [0.95751388 0.79801745 0.84135414 0.72624782 0.62539002 0.91954606]
Dice por clase: [0.97829588 0.88766374 0.91384283 0.8414178  0.7695261  0.958087  ]
mIoU macro = 0.8113 | Dice macro = 0.8915

--- Epoch 78/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0633]
Calculando métricas de entrenamiento...
IoU por clase : [0.96914411 0.86088017 0.89402179 0.79721204 0.82868467 0.94603341]
Dice por clase: [0.98433031 0.92523977 0.94404594 0.88716525 0.90631773 0.97226842]
mIoU macro = 0.8827 | Dice macro = 0.9366
Calculando métricas de validación...
IoU por clase : [0.95635943 0.78886263 0.83286013 0.7009648  0.64952068 0.92024377]
Dice por clase: [0.97769297 0.88197116 0.90880926 0.82419672 0.78752657 0.95846557]
mIoU macro = 0.8081 | Dice macro = 0.8898

--- Epoch 79/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0577]
Calculando métricas de entrenamiento...
IoU por clase : [0.97012253 0.86867026 0.89568524 0.81820193 0.83198311 0.94632383]
Dice por clase: [0.98483471 0.92972022 0.94497253 0.90001216 0.90828688 0.97242177]
mIoU macro = 0.8885 | Dice macro = 0.9400
Calculando métricas de validación...
IoU por clase : [0.95742013 0.79642823 0.84128133 0.72788899 0.62839158 0.92014207]
Dice por clase: [0.97824694 0.88667971 0.91379988 0.84251823 0.77179419 0.95841041]
mIoU macro = 0.8119 | Dice macro = 0.8919

--- Epoch 80/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0523]
Calculando métricas de entrenamiento...
IoU por clase : [0.97036661 0.87025057 0.89925897 0.8219729  0.83891548 0.94621323]
Dice por clase: [0.98496047 0.93062457 0.94695772 0.90228883 0.91240243 0.97236338]
mIoU macro = 0.8912 | Dice macro = 0.9416
Calculando métricas de validación...
IoU por clase : [0.95754276 0.7893699  0.82716355 0.72801877 0.65292607 0.92289726]
Dice por clase: [0.97831095 0.88228812 0.90540724 0.84260517 0.79002453 0.95990283]
mIoU macro = 0.8130 | Dice macro = 0.8931

--- Epoch 81/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0441]
Calculando métricas de entrenamiento...
IoU por clase : [0.97025411 0.87277303 0.89841176 0.81482178 0.836739   0.94679424]
Dice por clase: [0.98490251 0.93206493 0.94648777 0.89796341 0.91111366 0.97267007]
mIoU macro = 0.8900 | Dice macro = 0.9409
Calculando métricas de validación...
IoU por clase : [0.95761522 0.80327358 0.83036746 0.71794043 0.65113517 0.92331532]
Dice por clase: [0.97834877 0.89090595 0.90732323 0.83581528 0.78871213 0.96012891]
mIoU macro = 0.8139 | Dice macro = 0.8935

--- Epoch 82/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.04]
Calculando métricas de entrenamiento...
IoU por clase : [0.97073553 0.86952142 0.8989934  0.82589696 0.84501189 0.94675383]
Dice por clase: [0.98515048 0.9302075  0.94681045 0.90464794 0.91599615 0.97264874]
mIoU macro = 0.8928 | Dice macro = 0.9426
Calculando métricas de validación...
IoU por clase : [0.95842716 0.79773222 0.8348366  0.73011881 0.6482563  0.9226418 ]
Dice por clase: [0.97877233 0.88748726 0.90998468 0.84401002 0.78659648 0.95976463]
mIoU macro = 0.8153 | Dice macro = 0.8944

--- Epoch 83/200 ---
100% 210/210 [00:26<00:00,  7.96it/s, loss=0.0827]
Calculando métricas de entrenamiento...
IoU por clase : [0.97001817 0.8716778  0.89081132 0.81301364 0.83072119 0.94628122]
Dice por clase: [0.98478094 0.93144001 0.942253   0.89686434 0.90753436 0.97239927]
mIoU macro = 0.8871 | Dice macro = 0.9392
Calculando métricas de validación...
IoU por clase : [0.9573992  0.80488404 0.8225837  0.72022555 0.63392388 0.92136156]
Dice por clase: [0.97823602 0.89189557 0.90265671 0.83736176 0.77595277 0.95907151]
mIoU macro = 0.8101 | Dice macro = 0.8909

--- Epoch 84/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.0538]
Calculando métricas de entrenamiento...
IoU por clase : [0.97054623 0.87165395 0.89882291 0.82330548 0.8412938  0.9463859 ]
Dice por clase: [0.98505299 0.9314264  0.94671589 0.9030911  0.91380724 0.97245454]
mIoU macro = 0.8920 | Dice macro = 0.9421
Calculando métricas de validación...
IoU por clase : [0.95816696 0.81049604 0.84060455 0.73402414 0.64645089 0.92252867]
Dice por clase: [0.97863663 0.89533037 0.91340049 0.84661352 0.78526593 0.95970342]
mIoU macro = 0.8187 | Dice macro = 0.8965
🔹 Nuevo mejor mIoU: 0.8187 | Dice: 0.8965  →  guardando modelo…

--- Epoch 85/200 ---
100% 210/210 [00:28<00:00,  7.40it/s, loss=0.073]
Calculando métricas de entrenamiento...
IoU por clase : [0.97005964 0.86951066 0.8912281  0.81978969 0.83558959 0.94526763]
Dice por clase: [0.98480231 0.93020134 0.9424861  0.9009719  0.91043182 0.97186384]
mIoU macro = 0.8886 | Dice macro = 0.9401
Calculando métricas de validación...
IoU por clase : [0.95810972 0.79935373 0.82772041 0.72360304 0.65913644 0.92201834]
Dice por clase: [0.97860678 0.88848981 0.90574073 0.83964001 0.79455363 0.9594272 ]
mIoU macro = 0.8150 | Dice macro = 0.8944

--- Epoch 86/200 ---
100% 210/210 [00:26<00:00,  7.95it/s, loss=0.0605]
Calculando métricas de entrenamiento...
IoU por clase : [0.97053714 0.87216106 0.897349   0.82198778 0.83191788 0.94657623]
Dice por clase: [0.98504831 0.93171584 0.94589767 0.90229779 0.908248   0.97255501]
mIoU macro = 0.8901 | Dice macro = 0.9410
Calculando métricas de validación...
IoU por clase : [0.95823945 0.7972108  0.83874086 0.73187674 0.64394463 0.92160641]
Dice por clase: [0.97867444 0.88716449 0.91229915 0.8451834  0.78341401 0.95920414]
mIoU macro = 0.8153 | Dice macro = 0.8943

--- Epoch 87/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0716]
Calculando métricas de entrenamiento...
IoU por clase : [0.97084578 0.87205192 0.90106171 0.82807484 0.8449073  0.94658113]
Dice por clase: [0.98520726 0.93165356 0.9479563  0.90595289 0.91593469 0.97255759]
mIoU macro = 0.8939 | Dice macro = 0.9432
Calculando métricas de validación...
IoU por clase : [0.95840609 0.80440167 0.83778222 0.73322926 0.666169   0.92356005]
Dice por clase: [0.97876134 0.89159934 0.91173177 0.84608456 0.79964158 0.96026121]
mIoU macro = 0.8206 | Dice macro = 0.8980
🔹 Nuevo mejor mIoU: 0.8206 | Dice: 0.8980  →  guardando modelo…

--- Epoch 88/200 ---
100% 210/210 [00:29<00:00,  7.13it/s, loss=0.0602]
Calculando métricas de entrenamiento...
IoU por clase : [0.97078519 0.87265304 0.89962398 0.82380102 0.83954445 0.94665191]
Dice por clase: [0.98517606 0.9319965  0.94716006 0.90338914 0.9127743  0.97259495]
mIoU macro = 0.8922 | Dice macro = 0.9422
Calculando métricas de validación...
IoU por clase : [0.95813082 0.79764247 0.83774454 0.73832039 0.64551703 0.91977446]
Dice por clase: [0.97861778 0.88743171 0.91170946 0.84946411 0.78457654 0.95821096]
mIoU macro = 0.8162 | Dice macro = 0.8950

--- Epoch 89/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.0615]
Calculando métricas de entrenamiento...
IoU por clase : [0.97028954 0.8737563  0.89908488 0.81269143 0.83583603 0.94732158]
Dice por clase: [0.98492077 0.93262534 0.94686118 0.89666825 0.91057809 0.97294827]
mIoU macro = 0.8898 | Dice macro = 0.9408
Calculando métricas de validación...
IoU por clase : [0.95678481 0.8009906  0.83024852 0.71245597 0.639451   0.92199918]
Dice por clase: [0.97791521 0.88950004 0.90725222 0.83208676 0.78007943 0.95941683]
mIoU macro = 0.8103 | Dice macro = 0.8910

--- Epoch 90/200 ---
100% 210/210 [00:26<00:00,  8.03it/s, loss=0.046]
Calculando métricas de entrenamiento...
IoU por clase : [0.97109164 0.8735291  0.90141929 0.82800594 0.83933569 0.94745438]
Dice por clase: [0.98533383 0.9324959  0.94815414 0.90591165 0.9126509  0.97301831]
mIoU macro = 0.8935 | Dice macro = 0.9429
Calculando métricas de validación...
IoU por clase : [0.95814887 0.79699653 0.83972244 0.73238498 0.63854117 0.91939741]
Dice por clase: [0.9786272  0.88703179 0.91287949 0.8455222  0.77940205 0.95800631]
mIoU macro = 0.8142 | Dice macro = 0.8936

--- Epoch 91/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0683]
Calculando métricas de entrenamiento...
IoU por clase : [0.97025824 0.87105595 0.89706868 0.81798937 0.83604822 0.94601389]
Dice por clase: [0.98490464 0.93108488 0.94574191 0.89988355 0.91070399 0.97225811]
mIoU macro = 0.8897 | Dice macro = 0.9408
Calculando métricas de validación...
IoU por clase : [0.95823668 0.80167996 0.82270592 0.73102762 0.615957   0.920205  ]
Dice por clase: [0.97867299 0.88992494 0.90273029 0.84461694 0.7623433  0.95844454]
mIoU macro = 0.8083 | Dice macro = 0.8895

--- Epoch 92/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.092]
Calculando métricas de entrenamiento...
IoU por clase : [0.97106423 0.87051149 0.89863099 0.81931543 0.83848838 0.94709704]
Dice por clase: [0.98531972 0.93077374 0.94660942 0.90068541 0.91214977 0.97282983]
mIoU macro = 0.8909 | Dice macro = 0.9414
Calculando métricas de validación...
IoU por clase : [0.9582188  0.79337111 0.84071821 0.72543162 0.64355699 0.92149983]
Dice por clase: [0.97866367 0.88478186 0.91346759 0.84086974 0.78312708 0.95914641]
mIoU macro = 0.8138 | Dice macro = 0.8933

--- Epoch 93/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0599]
Calculando métricas de entrenamiento...
IoU por clase : [0.97084855 0.8628317  0.89663015 0.82909275 0.82200191 0.94712426]
Dice por clase: [0.98520868 0.92636571 0.94549815 0.90656174 0.90230631 0.97284419]
mIoU macro = 0.8881 | Dice macro = 0.9398
Calculando métricas de validación...
IoU por clase : [0.95825928 0.79186193 0.83316754 0.74115835 0.63335754 0.92294963]
Dice por clase: [0.97868478 0.88384258 0.90899225 0.8513394  0.77552835 0.95993115]
mIoU macro = 0.8135 | Dice macro = 0.8931

--- Epoch 94/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0712]
Calculando métricas de entrenamiento...
IoU por clase : [0.97136402 0.87366617 0.90320325 0.83243522 0.84374161 0.94792922]
Dice por clase: [0.98547403 0.93257399 0.94914009 0.90855623 0.9152493  0.97326865]
mIoU macro = 0.8954 | Dice macro = 0.9440
Calculando métricas de validación...
IoU por clase : [0.95839694 0.79889875 0.83538566 0.74317313 0.63909074 0.92112281]
Dice por clase: [0.97875658 0.88820869 0.91031076 0.85266703 0.7798113  0.95894214]
mIoU macro = 0.8160 | Dice macro = 0.8948

--- Epoch 95/200 ---
100% 210/210 [00:27<00:00,  7.75it/s, loss=0.0526]
Calculando métricas de entrenamiento...
IoU por clase : [0.97160527 0.87623946 0.90216799 0.83393385 0.8442067  0.94713021]
Dice por clase: [0.98559817 0.93403798 0.94856816 0.90944813 0.91552286 0.97284733]
mIoU macro = 0.8959 | Dice macro = 0.9443
Calculando métricas de validación...
IoU por clase : [0.95876371 0.8044837  0.83760537 0.7436434  0.65762864 0.92293892]
Dice por clase: [0.9789478  0.89164973 0.91162703 0.85297648 0.79345714 0.95992536]
mIoU macro = 0.8208 | Dice macro = 0.8981
🔹 Nuevo mejor mIoU: 0.8208 | Dice: 0.8981  →  guardando modelo…

--- Epoch 96/200 ---
100% 210/210 [00:28<00:00,  7.39it/s, loss=0.0524]
Calculando métricas de entrenamiento...
IoU por clase : [0.97169178 0.87494628 0.90266126 0.83145069 0.85038364 0.94829108]
Dice por clase: [0.98564267 0.93330277 0.94884074 0.9079695  0.91914306 0.97345935]
mIoU macro = 0.8966 | Dice macro = 0.9447
Calculando métricas de validación...
IoU por clase : [0.95887617 0.80168027 0.83950128 0.7402277  0.65777395 0.92136989]
Dice por clase: [0.97900642 0.88992512 0.91274879 0.85072511 0.7935629  0.95907601]
mIoU macro = 0.8199 | Dice macro = 0.8975

--- Epoch 97/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0513]
Calculando métricas de entrenamiento...
IoU por clase : [0.97145087 0.87682798 0.90129207 0.83130437 0.84740728 0.94563955]
Dice por clase: [0.98551872 0.93437224 0.94808376 0.90788226 0.91740169 0.97206037]
mIoU macro = 0.8957 | Dice macro = 0.9442
Calculando métricas de validación...
IoU por clase : [0.95832856 0.80066338 0.8305574  0.73361524 0.6722888  0.92187621]
Dice por clase: [0.97872092 0.88929823 0.90743661 0.84634148 0.80403433 0.95935025]
mIoU macro = 0.8196 | Dice macro = 0.8975

--- Epoch 98/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0607]
Calculando métricas de entrenamiento...
IoU por clase : [0.97147769 0.86954076 0.90502121 0.83272337 0.84725167 0.94755709]
Dice por clase: [0.98553252 0.93021856 0.95014292 0.90872784 0.91731049 0.97307247]
mIoU macro = 0.8956 | Dice macro = 0.9442
Calculando métricas de validación...
IoU por clase : [0.95821662 0.79901015 0.84168797 0.73411522 0.66053368 0.92032118]
Dice por clase: [0.97866253 0.88827753 0.91403971 0.8466741  0.79556794 0.95850755]
mIoU macro = 0.8190 | Dice macro = 0.8970

--- Epoch 99/200 ---
100% 210/210 [00:26<00:00,  7.84it/s, loss=0.0422]
Calculando métricas de entrenamiento...
IoU por clase : [0.97203979 0.87829129 0.90616076 0.8336322  0.84994259 0.94833173]
Dice por clase: [0.98582168 0.93520243 0.95077056 0.90926872 0.91888537 0.97348076]
mIoU macro = 0.8981 | Dice macro = 0.9456
Calculando métricas de validación...
IoU por clase : [0.95861947 0.80581592 0.83541249 0.73614029 0.64841714 0.92230601]
Dice por clase: [0.97887261 0.8924674  0.91032669 0.84801936 0.78671487 0.95958292]
mIoU macro = 0.8178 | Dice macro = 0.8960

--- Epoch 100/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0476]
Calculando métricas de entrenamiento...
IoU por clase : [0.97223056 0.87070272 0.90702112 0.83386741 0.84611745 0.9485479 ]
Dice por clase: [0.98591978 0.93088304 0.95124392 0.90940861 0.91664531 0.97359464]
mIoU macro = 0.8964 | Dice macro = 0.9446
Calculando métricas de validación...
IoU por clase : [0.95876268 0.79394224 0.83908121 0.73521539 0.65015413 0.92220381]
Dice por clase: [0.97894726 0.8851369  0.91250045 0.84740534 0.787992   0.95952761]
mIoU macro = 0.8166 | Dice macro = 0.8953

--- Epoch 101/200 ---
100% 210/210 [00:25<00:00,  8.10it/s, loss=0.0473]
Calculando métricas de entrenamiento...
IoU por clase : [0.9722239  0.87813534 0.90622189 0.83505975 0.85352114 0.94807066]
Dice por clase: [0.98591636 0.93511402 0.9508042  0.91011723 0.92097265 0.97334319]
mIoU macro = 0.8989 | Dice macro = 0.9460
Calculando métricas de validación...
IoU por clase : [0.95897233 0.80507507 0.82742869 0.73448943 0.64157042 0.92334116]
Dice por clase: [0.97905653 0.89201284 0.90556605 0.84692292 0.78165446 0.96014288]
mIoU macro = 0.8151 | Dice macro = 0.8942

--- Epoch 102/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0503]
Calculando métricas de entrenamiento...
IoU por clase : [0.97210042 0.87920398 0.9033327  0.83312701 0.84756528 0.949056  ]
Dice por clase: [0.98585286 0.93571958 0.94921156 0.90896812 0.91749427 0.97386222]
mIoU macro = 0.8974 | Dice macro = 0.9452
Calculando métricas de validación...
IoU por clase : [0.95853421 0.80151421 0.83617482 0.72663997 0.65050229 0.92304074]
Dice por clase: [0.97882815 0.8898228  0.91077909 0.84168093 0.78824766 0.95998043]
mIoU macro = 0.8161 | Dice macro = 0.8949

--- Epoch 103/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.054]
Calculando métricas de entrenamiento...
IoU por clase : [0.97223882 0.88158336 0.90437871 0.83317009 0.8405571  0.94878532]
Dice por clase: [0.98592403 0.93706543 0.94978872 0.90899377 0.91337248 0.97371969]
mIoU macro = 0.8968 | Dice macro = 0.9448
Calculando métricas de validación...
IoU por clase : [0.95895617 0.80664145 0.82800376 0.74131784 0.6200886  0.92396896]
Dice por clase: [0.97904811 0.89297348 0.90591035 0.8514446  0.76549962 0.96048219]
mIoU macro = 0.8132 | Dice macro = 0.8926

--- Epoch 104/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0413]
Calculando métricas de entrenamiento...
IoU por clase : [0.97203283 0.87647495 0.9021778  0.83136466 0.84485367 0.94850871]
Dice por clase: [0.9858181  0.93417176 0.94857358 0.9079182  0.91590318 0.973574  ]
mIoU macro = 0.8959 | Dice macro = 0.9443
Calculando métricas de validación...
IoU por clase : [0.95915423 0.7896805  0.84417814 0.73035384 0.64772128 0.92079604]
Dice por clase: [0.97915133 0.8824821  0.91550607 0.84416704 0.78620248 0.95876503]
mIoU macro = 0.8153 | Dice macro = 0.8944

--- Epoch 105/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.0493]
Calculando métricas de entrenamiento...
IoU por clase : [0.97164976 0.87728654 0.9029522  0.82959296 0.84081529 0.94761531]
Dice por clase: [0.98562106 0.93463254 0.94900145 0.90686068 0.91352489 0.97310316]
mIoU macro = 0.8950 | Dice macro = 0.9438
Calculando métricas de validación...
IoU por clase : [0.9585689  0.79721477 0.82938335 0.73482153 0.63758345 0.92192676]
Dice por clase: [0.97884624 0.88716695 0.90673543 0.84714366 0.77868819 0.95937762]
mIoU macro = 0.8132 | Dice macro = 0.8930

--- Epoch 106/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0371]
Calculando métricas de entrenamiento...
IoU por clase : [0.97132274 0.87419525 0.89820731 0.82794742 0.83232148 0.94772314]
Dice por clase: [0.98545278 0.93287532 0.9463743  0.90587663 0.90848848 0.97316002]
mIoU macro = 0.8920 | Dice macro = 0.9420
Calculando métricas de validación...
IoU por clase : [0.95895171 0.79394105 0.82190843 0.72451444 0.60019152 0.92145191]
Dice por clase: [0.97904579 0.88513616 0.90224999 0.84025326 0.75014961 0.95912045]
mIoU macro = 0.8035 | Dice macro = 0.8860

--- Epoch 107/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.0595]
Calculando métricas de entrenamiento...
IoU por clase : [0.97252746 0.88086114 0.904929   0.83687759 0.85271041 0.9492004 ]
Dice por clase: [0.98607242 0.93665728 0.9500921  0.91119582 0.92050048 0.97393824]
mIoU macro = 0.8995 | Dice macro = 0.9464
Calculando métricas de validación...
IoU por clase : [0.95885429 0.80029537 0.83230652 0.73433476 0.64015417 0.92321615]
Dice por clase: [0.97899501 0.88907119 0.90847957 0.84682009 0.78060243 0.96007529]
mIoU macro = 0.8149 | Dice macro = 0.8940

--- Epoch 108/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0499]
Calculando métricas de entrenamiento...
IoU por clase : [0.97192001 0.88037703 0.9055255  0.82780482 0.84711712 0.94772974]
Dice por clase: [0.98576008 0.93638352 0.95042076 0.90579126 0.91723163 0.9731635 ]
mIoU macro = 0.8967 | Dice macro = 0.9448
Calculando métricas de validación...
IoU por clase : [0.95852225 0.80300185 0.82987251 0.72079783 0.6270911  0.92183941]
Dice por clase: [0.97882192 0.8907388  0.90702768 0.83774842 0.77081253 0.95933032]
mIoU macro = 0.8102 | Dice macro = 0.8907

--- Epoch 109/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0523]
Calculando métricas de entrenamiento...
IoU por clase : [0.97244047 0.87998059 0.90493055 0.83351955 0.85356313 0.94912292]
Dice por clase: [0.9860277  0.93615923 0.95009296 0.9092017  0.92099709 0.97389745]
mIoU macro = 0.8989 | Dice macro = 0.9461
Calculando métricas de validación...
IoU por clase : [0.95896422 0.80144345 0.83992972 0.73171757 0.65214607 0.92332093]
Dice por clase: [0.97905231 0.8897792  0.91300196 0.84507726 0.78945329 0.96013194]
mIoU macro = 0.8179 | Dice macro = 0.8961

--- Epoch 110/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0496]
Calculando métricas de entrenamiento...
IoU por clase : [0.97290514 0.88210454 0.90809095 0.83684444 0.85686205 0.94992365]
Dice por clase: [0.98626652 0.93735977 0.95183194 0.91117617 0.92291407 0.97431882]
mIoU macro = 0.9011 | Dice macro = 0.9473
Calculando métricas de validación...
IoU por clase : [0.95987661 0.803612   0.83284987 0.73897054 0.65509155 0.92478283]
Dice por clase: [0.97952759 0.89111405 0.90880315 0.84989426 0.79160763 0.96092174]
mIoU macro = 0.8192 | Dice macro = 0.8970

--- Epoch 111/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0552]
Calculando métricas de entrenamiento...
IoU por clase : [0.97229723 0.8793433  0.90851924 0.8333268  0.85573869 0.9471633 ]
Dice por clase: [0.98595406 0.93579848 0.95206715 0.90908702 0.92226206 0.97286478]
mIoU macro = 0.8994 | Dice macro = 0.9463
Calculando métricas de validación...
IoU por clase : [0.95861075 0.79932167 0.83529356 0.73103946 0.65941161 0.92230331]
Dice por clase: [0.97886806 0.88847001 0.91025608 0.84462484 0.79475352 0.95958146]
mIoU macro = 0.8177 | Dice macro = 0.8961

--- Epoch 112/200 ---
100% 210/210 [00:26<00:00,  8.02it/s, loss=0.0426]
Calculando métricas de entrenamiento...
IoU por clase : [0.97292751 0.8822238  0.91025759 0.84027986 0.85843736 0.9499346 ]
Dice por clase: [0.98627801 0.9374271  0.95302078 0.91320878 0.92382706 0.97432457]
mIoU macro = 0.9023 | Dice macro = 0.9480
Calculando métricas de validación...
IoU por clase : [0.95952356 0.80851389 0.83574595 0.74408035 0.65584109 0.92557106]
Dice por clase: [0.97934373 0.89411963 0.91052463 0.85326384 0.79215463 0.96134708]
mIoU macro = 0.8215 | Dice macro = 0.8985
🔹 Nuevo mejor mIoU: 0.8215 | Dice: 0.8985  →  guardando modelo…

--- Epoch 113/200 ---
100% 210/210 [00:28<00:00,  7.38it/s, loss=0.0525]
Calculando métricas de entrenamiento...
IoU por clase : [0.97290062 0.88122229 0.9106034  0.8385762  0.86140224 0.94919559]
Dice por clase: [0.98626419 0.93686142 0.95321028 0.91220174 0.92554121 0.97393571]
mIoU macro = 0.9023 | Dice macro = 0.9480
Calculando métricas de validación...
IoU por clase : [0.95911806 0.80286557 0.8346859  0.73419126 0.64444852 0.92290537]
Dice por clase: [0.97913248 0.89065494 0.90989515 0.84672467 0.7837868  0.95990722]
mIoU macro = 0.8164 | Dice macro = 0.8950

--- Epoch 114/200 ---
100% 210/210 [00:26<00:00,  8.00it/s, loss=0.0557]
Calculando métricas de entrenamiento...
IoU por clase : [0.97273027 0.88103351 0.90490102 0.83914583 0.84656028 0.9486376 ]
Dice por clase: [0.98617666 0.93675472 0.95007668 0.91253865 0.91690511 0.97364189]
mIoU macro = 0.8988 | Dice macro = 0.9460
Calculando métricas de validación...
IoU por clase : [0.95894347 0.80473775 0.82556117 0.74054605 0.64666047 0.92371595]
Dice por clase: [0.97904149 0.89180575 0.90444646 0.85093532 0.78542053 0.96034547]
mIoU macro = 0.8167 | Dice macro = 0.8953

--- Epoch 115/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0501]
Calculando métricas de entrenamiento...
IoU por clase : [0.97254747 0.88119558 0.90583755 0.82935084 0.85150053 0.94921706]
Dice por clase: [0.9860827  0.93684632 0.95059261 0.906716   0.91979507 0.973947  ]
mIoU macro = 0.8983 | Dice macro = 0.9457
Calculando métricas de validación...
IoU por clase : [0.959202   0.80648376 0.82563341 0.72627316 0.66207961 0.92416423]
Dice por clase: [0.97917622 0.89287684 0.90448981 0.8414348  0.79668821 0.96058769]
mIoU macro = 0.8173 | Dice macro = 0.8959

--- Epoch 116/200 ---
100% 210/210 [00:27<00:00,  7.72it/s, loss=0.0563]
Calculando métricas de entrenamiento...
IoU por clase : [0.97169306 0.87764208 0.90111747 0.82932754 0.83868426 0.94827264]
Dice por clase: [0.98564333 0.93483427 0.94798715 0.90670208 0.91226566 0.97344963]
mIoU macro = 0.8945 | Dice macro = 0.9435
Calculando métricas de validación...
IoU por clase : [0.95853189 0.79596024 0.82298256 0.7268545  0.63008767 0.92377574]
Dice por clase: [0.97882694 0.8863896  0.9028968  0.84182483 0.77307213 0.96037778]
mIoU macro = 0.8097 | Dice macro = 0.8906

--- Epoch 117/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0555]
Calculando métricas de entrenamiento...
IoU por clase : [0.97244366 0.8828677  0.90435749 0.83598067 0.8520616  0.94974317]
Dice por clase: [0.98602934 0.93779048 0.94977702 0.91066391 0.92012231 0.97422387]
mIoU macro = 0.8996 | Dice macro = 0.9464
Calculando métricas de validación...
IoU por clase : [0.95926055 0.80047782 0.82454152 0.73197354 0.6486948  0.92439862]
Dice por clase: [0.97920672 0.88918376 0.90383421 0.84524795 0.78691921 0.96071428]
mIoU macro = 0.8149 | Dice macro = 0.8942

--- Epoch 118/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0674]
Calculando métricas de entrenamiento...
IoU por clase : [0.97268521 0.88122102 0.9085291  0.84012339 0.85231351 0.94958563]
Dice por clase: [0.9861535  0.9368607  0.95207257 0.91311636 0.92026917 0.97414098]
mIoU macro = 0.9007 | Dice macro = 0.9471
Calculando métricas de validación...
IoU por clase : [0.95942581 0.80539455 0.82460088 0.74082433 0.64267622 0.92554102]
Dice por clase: [0.97929282 0.89220891 0.90386987 0.851119   0.78247461 0.96133088]
mIoU macro = 0.8164 | Dice macro = 0.8950

--- Epoch 119/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0499]
Calculando métricas de entrenamiento...
IoU por clase : [0.97331369 0.88496033 0.90785663 0.83897976 0.86021929 0.95075403]
Dice por clase: [0.9864764  0.93896971 0.9517032  0.91244045 0.92485794 0.97475542]
mIoU macro = 0.9027 | Dice macro = 0.9482
Calculando métricas de validación...
IoU por clase : [0.959681   0.81135007 0.82258115 0.73278151 0.63663078 0.92596759]
Dice por clase: [0.97942573 0.8958512  0.90265517 0.84578639 0.77797728 0.96156093]
mIoU macro = 0.8148 | Dice macro = 0.8939

--- Epoch 120/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0506]
Calculando métricas de entrenamiento...
IoU por clase : [0.97288253 0.87925773 0.90884462 0.83890874 0.84538786 0.94923678]
Dice por clase: [0.9862549  0.93575002 0.95224578 0.91239845 0.91621699 0.97395739]
mIoU macro = 0.8991 | Dice macro = 0.9461
Calculando métricas de validación...
IoU por clase : [0.95895033 0.80169681 0.82986028 0.73077789 0.6416368  0.92444061]
Dice por clase: [0.97904507 0.88993532 0.90702038 0.84445022 0.78170373 0.96073696]
mIoU macro = 0.8146 | Dice macro = 0.8938

--- Epoch 121/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0498]
Calculando métricas de entrenamiento...
IoU por clase : [0.97327066 0.88039237 0.9076479  0.83943167 0.84904659 0.94911432]
Dice por clase: [0.9864543  0.9363922  0.9515885  0.91270764 0.91836149 0.97389292]
mIoU macro = 0.8998 | Dice macro = 0.9466
Calculando métricas de validación...
IoU por clase : [0.95922993 0.79971912 0.82509615 0.72904569 0.63581443 0.92313209]
Dice por clase: [0.97919077 0.88871548 0.90416732 0.84329257 0.77736743 0.96002983]
mIoU macro = 0.8120 | Dice macro = 0.8921

--- Epoch 122/200 ---
100% 210/210 [00:26<00:00,  7.95it/s, loss=0.0695]
Calculando métricas de entrenamiento...
IoU por clase : [0.97323254 0.88405308 0.9099     0.8387037  0.85650677 0.95081152]
Dice por clase: [0.98643471 0.93845878 0.95282476 0.91227717 0.92270794 0.97478563]
mIoU macro = 0.9022 | Dice macro = 0.9479
Calculando métricas de validación...
IoU por clase : [0.95881059 0.80429692 0.82527508 0.7258447  0.63786834 0.92459225]
Dice por clase: [0.97897223 0.89153499 0.90427475 0.84114718 0.77890063 0.96081884]
mIoU macro = 0.8128 | Dice macro = 0.8926

--- Epoch 123/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0426]
Calculando métricas de entrenamiento...
IoU por clase : [0.97343639 0.87571999 0.91082118 0.84114867 0.85997516 0.95051397]
Dice por clase: [0.98653942 0.93374277 0.95332958 0.91372162 0.92471682 0.97462924]
mIoU macro = 0.9019 | Dice macro = 0.9478
Calculando métricas de validación...
IoU por clase : [0.95947845 0.79617044 0.82161957 0.73371705 0.64523076 0.92660378]
Dice por clase: [0.97932024 0.88651992 0.90207592 0.84640922 0.78436506 0.96190383]
mIoU macro = 0.8138 | Dice macro = 0.8934

--- Epoch 124/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0537]
Calculando métricas de entrenamiento...
IoU por clase : [0.97375779 0.88408583 0.91377967 0.83657725 0.86446239 0.9507991 ]
Dice por clase: [0.98670444 0.93847724 0.95494762 0.91101776 0.92730472 0.9747791 ]
mIoU macro = 0.9039 | Dice macro = 0.9489
Calculando métricas de validación...
IoU por clase : [0.95976041 0.80631789 0.82917392 0.73182931 0.65336283 0.92581776]
Dice por clase: [0.97946709 0.89277518 0.90661026 0.84515178 0.79034416 0.96148013]
mIoU macro = 0.8177 | Dice macro = 0.8960

--- Epoch 125/200 ---
100% 210/210 [00:26<00:00,  7.98it/s, loss=0.0493]
Calculando métricas de entrenamiento...
IoU por clase : [0.97380082 0.88615328 0.90874304 0.84759652 0.86573118 0.95133132]
Dice por clase: [0.98672653 0.93964079 0.95219002 0.91751258 0.92803421 0.97505873]
mIoU macro = 0.9056 | Dice macro = 0.9499
Calculando métricas de validación...
IoU por clase : [0.95982645 0.80545211 0.82329444 0.74189133 0.64714518 0.92504423]
Dice por clase: [0.97950148 0.89224422 0.90308446 0.85182275 0.78577795 0.96106283]
mIoU macro = 0.8171 | Dice macro = 0.8956

--- Epoch 126/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0446]
Calculando métricas de entrenamiento...
IoU por clase : [0.97385487 0.88520077 0.91163468 0.84002693 0.8649963  0.95085   ]
Dice por clase: [0.98675428 0.93910504 0.95377499 0.91305939 0.92761181 0.97480585]
mIoU macro = 0.9044 | Dice macro = 0.9492
Calculando métricas de validación...
IoU por clase : [0.95986831 0.80661631 0.83481667 0.7336267  0.65973671 0.92561882]
Dice por clase: [0.97952327 0.89295807 0.90997284 0.8463491  0.7949896  0.96137284]
mIoU macro = 0.8200 | Dice macro = 0.8975

--- Epoch 127/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0468]
Calculando métricas de entrenamiento...
IoU por clase : [0.97393625 0.88660716 0.91403527 0.83993756 0.86117806 0.95089052]
Dice por clase: [0.98679605 0.93989589 0.95508718 0.91300659 0.92541179 0.97482715]
mIoU macro = 0.9044 | Dice macro = 0.9492
Calculando métricas de validación...
IoU por clase : [0.95954831 0.80142603 0.83612589 0.73157046 0.66178601 0.92396034]
Dice por clase: [0.97935662 0.88976846 0.91075007 0.84497914 0.79647561 0.96047753]
mIoU macro = 0.8191 | Dice macro = 0.8970

--- Epoch 128/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0539]
Calculando métricas de entrenamiento...
IoU por clase : [0.9708995  0.87306513 0.88601451 0.82689513 0.80956235 0.94373426]
Dice por clase: [0.98523491 0.93223147 0.93956277 0.90524641 0.89476038 0.97105276]
mIoU macro = 0.8850 | Dice macro = 0.9380
Calculando métricas de validación...
IoU por clase : [0.95736872 0.79331454 0.77364069 0.72186586 0.54446184 0.91646484]
Dice por clase: [0.97822011 0.88474668 0.87237589 0.83846933 0.70505056 0.95641185]
mIoU macro = 0.7845 | Dice macro = 0.8725

--- Epoch 129/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.072]
Calculando métricas de entrenamiento...
IoU por clase : [0.97213354 0.87423792 0.90080953 0.8352346  0.84293305 0.94816578]
Dice por clase: [0.98586989 0.93289962 0.94781672 0.91022107 0.91477338 0.97339332]
mIoU macro = 0.8956 | Dice macro = 0.9442
Calculando métricas de validación...
IoU por clase : [0.95900134 0.7880123  0.82951658 0.74461824 0.64134672 0.92302058]
Dice por clase: [0.97907165 0.88143946 0.90681505 0.8536174  0.78148841 0.95996953]
mIoU macro = 0.8143 | Dice macro = 0.8937

--- Epoch 130/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0537]
Calculando métricas de entrenamiento...
IoU por clase : [0.97278033 0.87958196 0.90452516 0.83495142 0.84970419 0.94990274]
Dice por clase: [0.98620238 0.9359336  0.94986948 0.91005289 0.91874603 0.97430782]
mIoU macro = 0.8986 | Dice macro = 0.9459
Calculando métricas de validación...
IoU por clase : [0.95883102 0.79773674 0.83447812 0.70351941 0.62893203 0.92224067]
Dice por clase: [0.97898288 0.88749006 0.90977168 0.82595996 0.77220169 0.95954756]
mIoU macro = 0.8076 | Dice macro = 0.8890

--- Epoch 131/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.048]
Calculando métricas de entrenamiento...
IoU por clase : [0.9735868  0.88496306 0.90887996 0.83967715 0.8578949  0.95066386]
Dice por clase: [0.98661665 0.93897125 0.95226518 0.91285272 0.92351284 0.97470803]
mIoU macro = 0.9026 | Dice macro = 0.9482
Calculando métricas de validación...
IoU por clase : [0.95978475 0.81109004 0.83211497 0.73796307 0.6602187  0.92547216]
Dice por clase: [0.97947976 0.89569267 0.90836545 0.84922756 0.79533944 0.96129373]
mIoU macro = 0.8211 | Dice macro = 0.8982

--- Epoch 132/200 ---
100% 210/210 [00:26<00:00,  7.94it/s, loss=0.0507]
Calculando métricas de entrenamiento...
IoU por clase : [0.97383957 0.8857067  0.91288137 0.84161781 0.85990577 0.95141992]
Dice por clase: [0.98674643 0.93938967 0.95445686 0.91399834 0.92467671 0.97510526]
mIoU macro = 0.9042 | Dice macro = 0.9491
Calculando métricas de validación...
IoU por clase : [0.95944778 0.80155804 0.83854307 0.73701511 0.65547004 0.92547584]
Dice por clase: [0.97930426 0.88984981 0.91218213 0.84859954 0.79188391 0.96129572]
mIoU macro = 0.8196 | Dice macro = 0.8972

--- Epoch 133/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0497]
Calculando métricas de entrenamiento...
IoU por clase : [0.9735635  0.88552434 0.91311325 0.84414842 0.86324568 0.95130514]
Dice por clase: [0.98660469 0.93928709 0.95458358 0.91548859 0.92660425 0.97504498]
mIoU macro = 0.9052 | Dice macro = 0.9496
Calculando métricas de validación...
IoU por clase : [0.95964632 0.80974567 0.83688706 0.74687692 0.6651679  0.92551457]
Dice por clase: [0.97940767 0.89487234 0.91120143 0.85509965 0.79891992 0.96131661]
mIoU macro = 0.8240 | Dice macro = 0.9001
🔹 Nuevo mejor mIoU: 0.8240 | Dice: 0.9001  →  guardando modelo…

--- Epoch 134/200 ---
100% 210/210 [00:28<00:00,  7.31it/s, loss=0.0444]
Calculando métricas de entrenamiento...
IoU por clase : [0.97434384 0.88834785 0.91362893 0.84841121 0.87011709 0.95115625]
Dice por clase: [0.98700522 0.9408731  0.9548653  0.91798968 0.93054825 0.97496676]
mIoU macro = 0.9077 | Dice macro = 0.9510
Calculando métricas de validación...
IoU por clase : [0.96003056 0.8076804  0.83641336 0.7459013  0.66083049 0.92411352]
Dice por clase: [0.97960775 0.89360974 0.91092058 0.85445987 0.79578319 0.96056029]
mIoU macro = 0.8225 | Dice macro = 0.8992

--- Epoch 135/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0549]
Calculando métricas de entrenamiento...
IoU por clase : [0.97453809 0.888963   0.91532594 0.84895423 0.86878731 0.95232741]
Dice por clase: [0.98710488 0.94121801 0.9557913  0.91830746 0.92978726 0.97558166]
mIoU macro = 0.9081 | Dice macro = 0.9513
Calculando métricas de validación...
IoU por clase : [0.96019284 0.80641354 0.83515028 0.7484785  0.65880796 0.92499095]
Dice por clase: [0.97969222 0.89283381 0.910171   0.85614836 0.79431493 0.96103408]
mIoU macro = 0.8223 | Dice macro = 0.8990

--- Epoch 136/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.0835]
Calculando métricas de entrenamiento...
IoU por clase : [0.97298691 0.88516395 0.87049755 0.83069991 0.8010452  0.92295088]
Dice por clase: [0.98630853 0.93908432 0.93076577 0.90752166 0.8895337  0.95993183]
mIoU macro = 0.8806 | Dice macro = 0.9355
Calculando métricas de validación...
IoU por clase : [0.9584995  0.78237853 0.78327622 0.7294462  0.64838796 0.88408123]
Dice por clase: [0.97881005 0.8779039  0.87846875 0.84356044 0.78669339 0.93847464]
mIoU macro = 0.7977 | Dice macro = 0.8840

--- Epoch 137/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0593]
Calculando métricas de entrenamiento...
IoU por clase : [0.97386069 0.88574953 0.9107588  0.8429582  0.85537046 0.95118383]
Dice por clase: [0.98675727 0.93941376 0.95329541 0.91478819 0.92204816 0.97498126]
mIoU macro = 0.9033 | Dice macro = 0.9485
Calculando métricas de validación...
IoU por clase : [0.95979319 0.79879583 0.8524969  0.74365164 0.70159965 0.92528817]
Dice por clase: [0.97948415 0.88814507 0.92037606 0.85298189 0.8246354  0.96119447]
mIoU macro = 0.8303 | Dice macro = 0.9045
🔹 Nuevo mejor mIoU: 0.8303 | Dice: 0.9045  →  guardando modelo…

--- Epoch 138/200 ---
100% 210/210 [00:28<00:00,  7.39it/s, loss=0.0356]
Calculando métricas de entrenamiento...
IoU por clase : [0.97407685 0.8874501  0.91151976 0.84550861 0.8655051  0.95157092]
Dice por clase: [0.98686822 0.94036934 0.9537121  0.91628791 0.9279043  0.97518456]
mIoU macro = 0.9059 | Dice macro = 0.9501
Calculando métricas de validación...
IoU por clase : [0.96022597 0.80373288 0.83928105 0.75038297 0.67377071 0.92499451]
Dice por clase: [0.97970947 0.89118837 0.9126186  0.8573929  0.8050932  0.961036  ]
mIoU macro = 0.8254 | Dice macro = 0.9012

--- Epoch 139/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0476]
Calculando métricas de entrenamiento...
IoU por clase : [0.97447351 0.88866129 0.91311606 0.84728871 0.86761918 0.95225991]
Dice por clase: [0.98707175 0.94104887 0.95458512 0.9173322  0.92911787 0.97554624]
mIoU macro = 0.9072 | Dice macro = 0.9508
Calculando métricas de validación...
IoU por clase : [0.96033633 0.80490594 0.83796732 0.74505216 0.68609962 0.92507953]
Dice por clase: [0.9797669  0.89190901 0.91184137 0.85390245 0.81383047 0.96108189]
mIoU macro = 0.8266 | Dice macro = 0.9021

--- Epoch 140/200 ---
100% 210/210 [00:26<00:00,  7.84it/s, loss=0.0521]
Calculando métricas de entrenamiento...
IoU por clase : [0.97452056 0.88878924 0.91569012 0.84962362 0.86603255 0.9519879 ]
Dice por clase: [0.98709588 0.94112061 0.95598981 0.91869893 0.92820733 0.97540349]
mIoU macro = 0.9078 | Dice macro = 0.9511
Calculando métricas de validación...
IoU por clase : [0.96008678 0.80508619 0.83791946 0.74658443 0.6731892  0.92373665]
Dice por clase: [0.97963701 0.89201966 0.91181304 0.85490792 0.80467792 0.96035666]
mIoU macro = 0.8244 | Dice macro = 0.9006

--- Epoch 141/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0456]
Calculando métricas de entrenamiento...
IoU por clase : [0.97462206 0.88810499 0.91642882 0.84406025 0.86600156 0.95204554]
Dice por clase: [0.98714795 0.94073687 0.95639223 0.91543674 0.92818953 0.97543374]
mIoU macro = 0.9069 | Dice macro = 0.9506
Calculando métricas de validación...
IoU por clase : [0.96017639 0.80467096 0.83855332 0.73899789 0.67774572 0.92508769]
Dice por clase: [0.97968366 0.89176473 0.9121882  0.84991235 0.80792424 0.96108629]
mIoU macro = 0.8242 | Dice macro = 0.9004

--- Epoch 142/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0432]
Calculando métricas de entrenamiento...
IoU por clase : [0.97476934 0.88880732 0.91569752 0.84981691 0.8706297  0.95246519]
Dice por clase: [0.98722349 0.94113074 0.95599384 0.91881192 0.93084131 0.97565395]
mIoU macro = 0.9087 | Dice macro = 0.9516
Calculando métricas de validación...
IoU por clase : [0.96001718 0.80651473 0.84008452 0.74121727 0.68562291 0.92434168]
Dice por clase: [0.97960078 0.89289582 0.91309341 0.85137827 0.813495   0.96068353]
mIoU macro = 0.8263 | Dice macro = 0.9019

--- Epoch 143/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0502]
Calculando métricas de entrenamiento...
IoU por clase : [0.97405798 0.8793942  0.91297835 0.84502649 0.86514845 0.95224525]
Dice por clase: [0.98685853 0.9358273  0.95450986 0.91600472 0.92769929 0.97553855]
mIoU macro = 0.9048 | Dice macro = 0.9494
Calculando métricas de validación...
IoU por clase : [0.95958858 0.79797575 0.82720189 0.73812597 0.64287083 0.92421238]
Dice por clase: [0.9793776  0.88763795 0.9054302  0.84933542 0.78261884 0.9606137 ]
mIoU macro = 0.8150 | Dice macro = 0.8942

--- Epoch 144/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0714]
Calculando métricas de entrenamiento...
IoU por clase : [0.97473806 0.89090733 0.91306461 0.85046496 0.86955148 0.95271876]
Dice por clase: [0.98720745 0.94230671 0.954557   0.91919056 0.9302247  0.97578697]
mIoU macro = 0.9086 | Dice macro = 0.9515
Calculando métricas de validación...
IoU por clase : [0.96002117 0.80842235 0.83864955 0.74549293 0.65835246 0.92448264]
Dice por clase: [0.97960286 0.89406366 0.91224513 0.85419186 0.79398376 0.96075966]
mIoU macro = 0.8226 | Dice macro = 0.8991

--- Epoch 145/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0419]
Calculando métricas de entrenamiento...
IoU por clase : [0.97492199 0.89017645 0.91667859 0.85060332 0.86988186 0.95199683]
Dice por clase: [0.98730177 0.94189772 0.95652823 0.91927136 0.93041371 0.97540817]
mIoU macro = 0.9090 | Dice macro = 0.9518
Calculando métricas de validación...
IoU por clase : [0.96057209 0.81019742 0.84111257 0.74476398 0.68523286 0.92527318]
Dice por clase: [0.97988959 0.89514813 0.91370032 0.85371315 0.81322039 0.96118638]
mIoU macro = 0.8279 | Dice macro = 0.9028

--- Epoch 146/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0395]
Calculando métricas de entrenamiento...
IoU por clase : [0.9750275  0.89042686 0.91624044 0.85121949 0.86998558 0.95302742]
Dice por clase: [0.98735587 0.94203789 0.95628964 0.91963108 0.93047304 0.97594884]
mIoU macro = 0.9093 | Dice macro = 0.9520
Calculando métricas de validación...
IoU por clase : [0.96030839 0.8088451  0.84067138 0.7500776  0.68494441 0.92592205]
Dice por clase: [0.97975236 0.89432213 0.91343994 0.85719354 0.81301722 0.96153637]
mIoU macro = 0.8285 | Dice macro = 0.9032

--- Epoch 147/200 ---
100% 210/210 [00:26<00:00,  7.94it/s, loss=0.0533]
Calculando métricas de entrenamiento...
IoU por clase : [0.97480559 0.89145202 0.91417647 0.85295634 0.87341645 0.95236307]
Dice por clase: [0.98724208 0.9426113  0.95516425 0.92064375 0.93243171 0.97560037]
mIoU macro = 0.9099 | Dice macro = 0.9523
Calculando métricas de validación...
IoU por clase : [0.96041584 0.8079628  0.83604243 0.74599227 0.66843103 0.92511585]
Dice por clase: [0.97980829 0.89378255 0.91070055 0.85451956 0.801269   0.96110149]
mIoU macro = 0.8240 | Dice macro = 0.9002

--- Epoch 148/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.0654]
Calculando métricas de entrenamiento...
IoU por clase : [0.97478831 0.89120838 0.91543468 0.84612449 0.86962678 0.95262164]
Dice por clase: [0.98723322 0.94247508 0.95585058 0.91664944 0.93026778 0.97573602]
mIoU macro = 0.9083 | Dice macro = 0.9514
Calculando métricas de validación...
IoU por clase : [0.96010641 0.81159575 0.82868061 0.73271224 0.67035615 0.92675324]
Dice por clase: [0.97964723 0.89600095 0.9063153  0.84574025 0.80265056 0.96198436]
mIoU macro = 0.8217 | Dice macro = 0.8987

--- Epoch 149/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.043]
Calculando métricas de entrenamiento...
IoU por clase : [0.97504794 0.89126892 0.91675702 0.85259801 0.8709374  0.95338423]
Dice por clase: [0.98736635 0.94250893 0.95657093 0.92043498 0.93101714 0.97613589]
mIoU macro = 0.9100 | Dice macro = 0.9523
Calculando métricas de validación...
IoU por clase : [0.96085401 0.81070926 0.83338358 0.7495298  0.67755672 0.92575924]
Dice por clase: [0.98003625 0.89546045 0.90912081 0.8568357  0.80778994 0.96144858]
mIoU macro = 0.8263 | Dice macro = 0.9018

--- Epoch 150/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.04]
Calculando métricas de entrenamiento...
IoU por clase : [0.97482256 0.89005268 0.91659906 0.85022119 0.86992007 0.95218271]
Dice por clase: [0.98725079 0.94182844 0.95648493 0.91904816 0.93043557 0.97550573]
mIoU macro = 0.9090 | Dice macro = 0.9518
Calculando métricas de validación...
IoU por clase : [0.96037046 0.81201547 0.83509048 0.7477857  0.67581429 0.92557297]
Dice por clase: [0.97978467 0.89625666 0.91013548 0.85569495 0.80655034 0.96134811]
mIoU macro = 0.8261 | Dice macro = 0.9016

--- Epoch 151/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0555]
Calculando métricas de entrenamiento...
IoU por clase : [0.97495457 0.89214906 0.91547515 0.85276183 0.86831878 0.95278943]
Dice por clase: [0.98731848 0.94300082 0.95587265 0.92053044 0.92951887 0.97582403]
mIoU macro = 0.9094 | Dice macro = 0.9520
Calculando métricas de validación...
IoU por clase : [0.96047108 0.81369809 0.83239104 0.74259908 0.65960675 0.92552954]
Dice por clase: [0.97983703 0.89728064 0.90852992 0.85228908 0.79489524 0.96132469]
mIoU macro = 0.8224 | Dice macro = 0.8990

--- Epoch 152/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0443]
Calculando métricas de entrenamiento...
IoU por clase : [0.97500953 0.89348473 0.9160236  0.85123705 0.86677135 0.95328407]
Dice por clase: [0.98734666 0.94374643 0.95617152 0.91964133 0.92863151 0.97608339]
mIoU macro = 0.9093 | Dice macro = 0.9519
Calculando métricas de validación...
IoU por clase : [0.96063315 0.80881177 0.83236051 0.74971016 0.65930447 0.92645587]
Dice por clase: [0.97992136 0.89430175 0.90851173 0.85695354 0.7946757  0.96182413]
mIoU macro = 0.8229 | Dice macro = 0.8994

--- Epoch 153/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0569]
Calculando métricas de entrenamiento...
IoU por clase : [0.9749696  0.89209404 0.91868313 0.84707555 0.86963066 0.95295496]
Dice por clase: [0.98732618 0.94297008 0.95761839 0.91720726 0.93027001 0.97591084]
mIoU macro = 0.9092 | Dice macro = 0.9519
Calculando métricas de validación...
IoU por clase : [0.96021892 0.81140083 0.84005163 0.73752854 0.67586454 0.92526622]
Dice por clase: [0.9797058  0.89588215 0.91307398 0.84893977 0.80658612 0.96118262]
mIoU macro = 0.8251 | Dice macro = 0.9009

--- Epoch 154/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0397]
Calculando métricas de entrenamiento...
IoU por clase : [0.97504035 0.89353387 0.916853   0.85187886 0.86483075 0.95380205]
Dice por clase: [0.98736246 0.94377384 0.95662317 0.92001575 0.92751661 0.97635485]
mIoU macro = 0.9093 | Dice macro = 0.9519
Calculando métricas de validación...
IoU por clase : [0.96022506 0.80413026 0.83368264 0.74766604 0.64954512 0.92626587]
Dice por clase: [0.97970899 0.89143259 0.90929872 0.8556166  0.78754453 0.96172173]
mIoU macro = 0.8203 | Dice macro = 0.8976

--- Epoch 155/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.0578]
Calculando métricas de entrenamiento...
IoU por clase : [0.97499164 0.89182701 0.9128129  0.85335907 0.87292024 0.95290829]
Dice por clase: [0.98733749 0.94282089 0.95441943 0.92087829 0.93214887 0.97588637]
mIoU macro = 0.9098 | Dice macro = 0.9522
Calculando métricas de validación...
IoU por clase : [0.96062806 0.81342661 0.84355326 0.74879957 0.65692597 0.92568677]
Dice por clase: [0.97991871 0.89711555 0.91513848 0.85635837 0.79294547 0.96140949]
mIoU macro = 0.8248 | Dice macro = 0.9005

--- Epoch 156/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0468]
Calculando métricas de entrenamiento...
IoU por clase : [0.97524859 0.89269213 0.91797889 0.85203677 0.86982747 0.95267503]
Dice por clase: [0.98746922 0.94330411 0.95723566 0.92010784 0.9303826  0.97576403]
mIoU macro = 0.9101 | Dice macro = 0.9524
Calculando métricas de validación...
IoU por clase : [0.96088181 0.81119269 0.84617743 0.74180733 0.65947218 0.92595998]
Dice por clase: [0.98005071 0.89575526 0.9166805  0.85176738 0.79479752 0.96155682]
mIoU macro = 0.8242 | Dice macro = 0.9001

--- Epoch 157/200 ---
100% 210/210 [00:26<00:00,  7.99it/s, loss=0.0532]
Calculando métricas de entrenamiento...
IoU por clase : [0.97556315 0.89191502 0.91707894 0.8533747  0.87079077 0.95402965]
Dice por clase: [0.98763044 0.94287007 0.95674614 0.92088739 0.93093336 0.97647408]
mIoU macro = 0.9105 | Dice macro = 0.9526
Calculando métricas de validación...
IoU por clase : [0.9610245  0.81035625 0.84012969 0.74554034 0.65800239 0.92781494]
Dice por clase: [0.98012493 0.89524507 0.91312009 0.85422298 0.79372912 0.96255602]
mIoU macro = 0.8238 | Dice macro = 0.8998

--- Epoch 158/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.069]
Calculando métricas de entrenamiento...
IoU por clase : [0.97471822 0.88698553 0.91475416 0.84591448 0.8626     0.95165592]
Dice por clase: [0.98719727 0.94010846 0.95547949 0.91652619 0.92623215 0.9752292 ]
mIoU macro = 0.9061 | Dice macro = 0.9501
Calculando métricas de validación...
IoU por clase : [0.96025223 0.80267763 0.83693984 0.72357967 0.66736144 0.9230099 ]
Dice por clase: [0.97972314 0.89053929 0.91123272 0.83962428 0.80050003 0.95996375]
mIoU macro = 0.8190 | Dice macro = 0.8969

--- Epoch 159/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0617]
Calculando métricas de entrenamiento...
IoU por clase : [0.97546804 0.88976624 0.91846394 0.85304415 0.87374181 0.95375794]
Dice por clase: [0.9875817  0.94166804 0.9574993  0.9206949  0.93261708 0.97633174]
mIoU macro = 0.9107 | Dice macro = 0.9527
Calculando métricas de validación...
IoU por clase : [0.96117407 0.805126   0.85217274 0.74578003 0.67630003 0.92598728]
Dice por clase: [0.98020271 0.8920441  0.92018711 0.85438029 0.80689616 0.96157154]
mIoU macro = 0.8278 | Dice macro = 0.9025

--- Epoch 160/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.0541]
Calculando métricas de entrenamiento...
IoU por clase : [0.97551559 0.89433693 0.91939235 0.85407746 0.87712781 0.953788  ]
Dice por clase: [0.98760607 0.94422161 0.95800356 0.92129642 0.93454245 0.97634749]
mIoU macro = 0.9124 | Dice macro = 0.9537
Calculando métricas de validación...
IoU por clase : [0.96113067 0.80750578 0.84786196 0.74779167 0.6976281  0.92627417]
Dice por clase: [0.98018014 0.89350285 0.91766807 0.85569886 0.82188566 0.9617262 ]
mIoU macro = 0.8314 | Dice macro = 0.9051
🔹 Nuevo mejor mIoU: 0.8314 | Dice: 0.9051  →  guardando modelo…

--- Epoch 161/200 ---
100% 210/210 [00:28<00:00,  7.46it/s, loss=0.0689]
Calculando métricas de entrenamiento...
IoU por clase : [0.97537396 0.89205043 0.91608393 0.85350623 0.87266945 0.9535131 ]
Dice por clase: [0.98753348 0.94294572 0.95620439 0.92096397 0.93200586 0.97620344]
mIoU macro = 0.9105 | Dice macro = 0.9526
Calculando métricas de validación...
IoU por clase : [0.96083569 0.80417736 0.83693541 0.74520263 0.66825396 0.92592051]
Dice por clase: [0.98002672 0.89146153 0.91123009 0.85400127 0.80114176 0.96153554]
mIoU macro = 0.8236 | Dice macro = 0.8999

--- Epoch 162/200 ---
100% 210/210 [00:27<00:00,  7.75it/s, loss=0.0574]
Calculando métricas de entrenamiento...
IoU por clase : [0.97519262 0.89502871 0.91503177 0.85407967 0.8753787  0.95391208]
Dice por clase: [0.98744053 0.94460702 0.95563091 0.9212977  0.93354873 0.97641249]
mIoU macro = 0.9114 | Dice macro = 0.9532
Calculando métricas de validación...
IoU por clase : [0.96052852 0.81211683 0.83455847 0.73908376 0.65621751 0.92670739]
Dice por clase: [0.97986692 0.8963184  0.90981943 0.84996913 0.79242915 0.96195966]
mIoU macro = 0.8215 | Dice macro = 0.8984

--- Epoch 163/200 ---
100% 210/210 [00:27<00:00,  7.75it/s, loss=0.0432]
Calculando métricas de entrenamiento...
IoU por clase : [0.97547533 0.89532555 0.91778733 0.85278247 0.86956385 0.95332561]
Dice por clase: [0.98758543 0.94477231 0.9571315  0.92054246 0.93023178 0.97610517]
mIoU macro = 0.9107 | Dice macro = 0.9527
Calculando métricas de validación...
IoU por clase : [0.96095207 0.81346091 0.82896388 0.74253163 0.64965223 0.92649096]
Dice por clase: [0.98008726 0.89713642 0.90648469 0.85224465 0.78762325 0.96184304]
mIoU macro = 0.8203 | Dice macro = 0.8976

--- Epoch 164/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0533]
Calculando métricas de entrenamiento...
IoU por clase : [0.97460867 0.88933109 0.91373687 0.84707651 0.86416371 0.95310656]
Dice por clase: [0.98714108 0.94142429 0.95492425 0.91720782 0.92713285 0.97599033]
mIoU macro = 0.9070 | Dice macro = 0.9506
Calculando métricas de validación...
IoU por clase : [0.96041364 0.80639573 0.83354466 0.74552945 0.6656419  0.92423481]
Dice por clase: [0.97980714 0.8928229  0.90921664 0.85421584 0.79926171 0.96062581]
mIoU macro = 0.8226 | Dice macro = 0.8993

--- Epoch 165/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0538]
Calculando métricas de entrenamiento...
IoU por clase : [0.9754728  0.8938374  0.91817234 0.85393544 0.87427187 0.95357817]
Dice por clase: [0.98758414 0.94394313 0.95734082 0.92121378 0.93291895 0.97623753]
mIoU macro = 0.9115 | Dice macro = 0.9532
Calculando métricas de validación...
IoU por clase : [0.96109528 0.8097233  0.83675884 0.74234549 0.67368949 0.92630057]
Dice por clase: [0.98016174 0.89485868 0.91112542 0.85212203 0.80503521 0.96174043]
mIoU macro = 0.8250 | Dice macro = 0.9008

--- Epoch 166/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0429]
Calculando métricas de entrenamiento...
IoU por clase : [0.97569759 0.89413362 0.92032945 0.85478693 0.87479617 0.95401351]
Dice por clase: [0.98769933 0.94410828 0.95851204 0.92170903 0.93321737 0.97646562]
mIoU macro = 0.9123 | Dice macro = 0.9536
Calculando métricas de validación...
IoU por clase : [0.96101695 0.81120188 0.8361412  0.74599498 0.67490822 0.92610464]
Dice por clase: [0.980121   0.89576087 0.91075915 0.85452133 0.80590472 0.96163482]
mIoU macro = 0.8259 | Dice macro = 0.9015

--- Epoch 167/200 ---
100% 210/210 [00:26<00:00,  7.94it/s, loss=0.0404]
Calculando métricas de entrenamiento...
IoU por clase : [0.97595062 0.89630474 0.92014379 0.85553729 0.87802467 0.95475098]
Dice por clase: [0.98782896 0.94531719 0.95841134 0.92214508 0.93505127 0.97685177]
mIoU macro = 0.9135 | Dice macro = 0.9543
Calculando métricas de validación...
IoU por clase : [0.96145036 0.8150588  0.83807201 0.74586342 0.67092283 0.92687164]
Dice por clase: [0.98034636 0.89810732 0.91190335 0.85443501 0.80305663 0.96204814]
mIoU macro = 0.8264 | Dice macro = 0.9016

--- Epoch 168/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0396]
Calculando métricas de entrenamiento...
IoU por clase : [0.97621075 0.89711632 0.92053221 0.85677427 0.87663241 0.95505917]
Dice por clase: [0.98796219 0.94576839 0.958622   0.92286314 0.93426119 0.97701306]
mIoU macro = 0.9137 | Dice macro = 0.9544
Calculando métricas de validación...
IoU por clase : [0.9614747  0.8133024  0.8365779  0.74602155 0.67362754 0.92630546]
Dice por clase: [0.98035901 0.89704001 0.91101815 0.85453877 0.80499098 0.96174307]
mIoU macro = 0.8262 | Dice macro = 0.9016

--- Epoch 169/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.052]
Calculando métricas de entrenamiento...
IoU por clase : [0.9747435  0.89322259 0.91331472 0.85107941 0.86710707 0.95297611]
Dice por clase: [0.98721024 0.94360018 0.95469366 0.91954932 0.92882415 0.97592193]
mIoU macro = 0.9087 | Dice macro = 0.9516
Calculando métricas de validación...
IoU por clase : [0.96012682 0.80431524 0.84355191 0.74178406 0.68371922 0.92477564]
Dice por clase: [0.97965786 0.89154625 0.91513768 0.85175203 0.81215349 0.96091785]
mIoU macro = 0.8264 | Dice macro = 0.9019

--- Epoch 170/200 ---
100% 210/210 [00:26<00:00,  7.84it/s, loss=0.0407]
Calculando métricas de entrenamiento...
IoU por clase : [0.97562571 0.89390803 0.91754626 0.85427533 0.87458281 0.95388756]
Dice por clase: [0.9876625  0.94398251 0.95700039 0.92141152 0.93309594 0.97639965]
mIoU macro = 0.9116 | Dice macro = 0.9533
Calculando métricas de validación...
IoU por clase : [0.96097604 0.8078683  0.84061876 0.74321197 0.66779382 0.9254145 ]
Dice por clase: [0.98009973 0.89372472 0.91340888 0.8526926  0.80081101 0.96126263]
mIoU macro = 0.8243 | Dice macro = 0.9003

--- Epoch 171/200 ---
100% 210/210 [00:27<00:00,  7.78it/s, loss=0.0484]
Calculando métricas de entrenamiento...
IoU por clase : [0.9759309  0.8970261  0.91879599 0.85678494 0.87722231 0.95443735]
Dice por clase: [0.98781885 0.94571825 0.95767971 0.92286934 0.93459608 0.97668759]
mIoU macro = 0.9134 | Dice macro = 0.9542
Calculando métricas de validación...
IoU por clase : [0.96121064 0.80934599 0.83992306 0.74595982 0.67021839 0.92495105]
Dice por clase: [0.98022172 0.89462822 0.91299802 0.85449827 0.8025518  0.96101254]
mIoU macro = 0.8253 | Dice macro = 0.9010

--- Epoch 172/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0566]
Calculando métricas de entrenamiento...
IoU por clase : [0.9757249  0.89209489 0.9193826  0.8562216  0.87740228 0.95460539]
Dice por clase: [0.98771332 0.94297056 0.95799826 0.92254244 0.93469822 0.97677556]
mIoU macro = 0.9126 | Dice macro = 0.9538
Calculando métricas de validación...
IoU por clase : [0.96105008 0.80942439 0.83839518 0.74792914 0.6726499  0.92616906]
Dice por clase: [0.98013823 0.89467611 0.91209462 0.85578886 0.80429252 0.96166954]
mIoU macro = 0.8259 | Dice macro = 0.9014

--- Epoch 173/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0439]
Calculando métricas de entrenamiento...
IoU por clase : [0.97594583 0.89553858 0.92081964 0.85448051 0.8781306  0.95429068]
Dice por clase: [0.9878265  0.9448909  0.95877783 0.92153086 0.93511133 0.97661079]
mIoU macro = 0.9132 | Dice macro = 0.9541
Calculando métricas de validación...
IoU por clase : [0.96146163 0.81147872 0.84256363 0.74239164 0.67505701 0.92650639]
Dice por clase: [0.98035222 0.89592962 0.9145558  0.85215244 0.80601079 0.96185136]
mIoU macro = 0.8266 | Dice macro = 0.9018

--- Epoch 174/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0565]
Calculando métricas de entrenamiento...
IoU por clase : [0.97620425 0.89770957 0.92163308 0.85838172 0.87854826 0.95503992]
Dice por clase: [0.98795886 0.94609795 0.95921858 0.92379484 0.93534809 0.97700299]
mIoU macro = 0.9146 | Dice macro = 0.9549
Calculando métricas de validación...
IoU por clase : [0.96186088 0.81414699 0.83978281 0.74750335 0.67670094 0.92665093]
Dice por clase: [0.98055972 0.8975535  0.91291516 0.85551006 0.80718144 0.96192924]
mIoU macro = 0.8278 | Dice macro = 0.9026

--- Epoch 175/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0422]
Calculando métricas de entrenamiento...
IoU por clase : [0.9763074  0.89751683 0.92119699 0.85620758 0.88173075 0.95497939]
Dice por clase: [0.98801168 0.9459909  0.95898234 0.9225343  0.93714868 0.97697131]
mIoU macro = 0.9147 | Dice macro = 0.9549
Calculando métricas de validación...
IoU por clase : [0.96164612 0.81140215 0.84177827 0.74534337 0.67360671 0.92581993]
Dice por clase: [0.98044811 0.89588295 0.91409295 0.85409368 0.80497611 0.96148131]
mIoU macro = 0.8266 | Dice macro = 0.9018

--- Epoch 176/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.0407]
Calculando métricas de entrenamiento...
IoU por clase : [0.9764484  0.89844719 0.92253109 0.85650982 0.87838501 0.9553456 ]
Dice por clase: [0.98808388 0.94650743 0.95970473 0.92270971 0.93525556 0.97716291]
mIoU macro = 0.9146 | Dice macro = 0.9549
Calculando métricas de validación...
IoU por clase : [0.96175871 0.81197893 0.83932137 0.74508506 0.65807154 0.9261034 ]
Dice por clase: [0.98050663 0.89623441 0.91264244 0.85392406 0.79377943 0.96163415]
mIoU macro = 0.8237 | Dice macro = 0.8998

--- Epoch 177/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0416]
Calculando métricas de entrenamiento...
IoU por clase : [0.97629437 0.89826699 0.92161518 0.85304837 0.87978248 0.95520923]
Dice por clase: [0.98800501 0.94640743 0.95920889 0.92069736 0.93604711 0.97709157]
mIoU macro = 0.9140 | Dice macro = 0.9546
Calculando métricas de validación...
IoU por clase : [0.96120643 0.81307522 0.84012419 0.74210337 0.67122096 0.9252684 ]
Dice por clase: [0.98021954 0.89690181 0.91311684 0.8519625  0.80327015 0.9611838 ]
mIoU macro = 0.8255 | Dice macro = 0.9011

--- Epoch 178/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0454]
Calculando métricas de entrenamiento...
IoU por clase : [0.9765581  0.89891222 0.92185393 0.86159913 0.88161392 0.95529196]
Dice por clase: [0.98814004 0.94676543 0.95933819 0.92565485 0.9370827  0.97713485]
mIoU macro = 0.9160 | Dice macro = 0.9557
Calculando métricas de validación...
IoU por clase : [0.96164243 0.8107865  0.84495494 0.74785098 0.67078589 0.92585044]
Dice por clase: [0.9804462  0.89550756 0.91596268 0.85573769 0.80295853 0.96149776]
mIoU macro = 0.8270 | Dice macro = 0.9020

--- Epoch 179/200 ---
100% 210/210 [00:26<00:00,  8.03it/s, loss=0.0511]
Calculando métricas de entrenamiento...
IoU por clase : [0.97623139 0.89825068 0.9222696  0.8582424  0.87979532 0.95452412]
Dice por clase: [0.98797276 0.94639837 0.95956321 0.92371415 0.93605438 0.97673302]
mIoU macro = 0.9149 | Dice macro = 0.9551
Calculando métricas de validación...
IoU por clase : [0.96135515 0.81111881 0.83979965 0.74559899 0.67457186 0.92616362]
Dice por clase: [0.98029686 0.89571022 0.91292511 0.85426148 0.80566487 0.96166661]
mIoU macro = 0.8264 | Dice macro = 0.9018

--- Epoch 180/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0499]
Calculando métricas de entrenamiento...
IoU por clase : [0.97648497 0.89753398 0.92120648 0.85213135 0.88327065 0.95548617]
Dice por clase: [0.9881026  0.94600043 0.95898748 0.92016298 0.93801775 0.97723644]
mIoU macro = 0.9144 | Dice macro = 0.9548
Calculando métricas de validación...
IoU por clase : [0.96132681 0.81543453 0.84183579 0.74024861 0.6698265  0.92710342]
Dice por clase: [0.98028213 0.89833537 0.91412687 0.85073892 0.80227077 0.96217298]
mIoU macro = 0.8260 | Dice macro = 0.9013

--- Epoch 181/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0418]
Calculando métricas de entrenamiento...
IoU por clase : [0.97641033 0.89984211 0.91959865 0.85868484 0.88288371 0.95590006]
Dice por clase: [0.98806438 0.94728094 0.95811554 0.92397035 0.93779951 0.97745286]
mIoU macro = 0.9156 | Dice macro = 0.9554
Calculando métricas de validación...
IoU por clase : [0.96156013 0.81498617 0.84549766 0.75002707 0.67526196 0.92797731]
Dice por clase: [0.98040342 0.89806323 0.91628148 0.85716054 0.80615686 0.96264339]
mIoU macro = 0.8292 | Dice macro = 0.9035

--- Epoch 182/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0558]
Calculando métricas de entrenamiento...
IoU por clase : [0.97677452 0.89934104 0.92297746 0.8595513  0.88468425 0.95585776]
Dice por clase: [0.98825082 0.94700322 0.95994621 0.92447172 0.93881429 0.97743075]
mIoU macro = 0.9165 | Dice macro = 0.9560
Calculando métricas de validación...
IoU por clase : [0.96167968 0.81295003 0.85077448 0.74498544 0.67375374 0.92734154]
Dice por clase: [0.98046555 0.89682564 0.91937131 0.85385863 0.80508109 0.9623012 ]
mIoU macro = 0.8286 | Dice macro = 0.9030

--- Epoch 183/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.0559]
Calculando métricas de entrenamiento...
IoU por clase : [0.97645904 0.89998322 0.92134904 0.85444771 0.8807485  0.95588683]
Dice por clase: [0.98808933 0.94735912 0.95906472 0.92151179 0.93659359 0.97744595]
mIoU macro = 0.9148 | Dice macro = 0.9550
Calculando métricas de validación...
IoU por clase : [0.96127533 0.81372422 0.84724825 0.73811216 0.67234907 0.92733819]
Dice por clase: [0.98025536 0.89729653 0.91730849 0.84932627 0.80407743 0.9622994 ]
mIoU macro = 0.8267 | Dice macro = 0.9018

--- Epoch 184/200 ---
100% 210/210 [00:26<00:00,  7.84it/s, loss=0.0546]
Calculando métricas de entrenamiento...
IoU por clase : [0.97595287 0.89662242 0.9155558  0.85447207 0.86750646 0.9548884 ]
Dice por clase: [0.98783011 0.94549385 0.95591661 0.92152595 0.92905323 0.9769237 ]
mIoU macro = 0.9108 | Dice macro = 0.9528
Calculando métricas de validación...
IoU por clase : [0.96089583 0.81718359 0.80483234 0.74198373 0.63024559 0.92556617]
Dice por clase: [0.98005801 0.89939574 0.89186383 0.85188365 0.77319098 0.96134445]
mIoU macro = 0.8135 | Dice macro = 0.8930

--- Epoch 185/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0424]
Calculando métricas de entrenamiento...
IoU por clase : [0.97674809 0.8992687  0.92196266 0.85843883 0.88363062 0.95586199]
Dice por clase: [0.98823729 0.94696311 0.95939706 0.92382791 0.9382207  0.97743296]
mIoU macro = 0.9160 | Dice macro = 0.9557
Calculando métricas de validación...
IoU por clase : [0.96177661 0.81577333 0.85618848 0.74096549 0.70118123 0.92670632]
Dice por clase: [0.98051593 0.89854093 0.92252321 0.85121215 0.8243463  0.96195908]
mIoU macro = 0.8338 | Dice macro = 0.9065
🔹 Nuevo mejor mIoU: 0.8338 | Dice: 0.9065  →  guardando modelo…

--- Epoch 186/200 ---
100% 210/210 [00:28<00:00,  7.31it/s, loss=0.0481]
Calculando métricas de entrenamiento...
IoU por clase : [0.97668063 0.89956816 0.92319075 0.86002161 0.88205078 0.95592117]
Dice por clase: [0.98820276 0.94712912 0.96006156 0.92474367 0.93732942 0.9774639 ]
mIoU macro = 0.9162 | Dice macro = 0.9558
Calculando métricas de validación...
IoU por clase : [0.9617248  0.81531347 0.83965818 0.74574062 0.6689541  0.9271301 ]
Dice por clase: [0.98048901 0.89826191 0.91284152 0.85435443 0.80164469 0.96218735]
mIoU macro = 0.8264 | Dice macro = 0.9016

--- Epoch 187/200 ---
100% 210/210 [00:27<00:00,  7.71it/s, loss=0.0445]
Calculando métricas de entrenamiento...
IoU por clase : [0.97697442 0.9000898  0.9221264  0.86300945 0.88473244 0.95606551]
Dice por clase: [0.98835312 0.94741817 0.9594857  0.92646814 0.93884142 0.97753936]
mIoU macro = 0.9172 | Dice macro = 0.9564
Calculando métricas de validación...
IoU por clase : [0.96185018 0.81841695 0.83844077 0.74624245 0.66559178 0.92689526]
Dice por clase: [0.98055416 0.90014223 0.9121216  0.85468367 0.79922558 0.96206087]
mIoU macro = 0.8262 | Dice macro = 0.9015

--- Epoch 188/200 ---
100% 210/210 [00:27<00:00,  7.75it/s, loss=0.0423]
Calculando métricas de entrenamiento...
IoU por clase : [0.97668185 0.89813749 0.92334599 0.85927764 0.88313417 0.95561723]
Dice por clase: [0.98820339 0.94633555 0.96014549 0.92431343 0.93794078 0.97730498]
mIoU macro = 0.9160 | Dice macro = 0.9557
Calculando métricas de validación...
IoU por clase : [0.96166239 0.79748136 0.84604458 0.75586141 0.68326321 0.92392427]
Dice por clase: [0.98045657 0.88733199 0.91660255 0.86095794 0.81183169 0.96045804]
mIoU macro = 0.8280 | Dice macro = 0.9029

--- Epoch 189/200 ---
100% 210/210 [00:26<00:00,  7.85it/s, loss=0.0539]
Calculando métricas de entrenamiento...
IoU por clase : [0.97535243 0.89613119 0.91342312 0.84880659 0.87150641 0.95496178]
Dice por clase: [0.98752244 0.94522067 0.95475288 0.91822108 0.93134216 0.9769621 ]
mIoU macro = 0.9100 | Dice macro = 0.9523
Calculando métricas de validación...
IoU por clase : [0.96099794 0.80911758 0.82552137 0.74049074 0.62495234 0.92649239]
Dice por clase: [0.98011112 0.89448866 0.90442258 0.8508988  0.76919467 0.96184381]
mIoU macro = 0.8146 | Dice macro = 0.8935

--- Epoch 190/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0613]
Calculando métricas de entrenamiento...
IoU por clase : [0.97653547 0.89826643 0.92092411 0.86207264 0.88510026 0.955504  ]
Dice por clase: [0.98812845 0.94640712 0.95883445 0.92592805 0.93904847 0.97724576]
mIoU macro = 0.9164 | Dice macro = 0.9559
Calculando métricas de validación...
IoU por clase : [0.96158694 0.80999736 0.84991991 0.75111865 0.68222777 0.92617984]
Dice por clase: [0.98041736 0.89502601 0.91887212 0.85787294 0.81110035 0.96167535]
mIoU macro = 0.8302 | Dice macro = 0.9042

--- Epoch 191/200 ---
100% 210/210 [00:27<00:00,  7.76it/s, loss=0.0385]
Calculando métricas de entrenamiento...
IoU por clase : [0.97606883 0.9007275  0.91946972 0.85437005 0.8806387  0.95465262]
Dice por clase: [0.9878895  0.94777131 0.95804556 0.92146662 0.93653151 0.97680029]
mIoU macro = 0.9143 | Dice macro = 0.9548
Calculando métricas de validación...
IoU por clase : [0.96127523 0.81092958 0.85472462 0.74456362 0.69188563 0.92667321]
Dice por clase: [0.98025531 0.89559482 0.9216728  0.85358151 0.817887   0.96194124]
mIoU macro = 0.8317 | Dice macro = 0.9052

--- Epoch 192/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0455]
Calculando métricas de entrenamiento...
IoU por clase : [0.97465874 0.89056742 0.91141081 0.84951915 0.86093864 0.94826518]
Dice por clase: [0.98716676 0.94211654 0.95365246 0.91863785 0.92527354 0.9734457 ]
mIoU macro = 0.9059 | Dice macro = 0.9500
Calculando métricas de validación...
IoU por clase : [0.96011141 0.7931336  0.84318005 0.7538291  0.67945275 0.91948845]
Dice por clase: [0.97964983 0.88463414 0.91491881 0.85963803 0.80913589 0.95805573]
mIoU macro = 0.8249 | Dice macro = 0.9010

--- Epoch 193/200 ---
100% 210/210 [00:27<00:00,  7.72it/s, loss=0.0472]
Calculando métricas de entrenamiento...
IoU por clase : [0.9757111  0.89496266 0.91646689 0.85550632 0.87107898 0.95382524]
Dice por clase: [0.98770625 0.94457023 0.95641296 0.92212709 0.93109804 0.97636699]
mIoU macro = 0.9113 | Dice macro = 0.9530
Calculando métricas de validación...
IoU por clase : [0.96045594 0.80862654 0.84072919 0.74620569 0.6601105  0.9251251 ]
Dice por clase: [0.97982915 0.89418851 0.91347407 0.85465956 0.79526092 0.96110648]
mIoU macro = 0.8235 | Dice macro = 0.8998

--- Epoch 194/200 ---
100% 210/210 [00:27<00:00,  7.67it/s, loss=0.0401]
Calculando métricas de entrenamiento...
IoU por clase : [0.97605047 0.89697418 0.91792    0.85285391 0.87789951 0.95487833]
Dice por clase: [0.9878801  0.94568939 0.95720364 0.92058408 0.93498029 0.97691843]
mIoU macro = 0.9128 | Dice macro = 0.9539
Calculando métricas de validación...
IoU por clase : [0.9610137  0.80770706 0.8424109  0.74370261 0.67844251 0.92498251]
Dice por clase: [0.98011931 0.89362605 0.91446582 0.85301542 0.80841912 0.96102952]
mIoU macro = 0.8264 | Dice macro = 0.9018

--- Epoch 195/200 ---
100% 210/210 [00:27<00:00,  7.68it/s, loss=0.0355]
Calculando métricas de entrenamiento...
IoU por clase : [0.97648503 0.90018168 0.91983245 0.85933066 0.87922285 0.95559615]
Dice por clase: [0.98810263 0.94746907 0.95824242 0.9243441  0.93573026 0.97729396]
mIoU macro = 0.9151 | Dice macro = 0.9552
Calculando métricas de validación...
IoU por clase : [0.9616223  0.81175079 0.8401367  0.75327397 0.67209527 0.92772544]
Dice por clase: [0.98043574 0.89609542 0.91312423 0.85927697 0.80389591 0.96250786]
mIoU macro = 0.8278 | Dice macro = 0.9026

--- Epoch 196/200 ---
100% 210/210 [00:27<00:00,  7.66it/s, loss=0.0376]
Calculando métricas de entrenamiento...
IoU por clase : [0.97672135 0.90047294 0.92312873 0.85479617 0.87251059 0.95575956]
Dice por clase: [0.9882236  0.94763038 0.96002802 0.9217144  0.93191525 0.97737941]
mIoU macro = 0.9139 | Dice macro = 0.9545
Calculando métricas de validación...
IoU por clase : [0.9613901  0.81113809 0.85165933 0.7464854  0.64216888 0.92729741]
Dice por clase: [0.98031503 0.89572197 0.91988771 0.85484299 0.78209846 0.96227744]
mIoU macro = 0.8234 | Dice macro = 0.8992

--- Epoch 197/200 ---
100% 210/210 [00:27<00:00,  7.54it/s, loss=0.0473]
Calculando métricas de entrenamiento...
IoU por clase : [0.97701527 0.89900731 0.92343327 0.86246042 0.88417085 0.95639354]
Dice por clase: [0.98837402 0.94681817 0.96019268 0.92615168 0.93852513 0.97771079]
mIoU macro = 0.9171 | Dice macro = 0.9563
Calculando métricas de validación...
IoU por clase : [0.96169456 0.80808232 0.85206053 0.75228658 0.67977528 0.92760243]
Dice por clase: [0.98047329 0.89385567 0.92012169 0.85863419 0.80936455 0.96244165]
mIoU macro = 0.8303 | Dice macro = 0.9041

--- Epoch 198/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.0552]
Calculando métricas de entrenamiento...
IoU por clase : [0.97704401 0.89735569 0.92413235 0.86119107 0.88358557 0.95632231]
Dice por clase: [0.98838873 0.94590139 0.96057046 0.9254193  0.93819531 0.97767357]
mIoU macro = 0.9166 | Dice macro = 0.9560
Calculando métricas de validación...
IoU por clase : [0.96183173 0.80764277 0.84984324 0.7549833  0.67953664 0.92595518]
Dice por clase: [0.98054458 0.8935867  0.91882731 0.86038802 0.80919538 0.96155423]
mIoU macro = 0.8300 | Dice macro = 0.9040

--- Epoch 199/200 ---
100% 210/210 [00:27<00:00,  7.72it/s, loss=0.0456]
Calculando métricas de entrenamiento...
IoU por clase : [0.97670821 0.90028561 0.92127323 0.86155399 0.88107293 0.95486288]
Dice por clase: [0.98821688 0.94752663 0.95902365 0.9256288  0.936777   0.97691034]
mIoU macro = 0.9160 | Dice macro = 0.9557
Calculando métricas de validación...
IoU por clase : [0.96133276 0.80738646 0.85384895 0.74571492 0.68781845 0.92360887]
Dice por clase: [0.98028522 0.8934298  0.92116345 0.85433757 0.81503843 0.9602876 ]
mIoU macro = 0.8300 | Dice macro = 0.9041

--- Epoch 200/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0508]
Calculando métricas de entrenamiento...
IoU por clase : [0.97719019 0.902317   0.92437907 0.86252725 0.88320074 0.95587396]
Dice por clase: [0.98846352 0.94865052 0.96070373 0.92619021 0.93797833 0.97743922]
mIoU macro = 0.9176 | Dice macro = 0.9566
Calculando métricas de validación...
IoU por clase : [0.96213546 0.81580287 0.85381521 0.75588811 0.69668816 0.92784208]
Dice por clase: [0.98070238 0.89855885 0.92114382 0.86097526 0.82123301 0.96257063]
mIoU macro = 0.8354 | Dice macro = 0.9075
🔹 Nuevo mejor mIoU: 0.8354 | Dice: 0.9075  →  guardando modelo…
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