import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ▼▼▼ PEGA AQUÍ TU TEXTO DE LOG COMPLETO ▼▼▼
log_text = """
Using device: cuda
/content/Proyecto_Cultivos/train_test4.py:249: UserWarning: Argument(s) 'shift_limit, scale_limit, rotate_limit' are not valid for transform Affine
  A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
/content/Proyecto_Cultivos/train_test4.py:260: UserWarning: Argument(s) 'num_shadows' are not valid for transform RandomShadow
  A.RandomShadow(shadow_roi=(0.0,0.5,1.0,1.0), num_shadows=2, p=0.3),
/content/Proyecto_Cultivos/train_test4.py:261: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
  A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
/content/Proyecto_Cultivos/train_test4.py:263: UserWarning: Argument(s) 'max_holes, max_height, max_width, min_holes, min_height, min_width, fill_value' are not valid for transform CoarseDropout
  A.CoarseDropout(

--- Distribución de clases en ENTRENAMIENTO (post-aug) ---
Analizando distribución: 100% 210/210 [00:07<00:00, 26.59it/s]
Número total de píxeles analizados: 110100480
Distribución final de píxeles por clase:
  Clase 0: 75.6598%  (83301844 píxeles)
  Clase 1: 3.2715%  (3601890 píxeles)
  Clase 2: 4.9559%  (5456480 píxeles)
  Clase 3: 2.9251%  (3220592 píxeles)
  Clase 4: 1.3522%  (1488746 píxeles)
  Clase 5: 11.8355%  (13030928 píxeles)

Pesos de importancia por clase (para CrossEntropy/Focal):
  Clase 0: 0.0470
  Clase 1: 1.0874
  Clase 2: 0.7178
  Clase 3: 1.2162
  Clase 4: 2.6310
  Clase 5: 0.3006
--------------------------------------------------------
model.safetensors: 100% 193M/193M [00:00<00:00, 281MB/s]
Unexpected keys (bn2.bias, bn2.num_batches_tracked, bn2.running_mean, bn2.running_var, bn2.weight, conv_head.weight) found while loading pretrained weights. This may be expected if model is being adapted.

--- Epoch 1/200 ---
  0% 0/210 [00:00<?, ?it/s]/content/Proyecto_Cultivos/train_test4.py:142: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
100% 210/210 [00:30<00:00,  6.99it/s, loss=0.983]
Calculando métricas de entrenamiento...
IoU por clase : [9.14614117e-01 4.75510529e-03 1.98752594e-01 6.60498872e-02
 1.38775198e-04 5.49647191e-01]
Dice por clase: [9.55403085e-01 9.46520254e-03 3.31599022e-01 1.23915190e-01
 2.77511885e-04 7.09383651e-01]
mIoU macro = 0.2890 | Dice macro = 0.3550
Calculando métricas de validación...
IoU por clase : [9.24389809e-01 5.67263217e-03 2.47241279e-01 8.39752172e-02
 1.33880996e-04 5.86362594e-01]
Dice por clase: [9.60709525e-01 1.12812698e-02 3.96461027e-01 1.54939367e-01
 2.67726148e-04 7.39254186e-01]
mIoU macro = 0.3080 | Dice macro = 0.3772
🔹 Nuevo mejor mIoU: 0.3080 | Dice: 0.3772  →  guardando modelo…

--- Epoch 2/200 ---
100% 210/210 [00:27<00:00,  7.63it/s, loss=0.759]
Calculando métricas de entrenamiento...
IoU por clase : [9.20834167e-01 2.29816808e-03 3.61855237e-01 1.83700631e-02
 2.28249348e-05 6.37587120e-01]
Dice por clase: [9.58785701e-01 4.58579723e-03 5.31415127e-01 3.60773824e-02
 4.56488276e-05 7.78690931e-01]
mIoU macro = 0.3235 | Dice macro = 0.3849
Calculando métricas de validación...
IoU por clase : [9.31740491e-01 3.23578321e-03 3.82928614e-01 1.86158406e-02
 4.82294436e-05 6.86960556e-01]
Dice por clase: [9.64664245e-01 6.45069337e-03 5.53793754e-01 3.65512489e-02
 9.64542352e-05 8.14435825e-01]
mIoU macro = 0.3373 | Dice macro = 0.3960
🔹 Nuevo mejor mIoU: 0.3373 | Dice: 0.3960  →  guardando modelo…

--- Epoch 3/200 ---
100% 210/210 [00:29<00:00,  7.21it/s, loss=0.597]
Calculando métricas de entrenamiento...
IoU por clase : [9.26211858e-01 5.44796501e-04 4.73204345e-01 1.09335602e-02
 6.71674673e-15 6.63132172e-01]
Dice por clase: [9.61692614e-01 1.08899972e-03 6.42415082e-01 2.16306206e-02
 6.71674673e-15 7.97449755e-01]
mIoU macro = 0.3457 | Dice macro = 0.4040
Calculando métricas de validación...
IoU por clase : [9.28248691e-01 6.99105517e-04 4.06348807e-01 1.47804054e-02
 5.36103918e-14 6.94270850e-01]
Dice por clase: [9.62789390e-01 1.39723422e-03 5.77877700e-01 2.91302538e-02
 5.36103918e-14 8.19551195e-01]
mIoU macro = 0.3407 | Dice macro = 0.3985
🔹 Nuevo mejor mIoU: 0.3407 | Dice: 0.3985  →  guardando modelo…

--- Epoch 4/200 ---
100% 210/210 [00:28<00:00,  7.39it/s, loss=0.626]
Calculando métricas de entrenamiento...
IoU por clase : [9.27760296e-01 8.24036569e-02 4.84501328e-01 9.72014502e-02
 6.71787930e-15 7.32348434e-01]
Dice por clase: [9.62526615e-01 1.52260492e-01 6.52746237e-01 1.77180681e-01
 6.71787930e-15 8.45497845e-01]
mIoU macro = 0.3874 | Dice macro = 0.4650
Calculando métricas de validación...
IoU por clase : [9.26155831e-01 1.11446530e-01 4.52039120e-01 1.50033130e-01
 5.36106792e-14 7.44979064e-01]
Dice por clase: [9.61662412e-01 2.00543214e-01 6.22626641e-01 2.60919666e-01
 5.36106792e-14 8.53854443e-01]
mIoU macro = 0.3974 | Dice macro = 0.4833
🔹 Nuevo mejor mIoU: 0.3974 | Dice: 0.4833  →  guardando modelo…

--- Epoch 5/200 ---
100% 210/210 [00:27<00:00,  7.51it/s, loss=0.478]
Calculando métricas de entrenamiento...
IoU por clase : [9.29113373e-01 1.20727090e-01 5.55921195e-01 8.82972263e-02
 6.71778453e-15 7.50701731e-01]
Dice por clase: [9.63254297e-01 2.15444226e-01 7.14587856e-01 1.62266749e-01
 6.71778453e-15 8.57600946e-01]
mIoU macro = 0.4075 | Dice macro = 0.4855
Calculando métricas de validación...
IoU por clase : [9.27353545e-01 1.49904030e-01 5.31876006e-01 1.62263830e-01
 5.36106792e-14 7.79989977e-01]
Dice por clase: [9.62307665e-01 2.60724419e-01 6.94411302e-01 2.79220304e-01
 5.36106792e-14 8.76398167e-01]
mIoU macro = 0.4252 | Dice macro = 0.5122
🔹 Nuevo mejor mIoU: 0.4252 | Dice: 0.5122  →  guardando modelo…

--- Epoch 6/200 ---
100% 210/210 [00:27<00:00,  7.55it/s, loss=0.343]
Calculando métricas de entrenamiento...
IoU por clase : [9.29118818e-01 4.49925200e-01 5.63407040e-01 1.22651932e-01
 6.71809142e-15 8.22424209e-01]
Dice por clase: [9.63257223e-01 6.20618498e-01 7.20742616e-01 2.18503935e-01
 6.71809142e-15 9.02560672e-01]
mIoU macro = 0.4813 | Dice macro = 0.5709
Calculando métricas de validación...
IoU por clase : [9.30540031e-01 4.89731973e-01 5.55890185e-01 2.52298175e-01
 5.36106792e-14 8.50731348e-01]
Dice por clase: [9.64020446e-01 6.57476623e-01 7.14562236e-01 4.02936265e-01
 5.36106792e-14 9.19346126e-01]
mIoU macro = 0.5132 | Dice macro = 0.6097
🔹 Nuevo mejor mIoU: 0.5132 | Dice: 0.6097  →  guardando modelo…

--- Epoch 7/200 ---
100% 210/210 [00:27<00:00,  7.50it/s, loss=0.445]
Calculando métricas de entrenamiento...
IoU por clase : [9.30228683e-01 4.24035856e-01 6.28340905e-01 1.84377524e-01
 6.71767622e-15 8.16450652e-01]
Dice por clase: [9.63853342e-01 5.95540982e-01 7.71755968e-01 3.11349246e-01
 6.71767622e-15 8.98951646e-01]
mIoU macro = 0.4972 | Dice macro = 0.5902
Calculando métricas de validación...
IoU por clase : [9.26967503e-01 4.76462540e-01 6.72299282e-01 2.85075017e-01
 5.36106792e-14 8.58012949e-01]
Dice por clase: [9.62099777e-01 6.45410943e-01 8.04041823e-01 4.43670623e-01
 5.36106792e-14 9.23581237e-01]
mIoU macro = 0.5365 | Dice macro = 0.6298
🔹 Nuevo mejor mIoU: 0.5365 | Dice: 0.6298  →  guardando modelo…

--- Epoch 8/200 ---
100% 210/210 [00:27<00:00,  7.55it/s, loss=0.241]
Calculando métricas de entrenamiento...
IoU por clase : [9.30263604e-01 6.11803700e-01 6.63456769e-01 2.74458120e-01
 6.71714828e-15 8.65907320e-01]
Dice por clase: [9.63872087e-01 7.59154108e-01 7.97684414e-01 4.30705592e-01
 6.71714828e-15 9.28135402e-01]
mIoU macro = 0.5576 | Dice macro = 0.6466
Calculando métricas de validación...
IoU por clase : [9.33394455e-01 6.19482798e-01 6.97780685e-01 4.09918005e-01
 5.36106792e-14 8.86276381e-01]
Dice por clase: [9.65549945e-01 7.65037824e-01 8.21991546e-01 5.81477793e-01
 5.36106792e-14 9.39709991e-01]
mIoU macro = 0.5911 | Dice macro = 0.6790
🔹 Nuevo mejor mIoU: 0.5911 | Dice: 0.6790  →  guardando modelo…

--- Epoch 9/200 ---
100% 210/210 [00:27<00:00,  7.51it/s, loss=0.421]
Calculando métricas de entrenamiento...
IoU por clase : [9.32324785e-01 6.60025138e-01 6.73830654e-01 3.21641316e-01
 6.71681891e-15 8.75492354e-01]
Dice por clase: [9.64977308e-01 7.95198967e-01 8.05135995e-01 4.86730116e-01
 6.71681891e-15 9.33613354e-01]
mIoU macro = 0.5772 | Dice macro = 0.6643
Calculando métricas de validación...
IoU por clase : [9.34593301e-01 7.02864942e-01 7.00669431e-01 4.56649708e-01
 5.36106792e-14 8.97864414e-01]
Dice por clase: [9.66190982e-01 8.25508735e-01 8.23992504e-01 6.26986304e-01
 5.36106792e-14 9.46183939e-01]
mIoU macro = 0.6154 | Dice macro = 0.6981
🔹 Nuevo mejor mIoU: 0.6154 | Dice: 0.6981  →  guardando modelo…

--- Epoch 10/200 ---
100% 210/210 [00:28<00:00,  7.50it/s, loss=0.243]
Calculando métricas de entrenamiento...
IoU por clase : [9.34198450e-01 6.49467826e-01 6.65039468e-01 3.71920613e-01
 6.71777099e-15 8.70942678e-01]
Dice por clase: [9.65979939e-01 7.87487717e-01 7.98827272e-01 5.42189700e-01
 6.71777099e-15 9.31020162e-01]
mIoU macro = 0.5819 | Dice macro = 0.6709
Calculando métricas de validación...
IoU por clase : [9.38210267e-01 6.53715453e-01 7.30969514e-01 5.09228142e-01
 5.36106792e-14 8.73847734e-01]
Dice por clase: [9.68120212e-01 7.90602098e-01 8.44578149e-01 6.74819304e-01
 5.36106792e-14 9.32677419e-01]
mIoU macro = 0.6177 | Dice macro = 0.7018
🔹 Nuevo mejor mIoU: 0.6177 | Dice: 0.7018  →  guardando modelo…

--- Epoch 11/200 ---
100% 210/210 [00:28<00:00,  7.39it/s, loss=0.278]
Calculando métricas de entrenamiento...
IoU por clase : [9.39101427e-01 6.59521464e-01 6.68280184e-01 4.26647940e-01
 6.71626855e-15 8.75071895e-01]
Dice por clase: [9.68594436e-01 7.94833304e-01 8.01160609e-01 5.98112439e-01
 6.71626855e-15 9.33374232e-01]
mIoU macro = 0.5948 | Dice macro = 0.6827
Calculando métricas de validación...
IoU por clase : [9.44545009e-01 7.19136936e-01 7.07056054e-01 5.91962496e-01
 5.36106792e-14 9.03097590e-01]
Dice por clase: [9.71481765e-01 8.36625543e-01 8.28392310e-01 7.43688997e-01
 5.36106792e-14 9.49081744e-01]
mIoU macro = 0.6443 | Dice macro = 0.7215
🔹 Nuevo mejor mIoU: 0.6443 | Dice: 0.7215  →  guardando modelo…

--- Epoch 12/200 ---
100% 210/210 [00:28<00:00,  7.38it/s, loss=0.233]
Calculando métricas de entrenamiento...
IoU por clase : [9.36457215e-01 7.01657509e-01 7.00297431e-01 4.50456006e-01
 6.71851118e-15 8.89849174e-01]
Dice por clase: [9.67186063e-01 8.24675360e-01 8.23735211e-01 6.21123294e-01
 6.71851118e-15 9.41714488e-01]
mIoU macro = 0.6131 | Dice macro = 0.6964
Calculando métricas de validación...
IoU por clase : [9.40309270e-01 7.07704348e-01 7.37838826e-01 5.40985581e-01
 5.36106792e-14 8.95706744e-01]
Dice por clase: [9.69236487e-01 8.28837086e-01 8.49145289e-01 7.02129322e-01
 5.36106792e-14 9.44984499e-01]
mIoU macro = 0.6371 | Dice macro = 0.7157

--- Epoch 13/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.32]
Calculando métricas de entrenamiento...
IoU por clase : [9.40178921e-01 6.98699810e-01 7.05551293e-01 4.97181445e-01
 6.71717084e-15 8.90486314e-01]
Dice por clase: [9.69167236e-01 8.22628938e-01 8.27358633e-01 6.64156568e-01
 6.71717084e-15 9.42071157e-01]
mIoU macro = 0.6220 | Dice macro = 0.7042
Calculando métricas de validación...
IoU por clase : [9.44742154e-01 7.21405368e-01 7.44969058e-01 6.09695227e-01
 5.36106792e-14 9.03521100e-01]
Dice por clase: [9.71586029e-01 8.38158613e-01 8.53847871e-01 7.57528776e-01
 5.36106792e-14 9.49315561e-01]
mIoU macro = 0.6541 | Dice macro = 0.7284
🔹 Nuevo mejor mIoU: 0.6541 | Dice: 0.7284  →  guardando modelo…

--- Epoch 14/200 ---
100% 210/210 [00:28<00:00,  7.41it/s, loss=0.271]
Calculando métricas de entrenamiento...
IoU por clase : [9.40739402e-01 7.09474976e-01 7.18738762e-01 4.74253334e-01
 6.71633621e-15 8.93308364e-01]
Dice por clase: [9.69464938e-01 8.30050145e-01 8.36356028e-01 6.43381056e-01
 6.71633621e-15 9.43648041e-01]
mIoU macro = 0.6228 | Dice macro = 0.7038
Calculando métricas de validación...
IoU por clase : [9.45692105e-01 7.27459489e-01 7.58478144e-01 5.82504759e-01
 5.36106792e-14 9.07456845e-01]
Dice por clase: [9.72088135e-01 8.42230447e-01 8.62652910e-01 7.36180735e-01
 5.36106792e-14 9.51483487e-01]
mIoU macro = 0.6536 | Dice macro = 0.7274

--- Epoch 15/200 ---
100% 210/210 [00:25<00:00,  8.09it/s, loss=0.188]
Calculando métricas de entrenamiento...
IoU por clase : [9.42621151e-01 7.04326633e-01 7.25044620e-01 5.41002035e-01
 6.52854189e-04 8.94403323e-01]
Dice por clase: [0.97046318 0.82651602 0.8406097  0.70214318 0.00130486 0.94425861]
mIoU macro = 0.6347 | Dice macro = 0.7142
Calculando métricas de validación...
IoU por clase : [9.46652195e-01 7.47984418e-01 7.58090381e-01 6.26334245e-01
 5.36106792e-14 9.04285693e-01]
Dice por clase: [9.72595102e-01 8.55825041e-01 8.62402058e-01 7.70240492e-01
 5.36106792e-14 9.49737423e-01]
mIoU macro = 0.6639 | Dice macro = 0.7351
🔹 Nuevo mejor mIoU: 0.6639 | Dice: 0.7351  →  guardando modelo…

--- Epoch 16/200 ---
100% 210/210 [00:29<00:00,  7.23it/s, loss=0.215]
Calculando métricas de entrenamiento...
IoU por clase : [0.9422819  0.72490945 0.71597112 0.48261498 0.00584223 0.89813496]
Dice por clase: [0.97028335 0.84051885 0.83447922 0.65103211 0.01161659 0.94633414]
mIoU macro = 0.6283 | Dice macro = 0.7090
Calculando métricas de validación...
IoU por clase : [0.94541748 0.74354602 0.74759746 0.56900079 0.00331225 0.91147803]
Dice por clase: [0.97194303 0.85291241 0.85557169 0.72530339 0.00660263 0.95368925]
mIoU macro = 0.6534 | Dice macro = 0.7277

--- Epoch 17/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.135]
Calculando métricas de entrenamiento...
IoU por clase : [0.94294163 0.71455553 0.74261954 0.55565185 0.01517872 0.89100221]
Dice por clase: [0.970633   0.83351693 0.85230255 0.7143653  0.02990354 0.94235978]
mIoU macro = 0.6437 | Dice macro = 0.7238
Calculando métricas de validación...
IoU por clase : [9.46887344e-01 7.14277680e-01 7.47759343e-01 6.27653907e-01
 7.44665463e-04 8.93164783e-01]
Dice por clase: [0.97271919 0.83332787 0.85567769 0.77123755 0.00148822 0.94356792]
mIoU macro = 0.6551 | Dice macro = 0.7297

--- Epoch 18/200 ---
100% 210/210 [00:26<00:00,  8.06it/s, loss=0.165]
Calculando métricas de entrenamiento...
IoU por clase : [0.94460002 0.74235036 0.73830444 0.54549584 0.03320285 0.90432161]
Dice por clase: [0.97151086 0.85212524 0.84945355 0.70591693 0.06427169 0.94975723]
mIoU macro = 0.6514 | Dice macro = 0.7322
Calculando métricas de validación...
IoU por clase : [0.94903513 0.75204509 0.76048331 0.62782558 0.01665784 0.91234243]
Dice por clase: [0.97385123 0.85847687 0.86394834 0.77136714 0.03276981 0.9541622 ]
mIoU macro = 0.6697 | Dice macro = 0.7424
🔹 Nuevo mejor mIoU: 0.6697 | Dice: 0.7424  →  guardando modelo…

--- Epoch 19/200 ---
100% 210/210 [00:28<00:00,  7.38it/s, loss=0.297]
Calculando métricas de entrenamiento...
IoU por clase : [0.94510611 0.71160833 0.74796981 0.55039827 0.02945114 0.89832598]
Dice por clase: [0.97177846 0.83150838 0.85581548 0.71000888 0.05721717 0.94644017]
mIoU macro = 0.6471 | Dice macro = 0.7288
Calculando métricas de validación...
IoU por clase : [0.94818761 0.75505374 0.75986162 0.62174218 0.03122381 0.91329719]
Dice por clase: [0.97340482 0.86043376 0.86354701 0.76675835 0.0605568  0.95468408]
mIoU macro = 0.6716 | Dice macro = 0.7466
🔹 Nuevo mejor mIoU: 0.6716 | Dice: 0.7466  →  guardando modelo…

--- Epoch 20/200 ---
100% 210/210 [00:28<00:00,  7.37it/s, loss=0.15]
Calculando métricas de entrenamiento...
IoU por clase : [0.94485596 0.73183769 0.74706342 0.57966058 0.12459652 0.90148991]
Dice por clase: [0.97164621 0.84515737 0.85522187 0.73390523 0.2215844  0.94819321]
mIoU macro = 0.6716 | Dice macro = 0.7626
Calculando métricas de validación...
IoU por clase : [0.94885011 0.72941949 0.7601064  0.63764297 0.13004491 0.90246265]
Dice por clase: [0.97375381 0.84354258 0.86370506 0.77873258 0.23015884 0.94873101]
mIoU macro = 0.6848 | Dice macro = 0.7731
🔹 Nuevo mejor mIoU: 0.6848 | Dice: 0.7731  →  guardando modelo…

--- Epoch 21/200 ---
100% 210/210 [00:28<00:00,  7.45it/s, loss=0.168]
Calculando métricas de entrenamiento...
IoU por clase : [0.94404695 0.71979912 0.74288389 0.62896659 0.33756331 0.90206652]
Dice por clase: [0.97121826 0.83707349 0.85247663 0.77222773 0.50474367 0.94851206]
mIoU macro = 0.7126 | Dice macro = 0.8144
Calculando métricas de validación...
IoU por clase : [0.94653987 0.72861953 0.73260018 0.66257606 0.22625872 0.90753866]
Dice por clase: [0.97253582 0.8430074  0.84566559 0.79704752 0.36902281 0.95152846]
mIoU macro = 0.7007 | Dice macro = 0.7965
🔹 Nuevo mejor mIoU: 0.7007 | Dice: 0.7965  →  guardando modelo…

--- Epoch 22/200 ---
100% 210/210 [00:28<00:00,  7.49it/s, loss=0.28]
Calculando métricas de entrenamiento...
IoU por clase : [0.94589159 0.72699716 0.74249212 0.63637772 0.27673022 0.90184812]
Dice por clase: [0.97219352 0.8419205  0.85221863 0.77778829 0.43349834 0.94839131]
mIoU macro = 0.7051 | Dice macro = 0.8043
Calculando métricas de validación...
IoU por clase : [0.94960887 0.74255351 0.73574492 0.65439674 0.22871432 0.91338829]
Dice por clase: [0.97415321 0.85225906 0.84775696 0.79110013 0.37228234 0.95473386]
mIoU macro = 0.7041 | Dice macro = 0.7987
🔹 Nuevo mejor mIoU: 0.7041 | Dice: 0.7987  →  guardando modelo…

--- Epoch 23/200 ---
100% 210/210 [00:28<00:00,  7.42it/s, loss=0.136]
Calculando métricas de entrenamiento...
IoU por clase : [0.94511616 0.73910281 0.76219173 0.64503908 0.39854172 0.89880839]
Dice por clase: [0.97178377 0.84998174 0.86504972 0.78422341 0.56993898 0.94670784]
mIoU macro = 0.7315 | Dice macro = 0.8313
Calculando métricas de validación...
IoU por clase : [0.94867862 0.7540858  0.76635653 0.67422316 0.37478724 0.91377679]
Dice por clase: [0.9736635  0.85980492 0.86772576 0.80541612 0.54522944 0.95494605]
mIoU macro = 0.7387 | Dice macro = 0.8345
🔹 Nuevo mejor mIoU: 0.7387 | Dice: 0.8345  →  guardando modelo…

--- Epoch 24/200 ---
100% 210/210 [00:28<00:00,  7.49it/s, loss=0.266]
Calculando métricas de entrenamiento...
IoU por clase : [0.9460676  0.74732617 0.76771789 0.65335699 0.39431702 0.90384741]
Dice por clase: [0.97228647 0.85539401 0.86859775 0.79033989 0.56560598 0.94949564]
mIoU macro = 0.7354 | Dice macro = 0.8336
Calculando métricas de validación...
IoU por clase : [0.9491112  0.74765317 0.77591957 0.67414713 0.3674431  0.90932614]
Dice por clase: [0.97389128 0.85560818 0.87382287 0.80536187 0.53741629 0.95251002]
mIoU macro = 0.7373 | Dice macro = 0.8331

--- Epoch 25/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.9474857  0.74622867 0.76697856 0.66176947 0.41755161 0.90334022]
Dice por clase: [0.97303482 0.85467463 0.86812436 0.79646363 0.58911663 0.94921571]
mIoU macro = 0.7406 | Dice macro = 0.8384
Calculando métricas de validación...
IoU por clase : [0.9502258  0.741658   0.76819768 0.67580794 0.44197875 0.9078104 ]
Dice por clase: [0.97447772 0.85166893 0.86890475 0.80654582 0.61301701 0.9516778 ]
mIoU macro = 0.7476 | Dice macro = 0.8444
🔹 Nuevo mejor mIoU: 0.7476 | Dice: 0.8444  →  guardando modelo…

--- Epoch 26/200 ---
100% 210/210 [00:27<00:00,  7.54it/s, loss=0.157]
Calculando métricas de entrenamiento...
IoU por clase : [0.94761425 0.75315315 0.76249655 0.64068101 0.42320952 0.90353759]
Dice por clase: [0.9731026  0.85919835 0.86524601 0.780994   0.59472553 0.94932466]
mIoU macro = 0.7384 | Dice macro = 0.8371
Calculando métricas de validación...
IoU por clase : [0.95057183 0.74053512 0.77490499 0.69149216 0.48250874 0.90377627]
Dice por clase: [0.97465965 0.8509281  0.87317912 0.81761202 0.65093544 0.94945639]
mIoU macro = 0.7573 | Dice macro = 0.8528
🔹 Nuevo mejor mIoU: 0.7573 | Dice: 0.8528  →  guardando modelo…

--- Epoch 27/200 ---
100% 210/210 [00:27<00:00,  7.55it/s, loss=0.191]
Calculando métricas de entrenamiento...
IoU por clase : [0.94792807 0.74208017 0.74532649 0.64352133 0.45098255 0.90444574]
Dice por clase: [0.97326804 0.85194721 0.8540826  0.78310067 0.62162367 0.94982569]
mIoU macro = 0.7390 | Dice macro = 0.8390
Calculando métricas de validación...
IoU por clase : [0.95101877 0.74707464 0.77933988 0.69043887 0.50440392 0.90964947]
Dice por clase: [0.97489454 0.85522922 0.87598765 0.81687529 0.67056981 0.95268737]
mIoU macro = 0.7637 | Dice macro = 0.8577
🔹 Nuevo mejor mIoU: 0.7637 | Dice: 0.8577  →  guardando modelo…

--- Epoch 28/200 ---
100% 210/210 [00:28<00:00,  7.49it/s, loss=0.119]
Calculando métricas de entrenamiento...
IoU por clase : [0.94611795 0.74018149 0.76269371 0.67947204 0.47318246 0.90219144]
Dice por clase: [0.97231306 0.85069459 0.86537293 0.80914957 0.64239491 0.94858112]
mIoU macro = 0.7506 | Dice macro = 0.8481
Calculando métricas de validación...
IoU por clase : [0.94944442 0.75906281 0.78144332 0.69885468 0.49795104 0.90906787]
Dice por clase: [0.97406667 0.86303093 0.87731483 0.82273627 0.66484288 0.95236831]
mIoU macro = 0.7660 | Dice macro = 0.8591
🔹 Nuevo mejor mIoU: 0.7660 | Dice: 0.8591  →  guardando modelo…

--- Epoch 29/200 ---
100% 210/210 [00:28<00:00,  7.38it/s, loss=0.208]
Calculando métricas de entrenamiento...
IoU por clase : [0.94849492 0.75133515 0.78461464 0.68696151 0.53129215 0.91183055]
Dice por clase: [0.97356674 0.85801413 0.87930988 0.81443649 0.6939135  0.95388219]
mIoU macro = 0.7691 | Dice macro = 0.8622
Calculando métricas de validación...
IoU por clase : [0.95062423 0.75976706 0.78810788 0.69913609 0.51997866 0.91433221]
Dice por clase: [0.97468719 0.86348594 0.88149925 0.82293125 0.68419205 0.95524926]
mIoU macro = 0.7720 | Dice macro = 0.8637
🔹 Nuevo mejor mIoU: 0.7720 | Dice: 0.8637  →  guardando modelo…

--- Epoch 30/200 ---
100% 210/210 [00:28<00:00,  7.47it/s, loss=0.186]
Calculando métricas de entrenamiento...
IoU por clase : [0.94694973 0.75529504 0.75457699 0.63745411 0.44545202 0.90395296]
Dice por clase: [0.97275211 0.86059041 0.86012412 0.77859173 0.61634979 0.94955388]
mIoU macro = 0.7406 | Dice macro = 0.8397
Calculando métricas de validación...
IoU por clase : [0.95064319 0.7538502  0.76624494 0.6707344  0.45245154 0.91253798]
Dice por clase: [0.97469716 0.85965176 0.86765422 0.80292163 0.62301775 0.95426913]
mIoU macro = 0.7511 | Dice macro = 0.8470

--- Epoch 31/200 ---
100% 210/210 [00:25<00:00,  8.13it/s, loss=0.17]
Calculando métricas de entrenamiento...
IoU por clase : [0.9467639  0.74016553 0.77882734 0.67883687 0.52715186 0.89835366]
Dice por clase: [0.97265406 0.85068405 0.87566378 0.80869903 0.69037255 0.94645553]
mIoU macro = 0.7617 | Dice macro = 0.8574
Calculando métricas de validación...
IoU por clase : [0.95076515 0.76453846 0.78239389 0.69751304 0.51154134 0.90890876]
Dice por clase: [0.97476126 0.86655913 0.87791357 0.82180581 0.6768473  0.95228099]
mIoU macro = 0.7693 | Dice macro = 0.8617

--- Epoch 32/200 ---
100% 210/210 [00:26<00:00,  8.05it/s, loss=0.168]
Calculando métricas de entrenamiento...
IoU por clase : [0.94850362 0.7530689  0.77411934 0.68641609 0.53272352 0.90965385]
Dice por clase: [0.97357132 0.85914353 0.87268012 0.81405306 0.69513323 0.95268977]
mIoU macro = 0.7674 | Dice macro = 0.8612
Calculando métricas de validación...
IoU por clase : [0.95106592 0.76498928 0.77403703 0.70088309 0.5256086  0.91422558]
Dice por clase: [0.97491931 0.86684864 0.87262782 0.82414023 0.68904777 0.95519106]
mIoU macro = 0.7718 | Dice macro = 0.8638

--- Epoch 33/200 ---
100% 210/210 [00:26<00:00,  8.01it/s, loss=0.149]
Calculando métricas de entrenamiento...
IoU por clase : [0.94874738 0.75814831 0.79367163 0.68073463 0.57424416 0.91487647]
Dice por clase: [0.97369971 0.86243954 0.88496871 0.81004415 0.72954904 0.9555462 ]
mIoU macro = 0.7784 | Dice macro = 0.8694
Calculando métricas de validación...
IoU por clase : [0.95118468 0.77338418 0.78683781 0.69862189 0.52968118 0.91167577]
Dice por clase: [0.9749817  0.87221279 0.88070423 0.82257493 0.69253801 0.95379748]
mIoU macro = 0.7752 | Dice macro = 0.8661
🔹 Nuevo mejor mIoU: 0.7752 | Dice: 0.8661  →  guardando modelo…

--- Epoch 34/200 ---
100% 210/210 [00:27<00:00,  7.57it/s, loss=0.111]
Calculando métricas de entrenamiento...
IoU por clase : [0.94899496 0.7573384  0.78254601 0.67084497 0.46537456 0.90568886]
Dice por clase: [0.97383008 0.86191527 0.87800932 0.80300086 0.63516124 0.95051074]
mIoU macro = 0.7551 | Dice macro = 0.8504
Calculando métricas de validación...
IoU por clase : [0.95114031 0.75630745 0.77581714 0.68192697 0.48764443 0.91285275]
Dice por clase: [0.97495839 0.86124721 0.87375792 0.81088773 0.65559273 0.95444121]
mIoU macro = 0.7609 | Dice macro = 0.8551

--- Epoch 35/200 ---
100% 210/210 [00:26<00:00,  7.99it/s, loss=0.187]
Calculando métricas de entrenamiento...
IoU por clase : [0.94938731 0.76563893 0.79941897 0.67182035 0.57984637 0.91448931]
Dice por clase: [0.97403662 0.86726558 0.88853011 0.80369921 0.73405412 0.95533499]
mIoU macro = 0.7801 | Dice macro = 0.8705
Calculando métricas de validación...
IoU por clase : [0.95146724 0.76351859 0.79229481 0.67920637 0.56646387 0.91680705]
Dice por clase: [0.97513012 0.86590365 0.88411215 0.80896117 0.72323898 0.95659816]
mIoU macro = 0.7783 | Dice macro = 0.8690
🔹 Nuevo mejor mIoU: 0.7783 | Dice: 0.8690  →  guardando modelo…

--- Epoch 36/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.172]
Calculando métricas de entrenamiento...
IoU por clase : [0.94867258 0.76443061 0.74080766 0.63042823 0.54225144 0.91152552]
Dice por clase: [0.97366032 0.86648985 0.85110799 0.77332841 0.70319459 0.95371525]
mIoU macro = 0.7564 | Dice macro = 0.8536
Calculando métricas de validación...
IoU por clase : [0.95133539 0.77154674 0.76382651 0.6723287  0.52094421 0.91788611]
Dice por clase: [0.97506087 0.87104305 0.86610163 0.80406286 0.68502738 0.95718521]
mIoU macro = 0.7663 | Dice macro = 0.8597

--- Epoch 37/200 ---
100% 210/210 [00:25<00:00,  8.09it/s, loss=0.187]
Calculando métricas de entrenamiento...
IoU por clase : [0.94865186 0.77616304 0.79188571 0.6684374  0.58447776 0.91371616]
Dice por clase: [0.9736494  0.87397724 0.88385738 0.80127358 0.73775445 0.95491294]
mIoU macro = 0.7806 | Dice macro = 0.8709
Calculando métricas de validación...
IoU por clase : [0.95108176 0.77785271 0.79337729 0.676966   0.5006838  0.89919503]
Dice por clase: [0.97492763 0.87504742 0.8847857  0.80736998 0.66727421 0.94692226]
mIoU macro = 0.7665 | Dice macro = 0.8594

--- Epoch 38/200 ---
100% 210/210 [00:26<00:00,  8.01it/s, loss=0.135]
Calculando métricas de entrenamiento...
IoU por clase : [0.94858054 0.76242421 0.78886978 0.69995713 0.5758958  0.90861992]
Dice por clase: [0.97361184 0.86519943 0.88197563 0.82349974 0.73088055 0.95212243]
mIoU macro = 0.7807 | Dice macro = 0.8712
Calculando métricas de validación...
IoU por clase : [0.95017591 0.771641   0.79145719 0.70907558 0.53357065 0.9056208 ]
Dice por clase: [0.97445149 0.87110312 0.8835904  0.82977674 0.69585402 0.95047325]
mIoU macro = 0.7769 | Dice macro = 0.8675

--- Epoch 39/200 ---
100% 210/210 [00:25<00:00,  8.12it/s, loss=0.132]
Calculando métricas de entrenamiento...
IoU por clase : [0.94991245 0.76861548 0.78854199 0.67520804 0.5929752  0.91285797]
Dice por clase: [0.97431292 0.86917195 0.88177073 0.80611843 0.74448767 0.95444407]
mIoU macro = 0.7814 | Dice macro = 0.8717
Calculando métricas de validación...
IoU por clase : [0.9517488  0.77600019 0.78205843 0.67994895 0.54486716 0.91246555]
Dice por clase: [0.97527797 0.87387399 0.87770234 0.80948763 0.70539031 0.95422953]
mIoU macro = 0.7745 | Dice macro = 0.8660

--- Epoch 40/200 ---
100% 210/210 [00:25<00:00,  8.11it/s, loss=0.0889]
Calculando métricas de entrenamiento...
IoU por clase : [0.95111339 0.77382089 0.79908064 0.70259351 0.59307532 0.91608059]
Dice por clase: [0.97494425 0.87249045 0.88832109 0.82532149 0.74456658 0.95620257]
mIoU macro = 0.7893 | Dice macro = 0.8770
Calculando métricas de validación...
IoU por clase : [0.95255935 0.77754863 0.78614614 0.69483135 0.57152883 0.91624848]
Dice por clase: [0.97570335 0.87485497 0.88027079 0.81994158 0.72735392 0.95629402]
mIoU macro = 0.7831 | Dice macro = 0.8724
🔹 Nuevo mejor mIoU: 0.7831 | Dice: 0.8724  →  guardando modelo…

--- Epoch 41/200 ---
100% 210/210 [00:27<00:00,  7.68it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.95056831 0.77429679 0.80796156 0.69677377 0.63464404 0.91543788]
Dice por clase: [0.9746578  0.87279286 0.89378179 0.82129248 0.77649203 0.95585233]
mIoU macro = 0.7966 | Dice macro = 0.8825
Calculando métricas de validación...
IoU por clase : [0.95213803 0.79023721 0.78459755 0.68790239 0.55807181 0.91374389]
Dice por clase: [0.97548228 0.88282961 0.87929914 0.81509736 0.71636212 0.95492808]
mIoU macro = 0.7811 | Dice macro = 0.8707

--- Epoch 42/200 ---
100% 210/210 [00:25<00:00,  8.16it/s, loss=0.137]
Calculando métricas de entrenamiento...
IoU por clase : [0.95032605 0.77380949 0.80026921 0.70347328 0.60374503 0.91588534]
Dice por clase: [0.97453044 0.8724832  0.88905504 0.82592817 0.75291897 0.95609619]
mIoU macro = 0.7913 | Dice macro = 0.8785
Calculando métricas de validación...
IoU por clase : [0.9523565  0.77109058 0.79345825 0.71282783 0.54640838 0.91022285]
Dice por clase: [0.97559692 0.87075228 0.88483604 0.83234032 0.70668057 0.95300174]
mIoU macro = 0.7811 | Dice macro = 0.8705

--- Epoch 43/200 ---
100% 210/210 [00:26<00:00,  8.03it/s, loss=0.12]
Calculando métricas de entrenamiento...
IoU por clase : [0.95024023 0.76678922 0.79974564 0.71353915 0.63531034 0.91711274]
Dice por clase: [0.97448531 0.86800306 0.88873186 0.83282503 0.77699055 0.95676453]
mIoU macro = 0.7971 | Dice macro = 0.8830
Calculando métricas de validación...
IoU por clase : [0.95249806 0.78272535 0.7822133  0.71092729 0.5726559  0.91811366]
Dice por clase: [0.9756712  0.8781222  0.87779987 0.83104325 0.72826599 0.95730892]
mIoU macro = 0.7865 | Dice macro = 0.8747
🔹 Nuevo mejor mIoU: 0.7865 | Dice: 0.8747  →  guardando modelo…

--- Epoch 44/200 ---
100% 210/210 [00:27<00:00,  7.58it/s, loss=0.138]
Calculando métricas de entrenamiento...
IoU por clase : [0.95004436 0.77953701 0.79123717 0.67429618 0.58058866 0.91064304]
Dice por clase: [0.9743823  0.87611216 0.88345328 0.80546822 0.73464864 0.953232  ]
mIoU macro = 0.7811 | Dice macro = 0.8712
Calculando métricas de validación...
IoU por clase : [0.95126206 0.77154839 0.77932176 0.66900475 0.52970557 0.9109577 ]
Dice por clase: [0.97502235 0.8710441  0.8759762  0.80168106 0.69255886 0.95340436]
mIoU macro = 0.7686 | Dice macro = 0.8616

--- Epoch 45/200 ---
100% 210/210 [00:25<00:00,  8.11it/s, loss=0.119]
Calculando métricas de entrenamiento...
IoU por clase : [0.95077291 0.76785273 0.8033334  0.71436751 0.61307881 0.91436464]
Dice por clase: [0.97476534 0.86868404 0.89094274 0.83338899 0.76013497 0.95526695]
mIoU macro = 0.7940 | Dice macro = 0.8805
Calculando métricas de validación...
IoU por clase : [0.95219713 0.7795485  0.80298529 0.71155222 0.58495062 0.91015863]
Dice por clase: [0.9755133  0.87611942 0.89072861 0.83147007 0.73813103 0.95296654]
mIoU macro = 0.7902 | Dice macro = 0.8775
🔹 Nuevo mejor mIoU: 0.7902 | Dice: 0.8775  →  guardando modelo…

--- Epoch 46/200 ---
100% 210/210 [00:27<00:00,  7.51it/s, loss=0.114]
Calculando métricas de entrenamiento...
IoU por clase : [0.95057392 0.77417814 0.80030934 0.70723299 0.6435871  0.91705754]
Dice por clase: [0.97466075 0.87271748 0.88907981 0.82851373 0.78314937 0.9567345 ]
mIoU macro = 0.7988 | Dice macro = 0.8841
Calculando métricas de validación...
IoU por clase : [0.95286065 0.77815777 0.80163906 0.70904927 0.6063049  0.91667616]
Dice por clase: [0.97586139 0.87524041 0.88989973 0.82975872 0.75490637 0.95652691]
mIoU macro = 0.7941 | Dice macro = 0.8804
🔹 Nuevo mejor mIoU: 0.7941 | Dice: 0.8804  →  guardando modelo…

--- Epoch 47/200 ---
100% 210/210 [00:28<00:00,  7.46it/s, loss=0.135]
Calculando métricas de entrenamiento...
IoU por clase : [0.95210639 0.77716069 0.80567959 0.70999217 0.62965992 0.91144245]
Dice por clase: [0.97546568 0.87460937 0.89238378 0.830404   0.77275009 0.95366978]
mIoU macro = 0.7977 | Dice macro = 0.8832
Calculando métricas de validación...
IoU por clase : [0.95403328 0.77837129 0.80517173 0.72583025 0.62925142 0.91666043]
Dice por clase: [0.97647598 0.87537546 0.89207217 0.84113748 0.77244238 0.95651834]
mIoU macro = 0.8016 | Dice macro = 0.8857
🔹 Nuevo mejor mIoU: 0.8016 | Dice: 0.8857  →  guardando modelo…

--- Epoch 48/200 ---
100% 210/210 [00:27<00:00,  7.55it/s, loss=0.137]
Calculando métricas de entrenamiento...
IoU por clase : [0.95195047 0.77194825 0.81215657 0.72197119 0.64579692 0.91924364]
Dice por clase: [0.97538384 0.87129886 0.89634261 0.83854038 0.78478324 0.95792282]
mIoU macro = 0.8038 | Dice macro = 0.8874
Calculando métricas de validación...
IoU por clase : [0.95330459 0.77170804 0.80520667 0.71423683 0.60681648 0.91740111]
Dice por clase: [0.97609415 0.87114584 0.89209361 0.83330006 0.75530278 0.95692143]
mIoU macro = 0.7948 | Dice macro = 0.8808

--- Epoch 49/200 ---
100% 210/210 [00:26<00:00,  8.02it/s, loss=0.148]
Calculando métricas de entrenamiento...
IoU por clase : [0.95187573 0.77175331 0.81586228 0.69432112 0.63477887 0.91570418]
Dice por clase: [0.9753446  0.87117468 0.89859489 0.81958622 0.77659295 0.95599747]
mIoU macro = 0.7974 | Dice macro = 0.8829
Calculando métricas de validación...
IoU por clase : [0.95305207 0.77243025 0.78181806 0.67373208 0.54664639 0.91446092]
Dice por clase: [0.97596176 0.87160581 0.87755094 0.80506562 0.7068796  0.95531949]
mIoU macro = 0.7737 | Dice macro = 0.8654

--- Epoch 50/200 ---
100% 210/210 [00:25<00:00,  8.11it/s, loss=0.104]
Calculando métricas de entrenamiento...
IoU por clase : [0.9513992  0.77289525 0.81992111 0.70736651 0.6454837  0.91715642]
Dice por clase: [0.97509438 0.87190177 0.90105127 0.82860535 0.78455192 0.9567883 ]
mIoU macro = 0.8024 | Dice macro = 0.8863
Calculando métricas de validación...
IoU por clase : [0.95186885 0.77342977 0.79144992 0.69510056 0.60338936 0.91505257]
Dice por clase: [0.97534099 0.87224178 0.88358587 0.82012899 0.75264234 0.95564225]
mIoU macro = 0.7884 | Dice macro = 0.8766

--- Epoch 51/200 ---
100% 210/210 [00:25<00:00,  8.14it/s, loss=0.132]
Calculando métricas de entrenamiento...
IoU por clase : [0.95187408 0.77043052 0.80415157 0.70634879 0.61298198 0.911721  ]
Dice por clase: [0.97534374 0.87033127 0.89144569 0.82790669 0.76006055 0.95382224]
mIoU macro = 0.7929 | Dice macro = 0.8798
Calculando métricas de validación...
IoU por clase : [0.9533596  0.78165126 0.75351423 0.6964396  0.50586644 0.91320324]
Dice por clase: [0.97612298 0.87744586 0.85943327 0.82106029 0.67186096 0.95463275]
mIoU macro = 0.7673 | Dice macro = 0.8601

--- Epoch 52/200 ---
100% 210/210 [00:26<00:00,  8.05it/s, loss=0.143]
Calculando métricas de entrenamiento...
IoU por clase : [0.95210244 0.77248706 0.82040087 0.71125907 0.64822375 0.91549688]
Dice por clase: [0.9754636  0.87164197 0.90134089 0.83126989 0.78657252 0.95588449]
mIoU macro = 0.8033 | Dice macro = 0.8870
Calculando métricas de validación...
IoU por clase : [0.95386021 0.7803849  0.80251233 0.70379211 0.59533098 0.91882649]
Dice por clase: [0.97638532 0.8766474  0.89043755 0.82614787 0.74634165 0.95769627]
mIoU macro = 0.7925 | Dice macro = 0.8789

--- Epoch 53/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.245]
Calculando métricas de entrenamiento...
IoU por clase : [0.95114279 0.77621837 0.82000989 0.68473529 0.67298643 0.91881054]
Dice por clase: [0.9749597  0.87401232 0.90110488 0.81286988 0.80453304 0.95768761]
mIoU macro = 0.8040 | Dice macro = 0.8875
Calculando métricas de validación...
IoU por clase : [0.95223355 0.77663677 0.78369978 0.65336912 0.5819138  0.91913025]
Dice por clase: [0.97553241 0.87427749 0.87873508 0.79034876 0.73570861 0.95786125]
mIoU macro = 0.7778 | Dice macro = 0.8687

--- Epoch 54/200 ---
100% 210/210 [00:26<00:00,  8.00it/s, loss=0.122]
Calculando métricas de entrenamiento...
IoU por clase : [0.95240101 0.76885633 0.80398112 0.69254165 0.59474114 0.91432272]
Dice por clase: [0.97562028 0.86932592 0.89134095 0.81834518 0.74587796 0.95524407]
mIoU macro = 0.7878 | Dice macro = 0.8760
Calculando métricas de validación...
IoU por clase : [0.95408086 0.77639303 0.79571795 0.67508193 0.56255122 0.91491566]
Dice por clase: [0.9765009  0.87412303 0.88623934 0.80602855 0.72004196 0.95556758]
mIoU macro = 0.7798 | Dice macro = 0.8698

--- Epoch 55/200 ---
100% 210/210 [00:26<00:00,  8.04it/s, loss=0.0924]
Calculando métricas de entrenamiento...
IoU por clase : [0.95312278 0.78812532 0.82767834 0.72271167 0.67018347 0.91836346]
Dice por clase: [0.97599884 0.88151016 0.90571554 0.83903961 0.80252676 0.9574447 ]
mIoU macro = 0.8134 | Dice macro = 0.8937
Calculando métricas de validación...
IoU por clase : [0.95452646 0.7808712  0.8041965  0.71390368 0.62807849 0.91509147]
Dice por clase: [0.97673424 0.87695416 0.8914733  0.83307328 0.771558   0.95566346]
mIoU macro = 0.7994 | Dice macro = 0.8842

--- Epoch 56/200 ---
100% 210/210 [00:27<00:00,  7.61it/s, loss=0.227]
Calculando métricas de entrenamiento...
IoU por clase : [0.9523974  0.7878094  0.82286588 0.69511763 0.66241293 0.91944872]
Dice por clase: [0.97561839 0.88131252 0.90282657 0.82014088 0.79692947 0.95803416]
mIoU macro = 0.8067 | Dice macro = 0.8891
Calculando métricas de validación...
IoU por clase : [0.95291127 0.77533402 0.8131723  0.69303584 0.61185667 0.91666629]
Dice por clase: [0.97588793 0.87345143 0.89696087 0.8186901  0.75919489 0.95652153]
mIoU macro = 0.7938 | Dice macro = 0.8801

--- Epoch 57/200 ---
100% 210/210 [00:26<00:00,  7.94it/s, loss=0.11]
Calculando métricas de entrenamiento...
IoU por clase : [0.95290615 0.78708549 0.82183393 0.70355115 0.6806341  0.91964901]
Dice por clase: [0.97588525 0.88085936 0.9022051  0.82598183 0.80997298 0.95814288]
mIoU macro = 0.8109 | Dice macro = 0.8922
Calculando métricas de validación...
IoU por clase : [0.95357525 0.76963377 0.8047042  0.70097454 0.5900265  0.91100569]
Dice por clase: [0.97623601 0.86982265 0.89178515 0.82420345 0.74215933 0.95343064]
mIoU macro = 0.7883 | Dice macro = 0.8763

--- Epoch 58/200 ---
100% 210/210 [00:26<00:00,  7.96it/s, loss=0.117]
Calculando métricas de entrenamiento...
IoU por clase : [0.95333722 0.78179736 0.81928645 0.72234127 0.67204548 0.92092023]
Dice por clase: [0.97611125 0.8775379  0.9006679  0.83878994 0.80386029 0.95883235]
mIoU macro = 0.8116 | Dice macro = 0.8926
Calculando métricas de validación...
IoU por clase : [0.95429407 0.77380001 0.79685371 0.71276281 0.56797243 0.91478666]
Dice por clase: [0.97661256 0.87247718 0.88694333 0.83229599 0.72446737 0.95549722]
mIoU macro = 0.7867 | Dice macro = 0.8747

--- Epoch 59/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0982]
Calculando métricas de entrenamiento...
IoU por clase : [0.95334765 0.79521246 0.823785   0.72912265 0.66882639 0.92020015]
Dice por clase: [0.97611672 0.88592574 0.90337951 0.84334405 0.80155299 0.95844191]
mIoU macro = 0.8151 | Dice macro = 0.8948
Calculando métricas de validación...
IoU por clase : [0.95429281 0.77757707 0.81362642 0.71580272 0.60212412 0.91834852]
Dice por clase: [0.9766119  0.87487298 0.89723706 0.83436483 0.75165727 0.95743658]
mIoU macro = 0.7970 | Dice macro = 0.8820

--- Epoch 60/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.95330196 0.78139144 0.82590318 0.73557035 0.66135871 0.9175736 ]
Dice por clase: [0.97609277 0.87728213 0.90465167 0.84764107 0.79616606 0.95701526]
mIoU macro = 0.8125 | Dice macro = 0.8931
Calculando métricas de validación...
IoU por clase : [0.95445328 0.77404081 0.80679481 0.72467445 0.61228338 0.91978321]
Dice por clase: [0.97669593 0.87263022 0.89306744 0.84036086 0.75952328 0.9582157 ]
mIoU macro = 0.7987 | Dice macro = 0.8834

--- Epoch 61/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.125]
Calculando métricas de entrenamiento...
IoU por clase : [0.95446645 0.7927356  0.83147721 0.73625137 0.68963923 0.9209927 ]
Dice por clase: [0.97670282 0.88438652 0.90798532 0.84809306 0.81631536 0.95887163]
mIoU macro = 0.8209 | Dice macro = 0.8987
Calculando métricas de validación...
IoU por clase : [0.95510469 0.78567696 0.81389667 0.72165706 0.59747468 0.91324969]
Dice por clase: [0.97703688 0.87997659 0.89740136 0.83832846 0.74802398 0.95465813]
mIoU macro = 0.7978 | Dice macro = 0.8826

--- Epoch 62/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0955]
Calculando métricas de entrenamiento...
IoU por clase : [0.95346252 0.79013802 0.82744851 0.69889659 0.6953144  0.91976751]
Dice por clase: [0.97617693 0.88276771 0.90557792 0.82276531 0.82027782 0.95820719]
mIoU macro = 0.8142 | Dice macro = 0.8943
Calculando métricas de validación...
IoU por clase : [0.95393647 0.78996182 0.80767213 0.69156478 0.58213493 0.91394809]
Dice por clase: [0.97642527 0.88265773 0.89360467 0.81766278 0.73588531 0.95503958]
mIoU macro = 0.7899 | Dice macro = 0.8769

--- Epoch 63/200 ---
100% 210/210 [00:26<00:00,  7.98it/s, loss=0.0919]
Calculando métricas de entrenamiento...
IoU por clase : [0.95287388 0.79052834 0.82855246 0.72420131 0.68458832 0.92114979]
Dice por clase: [0.97586832 0.88301126 0.90623866 0.84004264 0.81276631 0.95895676]
mIoU macro = 0.8170 | Dice macro = 0.8961
Calculando métricas de validación...
IoU por clase : [0.9541949  0.78209126 0.8069925  0.72243727 0.54937406 0.90760429]
Dice por clase: [0.97656063 0.87772302 0.89318855 0.83885467 0.70915614 0.95156453]
mIoU macro = 0.7871 | Dice macro = 0.8745

--- Epoch 64/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.95435486 0.78494598 0.83246387 0.71458968 0.69394689 0.91913745]
Dice por clase: [0.9766444  0.87951791 0.9085733  0.83354016 0.81932544 0.95786516]
mIoU macro = 0.8166 | Dice macro = 0.8959
Calculando métricas de validación...
IoU por clase : [0.9547364  0.78077814 0.81755352 0.71169455 0.60867581 0.91289722]
Dice por clase: [0.97684414 0.87689547 0.89961975 0.83156723 0.75674142 0.95446552]
mIoU macro = 0.7977 | Dice macro = 0.8827

--- Epoch 65/200 ---
100% 210/210 [00:27<00:00,  7.76it/s, loss=0.138]
Calculando métricas de entrenamiento...
IoU por clase : [0.95330587 0.79220402 0.82139331 0.7008902  0.70834144 0.9209366 ]
Dice por clase: [0.97609482 0.88405562 0.90193953 0.82414514 0.82927385 0.95884122]
mIoU macro = 0.8162 | Dice macro = 0.8957
Calculando métricas de validación...
IoU por clase : [0.95367548 0.78536175 0.80754587 0.69007755 0.62086166 0.9168943 ]
Dice por clase: [0.97628853 0.87977885 0.89352739 0.81662235 0.76608841 0.95664566]
mIoU macro = 0.7957 | Dice macro = 0.8815

--- Epoch 66/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.142]
Calculando métricas de entrenamiento...
IoU por clase : [0.95386981 0.79343284 0.82877393 0.7193845  0.68027704 0.91734577]
Dice por clase: [0.97639034 0.88482024 0.90637111 0.83679305 0.80972009 0.95689133]
mIoU macro = 0.8155 | Dice macro = 0.8952
Calculando métricas de validación...
IoU por clase : [0.95436258 0.78294031 0.81642916 0.70033917 0.62142935 0.9178768 ]
Dice por clase: [0.97664844 0.87825746 0.89893862 0.82376409 0.76652041 0.95718015]
mIoU macro = 0.7989 | Dice macro = 0.8836

--- Epoch 67/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0936]
Calculando métricas de entrenamiento...
IoU por clase : [0.95353208 0.79527937 0.83252967 0.73638564 0.67697234 0.91716165]
Dice por clase: [0.97621338 0.88596726 0.90861249 0.84818213 0.80737448 0.95679115]
mIoU macro = 0.8186 | Dice macro = 0.8972
Calculando métricas de validación...
IoU por clase : [0.95371865 0.78376213 0.81970552 0.72257265 0.58608503 0.91120707]
Dice por clase: [0.97631115 0.87877427 0.90092107 0.83894592 0.73903356 0.95354091]
mIoU macro = 0.7962 | Dice macro = 0.8813

--- Epoch 68/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.0982]
Calculando métricas de entrenamiento...
IoU por clase : [0.95484691 0.79980568 0.83237798 0.73377015 0.6880901  0.92204082]
Dice por clase: [0.97690198 0.88876893 0.90852214 0.84644455 0.81522912 0.95943937]
mIoU macro = 0.8218 | Dice macro = 0.8992
Calculando métricas de validación...
IoU por clase : [0.95483642 0.79069318 0.82454767 0.73047243 0.64155097 0.92251093]
Dice por clase: [0.97689649 0.88311408 0.9038379  0.84424625 0.78164002 0.95969382]
mIoU macro = 0.8108 | Dice macro = 0.8916
🔹 Nuevo mejor mIoU: 0.8108 | Dice: 0.8916  →  guardando modelo…

--- Epoch 69/200 ---
100% 210/210 [00:30<00:00,  6.99it/s, loss=0.0886]
Calculando métricas de entrenamiento...
IoU por clase : [0.95424408 0.79546446 0.83316491 0.71429662 0.69540666 0.92330814]
Dice por clase: [0.97658639 0.8860821  0.90899068 0.83334076 0.82034202 0.96012503]
mIoU macro = 0.8193 | Dice macro = 0.8976
Calculando métricas de validación...
IoU por clase : [0.95428964 0.78089256 0.81550036 0.70732867 0.6250176  0.91760785]
Dice por clase: [0.97661024 0.87696762 0.89837532 0.82857939 0.7692441  0.95703389]
mIoU macro = 0.8001 | Dice macro = 0.8845

--- Epoch 70/200 ---
100% 210/210 [00:27<00:00,  7.74it/s, loss=0.107]
Calculando métricas de entrenamiento...
IoU por clase : [0.95403989 0.78800944 0.82776269 0.74525628 0.69850848 0.9193302 ]
Dice por clase: [0.97647944 0.88143767 0.90576605 0.8540365  0.82249631 0.95796981]
mIoU macro = 0.8222 | Dice macro = 0.8997
Calculando métricas de validación...
IoU por clase : [0.95347924 0.77752566 0.81265338 0.71608636 0.63514066 0.91776224]
Dice por clase: [0.97618569 0.87484043 0.89664509 0.83455748 0.77686364 0.95711785]
mIoU macro = 0.8021 | Dice macro = 0.8860

--- Epoch 71/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0937]
Calculando métricas de entrenamiento...
IoU por clase : [0.95141347 0.78709877 0.82956258 0.65276643 0.68673965 0.91884665]
Dice por clase: [0.97510188 0.88086768 0.90684253 0.78990766 0.81428056 0.95770723]
mIoU macro = 0.8044 | Dice macro = 0.8875
Calculando métricas de validación...
IoU por clase : [0.95085204 0.76833525 0.80711823 0.63977601 0.60423648 0.9121917 ]
Dice por clase: [0.97480693 0.86899274 0.89326555 0.78032122 0.75330101 0.95407976]
mIoU macro = 0.7804 | Dice macro = 0.8708

--- Epoch 72/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.111]
Calculando métricas de entrenamiento...
IoU por clase : [0.95293479 0.76858544 0.83337962 0.71542112 0.70341564 0.91888131]
Dice por clase: [0.97590027 0.86915274 0.90911845 0.83410553 0.82588844 0.95772605]
mIoU macro = 0.8154 | Dice macro = 0.8953
Calculando métricas de validación...
IoU por clase : [0.9541799  0.77561291 0.81371435 0.71054492 0.61138369 0.91615723]
Dice por clase: [0.97655277 0.87362837 0.89729052 0.83078195 0.75883068 0.95624432]
mIoU macro = 0.7969 | Dice macro = 0.8822

--- Epoch 73/200 ---
100% 210/210 [00:26<00:00,  7.91it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.95428163 0.80113703 0.82763253 0.73475776 0.67999797 0.92074371]
Dice por clase: [0.97660605 0.88959032 0.90568811 0.84710128 0.80952237 0.95873667]
mIoU macro = 0.8198 | Dice macro = 0.8979
Calculando métricas de validación...
IoU por clase : [0.95421549 0.78412494 0.76805495 0.72434871 0.5335522  0.91583754]
Dice por clase: [0.97657141 0.87900228 0.86881344 0.8401418  0.69583833 0.95607015]
mIoU macro = 0.7800 | Dice macro = 0.8694

--- Epoch 74/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.106]
Calculando métricas de entrenamiento...
IoU por clase : [0.95493454 0.79250332 0.83931092 0.74678543 0.68587375 0.91880483]
Dice por clase: [0.97694784 0.88424195 0.91263626 0.85503968 0.81367154 0.95768451]
mIoU macro = 0.8230 | Dice macro = 0.9000
Calculando métricas de validación...
IoU por clase : [0.95508651 0.79007507 0.79999145 0.72374104 0.58881314 0.91850566]
Dice por clase: [0.97702736 0.88272842 0.88888361 0.83973291 0.74119873 0.95752197]
mIoU macro = 0.7960 | Dice macro = 0.8812

--- Epoch 75/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0902]
Calculando métricas de entrenamiento...
IoU por clase : [0.95489946 0.78777692 0.83224736 0.74654878 0.68860209 0.9211024 ]
Dice por clase: [0.97692948 0.88129219 0.90844433 0.85488455 0.81558834 0.95893108]
mIoU macro = 0.8219 | Dice macro = 0.8993
Calculando métricas de validación...
IoU por clase : [0.95488298 0.77877425 0.81242506 0.7256756  0.61993059 0.92006889]
Dice por clase: [0.97692086 0.87563023 0.8965061  0.84103362 0.7653792  0.95837071]
mIoU macro = 0.8020 | Dice macro = 0.8856

--- Epoch 76/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.145]
Calculando métricas de entrenamiento...
IoU por clase : [0.95503038 0.79840327 0.82641723 0.7365319  0.70637876 0.92090066]
Dice por clase: [0.97699799 0.88790238 0.90495996 0.84827915 0.82792728 0.95882175]
mIoU macro = 0.8239 | Dice macro = 0.9008
Calculando métricas de validación...
IoU por clase : [0.95469134 0.78279229 0.82457404 0.72527238 0.55824555 0.90538413]
Dice por clase: [0.97682056 0.87816432 0.90385374 0.84076276 0.71650524 0.95034289]
mIoU macro = 0.7918 | Dice macro = 0.8777

--- Epoch 77/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.121]
Calculando métricas de entrenamiento...
IoU por clase : [0.95239758 0.73998791 0.83371242 0.67960394 0.66900377 0.90218395]
Dice por clase: [0.97561848 0.85056673 0.90931643 0.80924309 0.80168036 0.94857697]
mIoU macro = 0.7961 | Dice macro = 0.8825
Calculando métricas de validación...
IoU por clase : [0.95203613 0.71639374 0.8288726  0.67543265 0.54814693 0.89090209]
Dice por clase: [0.9754288  0.8347662  0.90643011 0.80627849 0.70813296 0.94230377]
mIoU macro = 0.7686 | Dice macro = 0.8622

--- Epoch 78/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.118]
Calculando métricas de entrenamiento...
IoU por clase : [0.95543933 0.79479059 0.82471923 0.72372779 0.68620221 0.92291178]
Dice por clase: [0.97721194 0.88566387 0.90394096 0.83972399 0.81390263 0.95991068]
mIoU macro = 0.8180 | Dice macro = 0.8967
Calculando métricas de validación...
IoU por clase : [0.95502054 0.78274187 0.80700805 0.71720926 0.64151153 0.9207282 ]
Dice por clase: [0.97699284 0.87813259 0.89319807 0.83531958 0.78161075 0.95872826]
mIoU macro = 0.8040 | Dice macro = 0.8873

--- Epoch 79/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0881]
Calculando métricas de entrenamiento...
IoU por clase : [0.95519808 0.7940485  0.84084847 0.73769146 0.70649913 0.92226699]
Dice por clase: [0.97708574 0.88520294 0.91354447 0.84904769 0.82800995 0.9595618 ]
mIoU macro = 0.8261 | Dice macro = 0.9021
Calculando métricas de validación...
IoU por clase : [0.95481824 0.77847612 0.83017562 0.72566541 0.61618591 0.91313958]
Dice por clase: [0.97688698 0.87544175 0.9072087  0.84102678 0.7625186  0.95459797]
mIoU macro = 0.8031 | Dice macro = 0.8863

--- Epoch 80/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.141]
Calculando métricas de entrenamiento...
IoU por clase : [0.9536403  0.79508671 0.8424191  0.72227475 0.70798272 0.91332732]
Dice por clase: [0.97627009 0.88584769 0.91447066 0.83874509 0.82902797 0.95470055]
mIoU macro = 0.8225 | Dice macro = 0.8998
Calculando métricas de validación...
IoU por clase : [0.95302577 0.77660971 0.82610284 0.69758465 0.61475152 0.90555528]
Dice por clase: [0.97594797 0.87426035 0.90477143 0.82185551 0.76141934 0.95043717]
mIoU macro = 0.7956 | Dice macro = 0.8814

--- Epoch 81/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0989]
Calculando métricas de entrenamiento...
IoU por clase : [0.95523326 0.80092365 0.84400564 0.75244222 0.71416231 0.92085054]
Dice por clase: [0.97710414 0.88945875 0.91540462 0.85873555 0.83324934 0.95879458]
mIoU macro = 0.8313 | Dice macro = 0.9055
Calculando métricas de validación...
IoU por clase : [0.95491674 0.78189491 0.82659249 0.72815091 0.61302467 0.91380771]
Dice por clase: [0.97693853 0.87759935 0.90506503 0.84269366 0.76009336 0.95496293]
mIoU macro = 0.8031 | Dice macro = 0.8862

--- Epoch 82/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.95621755 0.80088314 0.84722322 0.74544665 0.72623394 0.92568078]
Dice por clase: [0.97761882 0.88943377 0.91729382 0.85416148 0.84140848 0.96140626]
mIoU macro = 0.8336 | Dice macro = 0.9069
Calculando métricas de validación...
IoU por clase : [0.95533037 0.77479202 0.82265298 0.7087928  0.64203673 0.91952789]
Dice por clase: [0.97715495 0.87310739 0.90269842 0.82958309 0.78200045 0.95807714]
mIoU macro = 0.8039 | Dice macro = 0.8871

--- Epoch 83/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.141]
Calculando métricas de entrenamiento...
IoU por clase : [0.95529789 0.7971686  0.84455109 0.75173534 0.71560907 0.9229694 ]
Dice por clase: [0.97713795 0.88713836 0.91572535 0.85827501 0.83423326 0.95994185]
mIoU macro = 0.8312 | Dice macro = 0.9054
Calculando métricas de validación...
IoU por clase : [0.95559778 0.79080998 0.80047266 0.72176088 0.64915544 0.92134509]
Dice por clase: [0.97729481 0.88318692 0.88918058 0.83839851 0.78725804 0.95906258]
mIoU macro = 0.8065 | Dice macro = 0.8891

--- Epoch 84/200 ---
100% 210/210 [00:27<00:00,  7.76it/s, loss=0.0878]
Calculando métricas de entrenamiento...
IoU por clase : [0.9559299  0.79688392 0.83107285 0.74080633 0.70264408 0.92363434]
Dice por clase: [0.97746847 0.88696204 0.90774417 0.85110712 0.82535638 0.96030137]
mIoU macro = 0.8252 | Dice macro = 0.9015
Calculando métricas de validación...
IoU por clase : [0.9562413  0.7872146  0.80907092 0.73285787 0.63376575 0.91992335]
Dice por clase: [0.97763123 0.88094021 0.89446015 0.84583726 0.77583429 0.95829175]
mIoU macro = 0.8065 | Dice macro = 0.8888

--- Epoch 85/200 ---
100% 210/210 [00:27<00:00,  7.74it/s, loss=0.0978]
Calculando métricas de entrenamiento...
IoU por clase : [0.95522723 0.79692148 0.83597024 0.74497424 0.69675023 0.92025195]
Dice por clase: [0.97710099 0.88698531 0.91065772 0.85385127 0.82127613 0.95847001]
mIoU macro = 0.8250 | Dice macro = 0.9014
Calculando métricas de validación...
IoU por clase : [0.95492533 0.78608549 0.81274145 0.72753305 0.63653895 0.91957786]
Dice por clase: [0.97694302 0.88023277 0.8966987  0.84227975 0.77790871 0.95810426]
mIoU macro = 0.8062 | Dice macro = 0.8887

--- Epoch 86/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.0982]
Calculando métricas de entrenamiento...
IoU por clase : [0.9552575  0.80123821 0.83609534 0.73885161 0.71714346 0.92083873]
Dice por clase: [0.97711682 0.88965269 0.91073195 0.84981559 0.83527495 0.95878817]
mIoU macro = 0.8282 | Dice macro = 0.9036
Calculando métricas de validación...
IoU por clase : [0.9555975  0.78875053 0.81166902 0.72430011 0.61459208 0.92068831]
Dice por clase: [0.97729466 0.8819011  0.89604559 0.8401091  0.76129703 0.95870663]
mIoU macro = 0.8026 | Dice macro = 0.8859

--- Epoch 87/200 ---
100% 210/210 [00:27<00:00,  7.75it/s, loss=0.125]
Calculando métricas de entrenamiento...
IoU por clase : [0.95594483 0.7986572  0.84509648 0.73731095 0.7186717  0.92517273]
Dice por clase: [0.97747627 0.88805938 0.91604584 0.8487956  0.83631062 0.96113218]
mIoU macro = 0.8301 | Dice macro = 0.9046
Calculando métricas de validación...
IoU por clase : [0.95592159 0.78455895 0.82883301 0.721376   0.63487389 0.91622343]
Dice por clase: [0.97746412 0.87927491 0.90640644 0.83813879 0.77666405 0.95628037]
mIoU macro = 0.8070 | Dice macro = 0.8890

--- Epoch 88/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.103]
Calculando métricas de entrenamiento...
IoU por clase : [0.95600056 0.79275758 0.84328111 0.74645355 0.71715226 0.92289568]
Dice por clase: [0.9775054  0.8844002  0.9149783  0.85482211 0.83528092 0.95990197]
mIoU macro = 0.8298 | Dice macro = 0.9045
Calculando métricas de validación...
IoU por clase : [0.95580831 0.78646556 0.81070994 0.72692532 0.57630285 0.91891856]
Dice por clase: [0.9774049  0.880471   0.89546086 0.84187233 0.73120829 0.95774629]
mIoU macro = 0.7959 | Dice macro = 0.8807

--- Epoch 89/200 ---
100% 210/210 [00:27<00:00,  7.74it/s, loss=0.116]
Calculando métricas de entrenamiento...
IoU por clase : [0.95690123 0.80779181 0.84024242 0.74342878 0.72288442 0.92300873]
Dice por clase: [0.97797601 0.89367791 0.91318667 0.85283527 0.83915602 0.95996312]
mIoU macro = 0.8324 | Dice macro = 0.9061
Calculando métricas de validación...
IoU por clase : [0.95584412 0.78754367 0.82718086 0.71840422 0.60674294 0.91673269]
Dice por clase: [0.97742362 0.88114622 0.90541761 0.83612949 0.75524581 0.95655768]
mIoU macro = 0.8021 | Dice macro = 0.8853

--- Epoch 90/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.9565888  0.80952851 0.84554739 0.74446143 0.75050307 0.92514769]
Dice por clase: [0.97781282 0.89473971 0.91631068 0.85351435 0.8574713  0.96111867]
mIoU macro = 0.8386 | Dice macro = 0.9102
Calculando métricas de validación...
IoU por clase : [0.95598278 0.78681614 0.82174622 0.72692881 0.60613188 0.91404748]
Dice por clase: [0.97749611 0.88069066 0.90215225 0.84187467 0.75477224 0.95509384]
mIoU macro = 0.8019 | Dice macro = 0.8853

--- Epoch 91/200 ---
100% 210/210 [00:27<00:00,  7.76it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.95583253 0.80508669 0.83036801 0.73989587 0.69742275 0.92363244]
Dice por clase: [0.97741756 0.89201997 0.90732356 0.85050592 0.82174314 0.96030034]
mIoU macro = 0.8254 | Dice macro = 0.9016
Calculando métricas de validación...
IoU por clase : [0.95612039 0.77457795 0.81890034 0.73299272 0.63840952 0.91440726]
Dice por clase: [0.97756804 0.87297146 0.90043453 0.84592706 0.77930397 0.95529021]
mIoU macro = 0.8059 | Dice macro = 0.8886

--- Epoch 92/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.12]
Calculando métricas de entrenamiento...
IoU por clase : [0.95425445 0.78415149 0.82496551 0.71588506 0.71587757 0.92072096]
Dice por clase: [0.97659182 0.87901896 0.90408888 0.83442077 0.83441567 0.95872433]
mIoU macro = 0.8193 | Dice macro = 0.8979
Calculando métricas de validación...
IoU por clase : [0.95527228 0.77829393 0.82019519 0.72953453 0.62081984 0.91763091]
Dice por clase: [0.97712456 0.87532653 0.90121675 0.8436195  0.76605657 0.95704643]
mIoU macro = 0.8036 | Dice macro = 0.8867

--- Epoch 93/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0898]
Calculando métricas de entrenamiento...
IoU por clase : [0.95619401 0.78908783 0.83192092 0.71611512 0.70749404 0.91855446]
Dice por clase: [0.97760652 0.88211189 0.90824982 0.83457702 0.82869284 0.95754849]
mIoU macro = 0.8199 | Dice macro = 0.8981
Calculando métricas de validación...
IoU por clase : [0.95599734 0.783768   0.81303316 0.70333458 0.57974003 0.91091832]
Dice por clase: [0.97750372 0.87877796 0.89687621 0.82583256 0.7339689  0.95338279]
mIoU macro = 0.7911 | Dice macro = 0.8777

--- Epoch 94/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0933]
Calculando métricas de entrenamiento...
IoU por clase : [0.95671341 0.77796993 0.84850913 0.75285774 0.71273446 0.92378585]
Dice por clase: [0.97787791 0.87512159 0.918047   0.85900609 0.83227666 0.96038325]
mIoU macro = 0.8288 | Dice macro = 0.9038
Calculando métricas de validación...
IoU por clase : [0.95646234 0.77242793 0.83898513 0.72908858 0.62819356 0.91660693]
Dice por clase: [0.97774674 0.87160433 0.91244362 0.84332126 0.77164482 0.95648922]
mIoU macro = 0.8070 | Dice macro = 0.8889

--- Epoch 95/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0886]
Calculando métricas de entrenamiento...
IoU por clase : [0.95671432 0.80464712 0.84840567 0.73982396 0.72935506 0.92676727]
Dice por clase: [0.97787838 0.89175009 0.91798644 0.85045841 0.84349949 0.96199192]
mIoU macro = 0.8343 | Dice macro = 0.9073
Calculando métricas de validación...
IoU por clase : [0.95687738 0.79005928 0.82328285 0.72628014 0.61969757 0.91900439]
Dice por clase: [0.97796356 0.88271857 0.90307749 0.84143949 0.76520158 0.9577929 ]
mIoU macro = 0.8059 | Dice macro = 0.8880

--- Epoch 96/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0726]
Calculando métricas de entrenamiento...
IoU por clase : [0.9571571  0.80844623 0.84774025 0.74987688 0.74773548 0.92786708]
Dice por clase: [0.97810963 0.89407826 0.91759678 0.85706245 0.85566207 0.96258408]
mIoU macro = 0.8398 | Dice macro = 0.9108
Calculando métricas de validación...
IoU por clase : [0.95623294 0.79235883 0.82811491 0.72216604 0.64587169 0.91883466]
Dice por clase: [0.97762687 0.88415201 0.90597687 0.83867179 0.78483844 0.95770071]
mIoU macro = 0.8106 | Dice macro = 0.8915

--- Epoch 97/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.124]
Calculando métricas de entrenamiento...
IoU por clase : [0.95676171 0.79915716 0.85317381 0.75628747 0.73529708 0.92540196]
Dice por clase: [0.97790314 0.88836837 0.92077042 0.86123426 0.8474596  0.96125586]
mIoU macro = 0.8377 | Dice macro = 0.9095
Calculando métricas de validación...
IoU por clase : [0.95621081 0.79473382 0.83049664 0.71933199 0.65670907 0.92385271]
Dice por clase: [0.9776153  0.88562862 0.90740034 0.83675753 0.79278744 0.96041938]
mIoU macro = 0.8136 | Dice macro = 0.8934
🔹 Nuevo mejor mIoU: 0.8136 | Dice: 0.8934  →  guardando modelo…

--- Epoch 98/200 ---
100% 210/210 [00:29<00:00,  7.23it/s, loss=0.142]
Calculando métricas de entrenamiento...
IoU por clase : [0.95734552 0.80954681 0.8387128  0.75121938 0.73474578 0.92624479]
Dice por clase: [0.978208   0.89475089 0.91228255 0.85793863 0.84709332 0.96171036]
mIoU macro = 0.8363 | Dice macro = 0.9087
Calculando métricas de validación...
IoU por clase : [0.95698069 0.79270266 0.82916273 0.73217511 0.6610343  0.92252204]
Dice por clase: [0.97801751 0.88436602 0.90660357 0.84538232 0.79593094 0.95969983]
mIoU macro = 0.8158 | Dice macro = 0.8950
🔹 Nuevo mejor mIoU: 0.8158 | Dice: 0.8950  →  guardando modelo…

--- Epoch 99/200 ---
100% 210/210 [00:29<00:00,  7.11it/s, loss=0.0956]
Calculando métricas de entrenamiento...
IoU por clase : [0.95764667 0.80203143 0.85060276 0.75231025 0.74007924 0.92563656]
Dice por clase: [0.97836518 0.89014144 0.91927104 0.8586496  0.85062706 0.96138241]
mIoU macro = 0.8381 | Dice macro = 0.9097
Calculando métricas de validación...
IoU por clase : [0.95683593 0.79127235 0.83018393 0.72524088 0.66049405 0.92206046]
Dice por clase: [0.97794191 0.8834752  0.90721366 0.84074159 0.7955392  0.95945001]
mIoU macro = 0.8143 | Dice macro = 0.8941

--- Epoch 100/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0704]
Calculando métricas de entrenamiento...
IoU por clase : [0.95796945 0.81173504 0.85719557 0.73949433 0.75492473 0.92868232]
Dice por clase: [0.9785336  0.89608582 0.92310749 0.85024057 0.86034998 0.96302259]
mIoU macro = 0.8417 | Dice macro = 0.9119
Calculando métricas de validación...
IoU por clase : [0.95658218 0.79794478 0.83230524 0.71566674 0.63749463 0.91778294]
Dice por clase: [0.97780935 0.88761878 0.90847881 0.83427244 0.77862195 0.95712911]
mIoU macro = 0.8096 | Dice macro = 0.8907

--- Epoch 101/200 ---
100% 210/210 [00:26<00:00,  7.95it/s, loss=0.0607]
Calculando métricas de entrenamiento...
IoU por clase : [0.95627754 0.80558332 0.85483147 0.72315162 0.74501761 0.92633747]
Dice por clase: [0.97765018 0.89232472 0.92173492 0.83933603 0.85387976 0.96176032]
mIoU macro = 0.8352 | Dice macro = 0.9078
Calculando métricas de validación...
IoU por clase : [0.95444131 0.78819825 0.83929716 0.68133433 0.63940359 0.91542883]
Dice por clase: [0.97668966 0.88155578 0.91262813 0.81046859 0.78004415 0.9558474 ]
mIoU macro = 0.8030 | Dice macro = 0.8862

--- Epoch 102/200 ---
100% 210/210 [00:25<00:00,  8.10it/s, loss=0.0973]
Calculando métricas de entrenamiento...
IoU por clase : [0.95755082 0.8033757  0.84741369 0.73743941 0.73774403 0.92392975]
Dice por clase: [0.97831516 0.89096875 0.91740545 0.84888072 0.84908251 0.96046101]
mIoU macro = 0.8346 | Dice macro = 0.9075
Calculando métricas de validación...
IoU por clase : [0.95597998 0.78998524 0.8263114  0.72081166 0.64320625 0.91801112]
Dice por clase: [0.97749465 0.88267235 0.9048965  0.83775776 0.78286734 0.95725318]
mIoU macro = 0.8091 | Dice macro = 0.8905

--- Epoch 103/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0928]
Calculando métricas de entrenamiento...
IoU por clase : [0.95774044 0.81036147 0.84932376 0.75202263 0.72507259 0.92563229]
Dice por clase: [0.97841412 0.89524825 0.9185236  0.85846223 0.8406285  0.96138011]
mIoU macro = 0.8367 | Dice macro = 0.9088
Calculando métricas de validación...
IoU por clase : [0.9563056  0.78774817 0.82897705 0.73329759 0.62271739 0.91792159]
Dice por clase: [0.97766484 0.8812742  0.90649257 0.84613005 0.76749949 0.9572045 ]
mIoU macro = 0.8078 | Dice macro = 0.8894

--- Epoch 104/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.131]
Calculando métricas de entrenamiento...
IoU por clase : [0.95757455 0.79385817 0.84736077 0.74742629 0.74093335 0.92396131]
Dice por clase: [0.97832754 0.88508466 0.91737443 0.85545959 0.85119094 0.96047806]
mIoU macro = 0.8352 | Dice macro = 0.9080
Calculando métricas de validación...
IoU por clase : [0.95647155 0.73847553 0.83515494 0.72765061 0.61494679 0.90163902]
Dice por clase: [0.97775156 0.84956678 0.91017376 0.84235853 0.76156911 0.94827568]
mIoU macro = 0.7957 | Dice macro = 0.8816

--- Epoch 105/200 ---
100% 210/210 [00:26<00:00,  7.95it/s, loss=0.163]
Calculando métricas de entrenamiento...
IoU por clase : [0.95795725 0.81595667 0.84650809 0.75553865 0.72648618 0.92792577]
Dice por clase: [0.97852724 0.89865213 0.9168745  0.86074852 0.84157775 0.96261566]
mIoU macro = 0.8384 | Dice macro = 0.9098
Calculando métricas de validación...
IoU por clase : [0.95742505 0.7905865  0.83684234 0.7319954  0.60343442 0.91186247]
Dice por clase: [0.97824951 0.88304754 0.91117493 0.84526252 0.75267739 0.95389965]
mIoU macro = 0.8054 | Dice macro = 0.8874

--- Epoch 106/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.107]
Calculando métricas de entrenamiento...
IoU por clase : [0.95698929 0.81168676 0.83737101 0.73314072 0.74924948 0.92841847]
Dice por clase: [0.978022   0.8960564  0.91148821 0.84602561 0.85665251 0.96288071]
mIoU macro = 0.8361 | Dice macro = 0.9085
Calculando métricas de validación...
IoU por clase : [0.95593247 0.80340506 0.82154552 0.71630916 0.61304958 0.91353608]
Dice por clase: [0.97746981 0.89098681 0.90203128 0.83470878 0.76011251 0.95481459]
mIoU macro = 0.8040 | Dice macro = 0.8867

--- Epoch 107/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.0826]
Calculando métricas de entrenamiento...
IoU por clase : [0.9570186  0.80739728 0.83662586 0.73851154 0.70290377 0.91944707]
Dice por clase: [0.97803731 0.89343642 0.91104659 0.84959061 0.82553552 0.95803326]
mIoU macro = 0.8270 | Dice macro = 0.9026
Calculando métricas de validación...
IoU por clase : [0.95516328 0.79247097 0.82892799 0.71338047 0.6072296  0.90701057]
Dice por clase: [0.97706753 0.88422182 0.90646323 0.83271694 0.75562272 0.95123811]
mIoU macro = 0.8007 | Dice macro = 0.8846

--- Epoch 108/200 ---
100% 210/210 [00:26<00:00,  7.98it/s, loss=0.0648]
Calculando métricas de entrenamiento...
IoU por clase : [0.95717764 0.81656572 0.85585771 0.76229638 0.73517202 0.92059352]
Dice por clase: [0.97812035 0.89902139 0.92233118 0.86511712 0.84737653 0.95865524]
mIoU macro = 0.8413 | Dice macro = 0.9118
Calculando métricas de validación...
IoU por clase : [0.95626084 0.79542192 0.83810444 0.73466411 0.63847442 0.91256723]
Dice por clase: [0.97764145 0.8860557  0.91192254 0.84703904 0.77935232 0.95428513]
mIoU macro = 0.8126 | Dice macro = 0.8927

--- Epoch 109/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0941]
Calculando métricas de entrenamiento...
IoU por clase : [0.95891546 0.81810319 0.84907887 0.77488259 0.74368456 0.93052844]
Dice por clase: [0.9790269  0.89995243 0.91838037 0.8731649  0.85300355 0.96401422]
mIoU macro = 0.8459 | Dice macro = 0.9146
Calculando métricas de validación...
IoU por clase : [0.95705427 0.79143155 0.82484789 0.73580958 0.61170134 0.91247816]
Dice por clase: [0.97805593 0.88357442 0.90401824 0.84779988 0.7590753  0.95423642]
mIoU macro = 0.8056 | Dice macro = 0.8878

--- Epoch 110/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.95822754 0.80244166 0.84629116 0.74946799 0.72116336 0.92275461]
Dice por clase: [0.97866823 0.89039404 0.91674724 0.85679532 0.83799525 0.95982566]
mIoU macro = 0.8334 | Dice macro = 0.9067
Calculando métricas de validación...
IoU por clase : [0.95680097 0.78271555 0.82917104 0.72794109 0.60760162 0.90957418]
Dice por clase: [0.97792365 0.87811603 0.90660854 0.84255313 0.75591068 0.95264608]
mIoU macro = 0.8023 | Dice macro = 0.8856

--- Epoch 111/200 ---
100% 210/210 [00:26<00:00,  8.01it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.95888401 0.8162707  0.85237773 0.75045269 0.734531   0.9314784 ]
Dice por clase: [0.9790105  0.89884256 0.92030661 0.85743841 0.84695056 0.96452376]
mIoU macro = 0.8407 | Dice macro = 0.9112
Calculando métricas de validación...
IoU por clase : [0.95701773 0.79873771 0.83349283 0.7269512  0.64602153 0.91715878]
Dice por clase: [0.97803685 0.88810915 0.90918581 0.84188968 0.78494907 0.95678959]
mIoU macro = 0.8132 | Dice macro = 0.8932

--- Epoch 112/200 ---
100% 210/210 [00:25<00:00,  8.08it/s, loss=0.108]
Calculando métricas de entrenamiento...
IoU por clase : [0.95851387 0.80463845 0.85248513 0.76073694 0.74317999 0.92837534]
Dice por clase: [0.97881754 0.89174477 0.9203692  0.86411198 0.85267155 0.96285751]
mIoU macro = 0.8413 | Dice macro = 0.9118
Calculando métricas de validación...
IoU por clase : [0.95707634 0.78658798 0.83190096 0.73105191 0.66231533 0.92070354]
Dice por clase: [0.97806746 0.88054771 0.90823792 0.84463315 0.79685884 0.95871489]
mIoU macro = 0.8149 | Dice macro = 0.8945

--- Epoch 113/200 ---
100% 210/210 [00:26<00:00,  7.98it/s, loss=0.0733]
Calculando métricas de entrenamiento...
IoU por clase : [0.95891406 0.814165   0.86081094 0.76437133 0.76150045 0.93028966]
Dice por clase: [0.97902617 0.89756445 0.92519978 0.86645177 0.86460432 0.96388607]
mIoU macro = 0.8483 | Dice macro = 0.9161
Calculando métricas de validación...
IoU por clase : [0.95778519 0.79766166 0.83200408 0.73688416 0.64411649 0.91630187]
Dice por clase: [0.97843747 0.88744359 0.90829937 0.84851273 0.78354118 0.9563231 ]
mIoU macro = 0.8141 | Dice macro = 0.8938

--- Epoch 114/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.135]
Calculando métricas de entrenamiento...
IoU por clase : [0.95800711 0.8132349  0.85287686 0.76663247 0.76074011 0.92938451]
Dice por clase: [0.97855325 0.89699895 0.92059745 0.86790262 0.86411402 0.96339999]
mIoU macro = 0.8468 | Dice macro = 0.9153
Calculando métricas de validación...
IoU por clase : [0.9564237  0.7930545  0.82856232 0.72959319 0.65369579 0.91684858]
Dice por clase: [0.97772655 0.88458493 0.90624455 0.84365872 0.79058772 0.95662077]
mIoU macro = 0.8130 | Dice macro = 0.8932

--- Epoch 115/200 ---
100% 210/210 [00:26<00:00,  7.96it/s, loss=0.117]
Calculando métricas de entrenamiento...
IoU por clase : [0.95916573 0.81557299 0.86116393 0.76540556 0.76856909 0.92823895]
Dice por clase: [0.97915731 0.89841939 0.92540363 0.86711583 0.86914229 0.96278415]
mIoU macro = 0.8497 | Dice macro = 0.9170
Calculando métricas de validación...
IoU por clase : [0.95706571 0.79206106 0.83462472 0.72693507 0.68115935 0.92256213]
Dice por clase: [0.97806191 0.8839666  0.9098588  0.84187887 0.81034478 0.95972153]
mIoU macro = 0.8191 | Dice macro = 0.8973
🔹 Nuevo mejor mIoU: 0.8191 | Dice: 0.8973  →  guardando modelo…

--- Epoch 116/200 ---
100% 210/210 [00:28<00:00,  7.37it/s, loss=0.0831]
Calculando métricas de entrenamiento...
IoU por clase : [0.95899303 0.80570125 0.85429095 0.76478283 0.75032776 0.92806589]
Dice por clase: [0.97906732 0.89239707 0.92142061 0.86671608 0.85735686 0.96269105]
mIoU macro = 0.8437 | Dice macro = 0.9133
Calculando métricas de validación...
IoU por clase : [0.95661816 0.77735794 0.82841672 0.7327126  0.64246981 0.91225672]
Dice por clase: [0.97782815 0.87473426 0.90615746 0.84574049 0.78232161 0.95411532]
mIoU macro = 0.8083 | Dice macro = 0.8901

--- Epoch 117/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0734]
Calculando métricas de entrenamiento...
IoU por clase : [0.95924148 0.8145033  0.85613817 0.76505604 0.7585253  0.92820476]
Dice por clase: [0.97919679 0.89776999 0.92249401 0.8668915  0.86268341 0.96276576]
mIoU macro = 0.8469 | Dice macro = 0.9153
Calculando métricas de validación...
IoU por clase : [0.95721129 0.79266179 0.82701485 0.7292472  0.63723349 0.91629439]
Dice por clase: [0.97813792 0.88434059 0.90531815 0.84342736 0.77842714 0.95631902]
mIoU macro = 0.8099 | Dice macro = 0.8910

--- Epoch 118/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.0736]
Calculando métricas de entrenamiento...
IoU por clase : [0.95922057 0.812584   0.85772592 0.76173517 0.76765512 0.92889739]
Dice por clase: [0.97918589 0.89660286 0.92341492 0.86475559 0.86855757 0.96313821]
mIoU macro = 0.8480 | Dice macro = 0.9159
Calculando métricas de validación...
IoU por clase : [0.95703218 0.79312082 0.82818206 0.72216977 0.62166377 0.91299074]
Dice por clase: [0.9780444  0.88462619 0.90601705 0.83867431 0.76669873 0.95451663]
mIoU macro = 0.8059 | Dice macro = 0.8881

--- Epoch 119/200 ---
100% 210/210 [00:26<00:00,  7.92it/s, loss=0.0972]
Calculando métricas de entrenamiento...
IoU por clase : [0.9569682  0.8129429  0.84557828 0.75269359 0.74031431 0.92789898]
Dice por clase: [0.97801099 0.8968213  0.91632882 0.85889923 0.8507823  0.96260124]
mIoU macro = 0.8394 | Dice macro = 0.9106
Calculando métricas de validación...
IoU por clase : [0.95621939 0.79647763 0.82265631 0.73345181 0.63123522 0.91792342]
Dice por clase: [0.97761979 0.88671032 0.90270042 0.84623271 0.77393525 0.95720549]
mIoU macro = 0.8097 | Dice macro = 0.8907

--- Epoch 120/200 ---
100% 210/210 [00:26<00:00,  7.96it/s, loss=0.101]
Calculando métricas de entrenamiento...
IoU por clase : [0.95376468 0.76012573 0.85322431 0.75303186 0.68585345 0.90334827]
Dice por clase: [0.97633526 0.86371754 0.92079982 0.85911942 0.81365726 0.94922015]
mIoU macro = 0.8182 | Dice macro = 0.8971
Calculando métricas de validación...
IoU por clase : [0.95362117 0.75860874 0.82566155 0.7331414  0.53138685 0.87592084]
Dice por clase: [0.97626007 0.86273737 0.9045067  0.84602607 0.69399427 0.93385693]
mIoU macro = 0.7797 | Dice macro = 0.8696

--- Epoch 121/200 ---
100% 210/210 [00:26<00:00,  7.99it/s, loss=0.125]
Calculando métricas de entrenamiento...
IoU por clase : [0.95858685 0.81458202 0.85870028 0.76531882 0.74263454 0.92734624]
Dice por clase: [0.97885559 0.8978178  0.92397929 0.86706017 0.85231243 0.96230373]
mIoU macro = 0.8445 | Dice macro = 0.9137
Calculando métricas de validación...
IoU por clase : [0.95719818 0.79821415 0.83484365 0.7297442  0.64278575 0.91610802]
Dice por clase: [0.97813108 0.88778542 0.90998887 0.84375967 0.78255579 0.95621751]
mIoU macro = 0.8131 | Dice macro = 0.8931

--- Epoch 122/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0981]
Calculando métricas de entrenamiento...
IoU por clase : [0.9589115  0.81749319 0.85063249 0.74522115 0.7231619  0.92858727]
Dice por clase: [0.97902483 0.89958322 0.9192884  0.85401343 0.83934295 0.96297148]
mIoU macro = 0.8373 | Dice macro = 0.9090
Calculando métricas de validación...
IoU por clase : [0.95713171 0.7939442  0.83018542 0.74689735 0.59239759 0.90733659]
Dice por clase: [0.97809637 0.88513812 0.90721455 0.85511304 0.74403226 0.95141738]
mIoU macro = 0.8046 | Dice macro = 0.8868

--- Epoch 123/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.106]
Calculando métricas de entrenamiento...
IoU por clase : [0.95995041 0.82425221 0.85265356 0.75863522 0.75016992 0.93120898]
Dice por clase: [0.97956602 0.90366036 0.92046736 0.86275449 0.85725382 0.9643793 ]
mIoU macro = 0.8461 | Dice macro = 0.9147
Calculando métricas de validación...
IoU por clase : [0.95773171 0.7942768  0.84095453 0.74137844 0.67732894 0.9207512 ]
Dice por clase: [0.97840956 0.88534478 0.91360706 0.85148458 0.80762804 0.95874073]
mIoU macro = 0.8221 | Dice macro = 0.8992
🔹 Nuevo mejor mIoU: 0.8221 | Dice: 0.8992  →  guardando modelo…

--- Epoch 124/200 ---
100% 210/210 [00:28<00:00,  7.37it/s, loss=0.0858]
Calculando métricas de entrenamiento...
IoU por clase : [0.95919319 0.74320609 0.86016353 0.76630292 0.75555296 0.90604671]
Dice por clase: [0.97917163 0.85268873 0.92482571 0.8676914  0.86075781 0.95070777]
mIoU macro = 0.8317 | Dice macro = 0.9060
Calculando métricas de validación...
IoU por clase : [0.95764973 0.76216626 0.84846139 0.74341692 0.63174371 0.89950892]
Dice por clase: [0.97836678 0.86503331 0.91801905 0.85282747 0.77431732 0.94709629]
mIoU macro = 0.8072 | Dice macro = 0.8893

--- Epoch 125/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.95945046 0.81293479 0.86065124 0.76228845 0.74452319 0.92524359]
Dice por clase: [0.97930566 0.89681636 0.92510753 0.86511201 0.85355494 0.96117042]
mIoU macro = 0.8442 | Dice macro = 0.9135
Calculando métricas de validación...
IoU por clase : [0.95676201 0.79173522 0.84453181 0.73396094 0.70384984 0.92082269]
Dice por clase: [0.97790329 0.88376364 0.91571401 0.84657148 0.82618764 0.95877948]
mIoU macro = 0.8253 | Dice macro = 0.9015
🔹 Nuevo mejor mIoU: 0.8253 | Dice: 0.9015  →  guardando modelo…

--- Epoch 126/200 ---
100% 210/210 [00:27<00:00,  7.50it/s, loss=0.145]
Calculando métricas de entrenamiento...
IoU por clase : [0.96025211 0.82027709 0.86015484 0.75352699 0.71407528 0.93113253]
Dice por clase: [0.97972307 0.90126618 0.92482069 0.85944156 0.8331901  0.9643383 ]
mIoU macro = 0.8399 | Dice macro = 0.9105
Calculando métricas de validación...
IoU por clase : [0.9577059  0.79654293 0.85001831 0.74821944 0.69765249 0.92356747]
Dice por clase: [0.97839609 0.88675079 0.91892962 0.85597886 0.82190259 0.96026522]
mIoU macro = 0.8290 | Dice macro = 0.9037
🔹 Nuevo mejor mIoU: 0.8290 | Dice: 0.9037  →  guardando modelo…

--- Epoch 127/200 ---
100% 210/210 [00:28<00:00,  7.47it/s, loss=0.0776]
Calculando métricas de entrenamiento...
IoU por clase : [0.95939352 0.8132327  0.86048372 0.77194024 0.74971369 0.9269172 ]
Dice por clase: [0.979276   0.89699761 0.92501075 0.87129377 0.85695585 0.96207268]
mIoU macro = 0.8469 | Dice macro = 0.9153
Calculando métricas de validación...
IoU por clase : [0.95787577 0.79967713 0.83555744 0.74294057 0.71443768 0.92363197]
Dice por clase: [0.97848473 0.88868955 0.91041274 0.85251394 0.83343675 0.96030008]
mIoU macro = 0.8290 | Dice macro = 0.9040
🔹 Nuevo mejor mIoU: 0.8290 | Dice: 0.9040  →  guardando modelo…

--- Epoch 128/200 ---
100% 210/210 [00:28<00:00,  7.38it/s, loss=0.1]
Calculando métricas de entrenamiento...
IoU por clase : [0.95901978 0.81687874 0.85972868 0.7779522  0.76520232 0.92913098]
Dice por clase: [0.97908126 0.89921107 0.92457431 0.87511037 0.8669854  0.96326376]
mIoU macro = 0.8513 | Dice macro = 0.9180
Calculando métricas de validación...
IoU por clase : [0.95684539 0.79948702 0.82531778 0.7341294  0.67770257 0.92476908]
Dice por clase: [0.97794685 0.88857215 0.90430038 0.84668353 0.80789358 0.96091431]
mIoU macro = 0.8197 | Dice macro = 0.8977

--- Epoch 129/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0933]
Calculando métricas de entrenamiento...
IoU por clase : [0.95964266 0.81960188 0.86229822 0.7687661  0.76474721 0.93122982]
Dice por clase: [0.97940577 0.90085847 0.92605815 0.86926825 0.8666932  0.96439048]
mIoU macro = 0.8510 | Dice macro = 0.9178
Calculando métricas de validación...
IoU por clase : [0.95739145 0.79808133 0.82612056 0.72268819 0.66873086 0.9221356 ]
Dice por clase: [0.97823197 0.88770326 0.90478206 0.83902379 0.80148438 0.95949068]
mIoU macro = 0.8159 | Dice macro = 0.8951

--- Epoch 130/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.213]
Calculando métricas de entrenamiento...
IoU por clase : [0.95952665 0.81908642 0.85479915 0.76822915 0.74522731 0.92832199]
Dice por clase: [0.97934535 0.90054701 0.92171614 0.86892488 0.85401748 0.96282882]
mIoU macro = 0.8459 | Dice macro = 0.9146
Calculando métricas de validación...
IoU por clase : [0.95662373 0.79433307 0.83573343 0.73127284 0.58119498 0.90187805]
Dice por clase: [0.97783106 0.88537974 0.9105172  0.84478058 0.73513386 0.94840787]
mIoU macro = 0.8002 | Dice macro = 0.8837

--- Epoch 131/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.0624]
Calculando métricas de entrenamiento...
IoU por clase : [0.95956212 0.81759774 0.85336393 0.77321836 0.75964034 0.92961007]
Dice por clase: [0.97936382 0.89964652 0.92088112 0.87210732 0.8634041  0.96352116]
mIoU macro = 0.8488 | Dice macro = 0.9165
Calculando métricas de validación...
IoU por clase : [0.95723255 0.80073314 0.83018205 0.73874053 0.62557344 0.91106359]
Dice por clase: [0.97814902 0.88934126 0.90721254 0.84974212 0.76966494 0.95346235]
mIoU macro = 0.8106 | Dice macro = 0.8913

--- Epoch 132/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.95901487 0.8208256  0.85900281 0.77099755 0.75374337 0.92638527]
Dice por clase: [0.9790787  0.90159717 0.9241544  0.87069296 0.85958229 0.96178608]
mIoU macro = 0.8483 | Dice macro = 0.9161
Calculando métricas de validación...
IoU por clase : [0.95659046 0.79943292 0.82383829 0.73386171 0.67865621 0.91956541]
Dice por clase: [0.97781368 0.88853873 0.90341155 0.84650547 0.80857081 0.9580975 ]
mIoU macro = 0.8187 | Dice macro = 0.8972

--- Epoch 133/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.106]
Calculando métricas de entrenamiento...
IoU por clase : [0.96041213 0.8097744  0.86145585 0.769083   0.76898879 0.92783071]
Dice por clase: [0.97980636 0.89488988 0.92557215 0.86947079 0.86941058 0.96256451]
mIoU macro = 0.8496 | Dice macro = 0.9170
Calculando métricas de validación...
IoU por clase : [0.95748397 0.77366675 0.83356568 0.72674911 0.67560994 0.91688984]
Dice por clase: [0.97828026 0.87239246 0.90922915 0.84175414 0.80640479 0.95664323]
mIoU macro = 0.8140 | Dice macro = 0.8941

--- Epoch 134/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0685]
Calculando métricas de entrenamiento...
IoU por clase : [0.9600543  0.82113804 0.86308366 0.77078726 0.76409184 0.93019607]
Dice por clase: [0.97962011 0.90178561 0.9265109  0.87055885 0.86627218 0.96383583]
mIoU macro = 0.8516 | Dice macro = 0.9181
Calculando métricas de validación...
IoU por clase : [0.95743677 0.8040428  0.84991574 0.72322383 0.7017225  0.92351765]
Dice por clase: [0.97825563 0.89137885 0.91886968 0.83938467 0.82472025 0.96023829]
mIoU macro = 0.8266 | Dice macro = 0.9021

--- Epoch 135/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0797]
Calculando métricas de entrenamiento...
IoU por clase : [0.96032968 0.81266498 0.86408547 0.77478125 0.764968   0.93128593]
Dice por clase: [0.97976345 0.89665216 0.92708782 0.87310056 0.86683498 0.96442056]
mIoU macro = 0.8514 | Dice macro = 0.9180
Calculando métricas de validación...
IoU por clase : [0.95825495 0.79768407 0.83431972 0.73213919 0.66394524 0.92258019]
Dice por clase: [0.97868252 0.88745746 0.90967753 0.84535838 0.79803737 0.9597313 ]
mIoU macro = 0.8182 | Dice macro = 0.8965

--- Epoch 136/200 ---
100% 210/210 [00:26<00:00,  7.89it/s, loss=0.137]
Calculando métricas de entrenamiento...
IoU por clase : [0.95930646 0.81932484 0.86336411 0.76708854 0.7599107  0.93130572]
Dice por clase: [0.97923064 0.9006911  0.92667247 0.8681948  0.8635787  0.96443117]
mIoU macro = 0.8501 | Dice macro = 0.9171
Calculando métricas de validación...
IoU por clase : [0.95750365 0.79807333 0.83997465 0.74254694 0.67921133 0.92458125]
Dice por clase: [0.97829054 0.88769831 0.9130285  0.85225474 0.80896468 0.96081291]
mIoU macro = 0.8236 | Dice macro = 0.9002

--- Epoch 137/200 ---
100% 210/210 [00:26<00:00,  7.90it/s, loss=0.0811]
Calculando métricas de entrenamiento...
IoU por clase : [0.95995333 0.81739159 0.86013229 0.76326235 0.76555183 0.92936534]
Dice por clase: [0.97956754 0.89952171 0.92480765 0.86573884 0.86720969 0.96338969]
mIoU macro = 0.8493 | Dice macro = 0.9167
Calculando métricas de validación...
IoU por clase : [0.95687807 0.79851797 0.83403581 0.72100644 0.64567242 0.91434584]
Dice por clase: [0.97796392 0.8879733  0.90950875 0.8378893  0.7846913  0.95525669]
mIoU macro = 0.8117 | Dice macro = 0.8922

--- Epoch 138/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.145]
Calculando métricas de entrenamiento...
IoU por clase : [0.95986142 0.82300365 0.86616391 0.77216956 0.77357715 0.92735293]
Dice por clase: [0.97951968 0.90290949 0.92828278 0.87143982 0.8723355  0.96230733]
mIoU macro = 0.8537 | Dice macro = 0.9195
Calculando métricas de validación...
IoU por clase : [0.95767385 0.80345441 0.8375852  0.73469989 0.69884914 0.91969487]
Dice por clase: [0.97837936 0.89101716 0.91161509 0.84706282 0.82273243 0.95816776]
mIoU macro = 0.8253 | Dice macro = 0.9015

--- Epoch 139/200 ---
100% 210/210 [00:26<00:00,  8.05it/s, loss=0.0904]
Calculando métricas de entrenamiento...
IoU por clase : [0.96087075 0.82831557 0.86871181 0.75563164 0.76718347 0.9317279 ]
Dice por clase: [0.98004496 0.90609694 0.92974401 0.86080887 0.8682556  0.9646575 ]
mIoU macro = 0.8521 | Dice macro = 0.9183
Calculando métricas de validación...
IoU por clase : [0.9570868  0.8016496  0.84405834 0.70625092 0.68633712 0.92322173]
Dice por clase: [0.97807292 0.88990623 0.91543561 0.82783946 0.81399752 0.96007831]
mIoU macro = 0.8198 | Dice macro = 0.8976

--- Epoch 140/200 ---
100% 210/210 [00:26<00:00,  7.97it/s, loss=0.102]
Calculando métricas de entrenamiento...
IoU por clase : [0.96121602 0.82085936 0.86842548 0.77903901 0.77595103 0.93319728]
Dice por clase: [0.98022453 0.90161753 0.92958    0.87579756 0.87384282 0.96544444]
mIoU macro = 0.8564 | Dice macro = 0.9211
Calculando métricas de validación...
IoU por clase : [0.95823234 0.80597321 0.85297637 0.74407373 0.70151878 0.9238929 ]
Dice por clase: [0.97867073 0.89256386 0.92065542 0.85325949 0.82457953 0.96044109]
mIoU macro = 0.8311 | Dice macro = 0.9050
🔹 Nuevo mejor mIoU: 0.8311 | Dice: 0.9050  →  guardando modelo…

--- Epoch 141/200 ---
100% 210/210 [00:28<00:00,  7.36it/s, loss=0.086]
Calculando métricas de entrenamiento...
IoU por clase : [0.96140984 0.8233688  0.8715596  0.78252538 0.78675356 0.93154309]
Dice por clase: [0.98032529 0.9031292  0.93137253 0.87799634 0.88065145 0.96455843]
mIoU macro = 0.8595 | Dice macro = 0.9230
Calculando métricas de validación...
IoU por clase : [0.95820909 0.80003001 0.84839965 0.74466544 0.6769694  0.92417316]
Dice por clase: [0.9786586  0.88890741 0.91798291 0.85364841 0.80737239 0.96059251]
mIoU macro = 0.8254 | Dice macro = 0.9012

--- Epoch 142/200 ---
100% 210/210 [00:26<00:00,  7.93it/s, loss=0.0972]
Calculando métricas de entrenamiento...
IoU por clase : [0.96147187 0.82958621 0.87333377 0.7810338  0.78508967 0.93162652]
Dice por clase: [0.98035754 0.90685665 0.93238459 0.87705669 0.8796081  0.96460316]
mIoU macro = 0.8604 | Dice macro = 0.9235
Calculando métricas de validación...
IoU por clase : [0.95818472 0.8019877  0.85044499 0.73492888 0.69555848 0.92368977]
Dice por clase: [0.9786459  0.89011451 0.9191789  0.84721499 0.82044765 0.96033132]
mIoU macro = 0.8275 | Dice macro = 0.9027

--- Epoch 143/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.0598]
Calculando métricas de entrenamiento...
IoU por clase : [0.9604208  0.83043373 0.86009875 0.75356394 0.76654827 0.93180982]
Dice por clase: [0.97981086 0.90736279 0.92478827 0.8594656  0.86784865 0.9647014 ]
mIoU macro = 0.8505 | Dice macro = 0.9173
Calculando métricas de validación...
IoU por clase : [0.9571826  0.79889828 0.83308055 0.72238866 0.69512537 0.92475188]
Dice por clase: [0.97812294 0.8882084  0.90894047 0.8388219  0.82014627 0.96090503]
mIoU macro = 0.8219 | Dice macro = 0.8992

--- Epoch 144/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.0952]
Calculando métricas de entrenamiento...
IoU por clase : [0.96139563 0.82689021 0.87133341 0.78243443 0.78392879 0.93047489]
Dice por clase: [0.98031791 0.90524346 0.93124336 0.87793909 0.87887902 0.96398549]
mIoU macro = 0.8594 | Dice macro = 0.9229
Calculando métricas de validación...
IoU por clase : [0.95844181 0.80110966 0.85543675 0.74176719 0.71068864 0.92296232]
Dice por clase: [0.97877997 0.88957344 0.92208668 0.85174092 0.83088018 0.95993802]
mIoU macro = 0.8317 | Dice macro = 0.9055
🔹 Nuevo mejor mIoU: 0.8317 | Dice: 0.9055  →  guardando modelo…

--- Epoch 145/200 ---
100% 210/210 [00:28<00:00,  7.25it/s, loss=0.128]
Calculando métricas de entrenamiento...
IoU por clase : [0.96071846 0.81727534 0.86745253 0.77599119 0.76904422 0.93030333]
Dice por clase: [0.97996574 0.89945131 0.92902231 0.87386829 0.86944601 0.96389341]
mIoU macro = 0.8535 | Dice macro = 0.9193
Calculando métricas de validación...
IoU por clase : [0.95737037 0.78723288 0.84042038 0.73059027 0.67523764 0.92120276]
Dice por clase: [0.97822097 0.88095165 0.91329176 0.84432495 0.80613953 0.95898546]
mIoU macro = 0.8187 | Dice macro = 0.8970

--- Epoch 146/200 ---
100% 210/210 [00:27<00:00,  7.74it/s, loss=0.0825]
Calculando métricas de entrenamiento...
IoU por clase : [0.96006895 0.81956135 0.86176454 0.76716036 0.76771786 0.93191673]
Dice por clase: [0.97962773 0.90083398 0.9257503  0.8682408  0.86859773 0.96475869]
mIoU macro = 0.8514 | Dice macro = 0.9180
Calculando métricas de validación...
IoU por clase : [0.95784334 0.80015761 0.83868524 0.74984555 0.65060431 0.91914388]
Dice por clase: [0.97846781 0.88898617 0.91226624 0.85704199 0.78832256 0.95786865]
mIoU macro = 0.8194 | Dice macro = 0.8972

--- Epoch 147/200 ---
100% 210/210 [00:27<00:00,  7.71it/s, loss=0.0791]
Calculando métricas de entrenamiento...
IoU por clase : [0.96103816 0.82412633 0.86732295 0.78320599 0.76758289 0.92771166]
Dice por clase: [0.98013203 0.90358471 0.92894799 0.87842459 0.86851134 0.96250044]
mIoU macro = 0.8552 | Dice macro = 0.9204
Calculando métricas de validación...
IoU por clase : [0.95819427 0.79636673 0.85329936 0.7441018  0.6749103  0.91696377]
Dice por clase: [0.97865087 0.88664159 0.92084353 0.85327795 0.8059062  0.95668347]
mIoU macro = 0.8240 | Dice macro = 0.9003

--- Epoch 148/200 ---
100% 210/210 [00:27<00:00,  7.65it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.96067508 0.81569857 0.86925416 0.76700416 0.77848165 0.92936575]
Dice por clase: [0.97994317 0.89849558 0.93005454 0.86814075 0.87544524 0.96338991]
mIoU macro = 0.8534 | Dice macro = 0.9192
Calculando métricas de validación...
IoU por clase : [0.95770012 0.76378413 0.82154695 0.72823935 0.65649512 0.91709332]
Dice por clase: [0.97839307 0.86607439 0.90203214 0.84275288 0.79263151 0.95675397]
mIoU macro = 0.8075 | Dice macro = 0.8898

--- Epoch 149/200 ---
100% 210/210 [00:27<00:00,  7.62it/s, loss=0.109]
Calculando métricas de entrenamiento...
IoU por clase : [0.95997533 0.82286731 0.84998787 0.76498058 0.74721219 0.93043114]
Dice por clase: [0.97957899 0.90282744 0.91891183 0.86684306 0.85531934 0.96396201]
mIoU macro = 0.8459 | Dice macro = 0.9146
Calculando métricas de validación...
IoU por clase : [0.95763805 0.80629896 0.84567737 0.74360171 0.67240792 0.92072884]
Dice por clase: [0.97836068 0.89276358 0.916387   0.85294905 0.80411951 0.9587286 ]
mIoU macro = 0.8244 | Dice macro = 0.9006

--- Epoch 150/200 ---
100% 210/210 [00:27<00:00,  7.63it/s, loss=0.0694]
Calculando métricas de entrenamiento...
IoU por clase : [0.95971903 0.8273973  0.86467715 0.73307517 0.75258367 0.93136065]
Dice por clase: [0.97944554 0.90554725 0.92742827 0.84598197 0.85882766 0.96446063]
mIoU macro = 0.8448 | Dice macro = 0.9136
Calculando métricas de validación...
IoU por clase : [0.95534255 0.7925535  0.8468374  0.68959849 0.65872882 0.91490165]
Dice por clase: [0.97716132 0.88427319 0.91706763 0.81628682 0.7942574  0.95555994]
mIoU macro = 0.8097 | Dice macro = 0.8908

--- Epoch 151/200 ---
100% 210/210 [00:27<00:00,  7.71it/s, loss=0.113]
Calculando métricas de entrenamiento...
IoU por clase : [0.96165091 0.8283492  0.8706899  0.769304   0.78554716 0.93288893]
Dice por clase: [0.9804506  0.90611706 0.93087571 0.86961201 0.87989517 0.9652794 ]
mIoU macro = 0.8581 | Dice macro = 0.9220
Calculando métricas de validación...
IoU por clase : [0.95817356 0.79980471 0.84798345 0.73742393 0.67123555 0.92216319]
Dice por clase: [0.97864007 0.88876833 0.91773923 0.84887047 0.8032806  0.95950562]
mIoU macro = 0.8228 | Dice macro = 0.8995

--- Epoch 152/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.0798]
Calculando métricas de entrenamiento...
IoU por clase : [0.96124074 0.82690344 0.87079202 0.76499299 0.7647972  0.93232009]
Dice por clase: [0.98023738 0.90525139 0.93093408 0.86685102 0.86672531 0.96497479]
mIoU macro = 0.8535 | Dice macro = 0.9192
Calculando métricas de validación...
IoU por clase : [0.95839469 0.80246224 0.85726301 0.73501764 0.6683448  0.91971169]
Dice por clase: [0.9787554  0.89040671 0.9231466  0.84727396 0.80120704 0.95817689]
mIoU macro = 0.8235 | Dice macro = 0.8998

--- Epoch 153/200 ---
100% 210/210 [00:27<00:00,  7.78it/s, loss=0.075]
Calculando métricas de entrenamiento...
IoU por clase : [0.96077782 0.8204265  0.86621806 0.7796545  0.78546103 0.93114133]
Dice por clase: [0.97999662 0.90135636 0.92831388 0.87618636 0.87984113 0.96434302]
mIoU macro = 0.8573 | Dice macro = 0.9217
Calculando métricas de validación...
IoU por clase : [0.9584208  0.79771729 0.85086074 0.7452791  0.67535224 0.9163228 ]
Dice por clase: [0.97876902 0.88747802 0.91942167 0.85405148 0.80622119 0.9563345 ]
mIoU macro = 0.8240 | Dice macro = 0.9004

--- Epoch 154/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.153]
Calculando métricas de entrenamiento...
IoU por clase : [0.96129458 0.82265288 0.8657464  0.77521919 0.78706075 0.93181621]
Dice por clase: [0.98026537 0.90269836 0.92804295 0.87337856 0.88084387 0.96470483]
mIoU macro = 0.8573 | Dice macro = 0.9217
Calculando métricas de validación...
IoU por clase : [0.95810295 0.79609089 0.849296   0.74329875 0.7009814  0.92073375]
Dice por clase: [0.97860325 0.8864706  0.91850737 0.85274971 0.82420819 0.95873127]
mIoU macro = 0.8281 | Dice macro = 0.9032

--- Epoch 155/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.0684]
Calculando métricas de entrenamiento...
IoU por clase : [0.96072652 0.79858219 0.86891904 0.77636899 0.77725064 0.93161056]
Dice por clase: [0.97996993 0.88801301 0.92986269 0.8741078  0.87466632 0.9645946 ]
mIoU macro = 0.8522 | Dice macro = 0.9185
Calculando métricas de validación...
IoU por clase : [0.95808681 0.78846792 0.85659413 0.73778929 0.67911186 0.92024858]
Dice por clase: [0.97859482 0.88172442 0.92275863 0.84911248 0.80889413 0.95846818]
mIoU macro = 0.8234 | Dice macro = 0.8999

--- Epoch 156/200 ---
100% 210/210 [00:27<00:00,  7.72it/s, loss=0.092]
Calculando métricas de entrenamiento...
IoU por clase : [0.9617546  0.82994474 0.86925348 0.77079078 0.78626966 0.93025445]
Dice por clase: [0.98050449 0.90707082 0.93005415 0.87056109 0.88034822 0.96386717]
mIoU macro = 0.8580 | Dice macro = 0.9221
Calculando métricas de validación...
IoU por clase : [0.95866368 0.80549026 0.85257943 0.73646392 0.69005904 0.91887549]
Dice por clase: [0.97889565 0.89226763 0.92042416 0.84823406 0.81660939 0.95772289]
mIoU macro = 0.8270 | Dice macro = 0.9024

--- Epoch 157/200 ---
100% 210/210 [00:27<00:00,  7.66it/s, loss=0.0698]
Calculando métricas de entrenamiento...
IoU por clase : [0.96230569 0.82954857 0.8780347  0.78415048 0.79015935 0.93313255]
Dice por clase: [0.98079081 0.90683416 0.93505695 0.87901832 0.88278102 0.9654098 ]
mIoU macro = 0.8629 | Dice macro = 0.9250
Calculando métricas de validación...
IoU por clase : [0.95888628 0.80598332 0.84916933 0.74761621 0.65515544 0.91787335]
Dice por clase: [0.97901168 0.89257006 0.91843328 0.85558397 0.79165427 0.95717827]
mIoU macro = 0.8224 | Dice macro = 0.8991

--- Epoch 158/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.104]
Calculando métricas de entrenamiento...
IoU por clase : [0.95978291 0.82172229 0.86891384 0.77520798 0.76088217 0.92608715]
Dice por clase: [0.9794788  0.90213782 0.92985971 0.87337145 0.86420566 0.96162539]
mIoU macro = 0.8521 | Dice macro = 0.9184
Calculando métricas de validación...
IoU por clase : [0.95728363 0.80013418 0.84621037 0.74844356 0.64877558 0.91545315]
Dice por clase: [0.97817568 0.88897171 0.91669983 0.8561255  0.78697864 0.95586065]
mIoU macro = 0.8194 | Dice macro = 0.8971

--- Epoch 159/200 ---
100% 210/210 [00:28<00:00,  7.49it/s, loss=0.13]
Calculando métricas de entrenamiento...
IoU por clase : [0.96145739 0.81994922 0.84408066 0.79520218 0.68815847 0.92853919]
Dice por clase: [0.98035001 0.90106824 0.91544874 0.88591936 0.8152771  0.96294563]
mIoU macro = 0.8396 | Dice macro = 0.9102
Calculando métricas de validación...
IoU por clase : [0.95835078 0.80320081 0.85210358 0.7436863  0.6462963  0.91472968]
Dice por clase: [0.9787325  0.89086119 0.92014679 0.8530047  0.78515186 0.95546613]
mIoU macro = 0.8197 | Dice macro = 0.8972

--- Epoch 160/200 ---
100% 210/210 [00:27<00:00,  7.63it/s, loss=0.084]
Calculando métricas de entrenamiento...
IoU por clase : [0.96149594 0.81425036 0.87137846 0.78799574 0.78148943 0.92891649]
Dice por clase: [0.98037005 0.89761631 0.93126909 0.8814291  0.87734388 0.96314848]
mIoU macro = 0.8576 | Dice macro = 0.9219
Calculando métricas de validación...
IoU por clase : [0.95839542 0.79696564 0.8542895  0.74196281 0.65095172 0.91297366]
Dice por clase: [0.97875578 0.88701266 0.92141976 0.85186986 0.78857754 0.9545073 ]
mIoU macro = 0.8193 | Dice macro = 0.8970

--- Epoch 161/200 ---
100% 210/210 [00:27<00:00,  7.66it/s, loss=0.0744]
Calculando métricas de entrenamiento...
IoU por clase : [0.9617065  0.82914023 0.86987638 0.78057591 0.79058638 0.9334739 ]
Dice por clase: [0.9804795  0.90659012 0.93041057 0.87676791 0.88304746 0.96559245]
mIoU macro = 0.8609 | Dice macro = 0.9238
Calculando métricas de validación...
IoU por clase : [0.95896011 0.80642544 0.85113065 0.74528799 0.66751474 0.91717324]
Dice por clase: [0.97905016 0.8928411  0.91957923 0.85405732 0.8006103  0.95679745]
mIoU macro = 0.8244 | Dice macro = 0.9005

--- Epoch 162/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.0678]
Calculando métricas de entrenamiento...
IoU por clase : [0.96170998 0.83011162 0.87158474 0.78675759 0.7844334  0.93210373]
Dice por clase: [0.9804813  0.90717048 0.93138689 0.88065398 0.87919605 0.96485889]
mIoU macro = 0.8611 | Dice macro = 0.9240
Calculando métricas de validación...
IoU por clase : [0.95845901 0.80712655 0.8457765  0.7527928  0.67106659 0.91588955]
Dice por clase: [0.97878894 0.89327065 0.91644519 0.85896382 0.8031596  0.95609849]
mIoU macro = 0.8252 | Dice macro = 0.9011

--- Epoch 163/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.0967]
Calculando métricas de entrenamiento...
IoU por clase : [0.9620344  0.82397216 0.87197612 0.78680804 0.78172068 0.9323836 ]
Dice por clase: [0.98064988 0.90349204 0.9316103  0.88068558 0.87748959 0.96500881]
mIoU macro = 0.8598 | Dice macro = 0.9232
Calculando métricas de validación...
IoU por clase : [0.95880477 0.80664483 0.86147922 0.74762865 0.62411442 0.90540769]
Dice por clase: [0.9789692  0.89297555 0.92558564 0.85559212 0.76855967 0.95035587]
mIoU macro = 0.8173 | Dice macro = 0.8953

--- Epoch 164/200 ---
100% 210/210 [00:27<00:00,  7.69it/s, loss=0.0628]
Calculando métricas de entrenamiento...
IoU por clase : [0.96143138 0.81546043 0.87578563 0.76323134 0.79680254 0.93077324]
Dice por clase: [0.98033649 0.89835109 0.93378008 0.86571889 0.88691164 0.96414558]
mIoU macro = 0.8572 | Dice macro = 0.9215
Calculando métricas de validación...
IoU por clase : [0.95781929 0.78084802 0.85454805 0.72494051 0.68551091 0.9103992 ]
Dice por clase: [0.97845526 0.87693954 0.92157014 0.84053973 0.81341617 0.95309839]
mIoU macro = 0.8190 | Dice macro = 0.8973

--- Epoch 165/200 ---
100% 210/210 [00:26<00:00,  7.87it/s, loss=0.0786]
Calculando métricas de entrenamiento...
IoU por clase : [0.96227839 0.83478108 0.86809176 0.78549503 0.78984409 0.93252241]
Dice por clase: [0.98077663 0.9099517  0.92938878 0.87986247 0.88258424 0.96508315]
mIoU macro = 0.8622 | Dice macro = 0.9246
Calculando métricas de validación...
IoU por clase : [0.95866615 0.80389153 0.84899781 0.74700943 0.64932364 0.91568252]
Dice por clase: [0.97889694 0.89128589 0.91833296 0.85518649 0.78738172 0.95598567]
mIoU macro = 0.8206 | Dice macro = 0.8978

--- Epoch 166/200 ---
100% 210/210 [00:26<00:00,  7.98it/s, loss=0.0708]
Calculando métricas de entrenamiento...
IoU por clase : [0.96204554 0.82717005 0.87387107 0.78579777 0.78820052 0.93347664]
Dice por clase: [0.98065567 0.90541113 0.93269071 0.88005236 0.8815572  0.96559392]
mIoU macro = 0.8618 | Dice macro = 0.9243
Calculando métricas de validación...
IoU por clase : [0.95870407 0.80238894 0.84459683 0.75577062 0.64564068 0.91628029]
Dice por clase: [0.97891671 0.89036159 0.91575223 0.86089904 0.78466786 0.95631134]
mIoU macro = 0.8206 | Dice macro = 0.8978

--- Epoch 167/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.0863]
Calculando métricas de entrenamiento...
IoU por clase : [0.9619275  0.83466811 0.87373582 0.78245752 0.78282338 0.93034594]
Dice por clase: [0.98059434 0.90988458 0.93261367 0.87795362 0.87818388 0.96391628]
mIoU macro = 0.8610 | Dice macro = 0.9239
Calculando métricas de validación...
IoU por clase : [0.95843219 0.80022233 0.84651429 0.745042   0.65446436 0.91458881]
Dice por clase: [0.97877496 0.88902611 0.91687814 0.85389578 0.79114954 0.95538928]
mIoU macro = 0.8199 | Dice macro = 0.8975

--- Epoch 168/200 ---
100% 210/210 [00:26<00:00,  7.82it/s, loss=0.0819]
Calculando métricas de entrenamiento...
IoU por clase : [0.96198491 0.82071215 0.8707186  0.78706705 0.77553859 0.93267531]
Dice por clase: [0.98062417 0.90152872 0.93089212 0.88084781 0.87358123 0.96516503]
mIoU macro = 0.8581 | Dice macro = 0.9221
Calculando métricas de validación...
IoU por clase : [0.95884217 0.80287465 0.83296972 0.73982681 0.62832492 0.9184764 ]
Dice por clase: [0.9789887  0.89066053 0.9088745  0.85046029 0.77174391 0.95750607]
mIoU macro = 0.8136 | Dice macro = 0.8930

--- Epoch 169/200 ---
100% 210/210 [00:27<00:00,  7.72it/s, loss=0.0634]
Calculando métricas de entrenamiento...
IoU por clase : [0.96190703 0.82741225 0.86697387 0.78167485 0.80023832 0.93249744]
Dice por clase: [0.9805837  0.9055562  0.92874773 0.87746072 0.88903598 0.96506978]
mIoU macro = 0.8618 | Dice macro = 0.9244
Calculando métricas de validación...
IoU por clase : [0.95907399 0.80267904 0.84567556 0.73827478 0.65780904 0.91807964]
Dice por clase: [0.97910951 0.89054016 0.91638593 0.84943392 0.79358844 0.95729043]
mIoU macro = 0.8203 | Dice macro = 0.8977

--- Epoch 170/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.0886]
Calculando métricas de entrenamiento...
IoU por clase : [0.96260639 0.83013929 0.87290623 0.7979324  0.7895462  0.93492254]
Dice por clase: [0.98094696 0.90718701 0.93214088 0.88761112 0.88239823 0.96636689]
mIoU macro = 0.8647 | Dice macro = 0.9261
Calculando métricas de validación...
IoU por clase : [0.9591826  0.80448461 0.85868367 0.75337487 0.67530992 0.91861553]
Dice por clase: [0.97916611 0.89165029 0.92396967 0.85934261 0.80619103 0.95758167]
mIoU macro = 0.8283 | Dice macro = 0.9030

--- Epoch 171/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0766]
Calculando métricas de entrenamiento...
IoU por clase : [0.96252126 0.83666465 0.87520415 0.77870246 0.78596296 0.93401945]
Dice por clase: [0.98090276 0.91106958 0.93344946 0.87558485 0.88015595 0.96588424]
mIoU macro = 0.8622 | Dice macro = 0.9245
Calculando métricas de validación...
IoU por clase : [0.95910193 0.80362379 0.85839398 0.74039018 0.65499925 0.91749853]
Dice por clase: [0.97912407 0.8911213  0.92380194 0.8508324  0.79154024 0.95697443]
mIoU macro = 0.8223 | Dice macro = 0.8989

--- Epoch 172/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0797]
Calculando métricas de entrenamiento...
IoU por clase : [0.9617597  0.82432061 0.87671645 0.78561027 0.78629349 0.93319792]
Dice por clase: [0.98050715 0.90370147 0.93430891 0.87993476 0.88036316 0.96544478]
mIoU macro = 0.8613 | Dice macro = 0.9240
Calculando métricas de validación...
IoU por clase : [0.95889091 0.80209799 0.86620938 0.75101241 0.66671307 0.91519148]
Dice por clase: [0.9790141  0.89018244 0.92830889 0.85780364 0.80003341 0.95571799]
mIoU macro = 0.8267 | Dice macro = 0.9018

--- Epoch 173/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.285]
Calculando métricas de entrenamiento...
IoU por clase : [0.96296416 0.83275282 0.87441625 0.78556885 0.78437185 0.93384145]
Dice por clase: [0.9811327  0.90874537 0.93300114 0.87990877 0.87915739 0.96578905]
mIoU macro = 0.8623 | Dice macro = 0.9246
Calculando métricas de validación...
IoU por clase : [0.95930435 0.80232841 0.86521778 0.74533643 0.68494805 0.91938506]
Dice por clase: [0.97922954 0.89032432 0.92773915 0.85408912 0.81301978 0.9579996 ]
mIoU macro = 0.8294 | Dice macro = 0.9037

--- Epoch 174/200 ---
100% 210/210 [00:27<00:00,  7.71it/s, loss=0.0629]
Calculando métricas de entrenamiento...
IoU por clase : [0.96291667 0.82575578 0.87472322 0.78667909 0.78397477 0.93340868]
Dice por clase: [0.98110805 0.90456324 0.93317585 0.8806048  0.87890791 0.96555755]
mIoU macro = 0.8612 | Dice macro = 0.9240
Calculando métricas de validación...
IoU por clase : [0.95889588 0.80019056 0.86546537 0.74926628 0.67634788 0.91940009]
Dice por clase: [0.97901669 0.88900651 0.92788146 0.85666349 0.80693022 0.95800776]
mIoU macro = 0.8283 | Dice macro = 0.9029

--- Epoch 175/200 ---
100% 210/210 [00:27<00:00,  7.64it/s, loss=0.0664]
Calculando métricas de entrenamiento...
IoU por clase : [0.96292217 0.82587111 0.87876727 0.79665656 0.79604456 0.93283524]
Dice por clase: [0.9811109  0.90463243 0.93547219 0.8868212  0.88644188 0.96525066]
mIoU macro = 0.8655 | Dice macro = 0.9266
Calculando métricas de validación...
IoU por clase : [0.95925389 0.80189651 0.85948496 0.75528731 0.67243909 0.91976253]
Dice por clase: [0.97920325 0.89005834 0.92443336 0.86058539 0.8041418  0.95820448]
mIoU macro = 0.8280 | Dice macro = 0.9028

--- Epoch 176/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.107]
Calculando métricas de entrenamiento...
IoU por clase : [0.96325876 0.81032597 0.88153573 0.79703626 0.79802482 0.92954222]
Dice por clase: [0.98128558 0.89522659 0.93703852 0.8870564  0.88766831 0.96348472]
mIoU macro = 0.8633 | Dice macro = 0.9253
Calculando métricas de validación...
IoU por clase : [0.95955634 0.80230287 0.85992269 0.75207203 0.69368123 0.92145638]
Dice por clase: [0.97936081 0.89030859 0.92468649 0.85849442 0.81914025 0.95912287]
mIoU macro = 0.8315 | Dice macro = 0.9052

--- Epoch 177/200 ---
100% 210/210 [00:26<00:00,  7.79it/s, loss=0.0689]
Calculando métricas de entrenamiento...
IoU por clase : [0.96312509 0.83582938 0.88094707 0.78570816 0.78982749 0.93272421]
Dice por clase: [0.98121622 0.91057414 0.93670586 0.87999615 0.88257387 0.96519121]
mIoU macro = 0.8647 | Dice macro = 0.9260
Calculando métricas de validación...
IoU por clase : [0.95909154 0.80740536 0.84825078 0.74514003 0.65503732 0.92006533]
Dice por clase: [0.97911866 0.89344137 0.91789576 0.85396016 0.79156803 0.95836878]
mIoU macro = 0.8225 | Dice macro = 0.8991

--- Epoch 178/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.0897]
Calculando métricas de entrenamiento...
IoU por clase : [0.96263581 0.83757369 0.87946688 0.79315401 0.80026403 0.93316675]
Dice por clase: [0.98096224 0.91160827 0.93586845 0.88464683 0.88905185 0.9654281 ]
mIoU macro = 0.8677 | Dice macro = 0.9279
Calculando métricas de validación...
IoU por clase : [0.95852025 0.80501068 0.85444451 0.74915169 0.6980787  0.92232711]
Dice por clase: [0.97882087 0.89197331 0.92150992 0.85658859 0.82219829 0.95959434]
mIoU macro = 0.8313 | Dice macro = 0.9051

--- Epoch 179/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0953]
Calculando métricas de entrenamiento...
IoU por clase : [0.96037668 0.82790291 0.84911862 0.74493005 0.75098995 0.93231015]
Dice por clase: [0.9797879  0.90584998 0.91840362 0.85382225 0.85778899 0.96496947]
mIoU macro = 0.8443 | Dice macro = 0.9134
Calculando métricas de validación...
IoU por clase : [0.95766882 0.79959168 0.84624952 0.72035026 0.68851803 0.92399617]
Dice por clase: [0.97837674 0.88863678 0.91672281 0.83744604 0.81552938 0.96049689]
mIoU macro = 0.8227 | Dice macro = 0.8995

--- Epoch 180/200 ---
100% 210/210 [00:27<00:00,  7.78it/s, loss=0.0565]
Calculando métricas de entrenamiento...
IoU por clase : [0.9630049  0.81642934 0.8758011  0.78814361 0.80718607 0.93046161]
Dice por clase: [0.98115384 0.89893873 0.93378888 0.8815216  0.8933071  0.96397836]
mIoU macro = 0.8635 | Dice macro = 0.9254
Calculando métricas de validación...
IoU por clase : [0.95890382 0.79759106 0.84682027 0.74128369 0.70404731 0.92322183]
Dice por clase: [0.97902083 0.88739989 0.91705759 0.85142208 0.82632366 0.96007836]
mIoU macro = 0.8286 | Dice macro = 0.9036

--- Epoch 181/200 ---
100% 210/210 [00:26<00:00,  7.84it/s, loss=0.0668]
Calculando métricas de entrenamiento...
IoU por clase : [0.96228442 0.83511572 0.87237713 0.76973086 0.8005153  0.93506652]
Dice por clase: [0.98077976 0.91015047 0.93183912 0.86988465 0.88920689 0.9664438 ]
mIoU macro = 0.8625 | Dice macro = 0.9247
Calculando métricas de validación...
IoU por clase : [0.95826995 0.79021076 0.85157369 0.73912659 0.70777112 0.91941689]
Dice por clase: [0.97869035 0.88281311 0.91983775 0.84999745 0.82888287 0.95801688]
mIoU macro = 0.8277 | Dice macro = 0.9030

--- Epoch 182/200 ---
100% 210/210 [00:27<00:00,  7.70it/s, loss=0.0742]
Calculando métricas de entrenamiento...
IoU por clase : [0.96301169 0.82789753 0.87575235 0.78843786 0.79281494 0.93042637]
Dice por clase: [0.98115737 0.90584676 0.93376116 0.88170562 0.88443589 0.96395945]
mIoU macro = 0.8631 | Dice macro = 0.9251
Calculando métricas de validación...
IoU por clase : [0.95916287 0.8071753  0.84649598 0.73692243 0.68986431 0.92061899]
Dice por clase: [0.97915583 0.8933005  0.91686739 0.8485381  0.81647302 0.95866905]
mIoU macro = 0.8267 | Dice macro = 0.9022

--- Epoch 183/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0961]
Calculando métricas de entrenamiento...
IoU por clase : [0.96297779 0.83691323 0.881805   0.78844733 0.79700217 0.93355154]
Dice por clase: [0.98113977 0.91121694 0.93719062 0.88171154 0.88703529 0.96563398]
mIoU macro = 0.8668 | Dice macro = 0.9273
Calculando métricas de validación...
IoU por clase : [0.9592237  0.80495338 0.84203562 0.74251857 0.67406568 0.92132537]
Dice por clase: [0.97918752 0.89193814 0.91424466 0.85223605 0.80530374 0.9590519 ]
mIoU macro = 0.8240 | Dice macro = 0.9003

--- Epoch 184/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.071]
Calculando métricas de entrenamiento...
IoU por clase : [0.96350036 0.84092594 0.87628693 0.78870318 0.79173863 0.93517985]
Dice por clase: [0.98141093 0.91359019 0.93406495 0.8818715  0.88376576 0.96650433]
mIoU macro = 0.8661 | Dice macro = 0.9269
Calculando métricas de validación...
IoU por clase : [0.95877114 0.80498495 0.84109431 0.72384808 0.67236968 0.92237305]
Dice por clase: [0.97895167 0.89195752 0.91368954 0.83980496 0.80409216 0.95961921]
mIoU macro = 0.8206 | Dice macro = 0.8980

--- Epoch 185/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.069]
Calculando métricas de entrenamiento...
IoU por clase : [0.96267626 0.84031203 0.88281874 0.77756771 0.79149177 0.93499105]
Dice por clase: [0.98098324 0.91322777 0.93776286 0.87486705 0.88361195 0.96640349]
mIoU macro = 0.8650 | Dice macro = 0.9261
Calculando métricas de validación...
IoU por clase : [0.9590298  0.80821892 0.84437559 0.73310986 0.65840397 0.92433265]
Dice por clase: [0.97908649 0.89393923 0.91562217 0.84600507 0.79402121 0.96067865]
mIoU macro = 0.8212 | Dice macro = 0.8982

--- Epoch 186/200 ---
100% 210/210 [00:26<00:00,  7.86it/s, loss=0.0935]
Calculando métricas de entrenamiento...
IoU por clase : [0.96251412 0.83261709 0.87483526 0.78083695 0.78619996 0.93437148]
Dice por clase: [0.98089905 0.90866455 0.93323961 0.87693256 0.88030453 0.96607243]
mIoU macro = 0.8619 | Dice macro = 0.9244
Calculando métricas de validación...
IoU por clase : [0.95881818 0.8013388  0.84939931 0.74833184 0.64040996 0.91400128]
Dice por clase: [0.97897619 0.8897147  0.91856778 0.85605241 0.78079258 0.95506862]
mIoU macro = 0.8187 | Dice macro = 0.8965

--- Epoch 187/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.062]
Calculando métricas de entrenamiento...
IoU por clase : [0.96350621 0.82830243 0.88162517 0.79144248 0.80614204 0.93387375]
Dice por clase: [0.98141397 0.90608908 0.93708905 0.88358124 0.89266738 0.96580632]
mIoU macro = 0.8675 | Dice macro = 0.9278
Calculando métricas de validación...
IoU por clase : [0.95920313 0.80047547 0.85910926 0.74583774 0.66760261 0.9168276 ]
Dice por clase: [0.9791768  0.88918231 0.924216   0.85441817 0.8006735  0.95660935]
mIoU macro = 0.8248 | Dice macro = 0.9007

--- Epoch 188/200 ---
100% 210/210 [00:26<00:00,  7.88it/s, loss=0.1]
Calculando métricas de entrenamiento...
IoU por clase : [0.96311855 0.83268377 0.87841557 0.79249664 0.78694847 0.93485862]
Dice por clase: [0.98121283 0.90870425 0.93527288 0.88423779 0.88077354 0.96633275]
mIoU macro = 0.8648 | Dice macro = 0.9261
Calculando métricas de validación...
IoU por clase : [0.95944536 0.80732177 0.86410086 0.74371232 0.67541107 0.92368979]
Dice por clase: [0.979303   0.89339019 0.92709668 0.85302181 0.80626311 0.96033133]
mIoU macro = 0.8289 | Dice macro = 0.9032

--- Epoch 189/200 ---
100% 210/210 [00:26<00:00,  7.81it/s, loss=0.104]
Calculando métricas de entrenamiento...
IoU por clase : [0.96324679 0.8323022  0.87631689 0.78842839 0.79672587 0.93436791]
Dice por clase: [0.98127937 0.908477   0.93408197 0.8816997  0.88686414 0.96607052]
mIoU macro = 0.8652 | Dice macro = 0.9264
Calculando métricas de validación...
IoU por clase : [0.95929466 0.79992898 0.85349692 0.7448721  0.68108587 0.92144075]
Dice por clase: [0.97922449 0.88884505 0.92095855 0.85378418 0.81029278 0.9591144 ]
mIoU macro = 0.8267 | Dice macro = 0.9020

--- Epoch 190/200 ---
100% 210/210 [00:27<00:00,  7.74it/s, loss=0.0787]
Calculando métricas de entrenamiento...
IoU por clase : [0.96410615 0.83776614 0.88104855 0.79456803 0.80286024 0.93710637]
Dice por clase: [0.9817251  0.91172225 0.93676322 0.88552567 0.89065167 0.96753217]
mIoU macro = 0.8696 | Dice macro = 0.9290
Calculando métricas de validación...
IoU por clase : [0.95967933 0.80468335 0.85852331 0.74914769 0.68377349 0.9257529 ]
Dice por clase: [0.97942486 0.89177235 0.92387683 0.85658598 0.81219178 0.96144516]
mIoU macro = 0.8303 | Dice macro = 0.9042

--- Epoch 191/200 ---
100% 210/210 [00:26<00:00,  7.80it/s, loss=0.108]
Calculando métricas de entrenamiento...
IoU por clase : [0.96270615 0.83026197 0.8713799  0.78582147 0.78936089 0.93160579]
Dice por clase: [0.98099876 0.90726025 0.93126991 0.88006722 0.88228249 0.96459205]
mIoU macro = 0.8619 | Dice macro = 0.9244
Calculando métricas de validación...
IoU por clase : [0.959648   0.80283895 0.83783534 0.74435204 0.65996873 0.91989281]
Dice por clase: [0.97940855 0.89063857 0.91176323 0.85344245 0.79515803 0.95827518]
mIoU macro = 0.8208 | Dice macro = 0.8981

--- Epoch 192/200 ---
100% 210/210 [00:26<00:00,  7.78it/s, loss=0.0883]
Calculando métricas de entrenamiento...
IoU por clase : [0.96323385 0.83987937 0.87437567 0.79059598 0.80016907 0.93439476]
Dice por clase: [0.98127266 0.91297222 0.93297804 0.88305345 0.88899325 0.96608487]
mIoU macro = 0.8671 | Dice macro = 0.9276
Calculando métricas de validación...
IoU por clase : [0.95978543 0.80986631 0.8470027  0.74701215 0.683106   0.92240877]
Dice por clase: [0.97948012 0.89494601 0.91716455 0.85518827 0.81172071 0.95963854]
mIoU macro = 0.8282 | Dice macro = 0.9030

--- Epoch 193/200 ---
100% 210/210 [00:27<00:00,  7.73it/s, loss=0.073]
Calculando métricas de entrenamiento...
IoU por clase : [0.96335391 0.83323515 0.8815845  0.79720322 0.8028678  0.93457871]
Dice por clase: [0.98133495 0.90903248 0.93706607 0.8871598  0.89065632 0.96618319]
mIoU macro = 0.8688 | Dice macro = 0.9286
Calculando métricas de validación...
IoU por clase : [0.95941745 0.80356502 0.85842581 0.75111861 0.66597034 0.92006448]
Dice por clase: [0.97928846 0.89108517 0.92382037 0.85787291 0.79949844 0.95836832]
mIoU macro = 0.8264 | Dice macro = 0.9017

--- Epoch 194/200 ---
100% 210/210 [00:27<00:00,  7.77it/s, loss=0.0648]
Calculando métricas de entrenamiento...
IoU por clase : [0.96264314 0.83191307 0.8823833  0.79244371 0.79190484 0.93175933]
Dice por clase: [0.98096604 0.90824514 0.93751714 0.88420485 0.8838693  0.96467434]
mIoU macro = 0.8655 | Dice macro = 0.9266
Calculando métricas de validación...
IoU por clase : [0.958419   0.80721043 0.84949505 0.73810325 0.63069531 0.91412913]
Dice por clase: [0.97876808 0.89332201 0.91862376 0.84932037 0.77352931 0.95513841]
mIoU macro = 0.8163 | Dice macro = 0.8948

--- Epoch 195/200 ---
100% 210/210 [00:26<00:00,  7.83it/s, loss=0.0609]
Calculando métricas de entrenamiento...
IoU por clase : [0.96391544 0.83878296 0.88019892 0.79398628 0.80035951 0.9359609 ]
Dice por clase: [0.98162622 0.91232405 0.93628276 0.88516427 0.88911077 0.96692128]
mIoU macro = 0.8689 | Dice macro = 0.9286
Calculando métricas de validación...
IoU por clase : [0.95987627 0.8074268  0.86500288 0.74624906 0.65992779 0.92026405]
Dice por clase: [0.97952742 0.89345449 0.9276156  0.854688   0.79512831 0.95847657]
mIoU macro = 0.8265 | Dice macro = 0.9015
"""
# ▲▲▲ PEGA AQUÍ TU TEXTO DE LOG COMPLETO ▲▲▲


def parse_log_data(text: str) -> tuple[list, list, list]:
    """
    Analiza el texto del log para extraer el número de época y los mIoU de
    entrenamiento y validación.
    """
    # Expresiones regulares para encontrar los datos necesarios
    epoch_pattern = re.compile(r"--- Epoch (\d+)/\d+ ---")
    train_miou_pattern = re.compile(r"Calculando métricas de entrenamiento.*?mIoU macro = ([\d.]+)", re.DOTALL)
    val_miou_pattern = re.compile(r"Calculando métricas de validación.*?mIoU macro = ([\d.]+)", re.DOTALL)

    epochs = []
    train_mious = []
    val_mious = []

    # Divide el log por cada bloque de época
    epoch_blocks = epoch_pattern.split(text)[1:] # Ignora el texto antes de la primera época

    for i in range(0, len(epoch_blocks), 2):
        epoch_num = int(epoch_blocks[i])
        epoch_content = epoch_blocks[i+1]
        
        # Busca el mIoU de entrenamiento en el bloque de la época
        train_match = train_miou_pattern.search(epoch_content)
        # Busca el mIoU de validación en el bloque de la época
        val_match = val_miou_pattern.search(epoch_content)
        
        # Si encuentra ambos, los añade a las listas
        if train_match and val_match:
            epochs.append(epoch_num)
            train_mious.append(float(train_match.group(1)))
            val_mious.append(float(val_match.group(1)))

    return epochs, train_mious, val_mious


def create_miou_plot(epochs: list, train_mious: list, val_mious: list, save_path: str = None):
    """
    Crea y muestra un gráfico de la evolución del mIoU.
    Opcionalmente, guarda el gráfico en un archivo.
    """
    if not epochs:
        print("No se encontraron datos de épocas para graficar.")
        return

    # Estilo de gráfico profesional con seaborn
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 8))

    # Graficar ambas curvas
    plt.plot(epochs, train_mious, 'o-', color="xkcd:sky blue", label='mIoU de Entrenamiento', markersize=5)
    plt.plot(epochs, val_mious, 'o-', color="xkcd:amber", label='mIoU de Validación', markersize=5)

    # Títulos y etiquetas
    plt.title('Evolución del mIoU (Mean Intersection over Union) por Época', fontsize=16, weight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('mIoU', fontsize=12)

    # Encontrar y marcar el mejor mIoU de validación
    best_val_miou = max(val_mious)
    best_epoch = epochs[val_mious.index(best_val_miou)]
    plt.axvline(x=best_epoch, color='xkcd:pale red', linestyle='--', linewidth=1.5, label=f'Mejor mIoU Val: {best_val_miou:.4f} (Época {best_epoch})')
    plt.scatter(best_epoch, best_val_miou, s=150, facecolors='none', edgecolors='red', zorder=5)

    # Leyenda, cuadrícula y límites
    plt.legend(fontsize=11, frameon=True, shadow=True, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)
    
    # Asegura que las épocas en el eje X se muestren como números enteros
    plt.xticks(np.arange(min(epochs), max(epochs)+1, step=max(1, len(epochs)//15)))
    plt.tight_layout()

    # Guardar el gráfico si se proporciona una ruta
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📈 Gráfico guardado en '{save_path}'")

    plt.show()


if __name__ == "__main__":
    # 1. Analizar el texto del log para extraer los datos
    epochs, train_mious, val_mious = parse_log_data(log_text)

    # 2. Imprimir los datos extraídos como verificación
    if epochs:
        print("Datos extraídos exitosamente:")
        for i in range(len(epochs)):
            print(f"  Época {epochs[i]:<3} -> mIoU Entrenamiento: {train_mious[i]:.4f} | mIoU Validación: {val_mious[i]:.4f}")
        print("-" * 60)
        
        # 3. Crear y mostrar el gráfico
        create_miou_plot(epochs, train_mious, val_mious, save_path="evolucion_miou.png")
    else:
        print("No se pudo extraer ninguna información de mIoU del log proporcionado.")
        print("Por favor, verifica que el texto del log contenga las líneas '--- Epoch ...' y 'mIoU macro = ...'.")