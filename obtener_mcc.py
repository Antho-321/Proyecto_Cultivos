from typing import Iterable, Union, Sequence, Optional
from pathlib import Path
import re, matplotlib.pyplot as plt
import matplotlib.ticker as mticker     # <- nuevo si quieres `MultipleLocator`


def plot_mcc_evolution(
    log: Union[str, Path, Iterable[str]],
    *,
    figsize: tuple[int, int] = (10, 5),
    save_path: str | None = None,
    epoch_ticks: Sequence[int] | None = None,            # ← NUEVO
    epoch_ticklabels: Sequence[str] | None = None        # ← NUEVO
) -> None:
    if isinstance(log, Path) or (isinstance(log, str) and Path(log).exists()):
        lines = Path(log).read_text(encoding="utf-8").splitlines()
    elif isinstance(log, str):
        lines = log.splitlines()
    else:
        lines = list(log)

    # Expresiones regulares ----------------------------------------------------
    epoch_re  = re.compile(r"^---\s*Epoch\s+(\d+)/", flags=re.I)
    mcc_train = re.compile(r"^\s*-\s*MCC:\s*([0-9.]+)")
    mcc_valid = re.compile(r"MCC.*?=\s*([0-9.]+)")        # <- ¡cambiado!

    epochs, train_mcc, val_mcc = [], [], []
    current_epoch = None

    for ln in lines:
        m = epoch_re.match(ln)
        if m:
            current_epoch = int(m.group(1))
            continue

        if current_epoch is not None:
            if (mt := mcc_train.match(ln)):
                epochs.append(current_epoch)
                train_mcc.append(float(mt.group(1)))
                continue

            if (mv := mcc_valid.search(ln)) and len(val_mcc) < len(epochs):
                val_mcc.append(float(mv.group(1)))

    # Sincronizar longitudes si el log se corta a mitad ------------------------
    if len(val_mcc) < len(train_mcc):
        train_mcc = train_mcc[: len(val_mcc)]
        epochs    = epochs[: len(val_mcc)]

    if not epochs:
        raise ValueError("No se encontraron épocas ni valores de MCC en el log.")

    # Gráfico ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_mcc, marker="o", label="Entrenamiento")
    plt.plot(epochs, val_mcc,   marker="s", label="Validación")

    ax = plt.gca()

    # ───────── ticks personalizados ─────────
    if epoch_ticks is not None:
        ax.set_xticks(epoch_ticks)
        if epoch_ticklabels is not None:
            ax.set_xticklabels(epoch_ticklabels)
        # Ampliar el rango si hace falta
        ax.set_xlim(min(epoch_ticks), max(epoch_ticks))

    plt.title("Evolución del MCC por época")
    plt.xlabel("Época")
    plt.ylabel("MCC")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Ejemplo de uso:
log_text = """
Using device: cuda

--- Distribución de clases en ENTRENAMIENTO (post-aug) ---
Analizando distribución: 100% 210/210 [00:05<00:00, 40.70it/s]
Número total de píxeles analizados: 110100480
Distribución final de píxeles por clase:
  Clase 0: 76.1844%  (83879378 píxeles)
  Clase 1: 3.1941%  (3516727 píxeles)
  Clase 2: 4.9363%  (5434867 píxeles)
  Clase 3: 2.8038%  (3086989 píxeles)
  Clase 4: 1.3248%  (1458575 píxeles)
  Clase 5: 11.5567%  (12723944 píxeles)

Pesos de importancia por clase (para CrossEntropy/Focal):
  Clase 0: 0.0456
  Clase 1: 1.0878
  Clase 2: 0.7039
  Clase 3: 1.2392
  Clase 4: 2.6228
  Clase 5: 0.3007
--------------------------------------------------------
Unexpected keys (bn2.bias, bn2.num_batches_tracked, bn2.running_mean, bn2.running_var, bn2.weight, conv_head.weight) found while loading pretrained weights. This may be expected if model is being adapted.
Compiling the model... (this may take a minute)

--- Epoch 1/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [01:09<00:00,  3.02it/s, loss=1.4]

Época de entrenamiento finalizada:
  - Dice por clase: [0.70856947 0.05608484 0.1279352  0.01413775 0.01539779 0.5297852 ]
  - IoU por clase : [0.5486702  0.02885148 0.06833909 0.0071192  0.00775863 0.36034542]
  - mIoU: 0.1702
  - MCC: 0.2447
Calculando métricas de validación...
IoU por clase : [0.78609278 0.03712388 0.12076543 0.00301474 0.00609046 0.57269416]
Dice por clase: [0.88023734 0.07159007 0.21550527 0.00601136 0.01210718 0.72829692]
mIoU macro = 0.2543 | Dice macro = 0.3190
Matthews Correlation Coefficient (MCC) = 0.4921
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.2543 | Dice: 0.3190  →  guardando modelo…

--- Epoch 2/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.34it/s, loss=1.26]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9369516  0.08770137 0.32956916 0.00182758 0.00647846 0.704836  ]
  - IoU por clase : [0.88138187 0.04586175 0.1972959  0.00091462 0.00324975 0.544206  ]
  - mIoU: 0.2788
  - MCC: 0.5755
Calculando métricas de validación...
IoU por clase : [9.15040922e-01 5.60353598e-02 2.36866781e-01 3.24967056e-04
 4.04033352e-03 6.35676227e-01]
Dice por clase: [9.55635894e-01 1.06124022e-01 3.83010984e-01 6.49722974e-04
 8.04814984e-03 7.77264126e-01]
mIoU macro = 0.3080 | Dice macro = 0.3718
Matthews Correlation Coefficient (MCC) = 0.6470
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3080 | Dice: 0.3718  →  guardando modelo…

--- Epoch 3/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.75it/s, loss=1.09]

Época de entrenamiento finalizada:
  - Dice por clase: [9.5932490e-01 9.0477526e-02 4.6614355e-01 2.7220129e-04 4.1836295e-03
 7.4314833e-01]
  - IoU por clase : [9.2182934e-01 4.7382277e-02 3.0390298e-01 1.3611917e-04 2.0961997e-03
 5.9127766e-01]
  - mIoU: 0.3111
  - MCC: 0.6559
Calculando métricas de validación...
IoU por clase : [9.27060830e-01 5.74325512e-02 3.20513856e-01 3.25395315e-05
 2.32604374e-03 6.72089359e-01]
Dice por clase: [9.62150043e-01 1.08626411e-01 4.85438081e-01 6.50769454e-05
 4.64129163e-03 8.03891677e-01]
mIoU macro = 0.3299 | Dice macro = 0.3941
Matthews Correlation Coefficient (MCC) = 0.6800
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3299 | Dice: 0.3941  →  guardando modelo…

--- Epoch 4/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.89it/s, loss=0.968]

Época de entrenamiento finalizada:
  - Dice por clase: [9.6278179e-01 8.4262729e-02 5.5655265e-01 5.0804254e-05 2.4605698e-03
 7.7268392e-01]
  - IoU por clase : [9.2823440e-01 4.3984491e-02 3.8557184e-01 2.5402773e-05 1.2318004e-03
 6.2957197e-01]
  - mIoU: 0.3314
  - MCC: 0.6875
Calculando métricas de validación...
IoU por clase : [9.29802798e-01 5.35119933e-02 3.65619511e-01 1.13898182e-05
 1.32689677e-03 6.82394320e-01]
Dice por clase: [9.63624676e-01 1.01587820e-01 5.35463221e-01 2.27793769e-05
 2.65027689e-03 8.11218050e-01]
mIoU macro = 0.3388 | Dice macro = 0.4024
Matthews Correlation Coefficient (MCC) = 0.6954
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3388 | Dice: 0.4024  →  guardando modelo…

--- Epoch 5/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.98it/s, loss=0.868]

Época de entrenamiento finalizada:
  - Dice por clase: [9.6443439e-01 1.0573163e-01 5.8185703e-01 3.8811049e-05 1.5687159e-03
 7.9376632e-01]
  - IoU por clase : [9.3131167e-01 5.5816602e-02 4.1029504e-01 1.9405901e-05 7.8497361e-04
 6.5805346e-01]
  - mIoU: 0.3427
  - MCC: 0.7002
Calculando métricas de validación...
IoU por clase : [9.31640086e-01 8.04950527e-02 3.91829061e-01 2.76602669e-05
 9.27094737e-04 7.43779570e-01]
Dice por clase: [9.64610429e-01 1.48996615e-01 5.63041931e-01 5.53190036e-05
 1.85247206e-03 8.53066044e-01]
mIoU macro = 0.3581 | Dice macro = 0.4219
Matthews Correlation Coefficient (MCC) = 0.7120
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3581 | Dice: 0.4219  →  guardando modelo…

--- Epoch 6/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.60it/s, loss=0.965]

Época de entrenamiento finalizada:
  - Dice por clase: [9.6488768e-01 1.5081346e-01 6.0281491e-01 8.8344619e-05 1.8365891e-03
 8.1852281e-01]
  - IoU por clase : [9.3215758e-01 8.1556655e-02 4.3144956e-01 4.4174260e-05 9.1913861e-04
 6.9279611e-01]
  - mIoU: 0.3565
  - MCC: 0.7107
Calculando métricas de validación...
IoU por clase : [9.28288090e-01 8.87461904e-02 4.08403642e-01 1.30109227e-04
 8.30953546e-04 7.48365588e-01]
Dice por clase: [9.62810583e-01 1.63024571e-01 5.79952550e-01 2.60184601e-04
 1.66052727e-03 8.56074488e-01]
mIoU macro = 0.3625 | Dice macro = 0.4273
Matthews Correlation Coefficient (MCC) = 0.7091
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3625 | Dice: 0.4273  →  guardando modelo…

--- Epoch 7/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.17it/s, loss=0.753]

Época de entrenamiento finalizada:
  - Dice por clase: [9.6539569e-01 1.9025517e-01 6.3455719e-01 7.5321086e-04 2.0332388e-03
 8.4077913e-01]
  - IoU por clase : [9.3310601e-01 1.0512817e-01 4.6472633e-01 3.7674731e-04 1.0176540e-03
 7.2529680e-01]
  - mIoU: 0.3716
  - MCC: 0.7197
Calculando métricas de validación...
IoU por clase : [9.32304540e-01 1.33128314e-01 4.48745062e-01 7.81799830e-04
 2.60153969e-04 8.21562208e-01]
Dice por clase: [9.64966464e-01 2.34974826e-01 6.19494863e-01 1.56237819e-03
 5.20172612e-04 9.02041341e-01]
mIoU macro = 0.3895 | Dice macro = 0.4539
Matthews Correlation Coefficient (MCC) = 0.7271
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.3895 | Dice: 0.4539  →  guardando modelo…

--- Epoch 8/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.81it/s, loss=0.747]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96609557 0.21155342 0.65789443 0.00620179 0.00363781 0.8538923 ]
  - IoU por clase : [0.93441474 0.11828893 0.49019572 0.00311054 0.00182222 0.7450367 ]
  - mIoU: 0.3821
  - MCC: 0.7262
Calculando métricas de validación...
IoU por clase : [0.93275138 0.15573801 0.51636711 0.00831722 0.00168774 0.80929554]
Dice por clase: [0.96520576 0.26950401 0.68105818 0.01649723 0.0033698  0.8945974 ]
mIoU macro = 0.4040 | Dice macro = 0.4717
Matthews Correlation Coefficient (MCC) = 0.7329
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.4040 | Dice: 0.4717  →  guardando modelo…

--- Epoch 9/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.12it/s, loss=0.597]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96695846 0.24597912 0.67070913 0.03736115 0.01098819 0.86393714]
  - IoU por clase : [0.93603057 0.14023727 0.5045616  0.01903618 0.00552445 0.7604659 ]
  - mIoU: 0.3943
  - MCC: 0.7343
Calculando métricas de validación...
IoU por clase : [0.93640859 0.169749   0.51402113 0.05157507 0.00674484 0.85907935]
Dice por clase: [0.96716013 0.29023149 0.67901447 0.09809109 0.01339931 0.92419869]
mIoU macro = 0.4229 | Dice macro = 0.4953
Matthews Correlation Coefficient (MCC) = 0.7466
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.4229 | Dice: 0.4953  →  guardando modelo…

--- Epoch 10/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.94it/s, loss=0.584]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96752894 0.24776818 0.66603994 0.10130432 0.01495482 0.8620894 ]
  - IoU por clase : [0.9371004  0.14140148 0.49929526 0.05335469 0.00753374 0.7576072 ]
  - mIoU: 0.3994
  - MCC: 0.7371
Calculando métricas de validación...
IoU por clase : [0.93555472 0.19345018 0.52794853 0.17395125 0.01770842 0.84815583]
Dice por clase: [0.96670449 0.32418643 0.69105538 0.29635174 0.03480057 0.91784017]
mIoU macro = 0.4495 | Dice macro = 0.5385
Matthews Correlation Coefficient (MCC) = 0.7596
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.4495 | Dice: 0.5385  →  guardando modelo…

--- Epoch 11/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.81it/s, loss=0.595]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96829295 0.25520113 0.6802044  0.22761749 0.01605636 0.8724088 ]
  - IoU por clase : [0.93853474 0.14626393 0.5153861  0.12842458 0.00809315 0.7736925 ]
  - mIoU: 0.4184
  - MCC: 0.7482
Calculando métricas de validación...
IoU por clase : [0.93555417 0.18179309 0.50652515 0.18639315 0.01394528 0.8259945 ]
Dice por clase: [0.9667042  0.30765638 0.67244167 0.31421818 0.02750696 0.90470645]
mIoU macro = 0.4417 | Dice macro = 0.5322
Matthews Correlation Coefficient (MCC) = 0.7570
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 12/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.51it/s, loss=0.626]

Época de entrenamiento finalizada:
  - Dice por clase: [0.969198   0.29048684 0.6957304  0.29737923 0.01903653 0.88009965]
  - IoU por clase : [0.94023675 0.16992372 0.5334253  0.1746597  0.00960973 0.7858732 ]
  - mIoU: 0.4356
  - MCC: 0.7600
Calculando métricas de validación...
IoU por clase : [0.93518342 0.21432109 0.58084714 0.31190914 0.01148872 0.82874319]
Dice por clase: [0.96650623 0.35298916 0.73485554 0.47550418 0.02271645 0.90635273]
mIoU macro = 0.4804 | Dice macro = 0.5765
Matthews Correlation Coefficient (MCC) = 0.7768
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.4804 | Dice: 0.5765  →  guardando modelo…

--- Epoch 13/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.92it/s, loss=0.631]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96973944 0.28251705 0.7063947  0.36823094 0.0174889  0.88984084]
  - IoU por clase : [0.9412565  0.16449483 0.54606664 0.22566363 0.00882159 0.8015435 ]
  - mIoU: 0.4480
  - MCC: 0.7682
Calculando métricas de validación...
IoU por clase : [0.93565836 0.25056641 0.60847897 0.38829453 0.00644426 0.84786961]
Dice por clase: [0.96675982 0.40072467 0.75658928 0.55938351 0.01280599 0.91767255]
mIoU macro = 0.5062 | Dice macro = 0.6023
Matthews Correlation Coefficient (MCC) = 0.7931
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5062 | Dice: 0.6023  →  guardando modelo…

--- Epoch 14/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.11it/s, loss=0.501]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9699397  0.29490626 0.72111756 0.43285275 0.02681208 0.88507557]
  - IoU por clase : [0.9416338  0.17295603 0.5638654  0.27620426 0.01358821 0.7938436 ]
  - mIoU: 0.4603
  - MCC: 0.7735
Calculando métricas de validación...
IoU por clase : [0.93467408 0.27498977 0.65538136 0.45924697 0.01819797 0.86633311]
Dice por clase: [0.96623415 0.43135996 0.79181918 0.62943008 0.03574545 0.92837994]
mIoU macro = 0.5348 | Dice macro = 0.6305
Matthews Correlation Coefficient (MCC) = 0.8063
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5348 | Dice: 0.6305  →  guardando modelo…

--- Epoch 15/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.80it/s, loss=0.533]

Época de entrenamiento finalizada:
  - Dice por clase: [0.96987164 0.297294   0.7377408  0.47939554 0.01761538 0.8941492 ]
  - IoU por clase : [0.94150555 0.17460088 0.5844607  0.31526643 0.00888595 0.80856234]
  - mIoU: 0.4722
  - MCC: 0.7811
Calculando métricas de validación...
IoU por clase : [0.93655471 0.25299871 0.61284331 0.50490902 0.01523873 0.86589585]
Dice por clase: [0.96723806 0.40382916 0.75995394 0.67101601 0.03001999 0.92812881]
mIoU macro = 0.5314 | Dice macro = 0.6267
Matthews Correlation Coefficient (MCC) = 0.8080
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 16/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.58it/s, loss=0.474]

Época de entrenamiento finalizada:
  - Dice por clase: [0.970664   0.29018784 0.76197964 0.5350898  0.01555683 0.90003353]
  - IoU por clase : [0.94300014 0.16971913 0.61548233 0.36527136 0.00783939 0.8182372 ]
  - mIoU: 0.4866
  - MCC: 0.7904
Calculando métricas de validación...
IoU por clase : [0.93781876 0.26054301 0.67903203 0.49616177 0.00516475 0.87833555]
Dice por clase: [0.96791174 0.41338219 0.80883749 0.66324616 0.01027642 0.93522752]
mIoU macro = 0.5428 | Dice macro = 0.6331
Matthews Correlation Coefficient (MCC) = 0.8157
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5428 | Dice: 0.6331  →  guardando modelo…

--- Epoch 17/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.71it/s, loss=0.41]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97083205 0.29586336 0.7742939  0.5488288  0.01645669 0.90210456]
  - IoU por clase : [0.94331753 0.17361481 0.6317125  0.37819713 0.00829661 0.821667  ]
  - mIoU: 0.4928
  - MCC: 0.7943
Calculando métricas de validación...
IoU por clase : [0.93692161 0.27876489 0.66100201 0.51762937 0.01099773 0.87581514]
Dice por clase: [0.96743369 0.43599084 0.79590754 0.68215518 0.02175619 0.93379686]
mIoU macro = 0.5469 | Dice macro = 0.6395
Matthews Correlation Coefficient (MCC) = 0.8176
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5469 | Dice: 0.6395  →  guardando modelo…

--- Epoch 18/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.23it/s, loss=0.457]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97117084 0.29839107 0.7681887  0.5686312  0.00825327 0.9020659 ]
  - IoU por clase : [0.94395727 0.17535819 0.62362534 0.3972639  0.00414374 0.821603  ]
  - mIoU: 0.4943
  - MCC: 0.7952
Calculando métricas de validación...
IoU por clase : [0.93799566 0.31195709 0.70094004 0.52327167 0.01015778 0.87361435]
Dice por clase: [0.96800594 0.4755599  0.8241796  0.68703657 0.02011127 0.93254447]
mIoU macro = 0.5597 | Dice macro = 0.6512
Matthews Correlation Coefficient (MCC) = 0.8223
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5597 | Dice: 0.6512  →  guardando modelo…

--- Epoch 19/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.64it/s, loss=0.398]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9716323  0.33366957 0.7864079  0.5673825  0.01369389 0.911309  ]
  - IoU por clase : [0.94482976 0.20024215 0.6480002  0.39604607 0.00689415 0.83706856]
  - mIoU: 0.5055
  - MCC: 0.8025
Calculando métricas de validación...
IoU por clase : [0.93905721 0.34506886 0.68941347 0.55719576 0.00182783 0.87824229]
Dice por clase: [0.96857092 0.51308728 0.81615719 0.71563997 0.00364899 0.93517465]
mIoU macro = 0.5685 | Dice macro = 0.6587
Matthews Correlation Coefficient (MCC) = 0.8276
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5685 | Dice: 0.6587  →  guardando modelo…

--- Epoch 20/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.17it/s, loss=0.47]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9720413  0.33713394 0.78331906 0.592122   0.00715752 0.91494286]
  - IoU por clase : [0.94560355 0.2027427  0.6438163  0.42057765 0.00359161 0.84322083]
  - mIoU: 0.5099
  - MCC: 0.8063
Calculando métricas de validación...
IoU por clase : [0.94089673 0.34006425 0.67183064 0.57280225 0.00656639 0.8746361 ]
Dice por clase: [0.96954848 0.50753424 0.80370657 0.72838432 0.01304711 0.93312627]
mIoU macro = 0.5678 | Dice macro = 0.6592
Matthews Correlation Coefficient (MCC) = 0.8298
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 21/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.50it/s, loss=0.346]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97216743 0.37486982 0.78450173 0.60421735 0.01094358 0.9071314 ]
  - IoU por clase : [0.9458424  0.23067065 0.6454157  0.43288785 0.0055019  0.8300462 ]
  - mIoU: 0.5151
  - MCC: 0.8065
Calculando métricas de validación...
IoU por clase : [0.94074937 0.41109577 0.71013183 0.58040161 0.00956338 0.86333999]
Dice por clase: [0.96947023 0.58266176 0.83049952 0.73449888 0.01894557 0.92665857]
mIoU macro = 0.5859 | Dice macro = 0.6771
Matthews Correlation Coefficient (MCC) = 0.8337
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5859 | Dice: 0.6771  →  guardando modelo…

--- Epoch 22/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.91it/s, loss=0.437]

Época de entrenamiento finalizada:
  - Dice por clase: [0.972553   0.40792507 0.80370045 0.61412984 0.01162526 0.9132349 ]
  - IoU por clase : [0.9465723  0.25622228 0.67182213 0.44313663 0.00584661 0.8403241 ]
  - mIoU: 0.5273
  - MCC: 0.8138
Calculando métricas de validación...
IoU por clase : [0.94079464 0.44166201 0.7077473  0.58357264 0.01797967 0.87831794]
Dice por clase: [0.96949427 0.61271228 0.82886654 0.73703299 0.03532423 0.93521754]
mIoU macro = 0.5950 | Dice macro = 0.6864
Matthews Correlation Coefficient (MCC) = 0.8385
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.5950 | Dice: 0.6864  →  guardando modelo…

--- Epoch 23/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.88it/s, loss=0.386]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97246075 0.47426054 0.80957335 0.63135546 0.02225767 0.9204441 ]
  - IoU por clase : [0.94639754 0.3108398  0.68006986 0.4612998  0.01125408 0.8526135 ]
  - mIoU: 0.5437
  - MCC: 0.8206
Calculando métricas de validación...
IoU por clase : [0.94183989 0.45458257 0.73051444 0.55953199 0.02055187 0.89528731]
Dice por clase: [0.97004897 0.62503509 0.84427431 0.71756398 0.040276   0.94475102]
mIoU macro = 0.6004 | Dice macro = 0.6903
Matthews Correlation Coefficient (MCC) = 0.8411
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6004 | Dice: 0.6903  →  guardando modelo…

--- Epoch 24/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.83it/s, loss=0.458]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9731663  0.51696175 0.8158815  0.63143545 0.02100842 0.9234696 ]
  - IoU por clase : [0.9477351  0.3485829  0.68902016 0.46138522 0.01061572 0.85782045]
  - mIoU: 0.5525
  - MCC: 0.8257
Calculando métricas de validación...
IoU por clase : [0.94170105 0.5196249  0.74640591 0.59933792 0.04397898 0.89217329]
Dice por clase: [0.96997532 0.68388574 0.85479086 0.74948254 0.08425261 0.94301436]
mIoU macro = 0.6239 | Dice macro = 0.7142
Matthews Correlation Coefficient (MCC) = 0.8490
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6239 | Dice: 0.7142  →  guardando modelo…

--- Epoch 25/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.07it/s, loss=0.258]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9733505  0.5733093  0.82463276 0.6585187  0.02696893 0.9243732 ]
  - IoU por clase : [0.9480844  0.40184554 0.70159584 0.49088916 0.01366878 0.859381  ]
  - mIoU: 0.5692
  - MCC: 0.8325
Calculando métricas de validación...
IoU por clase : [0.94270582 0.55988535 0.75147323 0.5940866  0.03824495 0.90333454]
Dice por clase: [0.97050805 0.71785449 0.85810416 0.74536302 0.07367231 0.94921258]
mIoU macro = 0.6316 | Dice macro = 0.7191
Matthews Correlation Coefficient (MCC) = 0.8527
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6316 | Dice: 0.7191  →  guardando modelo…

--- Epoch 26/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.67it/s, loss=0.276]

Época de entrenamiento finalizada:
  - Dice por clase: [0.973359   0.63006    0.8323797  0.6620666  0.05593923 0.9323623 ]
  - IoU por clase : [0.9481005  0.459918   0.7128856  0.4948427  0.02877443 0.8732948 ]
  - mIoU: 0.5863
  - MCC: 0.8399
Calculando métricas de validación...
IoU por clase : [0.94290211 0.59891113 0.75084786 0.60836145 0.05246811 0.89955739]
Dice por clase: [0.97061206 0.74914874 0.8576963  0.75649842 0.09970489 0.94712315]
mIoU macro = 0.6422 | Dice macro = 0.7301
Matthews Correlation Coefficient (MCC) = 0.8562
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6422 | Dice: 0.7301  →  guardando modelo…

--- Epoch 27/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.12it/s, loss=0.275]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9738344  0.6803626  0.8422401  0.68223417 0.05125604 0.93375045]
  - IoU por clase : [0.9490031  0.5155678  0.7274739  0.51772034 0.02630209 0.87573355]
  - mIoU: 0.6020
  - MCC: 0.8468
Calculando métricas de validación...
IoU por clase : [0.94376035 0.61076092 0.75366028 0.60591834 0.03495062 0.90593366]
Dice por clase: [0.97106657 0.75835081 0.85952825 0.75460666 0.06754066 0.95064553]
mIoU macro = 0.6425 | Dice macro = 0.7270
Matthews Correlation Coefficient (MCC) = 0.8579
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6425 | Dice: 0.7270  →  guardando modelo…

--- Epoch 28/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.73it/s, loss=0.535]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9741479  0.7180971  0.8497432  0.6944186  0.08953294 0.9361208 ]
  - IoU por clase : [0.9495988  0.56018054 0.7387422  0.5318846  0.04686443 0.8799127 ]
  - mIoU: 0.6179
  - MCC: 0.8529
Calculando métricas de validación...
IoU por clase : [0.94455425 0.59822885 0.76526019 0.6369619  0.08018638 0.89581625]
Dice por clase: [0.97148665 0.74861476 0.86702255 0.77822446 0.14846768 0.94504544]
mIoU macro = 0.6535 | Dice macro = 0.7431
Matthews Correlation Coefficient (MCC) = 0.8610
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6535 | Dice: 0.7431  →  guardando modelo…

--- Epoch 29/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.13it/s, loss=0.215]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97445244 0.7426246  0.8578713  0.7044564  0.07474899 0.9351216 ]
  - IoU por clase : [0.9501776  0.59061486 0.75111616 0.5437535  0.03882558 0.87814856]
  - mIoU: 0.6254
  - MCC: 0.8564
Calculando métricas de validación...
IoU por clase : [0.94426467 0.63486797 0.7722545  0.63356856 0.13421486 0.90249433]
Dice por clase: [0.97133346 0.77665962 0.87149391 0.77568653 0.23666568 0.94874851]
mIoU macro = 0.6703 | Dice macro = 0.7634
Matthews Correlation Coefficient (MCC) = 0.8645
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6703 | Dice: 0.7634  →  guardando modelo…

--- Epoch 30/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.98it/s, loss=0.266]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9740364  0.75364083 0.86528945 0.7263927  0.08978727 0.93846774]
  - IoU por clase : [0.9493869  0.60467386 0.762564   0.5703427  0.04700381 0.8840688 ]
  - mIoU: 0.6363
  - MCC: 0.8600
Calculando métricas de validación...
IoU por clase : [0.94476489 0.64276121 0.77971465 0.64363209 0.16745541 0.90412282]
Dice por clase: [0.97159805 0.7825376  0.87622434 0.78318268 0.28687247 0.94964759]
mIoU macro = 0.6804 | Dice macro = 0.7750
Matthews Correlation Coefficient (MCC) = 0.8677
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6804 | Dice: 0.7750  →  guardando modelo…

--- Epoch 31/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.85it/s, loss=0.247]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97451895 0.77438474 0.8710262  0.7225435  0.13508421 0.9449677 ]
  - IoU por clase : [0.9503042  0.63183343 0.7715204  0.56561106 0.07243448 0.89567643]
  - mIoU: 0.6479
  - MCC: 0.8652
Calculando métricas de validación...
IoU por clase : [0.94497203 0.6674935  0.74527178 0.62370984 0.17867232 0.90665701]
Dice por clase: [0.97170758 0.80059503 0.85404668 0.76825283 0.30317555 0.95104364]
mIoU macro = 0.6778 | Dice macro = 0.7748
Matthews Correlation Coefficient (MCC) = 0.8657
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 32/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.50it/s, loss=0.232]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9748798  0.779862   0.8716075  0.7377072  0.23330039 0.9416482 ]
  - IoU por clase : [0.9509907  0.6391588  0.7724329  0.5844185  0.13205436 0.8897308 ]
  - mIoU: 0.6615
  - MCC: 0.8675
Calculando métricas de validación...
IoU por clase : [0.94547996 0.61476931 0.77734516 0.63744302 0.14841134 0.90670246]
Dice por clase: [0.97197605 0.76143299 0.87472616 0.77858345 0.25846374 0.95106864]
mIoU macro = 0.6717 | Dice macro = 0.7660
Matthews Correlation Coefficient (MCC) = 0.8662
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 33/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.36it/s, loss=0.348]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9747415  0.78878665 0.8745542  0.7393059  0.15550615 0.94652027]
  - IoU por clase : [0.95072746 0.6512368  0.7770737  0.5864277  0.08430831 0.89847046]
  - mIoU: 0.6580
  - MCC: 0.8687
Calculando métricas de validación...
IoU por clase : [0.94543759 0.65783938 0.77890297 0.65322932 0.17367308 0.90718755]
Dice por clase: [0.97195366 0.79361051 0.87571158 0.79024647 0.29594796 0.95133543]
mIoU macro = 0.6860 | Dice macro = 0.7798
Matthews Correlation Coefficient (MCC) = 0.8704
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6860 | Dice: 0.7798  →  guardando modelo…

--- Epoch 34/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.97it/s, loss=0.243]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97489905 0.7995288  0.8800337  0.74605787 0.17186128 0.9468415 ]
  - IoU por clase : [0.95102733 0.66601247 0.7857679  0.59497    0.09400888 0.89904934]
  - mIoU: 0.6651
  - MCC: 0.8716
Calculando métricas de validación...
IoU por clase : [0.94515004 0.65876814 0.76857526 0.642723   0.1949149  0.90503966]
Dice por clase: [0.97180168 0.79428598 0.86914623 0.78250928 0.32624063 0.95015309]
mIoU macro = 0.6859 | Dice macro = 0.7824
Matthews Correlation Coefficient (MCC) = 0.8692
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 35/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.44it/s, loss=0.166]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97489536 0.79235774 0.87262744 0.7394189  0.18964766 0.9477479 ]
  - IoU por clase : [0.9510204  0.6561196  0.77403647 0.5865699  0.10475732 0.9006852 ]
  - mIoU: 0.6622
  - MCC: 0.8700
Calculando métricas de validación...
IoU por clase : [0.94485356 0.68397519 0.77007317 0.64744673 0.21734473 0.91003401]
Dice por clase: [0.97164494 0.81233405 0.8701032  0.7860002  0.35708    0.95289822]
mIoU macro = 0.6956 | Dice macro = 0.7917
Matthews Correlation Coefficient (MCC) = 0.8712
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.6956 | Dice: 0.7917  →  guardando modelo…

--- Epoch 36/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.84it/s, loss=0.198]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97476447 0.8034188  0.8757386  0.7394717  0.2703911  0.94732815]
  - IoU por clase : [0.9507713  0.67142856 0.77894574 0.5866363  0.15633076 0.8999274 ]
  - mIoU: 0.6740
  - MCC: 0.8719
Calculando métricas de validación...
IoU por clase : [0.94582851 0.66601128 0.77213143 0.64639625 0.23472196 0.90752752]
Dice por clase: [0.97216019 0.79952794 0.87141553 0.78522561 0.38020212 0.95152233]
mIoU macro = 0.6954 | Dice macro = 0.7933
Matthews Correlation Coefficient (MCC) = 0.8719
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 37/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.15it/s, loss=0.327]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97534245 0.8120212  0.8777194  0.75124764 0.28437155 0.95100033]
  - IoU por clase : [0.9518719  0.68353176 0.7820855  0.60159856 0.16575357 0.9065783 ]
  - mIoU: 0.6819
  - MCC: 0.8761
Calculando métricas de validación...
IoU por clase : [0.94571973 0.70820483 0.78778088 0.63928075 0.22022015 0.91125823]
Dice por clase: [0.97210273 0.82918022 0.88129467 0.77995274 0.36095151 0.95356893]
mIoU macro = 0.7021 | Dice macro = 0.7962
Matthews Correlation Coefficient (MCC) = 0.8741
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7021 | Dice: 0.7962  →  guardando modelo…

--- Epoch 38/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.87it/s, loss=0.261]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97555083 0.81576943 0.8862074  0.76226836 0.28136325 0.9530264 ]
  - IoU por clase : [0.9522687  0.6888603  0.79566646 0.61585915 0.16371304 0.9102679 ]
  - mIoU: 0.6878
  - MCC: 0.8790
Calculando métricas de validación...
IoU por clase : [0.94604686 0.69015496 0.78518241 0.65207192 0.24832304 0.91003289]
Dice por clase: [0.97227552 0.81667655 0.87966631 0.78939895 0.39785061 0.95289761]
mIoU macro = 0.7053 | Dice macro = 0.8015
Matthews Correlation Coefficient (MCC) = 0.8750
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7053 | Dice: 0.8015  →  guardando modelo…

--- Epoch 39/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:20<00:00, 10.38it/s, loss=0.174]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9754428  0.81779766 0.8839932  0.7655082  0.26518196 0.95053023]
  - IoU por clase : [0.952063   0.6917578  0.7921038  0.62009984 0.15285864 0.9057241 ]
  - mIoU: 0.6858
  - MCC: 0.8780
Calculando métricas de validación...
IoU por clase : [0.94684496 0.69182728 0.79534805 0.67010616 0.24030306 0.91066403]
Dice por clase: [0.97269683 0.81784623 0.88600987 0.80247134 0.38749088 0.9532435 ]
mIoU macro = 0.7092 | Dice macro = 0.8033
Matthews Correlation Coefficient (MCC) = 0.8778
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7092 | Dice: 0.8033  →  guardando modelo…

--- Epoch 40/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.12it/s, loss=0.129]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97580993 0.82534224 0.8864625  0.7641517  0.34342304 0.9529774 ]
  - IoU por clase : [0.9527625  0.7026236  0.79607785 0.61832154 0.20730883 0.9101785 ]
  - mIoU: 0.6979
  - MCC: 0.8811
Calculando métricas de validación...
IoU por clase : [0.94636045 0.6873531  0.79500213 0.6659716  0.26456932 0.91018964]
Dice por clase: [0.9724411  0.81471163 0.88579519 0.79949934 0.41843388 0.95298354]
mIoU macro = 0.7116 | Dice macro = 0.8073
Matthews Correlation Coefficient (MCC) = 0.8770
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7116 | Dice: 0.8073  →  guardando modelo…

--- Epoch 41/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.89it/s, loss=0.297]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9758517  0.82110703 0.88995516 0.7752572  0.35325986 0.9526855 ]
  - IoU por clase : [0.95284224 0.69650686 0.801729   0.6329959  0.21452071 0.909646  ]
  - mIoU: 0.7014
  - MCC: 0.8823
Calculando métricas de validación...
IoU por clase : [0.94665894 0.68570251 0.79556665 0.67009074 0.28834496 0.91054595]
Dice por clase: [0.97259866 0.81355103 0.8861455  0.80246028 0.44762074 0.9531788 ]
mIoU macro = 0.7162 | Dice macro = 0.8126
Matthews Correlation Coefficient (MCC) = 0.8777
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7162 | Dice: 0.8126  →  guardando modelo…

--- Epoch 42/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.87it/s, loss=0.22]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9760348  0.83289105 0.8868079  0.7770324  0.40161395 0.9538136 ]
  - IoU por clase : [0.9531915  0.71363604 0.79663515 0.63536626 0.2512622  0.91170526]
  - mIoU: 0.7103
  - MCC: 0.8843
Calculando métricas de validación...
IoU por clase : [0.94676617 0.70142509 0.78354475 0.67285298 0.29458187 0.91335805]
Dice por clase: [0.97265525 0.8245148  0.87863761 0.80443768 0.45509964 0.95471733]
mIoU macro = 0.7188 | Dice macro = 0.8150
Matthews Correlation Coefficient (MCC) = 0.8786
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7188 | Dice: 0.8150  →  guardando modelo…

--- Epoch 43/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.96it/s, loss=0.152]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9760377  0.8314763  0.8883758  0.78131944 0.45376673 0.95420873]
  - IoU por clase : [0.95319694 0.7115613  0.7991692  0.6411191  0.2934659  0.9124276 ]
  - mIoU: 0.7185
  - MCC: 0.8857
Calculando métricas de validación...
IoU por clase : [0.94623026 0.69354364 0.79053053 0.66330985 0.28419272 0.90990063]
Dice por clase: [0.97237236 0.81904431 0.88301263 0.79757821 0.44260135 0.9528251 ]
mIoU macro = 0.7146 | Dice macro = 0.8112
Matthews Correlation Coefficient (MCC) = 0.8767
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 44/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.27it/s, loss=0.154]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97598726 0.83719325 0.8841196  0.7751114  0.46861395 0.9555362 ]
  - IoU por clase : [0.9531007  0.7199763  0.7923067  0.63280153 0.30600643 0.9148582 ]
  - mIoU: 0.7198
  - MCC: 0.8856
Calculando métricas de validación...
IoU por clase : [0.94682571 0.69974323 0.77686334 0.66994357 0.2649299  0.91040544]
Dice por clase: [0.97268667 0.82335169 0.87442103 0.80235474 0.41888471 0.95310181]
mIoU macro = 0.7115 | Dice macro = 0.8075
Matthews Correlation Coefficient (MCC) = 0.8776
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 45/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.47it/s, loss=0.16]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97637796 0.83992815 0.8919998  0.7855592  0.4848483  0.95689285]
  - IoU por clase : [0.9538461  0.72403115 0.8050538  0.64684844 0.31999984 0.9173485 ]
  - mIoU: 0.7279
  - MCC: 0.8889
Calculando métricas de validación...
IoU por clase : [0.9481924  0.69807739 0.79501043 0.67439374 0.34342195 0.91116281]
Dice por clase: [0.97340735 0.82219738 0.88580035 0.80553782 0.51126446 0.95351668]
mIoU macro = 0.7284 | Dice macro = 0.8253
Matthews Correlation Coefficient (MCC) = 0.8816
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7284 | Dice: 0.8253  →  guardando modelo…

--- Epoch 46/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.75it/s, loss=0.202]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9763629  0.8451817  0.89626324 0.79410464 0.5246826  0.95523804]
  - IoU por clase : [0.9538174  0.73187417 0.81202626 0.6585187  0.3556405  0.9143117 ]
  - mIoU: 0.7377
  - MCC: 0.8906
Calculando métricas de validación...
IoU por clase : [0.9471359  0.71778183 0.78409602 0.6711542  0.34718861 0.91027551]
Dice por clase: [0.97285033 0.83570779 0.8789841  0.80322235 0.51542687 0.9530306 ]
mIoU macro = 0.7296 | Dice macro = 0.8265
Matthews Correlation Coefficient (MCC) = 0.8808
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7296 | Dice: 0.8265  →  guardando modelo…

--- Epoch 47/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.80it/s, loss=0.192]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97673136 0.8517966  0.89710855 0.8018     0.56692916 0.95722264]
  - IoU por clase : [0.95452094 0.7418517  0.8134151  0.66917044 0.39560443 0.91795516]
  - mIoU: 0.7488
  - MCC: 0.8937
Calculando métricas de validación...
IoU por clase : [0.94812115 0.71315768 0.78910608 0.68520084 0.36992426 0.90976579]
Dice por clase: [0.9733698  0.83256514 0.8821233  0.81319784 0.54006528 0.95275117]
mIoU macro = 0.7359 | Dice macro = 0.8323
Matthews Correlation Coefficient (MCC) = 0.8828
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7359 | Dice: 0.8323  →  guardando modelo…

--- Epoch 48/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.57it/s, loss=0.201]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97669    0.85371566 0.89775765 0.7992582  0.56551677 0.9580189 ]
  - IoU por clase : [0.9544418 0.7447678 0.814483  0.665637  0.3942303 0.9194206]
  - mIoU: 0.7488
  - MCC: 0.8939
Calculando métricas de validación...
IoU por clase : [0.94785771 0.71149004 0.79453617 0.67263288 0.31718636 0.91366217]
Dice por clase: [0.97323096 0.83142762 0.88550589 0.80428035 0.48161197 0.95488345]
mIoU macro = 0.7262 | Dice macro = 0.8218
Matthews Correlation Coefficient (MCC) = 0.8816
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 49/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:17<00:00, 11.68it/s, loss=0.172]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9768252  0.85377586 0.89937997 0.80878687 0.60422885 0.9584849 ]
  - IoU por clase : [0.9547001  0.7448594  0.8171575  0.6789607  0.43289965 0.9202793 ]
  - mIoU: 0.7581
  - MCC: 0.8960
Calculando métricas de validación...
IoU por clase : [0.9485258  0.68830463 0.78635223 0.67813979 0.38870195 0.90667657]
Dice por clase: [0.973583   0.81537966 0.88039998 0.80820417 0.55980616 0.9510544 ]
mIoU macro = 0.7328 | Dice macro = 0.8314
Matthews Correlation Coefficient (MCC) = 0.8810
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 50/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.22it/s, loss=0.161]

Época de entrenamiento finalizada:
  - Dice por clase: [0.976965  0.8598083 0.9017274 0.8124736 0.6410047 0.9589464]
  - IoU por clase : [0.95496726 0.754091   0.8210415  0.68417305 0.47167543 0.92113066]
  - mIoU: 0.7678
  - MCC: 0.8982
Calculando métricas de validación...
IoU por clase : [0.94911867 0.72180616 0.79367758 0.69208987 0.40475772 0.91061282]
Dice por clase: [0.97389521 0.83842906 0.8849724  0.81802968 0.57626695 0.95321544]
mIoU macro = 0.7453 | Dice macro = 0.8408
Matthews Correlation Coefficient (MCC) = 0.8858
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7453 | Dice: 0.8408  →  guardando modelo…

--- Epoch 51/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 11.04it/s, loss=0.155]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9771505  0.86151475 0.90095586 0.81318444 0.6209872  0.9584642 ]
  - IoU por clase : [0.9553218  0.7567201  0.8197631  0.6851818  0.45031282 0.9202411 ]
  - mIoU: 0.7646
  - MCC: 0.8980
Calculando métricas de validación...
IoU por clase : [0.94892125 0.73153192 0.79632439 0.66422421 0.41996171 0.91278537]
Dice por clase: [0.97379127 0.84495343 0.88661535 0.79823885 0.59151131 0.95440438]
mIoU macro = 0.7456 | Dice macro = 0.8416
Matthews Correlation Coefficient (MCC) = 0.8847
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7456 | Dice: 0.8416  →  guardando modelo…

--- Epoch 52/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.64it/s, loss=0.15]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9771011 0.859844  0.8944041 0.815152  0.6153131 0.9574424]
  - IoU por clase : [0.95522743 0.7541459  0.8089792  0.68798023 0.44436985 0.9183593 ]
  - mIoU: 0.7615
  - MCC: 0.8966
Calculando métricas de validación...
IoU por clase : [0.94907327 0.71571918 0.79441982 0.67514144 0.42671245 0.90931174]
Dice por clase: [0.97387131 0.83430807 0.88543362 0.80607097 0.59817583 0.95250212]
mIoU macro = 0.7451 | Dice macro = 0.8417
Matthews Correlation Coefficient (MCC) = 0.8844
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 53/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.61it/s, loss=0.151]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9772723  0.86491007 0.89778215 0.8144117  0.6707056  0.9604101 ]
  - IoU por clase : [0.9555547  0.7619749  0.8145233  0.68692625 0.5045576  0.9238356 ]
  - mIoU: 0.7746
  - MCC: 0.8998
Calculando métricas de validación...
IoU por clase : [0.94939007 0.73298487 0.80263616 0.68334825 0.44224351 0.9137352 ]
Dice por clase: [0.97403807 0.84592184 0.89051377 0.81189172 0.61327162 0.95492333]
mIoU macro = 0.7541 | Dice macro = 0.8484
Matthews Correlation Coefficient (MCC) = 0.8877
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7541 | Dice: 0.8484  →  guardando modelo…

--- Epoch 54/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.67it/s, loss=0.146]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9773844  0.86277634 0.8936367  0.8143843  0.65107465 0.9577737 ]
  - IoU por clase : [0.9557692  0.758669   0.80772454 0.6868872  0.48266172 0.9189691 ]
  - mIoU: 0.7684
  - MCC: 0.8979
Calculando métricas de validación...
IoU por clase : [0.94936514 0.7407996  0.79367144 0.68729561 0.45048256 0.91257856]
Dice por clase: [0.97402495 0.85110268 0.88496859 0.81467125 0.62114854 0.95429132]
mIoU macro = 0.7557 | Dice macro = 0.8500
Matthews Correlation Coefficient (MCC) = 0.8876
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7557 | Dice: 0.8500  →  guardando modelo…

--- Epoch 55/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.94it/s, loss=0.208]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9773548  0.8632462  0.9013018  0.81772125 0.69544417 0.9608822 ]
  - IoU por clase : [0.95571256 0.75939596 0.8203361  0.6916485  0.5330888  0.92470956]
  - mIoU: 0.7808
  - MCC: 0.9012
Calculando métricas de validación...
IoU por clase : [0.95039645 0.72425203 0.80240412 0.69516582 0.45973712 0.91307399]
Dice por clase: [0.97456745 0.84007676 0.89037094 0.82017441 0.62989029 0.95456213]
mIoU macro = 0.7575 | Dice macro = 0.8516
Matthews Correlation Coefficient (MCC) = 0.8891
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7575 | Dice: 0.8516  →  guardando modelo…

--- Epoch 56/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.97it/s, loss=0.0984]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9775813  0.86655223 0.9039819  0.82011557 0.69650835 0.96100163]
  - IoU por clase : [0.9561459  0.76452774 0.8247874  0.6950813  0.5343405  0.92493093]
  - mIoU: 0.7833
  - MCC: 0.9023
Calculando métricas de validación...
IoU por clase : [0.94998236 0.72932961 0.78934592 0.67951034 0.46686536 0.91252733]
Dice por clase: [0.97434969 0.84348248 0.88227314 0.80917673 0.63654834 0.95426331]
mIoU macro = 0.7546 | Dice macro = 0.8500
Matthews Correlation Coefficient (MCC) = 0.8873
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 57/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.31it/s, loss=0.216]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9778661  0.87562585 0.9050228  0.82126874 0.70481706 0.9609239 ]
  - IoU por clase : [0.9566908  0.7787673  0.82652205 0.69673955 0.5441834  0.924787  ]
  - mIoU: 0.7879
  - MCC: 0.9040
Calculando métricas de validación...
IoU por clase : [0.9501877  0.74768179 0.79086904 0.68924043 0.4679358  0.91336651]
Dice por clase: [0.97445769 0.85562692 0.88322376 0.81603591 0.6375426  0.95472196]
mIoU macro = 0.7599 | Dice macro = 0.8536
Matthews Correlation Coefficient (MCC) = 0.8890
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7599 | Dice: 0.8536  →  guardando modelo…

--- Epoch 58/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.91it/s, loss=0.111]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97782594 0.87291616 0.9046153  0.8233608  0.7079682  0.96085966]
  - IoU por clase : [0.95661396 0.7744909  0.82584256 0.69975644 0.5479495  0.92466784]
  - mIoU: 0.7882
  - MCC: 0.9038
Calculando métricas de validación...
IoU por clase : [0.94983671 0.74181168 0.7875736  0.69533845 0.48505633 0.91176645]
Dice por clase: [0.97427308 0.85177025 0.88116495 0.82029455 0.65324974 0.95384711]
mIoU macro = 0.7619 | Dice macro = 0.8558
Matthews Correlation Coefficient (MCC) = 0.8892
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7619 | Dice: 0.8558  →  guardando modelo…

--- Epoch 59/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.61it/s, loss=0.154]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97794735 0.87533385 0.9053759  0.8310709  0.7313464  0.9614586 ]
  - IoU por clase : [0.95684624 0.7783055  0.8271113  0.7109678  0.5764745  0.92577803]
  - mIoU: 0.7959
  - MCC: 0.9056
Calculando métricas de validación...
IoU por clase : [0.95080073 0.73633053 0.79364032 0.68831567 0.48476886 0.91327005]
Dice por clase: [0.97477996 0.84814557 0.88494924 0.81538741 0.652989   0.95466926]
mIoU macro = 0.7612 | Dice macro = 0.8552
Matthews Correlation Coefficient (MCC) = 0.8898
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 60/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.52it/s, loss=0.121]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97801703 0.87513673 0.90398484 0.83101106 0.7303088  0.96025693]
  - IoU por clase : [0.95697993 0.7779939  0.82479227 0.7108802  0.5751861  0.9235522 ]
  - mIoU: 0.7949
  - MCC: 0.9051
Calculando métricas de validación...
IoU por clase : [0.95068429 0.74589715 0.78950693 0.69002986 0.49471887 0.91304647]
Dice por clase: [0.97471876 0.85445715 0.88237371 0.81658896 0.66195575 0.95454709]
mIoU macro = 0.7640 | Dice macro = 0.8574
Matthews Correlation Coefficient (MCC) = 0.8903
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7640 | Dice: 0.8574  →  guardando modelo…

--- Epoch 61/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.78it/s, loss=0.103]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9779938  0.88022834 0.9037156  0.82890844 0.712612   0.9614707 ]
  - IoU por clase : [0.9569354  0.7860784  0.82434416 0.70780843 0.5535332  0.9258002 ]
  - mIoU: 0.7924
  - MCC: 0.9052
Calculando métricas de validación...
IoU por clase : [0.95094767 0.75375837 0.78817195 0.67627181 0.52232955 0.9137666 ]
Dice por clase: [0.97485718 0.85959204 0.88153933 0.80687607 0.68622401 0.95494048]
mIoU macro = 0.7675 | Dice macro = 0.8607
Matthews Correlation Coefficient (MCC) = 0.8902
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7675 | Dice: 0.8607  →  guardando modelo…

--- Epoch 62/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.09it/s, loss=0.126]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9781836  0.875902   0.9056047  0.82858795 0.7332668  0.96136034]
  - IoU por clase : [0.9572989 0.7792043 0.8274932 0.7073412 0.5788644 0.9255955]
  - mIoU: 0.7960
  - MCC: 0.9059
Calculando métricas de validación...
IoU por clase : [0.95098301 0.75727896 0.78413331 0.66117771 0.48434135 0.91330551]
Dice por clase: [0.97487575 0.86187678 0.87900754 0.79603489 0.65260103 0.95468863]
mIoU macro = 0.7585 | Dice macro = 0.8532
Matthews Correlation Coefficient (MCC) = 0.8878
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 63/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.39it/s, loss=0.0912]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9784995  0.8801577  0.9085593  0.83364516 0.7449708  0.96220523]
  - IoU por clase : [0.9579039  0.78596574 0.8324404  0.71474403 0.5935885  0.92716324]
  - mIoU: 0.8020
  - MCC: 0.9081
Calculando métricas de validación...
IoU por clase : [0.95157697 0.75641761 0.79528981 0.68895224 0.5230642  0.91435445]
Dice por clase: [0.97518774 0.86131863 0.88597374 0.8158339  0.68685772 0.9552614 ]
mIoU macro = 0.7716 | Dice macro = 0.8634
Matthews Correlation Coefficient (MCC) = 0.8922
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7716 | Dice: 0.8634  →  guardando modelo…

--- Epoch 64/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.28it/s, loss=0.142]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9781951  0.8804955  0.9073785  0.8275693  0.75080746 0.9617211 ]
  - IoU por clase : [0.9573207 0.7865047 0.83046   0.7058577 0.6010342 0.9262647]
  - mIoU: 0.8012
  - MCC: 0.9069
Calculando métricas de validación...
IoU por clase : [0.95154623 0.75417439 0.79141394 0.68615172 0.4969308  0.91542696]
Dice por clase: [0.9751716  0.8598625  0.88356345 0.81386711 0.66393289 0.95584638]
mIoU macro = 0.7659 | Dice macro = 0.8587
Matthews Correlation Coefficient (MCC) = 0.8914
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 65/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.44it/s, loss=0.138]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9785404  0.88652515 0.9098588  0.8401882  0.7780945  0.96260166]
  - IoU por clase : [0.9579825 0.7961788 0.8346248 0.7244177 0.6367878 0.9278998]
  - mIoU: 0.8130
  - MCC: 0.9102
Calculando métricas de validación...
IoU por clase : [0.95140832 0.74935681 0.79407679 0.68427765 0.51593981 0.91262558]
Dice por clase: [0.97509917 0.85672266 0.88522051 0.81254732 0.68068641 0.95431703]
mIoU macro = 0.7679 | Dice macro = 0.8608
Matthews Correlation Coefficient (MCC) = 0.8906
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 66/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.66it/s, loss=0.09]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9787219  0.88319236 0.910477   0.8384837  0.77097064 0.96277356]
  - IoU por clase : [0.95833045 0.79081875 0.83566564 0.7218872  0.62730044 0.92821944]
  - mIoU: 0.8104
  - MCC: 0.9101
Calculando métricas de validación...
IoU por clase : [0.95201287 0.75601147 0.80318097 0.68928281 0.52550187 0.91538303]
Dice por clase: [0.97541659 0.86105528 0.89084899 0.81606562 0.68895604 0.95582243]
mIoU macro = 0.7736 | Dice macro = 0.8647
Matthews Correlation Coefficient (MCC) = 0.8934
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7736 | Dice: 0.8647  →  guardando modelo…

--- Epoch 67/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.69it/s, loss=0.0875]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9783938 0.8852704 0.9076285 0.8347178 0.756717  0.960375 ]
  - IoU por clase : [0.95770144 0.7941571  0.8308789  0.7163225  0.60864425 0.92377067]
  - mIoU: 0.8052
  - MCC: 0.9080
Calculando métricas de validación...
IoU por clase : [0.9520085  0.75397033 0.79790216 0.69611458 0.52777635 0.9132162 ]
Dice por clase: [0.9754143  0.85972985 0.88759241 0.82083438 0.69090787 0.95463984]
mIoU macro = 0.7735 | Dice macro = 0.8649
Matthews Correlation Coefficient (MCC) = 0.8928
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 68/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.42it/s, loss=0.107]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97857434 0.8872062  0.9101705  0.8375727  0.7767727  0.96204334]
  - IoU por clase : [0.9580475  0.79727817 0.8351494  0.7205377  0.6350191  0.9268627 ]
  - mIoU: 0.8121
  - MCC: 0.9100
Calculando métricas de validación...
IoU por clase : [0.95183922 0.75444902 0.80420247 0.69353009 0.52463075 0.91498002]
Dice por clase: [0.97532544 0.86004097 0.89147696 0.81903486 0.68820697 0.95560268]
mIoU macro = 0.7739 | Dice macro = 0.8649
Matthews Correlation Coefficient (MCC) = 0.8936
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7739 | Dice: 0.8649  →  guardando modelo…

--- Epoch 69/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.91it/s, loss=0.0905]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97887737 0.8883814  0.9136058  0.84648794 0.7859376  0.96258646]
  - IoU por clase : [0.9586284  0.79917824 0.8409524  0.73383534 0.6473618  0.92787135]
  - mIoU: 0.8180
  - MCC: 0.9122
Calculando métricas de validación...
IoU por clase : [0.95205253 0.75363038 0.80872959 0.68840115 0.5368173  0.91412991]
Dice por clase: [0.97543741 0.85950881 0.89425152 0.81544739 0.69860913 0.95513884]
mIoU macro = 0.7756 | Dice macro = 0.8664
Matthews Correlation Coefficient (MCC) = 0.8938
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7756 | Dice: 0.8664  →  guardando modelo…

--- Epoch 70/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.79it/s, loss=0.116]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9789874 0.8859873 0.9108466 0.8443072 0.7812707 0.962328 ]
  - IoU por clase : [0.9588397  0.7953116  0.83628863 0.73056364 0.6410535  0.92739135]
  - mIoU: 0.8149
  - MCC: 0.9115
Calculando métricas de validación...
IoU por clase : [0.9521739  0.75845957 0.80526343 0.69926035 0.54877378 0.91157917]
Dice por clase: [0.97550111 0.8626409  0.89212845 0.82301732 0.70865582 0.95374461]
mIoU macro = 0.7793 | Dice macro = 0.8693
Matthews Correlation Coefficient (MCC) = 0.8945
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7793 | Dice: 0.8693  →  guardando modelo…

--- Epoch 71/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.94it/s, loss=0.139]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97918695 0.89122885 0.9159839  0.84762996 0.79602945 0.963811  ]
  - IoU por clase : [0.9592226  0.80379874 0.8449911  0.7355537  0.6611702  0.93014973]
  - mIoU: 0.8225
  - MCC: 0.9140
Calculando métricas de validación...
IoU por clase : [0.95247996 0.7629119  0.80669066 0.69204212 0.55669064 0.91517881]
Dice por clase: [0.9756617  0.86551336 0.89300363 0.81799633 0.71522322 0.95571108]
mIoU macro = 0.7810 | Dice macro = 0.8705
Matthews Correlation Coefficient (MCC) = 0.8949
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7810 | Dice: 0.8705  →  guardando modelo…

--- Epoch 72/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.64it/s, loss=0.114]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9791263  0.888403   0.91190326 0.84357136 0.79029715 0.963048  ]
  - IoU por clase : [0.9591063  0.79921323 0.8380718  0.7294625  0.65329856 0.92872965]
  - mIoU: 0.8180
  - MCC: 0.9125
Calculando métricas de validación...
IoU por clase : [0.95188598 0.74746556 0.79828847 0.70260387 0.51345063 0.91585598]
Dice por clase: [0.97534998 0.85548531 0.88783139 0.82532864 0.67851652 0.9560802 ]
mIoU macro = 0.7716 | Dice macro = 0.8631
Matthews Correlation Coefficient (MCC) = 0.8935
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 73/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:17<00:00, 11.69it/s, loss=0.19]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9789316  0.89048785 0.9080982  0.83635384 0.7713165  0.963031  ]
  - IoU por clase : [0.9587328  0.802594   0.8316666  0.7187355  0.62775856 0.92869806]
  - mIoU: 0.8114
  - MCC: 0.9107
Calculando métricas de validación...
IoU por clase : [0.95161865 0.7494794  0.80066831 0.68069543 0.53478949 0.91293657]
Dice por clase: [0.97520963 0.85680277 0.88930127 0.8100164  0.6968897  0.95448703]
mIoU macro = 0.7717 | Dice macro = 0.8638
Matthews Correlation Coefficient (MCC) = 0.8915
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 74/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.26it/s, loss=0.0913]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9790955  0.8905975  0.9127643  0.8453434  0.7944582  0.96304756]
  - IoU por clase : [0.9590471  0.8027723  0.83952755 0.73211676 0.6590051  0.9287287 ]
  - mIoU: 0.8202
  - MCC: 0.9130
Calculando métricas de validación...
IoU por clase : [0.95283692 0.76537131 0.80667774 0.68782171 0.55426066 0.91513357]
Dice por clase: [0.97584894 0.86709386 0.89299572 0.81504072 0.71321455 0.95568642]
mIoU macro = 0.7804 | Dice macro = 0.8700
Matthews Correlation Coefficient (MCC) = 0.8950
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 75/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:17<00:00, 11.74it/s, loss=0.0924]

Época de entrenamiento finalizada:
  - Dice por clase: [0.979097  0.887704  0.9108543 0.8475824 0.7926465 0.9613735]
  - IoU por clase : [0.9590499  0.79808253 0.8363016  0.73548204 0.6565157  0.9256201 ]
  - mIoU: 0.8185
  - MCC: 0.9122
Calculando métricas de validación...
IoU por clase : [0.95278111 0.76146313 0.80455429 0.71068577 0.58091736 0.91523445]
Dice por clase: [0.97581967 0.86458027 0.89169309 0.83087821 0.73491173 0.95574143]
mIoU macro = 0.7876 | Dice macro = 0.8756
Matthews Correlation Coefficient (MCC) = 0.8970
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7876 | Dice: 0.8756  →  guardando modelo…

--- Epoch 76/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.73it/s, loss=0.11]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97929    0.89269406 0.9149513  0.8489389  0.7979676  0.9632928 ]
  - IoU por clase : [0.9594205  0.80618554 0.8432353  0.73752725 0.6638487  0.92918503]
  - mIoU: 0.8232
  - MCC: 0.9142
Calculando métricas de validación...
IoU por clase : [0.95331265 0.76693387 0.81290676 0.68783454 0.53330914 0.91580503]
Dice por clase: [0.97609837 0.86809572 0.8967993  0.81504972 0.69563159 0.95605243]
mIoU macro = 0.7784 | Dice macro = 0.8680
Matthews Correlation Coefficient (MCC) = 0.8954
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 77/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:17<00:00, 11.75it/s, loss=0.139]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9793546  0.8929184  0.91438943 0.8500041  0.79515916 0.9634212 ]
  - IoU por clase : [0.9595444  0.80655164 0.8422812  0.73913664 0.6599703  0.929424  ]
  - mIoU: 0.8228
  - MCC: 0.9143
Calculando métricas de validación...
IoU por clase : [0.95324219 0.76650912 0.81002705 0.70380116 0.55639804 0.9161545 ]
Dice por clase: [0.97606144 0.86782357 0.89504414 0.82615411 0.71498168 0.95624283]
mIoU macro = 0.7844 | Dice macro = 0.8727
Matthews Correlation Coefficient (MCC) = 0.8970
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 78/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.36it/s, loss=0.15]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97927415 0.8877187  0.9119437  0.84759045 0.7984605  0.9608585 ]
  - IoU por clase : [0.95939    0.79810625 0.83814013 0.7354941  0.6645312  0.92466563]
  - mIoU: 0.8201
  - MCC: 0.9126
Calculando métricas de validación...
IoU por clase : [0.95275606 0.77022022 0.81590684 0.7031305  0.55426752 0.91526512]
Dice por clase: [0.97580653 0.87019707 0.89862191 0.82569187 0.71322023 0.95575814]
mIoU macro = 0.7853 | Dice macro = 0.8732
Matthews Correlation Coefficient (MCC) = 0.8965
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 79/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.62it/s, loss=0.0808]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97935396 0.8910066  0.9149712  0.8513995  0.8044262  0.96397495]
  - IoU por clase : [0.9595434  0.80343723 0.84326905 0.7412494  0.6728369  0.93045527]
  - mIoU: 0.8251
  - MCC: 0.9148
Calculando métricas de validación...
IoU por clase : [0.95315294 0.76556787 0.81563483 0.70392551 0.56510859 0.91530553]
Dice por clase: [0.97601465 0.86721998 0.89845691 0.82623977 0.72213339 0.95578018]
mIoU macro = 0.7864 | Dice macro = 0.8743
Matthews Correlation Coefficient (MCC) = 0.8974
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 80/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.38it/s, loss=0.0799]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9795701 0.8941336 0.9180725 0.8552674 0.8064503 0.9643112]
  - IoU por clase : [0.9599585  0.80853677 0.8485527  0.7471329  0.6756738  0.931082  ]
  - mIoU: 0.8285
  - MCC: 0.9162
Calculando métricas de validación...
IoU por clase : [0.95297038 0.76753102 0.80439138 0.71378522 0.54925431 0.91608531]
Dice por clase: [0.97591893 0.86847813 0.89159302 0.83299262 0.70905636 0.95620514]
mIoU macro = 0.7840 | Dice macro = 0.8724
Matthews Correlation Coefficient (MCC) = 0.8970
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 81/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.34it/s, loss=0.11]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9795158  0.89583766 0.91431594 0.8497688  0.81416476 0.9636425 ]
  - IoU por clase : [0.9598539  0.8113279  0.8421566  0.7387809  0.68657494 0.92983603]
  - mIoU: 0.8281
  - MCC: 0.9154
Calculando métricas de validación...
IoU por clase : [0.95317609 0.76912378 0.8153515  0.69987007 0.5799869  0.91619179]
Dice por clase: [0.97602679 0.86949685 0.89828499 0.82343949 0.73416672 0.95626314]
mIoU macro = 0.7890 | Dice macro = 0.8763
Matthews Correlation Coefficient (MCC) = 0.8980
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7890 | Dice: 0.8763  →  guardando modelo…

--- Epoch 82/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.15it/s, loss=0.0858]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9796543  0.8951823  0.91681814 0.8520524  0.81515896 0.9644575 ]
  - IoU por clase : [0.96011996 0.81025344 0.846412   0.7422398  0.6879901  0.9313549 ]
  - mIoU: 0.8297
  - MCC: 0.9163
Calculando métricas de validación...
IoU por clase : [0.95355865 0.77157073 0.80851118 0.70717129 0.59051424 0.91607249]
Dice por clase: [0.97622731 0.87105834 0.89411798 0.8284714  0.74254506 0.95619815]
mIoU macro = 0.7912 | Dice macro = 0.8781
Matthews Correlation Coefficient (MCC) = 0.8984
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7912 | Dice: 0.8781  →  guardando modelo…

--- Epoch 83/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.82it/s, loss=0.0764]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97953445 0.8946324  0.91638374 0.85279685 0.8091956  0.96352994]
  - IoU por clase : [0.9598897  0.8093528  0.84567183 0.7433704  0.67953694 0.92962635]
  - mIoU: 0.8279
  - MCC: 0.9157
Calculando métricas de validación...
IoU por clase : [0.95368134 0.75867605 0.80282793 0.69958532 0.58765963 0.91616295]
Dice por clase: [0.9762916  0.86278089 0.89063179 0.82324236 0.74028415 0.95624743]
mIoU macro = 0.7864 | Dice macro = 0.8749
Matthews Correlation Coefficient (MCC) = 0.8973
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 84/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.51it/s, loss=0.107]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9797057 0.8947344 0.910575  0.8536309 0.8037774 0.963963 ]
  - IoU por clase : [0.9602187  0.8095197  0.83583075 0.7446388  0.6719296  0.930433  ]
  - mIoU: 0.8254
  - MCC: 0.9153
Calculando métricas de validación...
IoU por clase : [0.95306748 0.7667825  0.82266132 0.68870005 0.56929049 0.91540641]
Dice por clase: [0.97596984 0.86799875 0.90270344 0.81565705 0.7255387  0.95583518]
mIoU macro = 0.7860 | Dice macro = 0.8740
Matthews Correlation Coefficient (MCC) = 0.8965
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 85/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.49it/s, loss=0.0917]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9797134  0.8988183  0.9186171  0.85590845 0.82260114 0.9644544 ]
  - IoU por clase : [0.9602334  0.8162307  0.8494837  0.7481119  0.69865966 0.9313491 ]
  - mIoU: 0.8340
  - MCC: 0.9175
Calculando métricas de validación...
IoU por clase : [0.95336848 0.7595292  0.81559687 0.71262333 0.59774355 0.9146657 ]
Dice por clase: [0.97612764 0.8633323  0.89843388 0.83220089 0.74823466 0.95543123]
mIoU macro = 0.7923 | Dice macro = 0.8790
Matthews Correlation Coefficient (MCC) = 0.8988
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7923 | Dice: 0.8790  →  guardando modelo…

--- Epoch 86/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.10it/s, loss=0.109]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9798577  0.8987593  0.9198732  0.85713655 0.817103   0.96498036]
  - IoU por clase : [0.96051097 0.8161334  0.85163444 0.74999034 0.6907643  0.9323304 ]
  - mIoU: 0.8336
  - MCC: 0.9180
Calculando métricas de validación...
IoU por clase : [0.95325962 0.7686008  0.81567807 0.7135212  0.61226048 0.91604383]
Dice por clase: [0.97607057 0.86916256 0.89848314 0.83281281 0.75950566 0.95618254]
mIoU macro = 0.7966 | Dice macro = 0.8820
Matthews Correlation Coefficient (MCC) = 0.8996
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
🔹 Nuevo mejor mIoU: 0.7966 | Dice: 0.8820  →  guardando modelo…

--- Epoch 87/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:19<00:00, 10.63it/s, loss=0.0797]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97995377 0.8973735  0.9193775  0.85674125 0.81966233 0.96478075]
  - IoU por clase : [0.96069556 0.8138509  0.85078514 0.7493853  0.6944304  0.9319578 ]
  - mIoU: 0.8335
  - MCC: 0.9180
Calculando métricas de validación...
IoU por clase : [0.95321823 0.77403906 0.81782689 0.70223579 0.57652103 0.91559933]
Dice por clase: [0.97604888 0.87262911 0.89978523 0.82507464 0.73138387 0.95594033]
mIoU macro = 0.7899 | Dice macro = 0.8768
Matthews Correlation Coefficient (MCC) = 0.8978
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 88/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.14it/s, loss=0.0759]

Época de entrenamiento finalizada:
  - Dice por clase: [0.98003876 0.8991665  0.9209189  0.861079   0.82807654 0.96381146]
  - IoU por clase : [0.9608588  0.8168052  0.8534288  0.756048   0.70659614 0.93015075]
  - mIoU: 0.8373
  - MCC: 0.9187
Calculando métricas de validación...
IoU por clase : [0.95361054 0.77123411 0.81427403 0.71901973 0.60541955 0.91471258]
Dice por clase: [0.9762545  0.87084379 0.8976307  0.83654622 0.75421973 0.9554568 ]
mIoU macro = 0.7964 | Dice macro = 0.8818
Matthews Correlation Coefficient (MCC) = 0.8999
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 89/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.37it/s, loss=0.104]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9799827 0.9007799 0.9201669 0.8573235 0.8256149 0.9647273]
  - IoU por clase : [0.96075106 0.81947184 0.8521381  0.75027674 0.7030189  0.9318583 ]
  - mIoU: 0.8363
  - MCC: 0.9186
Calculando métricas de validación...
IoU por clase : [0.95385548 0.77078515 0.81640673 0.708236   0.59817915 0.91621024]
Dice por clase: [0.97638284 0.8705575  0.89892502 0.82920158 0.74857584 0.95627319]
mIoU macro = 0.7939 | Dice macro = 0.8800
Matthews Correlation Coefficient (MCC) = 0.8993
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 90/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.42it/s, loss=0.0979]

Época de entrenamiento finalizada:
  - Dice por clase: [0.97999126 0.8997648  0.9185553  0.86048836 0.8316193  0.9646997 ]
  - IoU por clase : [0.96076757 0.81779313 0.849378   0.75513786 0.7117709  0.93180656]
  - mIoU: 0.8378
  - MCC: 0.9187
Calculando métricas de validación...
IoU por clase : [0.95407228 0.77378754 0.8214004  0.71155158 0.59131867 0.91726585]
Dice por clase: [0.97649641 0.87246925 0.9019438  0.83146963 0.74318071 0.95684785]
mIoU macro = 0.7949 | Dice macro = 0.8804
Matthews Correlation Coefficient (MCC) = 0.9001
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:243: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)

--- Epoch 91/200 ---
Calculando métricas de entrenamiento...
100% 210/210 [00:18<00:00, 11.25it/s, loss=0.0667]

Época de entrenamiento finalizada:
  - Dice por clase: [0.9800041  0.89342237 0.91526824 0.8586933  0.8331697  0.96430933]
  - IoU por clase : [0.96079236 0.8073743  0.8437738  0.7523774  0.7140453  0.93107873]
  - mIoU: 0.8349
"""
# 2) Ticks arbitrarios y etiquetas de texto libre
plot_mcc_evolution(
    log_text,
    epoch_ticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
    epoch_ticklabels=['Inicio', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180']
)