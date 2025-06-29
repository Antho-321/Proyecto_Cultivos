# copypaste_class4.py
import random
import numpy as np
import albumentations as A

class CopyPasteClass4(A.DualTransform):
    """
    Copia todos los píxeles con valor 4 de la máscara, recorta el parche mínimo
    que los contiene y lo pega N veces en posiciones aleatorias de la misma imagen
    (in-place).  Preserva la coherencia máscara-imagen.

    Args
    ----
    num_pastes : int
        Nº de veces que se pegará el parche en cada llamada.
    prob_flip   : float
        Probabilidad de hacer un flip horizontal/vertical al parche antes de pegar.
    always_apply : bool
    p            : float
        Probabilidad global de aplicar la transformación (por defecto 1.0).
    """
    def __init__(self,
                 num_pastes: int = 3,
                 prob_flip: float = 0.5,
                 always_apply: bool = False,
                 p: float = 1.0):
        super().__init__(always_apply, p)
        self.num_pastes = num_pastes
        self.prob_flip  = prob_flip

    # ------------- implementación interna -----------------
    @staticmethod
    def _extract_patch(img: np.ndarray, msk: np.ndarray):
        ys, xs = np.where(msk == 4)
        if len(xs) == 0:               # la clase 4 no está presente
            return None
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        return img[y0:y1, x0:x1].copy(), msk[y0:y1, x0:x1].copy()

    @staticmethod
    def _random_pos(h_img, w_img, h_patch, w_patch):
        """Devuelve una coordenada (top, left) válida dentro de la imagen."""
        if h_patch >= h_img or w_patch >= w_img:
            return None
        top  = random.randint(0, h_img - h_patch)
        left = random.randint(0, w_img - w_patch)
        return top, left

    # ------------- métodos heredados de Albumentations ----
    def apply(self, img, patch=None, top=None, left=None, **params):
        # Pega el parche en la imagen
        if patch is not None and top is not None:
            h, w = patch.shape[:2]
            img[top:top + h, left:left + w] = patch
        return img

    def apply_to_mask(self, mask, patch=None, top=None, left=None, **params):
        if patch is not None and top is not None:
            h, w = patch.shape[:2]
            mask[top:top + h, left:left + w] = patch
        return mask

    # ------------- punto de entrada -----------------------
    def get_params_dependent_on_targets(self, params):
        """
        - Extrae el parche con clase 4.
        - Genera coordenadas y, opcionalmente, flippea el parche.
        """
        image, mask = params["image"], params["mask"]
        patch_data = self._extract_patch(image, mask)

        if patch_data is None:         # no hay píxeles de clase 4
            return {"patches": [], "tops": [], "lefts": []}

        img_h, img_w = image.shape[:2]
        patch_img, patch_msk = patch_data
        # Aplicar flip aleatorio al parche
        if random.random() < self.prob_flip:
            if random.random() < 0.5:                        # horizontal
                patch_img = patch_img[:, ::-1]
                patch_msk = patch_msk[:, ::-1]
            else:                                            # vertical
                patch_img = patch_img[::-1, :]
                patch_msk = patch_msk[::-1, :]

        h_p, w_p = patch_img.shape[:2]
        patches, tops, lefts = [], [], []

        for _ in range(self.num_pastes):
            pos = self._random_pos(img_h, img_w, h_p, w_p)
            if pos is None:
                break
            top, left = pos
            patches.append((patch_img, patch_msk))
            tops.append(top)
            lefts.append(left)

        return {"patches": patches, "tops": tops, "lefts": lefts}

    def apply_with_params(self, params, **kwargs):
        """
        Llama repetidamente a `apply`/`apply_to_mask` tantas veces como parches
        se hayan generado.
        """
        img   = kwargs["image"]
        mask  = kwargs["mask"]
        for (p_img, p_msk), t, l in zip(params["patches"],
                                        params["tops"],
                                        params["lefts"]):
            img  = self.apply(img,  patch=p_img, top=t, left=l)
            mask = self.apply_to_mask(mask, patch=p_msk, top=t, left=l)
        return {"image": img, "mask": mask}

    @property
    def targets_as_params(self):
        # Queremos tanto imagen como máscara para generar los parches
        return ["image", "mask"]

    def get_transform_init_args_names(self):
        return ("num_pastes", "prob_flip", "always_apply", "p")
