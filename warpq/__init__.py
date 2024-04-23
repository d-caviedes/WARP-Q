# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Content copied and modified from
# https://github.com/wjassim/WARP-Q/blob/main/WARPQ/WARPQmetric.py
import pickle
from pathlib import Path

import keras
import librosa
import numpy as np
import pandas as pd
import speechpy
from pyvad import vad
from skimage.util.shape import view_as_windows
from torch import Tensor
from typing_extensions import Literal
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="sklearn")


def warpq(
    preds: Tensor,
    target: Tensor,
    fs: int,
    n_mfcc: int = 12,
    lifter: int = 3,
    metric: str = "euclidean",
    fmax: int = 5000,
    patch_size: float = 0.4,
    sigma: Tensor = None,
    band_rad: float = 0.25,
    weights_mul: Tensor = None,
    apply_vad: bool = True,
    hop_size_vad: int = 30,
    aggresive: int = 0,
    keep_same_device: bool = False,
    net_type: str = "RF_Genspeech",
) -> Tensor:
    r"""Calculate `WARP-Q`_ (WARPQ).

    WARP-Q a full-reference objective speech quality metric that uses dynamic time
    warping cost for MFCC speech representations. It is robust to small perceptual signal changes.
    It is originally developed for quality prediction for generative neural speech codecs.
    This code is translated from `https://github.com/wjassim/WARP-Q`.

    .. note:: using this metric requires you to have ``scikit-learn>=1.1.1 <=1.2.2``, ``pyvad``, ``speechpy`` and
        ``scikit-image`` installed. Install as ``pip install torchmetrics[audio]``.

    .. note::
        This implementation has been uptaded for speech files that are too short. It parses NaN values to 0
        in order to avoid errors in the MOS score mappging.

    Args:
        preds: float tensor with shape ``(1,time)``
        target: float tensor with shape ``(1,time)``
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        n_mfcc: integer specifying the number of MFCC features to extract
        fmax: integer specifying the maximum frequency to include in the mel spectrum
        patch_size: float specifying the size of the patch in seconds
        sigma: array of shape (3, 2) specifying the step sizes and weights for the DTW algorithm
        apply_vad: boolean specifying whether to apply voice activity detection (VAD) to the signals
        hop_size_vad: integer specifying the hop size for the VAD algorithm
        aggresive: integer specifying the aggresiveness of the VAD algorithm
        lifter: integer specifying the lifter parameter for the MFCC algorithm
        metric: string specifying the metric to use for the DTW algorithm
        band_rad: float specifying the band radius for the DTW algorithm
        weights_mul: array of shape (3,) specifying the weights for the DTW algorithm
        keep_same_device: whether to move the pesq value to the device of preds
        net_type: string specifying the type of the model to use for mapping the raw WARP-Q scores onto MOS

    Returns:
        A tuples of tensors ``(Tensor, Tensor)`` with the WARPQ value and its MOS equivalent

    Raises:
        ValueError:
            If ``fs`` is not either  ``8000`` or ``16000``
        RuntimeError:
            If ``preds`` and ``target`` do not have the same shape

    Example:
        >>> from torch import randn
        >>> from torchmetrics.functional.audio.warpq import warpq
        >>> import torch
        >>> g = torch.manual_seed(1)
        >>> preds = randn(8000)
        >>> target = randn(8000)
        >>> warpq(preds, target, fs=16000)
        (tensor(1.4610), tensor(4.2980))

    """
    if fs not in (8000, 16000):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")

    if preds.shape != target.shape:
        raise RuntimeError("Expected `preds` and `target` to have the same shape")
    if len(preds.shape) > 1:
        if preds.shape[0] != 1:
            raise RuntimeError("Expected `preds` and `target` to be Mono")
        else:
            preds = preds.squeeze(0)
            target = target.squeeze(0)
    if sigma is None:
        sigma = Tensor([[1, 1], [3, 2], [1, 3]])
    if weights_mul is None:
        weights_mul = Tensor([1, 1, 1])

    _warpq_arg_validate(
        fs=fs,
        n_mfcc=n_mfcc,
        lifter=lifter,
        metric=metric,
        fmax=fmax,
        patch_size=patch_size,
        sigma=sigma,
        band_rad=band_rad,
        weights_mul=weights_mul,
        apply_vad=apply_vad,
        hop_size_vad=hop_size_vad,
        aggresive=aggresive,
        keep_same_device=keep_same_device,
        net_type=net_type,
    )
    target[target > 1] = 1.0
    target[target < -1] = -1.0

    preds[preds > 1] = 1.0
    preds[preds < -1] = -1.0
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    win_length = int(0.032 * fs)  # 32 ms frame
    hop_length = int(0.004 * fs)  # 4 ms overlap
    n_fft = 2 * win_length

    if apply_vad:
        # VAD for Ref speech
        vact1 = vad(
            target,
            fs,
            fs_vad=fs,
            hop_length=hop_size_vad,
            vad_mode=aggresive,
        )

        target = target[vact1 == 1]

        # VAD for Coded speech
        vact2 = vad(
            preds,
            fs,
            fs_vad=fs,
            hop_length=hop_size_vad,
            vad_mode=aggresive,
        )

        preds = preds[vact2 == 1]

    # Compute MFCC features for the two signals
    mfcc_ref = librosa.feature.mfcc(
        y=target,
        sr=fs,
        n_mfcc=n_mfcc,
        fmax=fmax,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        lifter=lifter,
    )

    mfcc_coded = librosa.feature.mfcc(
        y=preds,
        sr=fs,
        n_mfcc=n_mfcc,
        fmax=fmax,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        lifter=lifter,
    )

    # Feature Normalisation using CMVNW method
    mfcc_ref = speechpy.processing.cmvnw(mfcc_ref.T, win_size=201, variance_normalization=True).T

    mfcc_coded = speechpy.processing.cmvnw(mfcc_coded.T, win_size=201, variance_normalization=True).T

    # Divide MFCC features of Coded speech into patches
    cols = int(patch_size / (hop_length / fs))
    window_shape = (np.size(mfcc_ref, 0), cols)
    step = int(cols / 2)

    while window_shape[-1] > mfcc_coded.shape[-1]:
        mfcc_coded = np.append(mfcc_coded, mfcc_coded, axis=-1)

    mfcc_coded_patch = view_as_windows(mfcc_coded, window_shape, step)

    acc = []

    # Compute alignment cost between each patch and Ref MFCC
    for i in range(mfcc_coded_patch.shape[1]):
        patch = mfcc_coded_patch[0][i]
        cost = _compute_alignment_cost(
            patch,
            mfcc_ref,
            sigma=sigma,
            band_rad=band_rad,
            metric=metric,
            weights_mul=weights_mul,
        )
        acc.append(cost)
    mos_score = _warpq_to_mos_score(acc, net_type, keep_same_device)
    # Final score
    score = Tensor(np.array(round(np.median(acc), 3)))
    mos_score = Tensor(np.array(round(np.median(mos_score), 3)))
    if keep_same_device:
        return score.to(preds.device), mos_score.to(preds.device)
    # Return scores
    return score, mos_score


def _warpq_to_mos_score(score: Tensor, net_type: str, keep_same_device: bool) -> Tensor:
    r"""Map raw WARP-Q scores onto MOS.

    For more details, please see Section 8 of our paper
    "W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
    with WARP‐Q: From similarity to subsequence dynamic time warp cost,
    IET Signal Processing, 1– 21 (2022)". Available on:
    https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/sil2.12151
    """
    fea = _get_feaures_for_score_mapping(score)
    mapping_model, data_scaler = _load_model(net_type)
    fea_scaled = data_scaler.transform(fea)
    mapped_score = (
        mapping_model.predict(fea_scaled, verbose=0) if "NN" in net_type else mapping_model.predict(fea_scaled)
    )
    mapped_score = Tensor(np.array(round(mapped_score.item(), 3)))
    if mapped_score > 5:
        mapped_score = 5
    if mapped_score < 1:
        mapped_score = 1
    if keep_same_device:
        return mapped_score.to(score.device)
    return mapped_score


def _get_feaures_for_score_mapping(acc: np.array) -> pd.DataFrame:
    r"""Extract features from alignment costs vector to map raw WARP-Q scores onto MOS.

    For more details, please see Section 8 of our paper
    "W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
    with WARP‐Q: From similarity to subsequence dynamic time warp cost,
    IET Signal Processing, 1– 21 (2022)". Available on:
    https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/sil2.12151
    """
    acc_df = pd.DataFrame({"acc": acc})

    acc_fea = dict()
    acc_fea["warpq_count"] = acc_df["acc"].count()
    acc_fea["warpq_mean"] = acc_df["acc"].mean()
    acc_fea["warpq_median"] = acc_df["acc"].median()
    acc_fea["warpq_var"] = acc_df["acc"].var()
    acc_fea["warpq_std"] = acc_df["acc"].std()
    acc_fea["warpq_min"] = acc_df["acc"].min()
    acc_fea["warpq_max"] = acc_df["acc"].max()

    quantile = acc_df.quantile([0.25, 0.5, 0.75])

    acc_fea["warpq_25%"] = quantile.iloc[0, 0]
    acc_fea["warpq_50%"] = quantile.iloc[1, 0]
    acc_fea["warpq_75%"] = quantile.iloc[2, 0]
    acc_fea["warpq_skewness"] = acc_df["acc"].skew()
    acc_fea["warpq_kurtosis"] = acc_df["acc"].kurtosis()

    return pd.DataFrame.from_dict([acc_fea]).fillna(0)


def _compute_alignment_cost(
    patch: np.array,
    mfcc_ref: np.array,
    metric: str = "euclidean",
    sigma: np.array = np.array([[1, 1], [3, 2], [1, 3]]),  # noqa: B008
    band_rad: float = 0.25,
    weights_mul: np.array = np.array([1, 1, 1]),  # noqa: B008
) -> float:
    """Compute the alignment cost between two spectral representations using subsequence DTW.

    For more details, please see Subsection 3.3 of our paper
    "W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
    with WARP-Q: From similarity to subsequence dynamic time warp cost,
    IET Signal Processing, 1- 21 (2022)". Available on:
    https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/sil2.12151.
    """
    d, p = librosa.sequence.dtw(
        X=patch,
        Y=mfcc_ref,
        metric=metric,
        step_sizes_sigma=sigma.detach().cpu().numpy().astype(int),
        weights_mul=weights_mul.detach().cpu().numpy(),
        band_rad=band_rad,
        subseq=True,
        backtrack=True,
    )
    p_librosa = p[::-1, :]
    b_ast = p_librosa[-1, 1]

    return d[-1, b_ast] / d.shape[0]


def _warpq_arg_validate(
    fs: int,
    n_mfcc: int,
    lifter: int,
    metric: str,
    fmax: int,
    patch_size: float,
    sigma: Tensor,
    band_rad: float,
    weights_mul: Tensor,
    apply_vad: bool,
    hop_size_vad: int,
    aggresive: int,
    keep_same_device: bool,
    net_type: str,
) -> None:
    """Validate the arguments for warpq.

    Args:
        fs: sampling frequency, should be 16000 or 8000 (Hz)
        n_mfcc: integer specifying the number of MFCC features to extract
        fmax: integer specifying the maximum frequency to include in the mel spectrum
        patch_size: float specifying the size of the patch in seconds
        sigma: array of shape (3, 2) specifying the step sizes and weights for the DTW algorithm
        apply_vad: boolean specifying whether to apply voice activity detection (VAD) to the signals
        hop_size_vad: integer specifying the hop size for the VAD algorithm
        aggresive: integer specifying the aggresiveness of the VAD algorithm
        lifter: integer specifying the lifter parameter for the MFCC algorithm
        metric: string specifying the metric to use for the DTW algorithm
        band_rad: float specifying the band radius for the DTW algorithm
        weights_mul: array of shape (3,) specifying the weights for the DTW algorithm
        keep_same_device: whether to move the pesq value to the device of preds
        net_type: string specifying the type of the model to use for mapping the raw WARP-Q scores onto MOS

    """
    if not (isinstance(fs, int) and (fs in (8000, 16000)) and fs > 0):
        raise ValueError(f"Expected argument `fs` to either be 8000 or 16000 but got {fs}")
    if not (isinstance(n_mfcc, int) and n_mfcc > 0):
        raise ValueError(f"Expected argument `n_mfcc` to be an int larger than 0, but got {n_mfcc}")
    if not (isinstance(lifter, int)):
        raise ValueError(f"Expected argument `lifter` to be an int, but got {lifter}")
    if not (isinstance(metric, str)):
        raise ValueError(f"Expected argument `metric` to be a string, but got {metric}")
    if not (isinstance(fmax, int) and fmax > 0):
        raise ValueError(f"Expected argument `fmax` to be an int larger than 0, but got {fmax}")
    if not (isinstance(patch_size, float) and patch_size > 0):
        raise ValueError(f"Expected argument `patch_size` to be a float larger than 0, but got {patch_size}")
    if not (isinstance(sigma, Tensor) and sigma.shape == (3, 2)):
        raise ValueError(f"Expected argument `sigma` to be an array of shape (3, 2), but got {sigma}")
    if not (isinstance(band_rad, float) and band_rad > 0):
        raise ValueError(f"Expected argument `band_rad` to be a float larger than 0, but got {band_rad}")
    if not (isinstance(weights_mul, Tensor) and weights_mul.shape == (3,)):
        raise ValueError(f"Expected argument `weights_mul` to be an array of shape (3,), but got {weights_mul}")
    if not (isinstance(apply_vad, bool)):
        raise ValueError(f"Expected argument `apply_vad` to be a boolean, but got {apply_vad}")
    if not (isinstance(hop_size_vad, int) and hop_size_vad > 0):
        raise ValueError(f"Expected argument `hop_size_vad` to be an int larger than 0, but got {hop_size_vad}")
    if not (isinstance(aggresive, int) and aggresive in (0, 1, 2, 3)):
        raise ValueError(f"Expected argument `aggresive` to be an int in (0, 1, 2, 3), but got {aggresive}")
    if not (isinstance(keep_same_device, bool)):
        raise ValueError(f"Expected argument `keep_same_device` to be a boolean, but got {keep_same_device}")
    if not (isinstance(net_type, str)) and (
        net_type
        not in [
            "RF_Genspeech",
            "RF_Genspeech_TCDVoIP_PSup23",
            "RF_PSup23",
            "RF_TCDVoIP",
            "NN_Genspeech",
            "NN_Genspeech_TCDVoIP_PSup23",
            "NN_PSup23",
            "NN_TCDVoIP",
        ]
    ):
        raise ValueError(
            f"Expected argument `net_type` to be a string in "
            f"['RF_Genspeech', 'RF_Genspeech_TCDVoIP_PSup23', 'RF_PSup23', 'RF_TCDVoIP', "
            f"'NN_Genspeech', 'NN_Genspeech_TCDVoIP_PSup23', 'NN_PSup23', 'NN_TCDVoIP'], but got {net_type}"
        )


def _load_model(
    net_type: Literal[
        "RF_Genspeech",
        "RF_Genspeech_TCDVoIP_PSup23",
        "RF_PSup23",
        "RF_TCDVoIP",
        "NN_Genspeech",
        "NN_Genspeech_TCDVoIP_PSup23",
        "NN_PSup23",
        "NN_TCDVoIP",
    ] = "RF_Genspeech",
) -> None:
    """Load trained mapping model with its data scaler."""
    # Create a temp dir
    # Extract the zip file
    # Read model
    if "NN" in net_type:
        try:
            mapping_model = keras.models.load_model(
                Path(__file__).parent.resolve() / "warpq_models" / net_type / "mapping_model.h5"
            )
        except Exception as e:
            print(e)
            exit()

    else:
        try:
            mapping_model = pickle.load(
                open(
                    (Path(__file__).parent.resolve() / "warpq_models" / net_type / "mapping_model.pkl"),
                    "rb",
                )
            )
        except Exception as e:
            print(e)
            exit()

    try:  # Read data scaler
        data_scaler = pickle.load(
            open(
                (Path(__file__).parent.resolve() / "warpq_models" / net_type / "data_scaler.pkl"),
                "rb",
            )
        )
    except Exception as e:
        print(e)
        exit()
    return mapping_model, data_scaler
