import numpy as np

from rave.dataset import build_audio_transforms
from rave.transforms import RandomApply, RandomCrop, RandomGain, RandomTargetLoudness, StartCrop


def test_start_crop_is_prefix():
    x = np.arange(10, dtype=np.float32)[None]
    y = StartCrop(4)(x)
    np.testing.assert_array_equal(y, x[..., :4])


def test_train_transforms_include_phase_and_loudness():
    augs = [RandomTargetLoudness(loudness_range=(-30, -20), prob=1.0)]
    t = build_audio_transforms(
        1024,
        sr_dataset=44100,
        sr=44100,
        augmentations=augs,
        stochastic=True,
    )
    kinds = [type(x) for x in t.transform_list]
    assert RandomCrop in kinds
    assert RandomApply in kinds
    assert RandomTargetLoudness in kinds
    assert StartCrop not in kinds


def test_val_transforms_skip_phase_and_loudness():
    augs = [RandomTargetLoudness(loudness_range=(-30, -20), prob=1.0)]
    t = build_audio_transforms(
        1024,
        sr_dataset=44100,
        sr=44100,
        augmentations=augs,
        stochastic=False,
    )
    kinds = [type(x) for x in t.transform_list]
    assert StartCrop in kinds
    assert RandomCrop not in kinds
    assert RandomApply not in kinds
    assert RandomTargetLoudness not in kinds
    assert RandomGain not in kinds