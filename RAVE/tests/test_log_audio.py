import gin

from rave.core import bind_log_audio_every_n_steps, should_log_audio


class _Module:
    pass


def test_should_log_audio_first_then_throttle():
    module = _Module()
    assert should_log_audio(module, "audio_val", 500, 1000)
    assert not should_log_audio(module, "audio_val", 999, 1000)
    assert not should_log_audio(module, "audio_val", 1499, 1000)
    assert should_log_audio(module, "audio_val", 1500, 1000)


def test_should_log_audio_keys_are_independent():
    module = _Module()
    assert should_log_audio(module, "val/audio_x", 500, 1000)
    assert should_log_audio(module, "val/audio_y", 500, 1000)
    assert not should_log_audio(module, "val/audio_x", 999, 1000)
    assert not should_log_audio(module, "val/audio_y", 999, 1000)


def test_should_log_audio_zero_logs_every_call():
    module = _Module()
    assert should_log_audio(module, "audio_val", 1, 0)
    assert should_log_audio(module, "audio_val", 2, 0)


def test_should_log_audio_skips_sanity_checking():
    class _Stage:
        def __str__(self):
            return "RunningStage.SANITY_CHECKING"

    class _Trainer:
        class state:
            stage = _Stage()

    module = _Module()
    module.trainer = _Trainer()
    assert not should_log_audio(module, "audio_val", 0, 1000)
    assert not should_log_audio(module, "audio_val", 0, 0)


def test_gin_binds_log_audio_every_n_steps():
    gin.clear_config()
    gin.parse_config("core.log_audio.every_n_steps = 25000")
    assert int(gin.query_parameter("core.log_audio.every_n_steps")) == 25000
    gin.finalize()
    bind_log_audio_every_n_steps(5000)
    assert int(gin.query_parameter("core.log_audio.every_n_steps")) == 5000
    gin.clear_config()
