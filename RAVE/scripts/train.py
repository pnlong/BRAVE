import hashlib
import os
import sys
from typing import Any, Dict

_RAVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAVE_ROOT not in sys.path:
    sys.path.insert(0, _RAVE_ROOT)

import gin
import pytorch_lightning as pl
import torch
from absl import flags, app
from torch.utils.data import DataLoader

import rave
import rave.core
import rave.dataset
from rave.transforms import get_augmentations, add_augmentation


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_multi_string('augment',
                           default = [],
                            help = 'augmentation configurations to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     500000,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 0, help="number of audio channels")
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")
flags.DEFINE_string('wandb_project',
                    default='brave',
                    help='Weights & Biases project name')
flags.DEFINE_string('wandb_entity',
                    default=None,
                    help='Weights & Biases entity (team or user)')
flags.DEFINE_bool('wandb_offline',
                  default=False,
                  help='Log to W&B in offline mode')
flags.DEFINE_integer(
    'log_every_n_steps',
    default=None,
    help='Lightning/W&B flush interval (default: min(50, batches per epoch))',
)
flags.DEFINE_bool(
    'reject_silent',
    default=None,
    help='Enable RMS gate on train loader (default: gin maybe_reject_silent.enabled)',
)
flags.DEFINE_bool(
    'noreject_silent',
    default=False,
    help='Disable RMS gate even if enabled in gin',
)
flags.DEFINE_float(
    'reject_silent_rms_db',
    default=None,
    help='Override gin rms_db_threshold for silent rejection',
)
flags.DEFINE_integer(
    'reject_silent_max_tries',
    default=None,
    help='Override gin max_tries for silent rejection',
)


class EMA(pl.Callback):

    def __init__(self, factor=.999) -> None:
        super().__init__()
        self.weights = {}
        self.factor = factor

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        for n, p in pl_module.named_parameters():
            if n not in self.weights:
                self.weights[n] = p.data.clone()
                continue

            self.weights[n] = self.weights[n] * self.factor + p.data * (
                1 - self.factor)

    def swap_weights(self, module):
        for n, p in module.named_parameters():
            current = p.data.clone()
            p.data.copy_(self.weights[n])
            self.weights[n] = current

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def state_dict(self) -> Dict[str, Any]:
        return self.weights.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights.update(state_dict)

def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name

def parse_augmentations(augmentations):
    for a in augmentations:
        gin.parse_config_file(a)
        add_augmentation()
        gin.clear_config()
    return get_augmentations()

def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # check dataset channels
    n_channels = rave.dataset.get_training_channels(FLAGS.db_path, FLAGS.channels)
    gin.bind_parameter('RAVE.n_channels', n_channels)

    # parse augmentations
    augmentations = parse_augmentations(map(add_gin_extension, FLAGS.augment))
    gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)

    # parse configuration
    if FLAGS.ckpt:
        config_file = rave.core.search_for_config(FLAGS.ckpt)
        if config_file is None:
            print('Config file not found in %s'%FLAGS.run)
        gin.parse_config_file(config_file)
    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config),
            FLAGS.override,
        )

    model = rave.training.build_training_model(n_channels=n_channels)
    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    # parse datasset
    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       derivative=FLAGS.derivative,
                                       normalize=FLAGS.normalize,
                                       rand_pitch=FLAGS.rand_pitch,
                                       n_channels=n_channels)
    train, val = rave.dataset.split_dataset(dataset, 98)

    reject_kwargs = {}
    if FLAGS.noreject_silent:
        reject_kwargs['enabled'] = False
    elif FLAGS.reject_silent is not None:
        reject_kwargs['enabled'] = FLAGS.reject_silent
    if FLAGS.reject_silent_rms_db is not None:
        reject_kwargs['rms_db_threshold'] = FLAGS.reject_silent_rms_db
    if FLAGS.reject_silent_max_tries is not None:
        reject_kwargs['max_tries'] = FLAGS.reject_silent_max_tries
    train = rave.dataset.maybe_reject_silent(train, **reject_kwargs)

    train, val = rave.training.wrap_training_datasets(
        train,
        val,
        sampling_rate=model.sr,
        n_signal=FLAGS.n_signal,
        db_path=FLAGS.db_path,
    )
    rave.training.finalize_training_model(
        model,
        db_path=FLAGS.db_path,
        n_signal=FLAGS.n_signal,
        smoke_test=FLAGS.smoke_test,
        rave_root=_RAVE_ROOT,
    )

    # get data-loader
    num_workers = FLAGS.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

    steps_per_epoch = len(train)
    if FLAGS.log_every_n_steps is not None:
        log_every_n_steps = max(1, FLAGS.log_every_n_steps)
    else:
        log_every_n_steps = min(50, max(1, steps_per_epoch))
    if steps_per_epoch < 50:
        print(
            f"W&B: log_every_n_steps={log_every_n_steps} "
            f"({steps_per_epoch} train batches/epoch; default 50 would skip charts)"
        )

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]
    RUN_NAME = f'{FLAGS.name}_{gin_hash}'
    RUN_DIR = os.path.join(FLAGS.out_path, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f'Run directory: {RUN_DIR}')

    # CHECKPOINT CALLBACKS (dirpath = RUN_DIR so ckpts sit with config.gin)
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=RUN_DIR,
        monitor="val/loss",
        filename="best",
    )
    last_filename = "last" if FLAGS.save_every is None else "epoch-{epoch:04d}"
    last_checkpoint = rave.core.ModelCheckpoint(
        dirpath=RUN_DIR,
        filename=last_filename,
        step_period=FLAGS.save_every,
    )

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = 1 if FLAGS.smoke_test else FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        # rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        rave.model.BetaWarmupCallback(),
    ]
    callbacks.extend(rave.training.extra_training_callbacks())

    if FLAGS.ema is not None:
        callbacks.append(EMA(FLAGS.ema))

    wandb_kwargs = dict(
        project=FLAGS.wandb_project,
        name=RUN_NAME,
        save_dir=RUN_DIR,
        offline=FLAGS.wandb_offline,
        config={
            'db_path': FLAGS.db_path,
            'batch': FLAGS.batch,
            'n_signal': FLAGS.n_signal,
            'max_steps': FLAGS.max_steps,
        },
    )
    if FLAGS.wandb_entity:
        wandb_kwargs['entity'] = FLAGS.wandb_entity

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(**wandb_kwargs),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        log_every_n_steps=log_every_n_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        print('loading state from file %s'%run)
        loaded = torch.load(run, map_location='cpu')
        # model = model.load_state_dict(loaded)
        trainer.fit_loop.epoch_loop._batches_that_stepped = loaded['global_step']
        # model = model.load_state_dict(loaded['state_dict'])
    
    with open(os.path.join(RUN_DIR, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    app.run(main)
