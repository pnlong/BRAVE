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
from pytorch_lightning.strategies import DDPStrategy
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
flags.DEFINE_string(
    'db_path',
    None,
    help='Preprocessed dataset path (single-domain). Alias for --db_path_x.',
)
flags.DEFINE_string(
    'db_path_x',
    None,
    help='Domain-X LMDB (joint / stratified). Falls back to --db_path.',
)
flags.DEFINE_string(
    'db_path_y',
    None,
    help='Domain-Y LMDB. When set with --db_path_x (or --db_path), train with '
    'stratified X/Y batches (joint embedding).',
)
flags.DEFINE_float(
    'domain_x_fraction',
    0.5,
    help='Fraction of each batch from --db_path_x when --db_path_y is set '
    '(default 0.5 = balanced). Requires --batch >= 2.',
)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Run validation every n steps')
flags.DEFINE_integer('save_every',
                     500000,
                     help='Write epoch_{global_step}.ckpt every n steps and at train end')
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
flags.DEFINE_string(
    'wandb_run_id',
    default=None,
    help='Resume a specific W&B run id (auto-detected from --ckpt run dir when omitted)',
)
flags.DEFINE_bool('wandb_offline',
                  default=False,
                  help='Log to W&B in offline mode')
flags.DEFINE_integer(
    'log_every_n_steps',
    default=None,
    help='Lightning/W&B flush interval (default: min(50, batches per epoch))',
)
flags.DEFINE_integer(
    'log_audio_every_n_steps',
    default=20000,
    help='W&B audio at most every N train steps (default: 20000; 0 = every val epoch)',
)
flags.DEFINE_bool(
    'reject_silent',
    default=None,
    help='Enable RMS gate on train loader (gin default is off)',
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
    if os.environ.get('CUDNN_BENCHMARK', '1') == '0':
        torch.backends.cudnn.benchmark = False
        print('cudnn.benchmark=False (CUDNN_BENCHMARK=0)', flush=True)
    else:
        torch.backends.cudnn.benchmark = True

    db_path_x = FLAGS.db_path_x or FLAGS.db_path
    if not db_path_x:
        raise ValueError("Pass --db_path or --db_path_x")
    db_path_y = FLAGS.db_path_y

    # check dataset channels
    n_channels = rave.dataset.get_training_channels(db_path_x, FLAGS.channels)
    gin.bind_parameter('RAVE.n_channels', n_channels)

    # parse CLI --augment files (skipped when empty so gin can bind RandomGain)
    augmentations = parse_augmentations(map(add_gin_extension, FLAGS.augment))
    if FLAGS.augment:
        gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)

    # parse configuration
    if FLAGS.ckpt:
        config_file = rave.core.search_for_config(FLAGS.ckpt)
        if config_file is None:
            if FLAGS.config:
                print(
                    'Config not found near %s; using --config %s'
                    % (FLAGS.ckpt, FLAGS.config))
                gin.parse_config_files_and_bindings(
                    map(add_gin_extension, FLAGS.config),
                    FLAGS.override,
                )
            else:
                raise FileNotFoundError(
                    'No config.gin found near %s and no --config provided'
                    % FLAGS.ckpt)
        else:
            gin.parse_config_file(config_file)
    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config),
            FLAGS.override,
        )

    rave.core.bind_log_audio_every_n_steps(FLAGS.log_audio_every_n_steps)

    model = rave.training.build_training_model(n_channels=n_channels)
    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    # parse datasset
    train, val = rave.dataset.split_train_val(
        db_path_x,
        model.sr,
        FLAGS.n_signal,
        percent=98,
        derivative=FLAGS.derivative,
        normalize=FLAGS.normalize,
        rand_pitch=FLAGS.rand_pitch,
        n_channels=n_channels,
    )

    train_y = val_y = None
    if db_path_y:
        n_channels_y = rave.dataset.get_training_channels(
            db_path_y, FLAGS.channels)
        if n_channels_y != n_channels:
            raise ValueError(
                f"--db_path_x channels={n_channels} != "
                f"--db_path_y channels={n_channels_y}")
        train_y, val_y = rave.dataset.split_train_val(
            db_path_y,
            model.sr,
            FLAGS.n_signal,
            percent=98,
            derivative=FLAGS.derivative,
            normalize=FLAGS.normalize,
            rand_pitch=FLAGS.rand_pitch,
            n_channels=n_channels,
            show_progress=False,
        )

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
    if train_y is not None:
        train_y = rave.dataset.maybe_reject_silent(train_y, **reject_kwargs)

    train, val = rave.training.wrap_training_datasets(
        train,
        val,
        sampling_rate=model.sr,
        n_signal=FLAGS.n_signal,
        db_path=db_path_x,
    )
    if train_y is not None:
        train_y, val_y = rave.training.wrap_training_datasets(
            train_y,
            val_y,
            sampling_rate=model.sr,
            n_signal=FLAGS.n_signal,
            db_path=db_path_y,
        )
    rave.training.finalize_training_model(
        model,
        db_path=db_path_x,
        n_signal=FLAGS.n_signal,
        smoke_test=FLAGS.smoke_test,
        rave_root=_RAVE_ROOT,
    )

    # get data-loader
    num_workers = FLAGS.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    if train_y is not None:
        if FLAGS.batch < 2:
            raise ValueError(
                "stratified dual-domain training requires --batch >= 2")
        print(
            f"Stratified dual-domain: "
            f"X={db_path_x} ({len(train)} train) "
            f"Y={db_path_y} ({len(train_y)} train) "
            f"domain_x_fraction={FLAGS.domain_x_fraction}",
            flush=True,
        )
        train = rave.dataset.build_stratified_dual_dataloader(
            train,
            train_y,
            FLAGS.batch,
            domain_x_fraction=FLAGS.domain_x_fraction,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        val = rave.dataset.build_stratified_dual_dataloader(
            val,
            val_y,
            FLAGS.batch,
            domain_x_fraction=FLAGS.domain_x_fraction,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
    else:
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
    ckpt_path = rave.core.search_for_run(FLAGS.ckpt)
    if ckpt_path:
        resume_run_dir = rave.core.run_dir_from_checkpoint(ckpt_path)
        if os.path.isfile(os.path.join(resume_run_dir, "config.gin")):
            RUN_DIR = resume_run_dir
            RUN_NAME = os.path.basename(RUN_DIR)
        else:
            RUN_NAME = f'{FLAGS.name}_{gin_hash}'
            RUN_DIR = os.path.join(FLAGS.out_path, RUN_NAME)
    else:
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

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    strategy = None
    n_devices = 1
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
        if isinstance(devices, int):
            n_devices = devices if devices > 0 else 1
        elif isinstance(devices, (list, tuple)):
            n_devices = len(devices)
        else:
            n_devices = 1
        if n_devices > 1:
            # Alternating gen/disc optimizers leave D unused on gen steps (and
            # vice versa). Required for phase-2 GAN training under DDP.
            strategy = DDPStrategy(find_unused_parameters=True)
            print(
                f'Multi-GPU DDP (find_unused_parameters=True): {n_devices} devices, '
                f'per-GPU batch={FLAGS.batch}, '
                f'global batch≈{FLAGS.batch * n_devices}',
            )
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    batches_per_rank = max(1, len(train) // max(1, n_devices))
    val_check = {}
    if FLAGS.smoke_test:
        val_check["val_check_interval"] = 1
        val_check["limit_train_batches"] = 1
        val_check["limit_val_batches"] = 1
    elif batches_per_rank >= FLAGS.val_every:
        val_check["val_check_interval"] = FLAGS.val_every
    else:
        val_check["check_val_every_n_epoch"] = max(1, FLAGS.val_every // batches_per_rank)

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        # rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        rave.model.BetaWarmupCallback(),
        rave.core.SaveWandbRunIdCallback(RUN_DIR),
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
            'db_path': db_path_x,
            'db_path_x': db_path_x,
            'db_path_y': db_path_y,
            'domain_x_fraction': FLAGS.domain_x_fraction if db_path_y else None,
            'batch': FLAGS.batch,
            'n_signal': FLAGS.n_signal,
            'max_steps': FLAGS.max_steps,
        },
    )
    if FLAGS.wandb_entity:
        wandb_kwargs['entity'] = FLAGS.wandb_entity

    wandb_run_id = FLAGS.wandb_run_id
    if wandb_run_id is None and ckpt_path is not None:
        wandb_run_id = rave.core.find_wandb_run_id(RUN_DIR)
    if wandb_run_id:
        wandb_kwargs['id'] = wandb_run_id
        wandb_kwargs['resume'] = 'must'
        print(f'W&B: resuming run id={wandb_run_id}')

    trainer = pl.Trainer(
        logger=pl.loggers.WandbLogger(**wandb_kwargs),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        log_every_n_steps=log_every_n_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = ckpt_path
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
