import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.parts.preprocessing.perturb import Perturbation, register_perturbation, process_augmentations
from nemo.collections.asr.data import audio_to_label_dataset
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.asr.modules import ConvASRDecoderClassification
from nemo.core.classes.common import typecheck
import pandas as pd
from typing import Any, Callable, Dict, Optional, Union
from pytorch_lightning.core.saving import inspect, parse_class_init_keys, CHECKPOINT_PAST_HPARAMS_KEYS, \
    _convert_loaded_hparams, load_hparams_from_tags_csv, load_hparams_from_yaml, pl_load
from pytorch_lightning.utilities import rank_zero_warn
from omegaconf import OmegaConf
import logging

MAXSIZE = 2**32-1

class Decoder(ConvASRDecoderClassification):

    @typecheck()
    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if not self._return_logits:
            raise Exception("return logits param must be TRUE")

        return logits

class Model(EncDecClassificationModel):

    def __init__(self, cfg, trainer = None, class_weight=None, loading=False):
        try:
            self.last_layer = cfg.last_layer
        except Exception:
            self.last_layer = None
        self.class_weight = class_weight
        if self.last_layer == None:
            print("last_layer param not set, will be set the softmax")
        elif self.last_layer != "sigmoid" and self.last_layer != "softmax":
            raise Exception("Last layer must be softmax or sigmoid")
        if "last_layer" in cfg.keys():
            del cfg["last_layer"]
        if not loading:
            self._load_dataset_csv(cfg)
        if loading:
            cfg.train_ds = None
        super().__init__(cfg, trainer)

    def _load_dataset_csv(self, cfg):
        mainfest_path = Path(cfg.train_ds.manifest_filepath)
        csv_path = mainfest_path.parent.joinpath(f"train_{cfg.lang}_manifest.csv")
        self.dataset_csv = pd.read_csv(csv_path, index_col=0)

    def decode(self, input_signal, input_signal_length):
        logits = self.forward(input_signal=input_signal, input_signal_length=input_signal_length)
        assert logits.size(0) == 1
        logits = logits[0]
        if self.last_layer == "softmax":
            predicted_values = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
        else:
            predicted_values = torch.nn.functional.sigmoid(logits).cpu().detach().numpy()
        return predicted_values

    def predict(self, logits):
        if self.last_layer == "softmax" or self.last_layer is None:
            predicted_values = torch.nn.functional.softmax(logits, dim=-1)
        else:
            predicted_values = torch.nn.functional.sigmoid(logits)
        return predicted_values

    def validation_step(self, batch, batch_idx, dataloader_idx=0): #Get singleton class
        raise Exception("Use the validation callback to perform the validation task")

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        raise Exception("Use the validation callback to perform the validation task")

    def _setup_loss(self):
        weights = list(self.class_weight) if self.class_weight is not None else None
        return CrossEntropyLoss(weight=weights)

    def save_backup(self, exp_dir, epoch ,loss=None):
        save_dict = {}
        save_dict["state_dict"] = self.state_dict()
        save_dict["cfg"] = self.cfg
        text = "matchcboxnet--val_loss={}-epoch={}.model"
        text = text.format(round(loss, 4), epoch) if loss is not None else text.format("nope", epoch)
        path = Path(exp_dir).joinpath("checkpoints_backup")
        path.mkdir(exist_ok=True, parents=True)
        path_save = path.joinpath(text)
        torch.save(save_dict, path_save)

    @classmethod
    def load_backup(cls, ckpt_name, exp_dir):
        ckpt_path = Path(exp_dir).joinpath("checkpoints_backup", ckpt_name)
        try:
            checkpoint = torch.load(ckpt_path)
        except RuntimeError:
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"]
        class_weight = state_dict["loss.weight"] if "loss.weight" in state_dict.keys() else None
        # del state_dict["loss.weight"]
        model = cls(checkpoint["cfg"], class_weight=class_weight, loading=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
        cls_spec = inspect.getfullargspec(cls.__init__)
        cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

        self_var, args_var, kwargs_var = parse_class_init_keys(cls)
        drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
        cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

        cls_kwargs_loaded = {}
        # pass in the values we saved automatically
        if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

            # 2. Try to restore model hparams from checkpoint using the new key
            _new_hparam_key = cls.CHECKPOINT_HYPER_PARAMS_KEY
            cls_kwargs_loaded.update(checkpoint.get(_new_hparam_key))

            # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
            cls_kwargs_loaded = _convert_loaded_hparams(
                cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE)
            )

            # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
            args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
            if args_name and args_name in cls_init_args_name:
                cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

        _cls_kwargs = {}
        _cls_kwargs.update(cls_kwargs_loaded)
        _cls_kwargs.update(cls_kwargs_new)

        if not cls_spec.varkw:
            # filter kwargs according to class init unless it allows any argument via kwargs
            _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

        _cls_kwargs["cfg"]["train_ds"]["augmentor"] = None
        loss_weight = checkpoint["state_dict"].get("loss.weight")
        loss_weight = loss_weight.cpu().detach().numpy()
        _cls_kwargs["class_weight"] = loss_weight
        _cls_kwargs["loading"] = True
        model = cls(**_cls_kwargs)

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        # load the state_dict on the model automatically
        keys = model.load_state_dict(checkpoint["state_dict"], strict=strict)

        if not strict:
            if keys.missing_keys:
                rank_zero_warn(
                    f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
                )
            if keys.unexpected_keys:
                rank_zero_warn(
                    f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
                )

        return model


    def _setup_dataloader_from_config(self, config):

        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        featurizer = MyWaveFormFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.warning("VAD inference does not support tarred dataset now")
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_label_dataset.get_tarred_classification_label_dataset(
                featurizer=featurizer,
                config=OmegaConf.to_container(config),
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
            batch_size = config['batch_size']
            collate_func = dataset.collate_fn

        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.info("Perform streaming frame-level VAD")
                dataset = audio_to_label_dataset.get_speech_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = 1
                collate_func = dataset.vad_frame_seq_collate_fn
            else:
                dataset = audio_to_label_dataset.get_classification_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = config['batch_size']
                collate_func = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=DatasetSampler(dataset, batch_size, self.cfg.labels, self.cfg.train_ds.augmentor.mynoise.seed),
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        try:
            register_perturbation("mynoise", Augmentation)
        except KeyError: pass

        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        if hparams_file is not None:
            extension = hparams_file.split(".")[-1]
            if extension.lower() == "csv":
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ("yml", "yaml"):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

            hparams["on_gpu"] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

        # for past checkpoint need to add the new key
        if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
        # override the hparams with values that were passed in
        structure = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        for k, v in kwargs.items(): #Adding last_layer
            OmegaConf.update(structure, f"{k}", v, force_add=True)

        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model
