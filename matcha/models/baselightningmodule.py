"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.utils import plot_tensor

import tempfile
import os
import torchaudio
from time import perf_counter
log = utils.get_pylogger(__name__)


torch.set_float32_matmul_precision("high")

class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))


    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",#self.hparams.scheduler.lightning_args.interval,
                    "frequency": 1,#self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]

        dur_loss, prior_loss, diff_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }
    
    # @abstractmethod
    # def synthesize(self):
    #     pass

    # @abstractmethod
    # def generate_waveform(self):
    #     pass
    
    # @abstractmethod
    # def compute_quality_scores(self):
    #     pass
    
    def get_quality_scores(self, batch):
        x, x_lengths, spks = batch["x"], batch["x_lengths"], batch["spks"]
        # 'UTMoS_score', 'STOI', 'PESQ', 'SI-SDR'

        # utmos = []
        pesq = []
        sisdr = []
        scoreq = []
        scores_dict = {}

        # synthesize
        out_dict = self.synthesise(x, x_lengths, n_timesteps=10, spks=spks, temperature=0.7, length_scale=0.95)  # not sway sampling for now
        mels, mel_lengths = out_dict["mel"], out_dict["mel_lengths"]

        # with tempfile.TemporaryDirectory() as temp_dir:
        # generate_waveform
        for idx, m in enumerate(mels):
            m = m.unsqueeze(0)
            # print("Mel with shape: ", m.shape)
            # print("Mel spectrogram tensor type", m.dtype)
            wf, _ = self.generate_waveform(m, mel_lengths[idx])
            # compute_quality_scores


            obj_scores = self.compute_quality_scores(wf)  # utmos_score
            
            # create temp wav for utmos
            # temp_wav_path = os.path.join(temp_dir, f"temp_audio_{idx}.wav")
            # torchaudio.save(temp_wav_path, wf.cpu(), 16000)

            # scoreq
            ort_inputs = {"audio_input": wf.unsqueeze(0).cpu().numpy()}
            scoreq_score = self._scoreq_onnx.run(None, ort_inputs)[0][0][0]

            print(scoreq_score)

            # utmos
            # utmos_score = self.utmosv2_model.predict(input_path=temp_wav_path)

            # utmos.append(utmos_score)
            scoreq.append(scoreq_score)
            pesq.append(obj_scores[1].item())
            sisdr.append(obj_scores[2].item())

        scores_dict["pesq"] = round(sum(pesq) / len(pesq), 3)
        scores_dict["sisdr"] = round(sum(sisdr) / len(sisdr), 3)
        # scores_dict["utmos"] = round(sum(utmos) / len(utmos), 3)
        scores_dict["scoreq"] = round(sum(scoreq) / len(scoreq), 3)


        return scores_dict


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        #lr log with scheduler
        #log.info(self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"])
        if self.hparams.scheduler not in (None, {}):
            #log.info(self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"])
            self.log(
                "model/lr",
                self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
                sync_dist=True,
            )


        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)

        if self._squim_model is None:
            self.load_squim()
        if self._scoreq_onnx is None:
            self.load_scoreq_onnx()

        quality_dict = self.get_quality_scores(batch)

        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        
        # Quality scores
        
        self.log(
            "quality_scores/val_scoreq",
            quality_dict["scoreq"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        
        

        self.log(
            "quality_scores/val_pesq",
            quality_dict["pesq"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "quality_scores/val_sisdr",
            quality_dict["sisdr"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks, length_scale=0.95)
                y_enc, y_dec, mel = output["encoder_outputs"], output["decoder_outputs"], output["mel"]
                attn = output["attn"]

                if self._vocoder is None:
                    self.load_vocoder()
                # vocoder = self._vocoder.to(self.device)
                y_hat_audio = self._vocoder.decode(mel)

                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

                self.logger.experiment.add_audio(
                            f"audio_{i}_spk_{spks.item()}",
                            y_hat_audio.squeeze().cpu() ,
                            self.global_step,
                            22050, # sample rate
                        )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
