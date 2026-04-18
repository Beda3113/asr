from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        # Skip if batch is None or not a dict
        if batch is None or not isinstance(batch, dict):
            return None

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # Pass arguments directly to model (not **batch to avoid extra keys)
        outputs = self.model(
            spectrogram=batch.get("spectrogram"),
            spectrogram_length=batch.get("spectrogram_length")
        )
        batch.update(outputs)

        # Pass arguments directly to loss function
        all_losses = self.criterion(
            log_probs=batch.get("log_probs"),
            log_probs_length=batch.get("log_probs_length"),
            text_encoded=batch.get("text_encoded"),
            text_encoded_length=batch.get("text_encoded_length")
        )
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            try:
                metrics.update(met.name, met(**batch))
            except Exception as e:
                self.logger.warning(f"Failed to compute metric {met.name}: {e}")

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if batch is None or not isinstance(batch, dict):
            return

        if mode == "train":
            if "spectrogram" in batch and batch["spectrogram"] is not None:
                self.log_spectrogram(**batch)
        else:
            if "spectrogram" in batch and batch["spectrogram"] is not None:
                self.log_spectrogram(**batch)
            if "log_probs" in batch and "text" in batch:
                self.log_predictions(**batch)
            if "audio" in batch and batch["audio"] is not None:
                self.log_audio(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        """Log spectrogram to writer."""
        if spectrogram is not None and len(spectrogram) > 0:
            spectrogram_for_plot = spectrogram[0].detach().cpu()
            image = plot_spectrogram(spectrogram_for_plot)
            self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, **batch):
        """Log audio sample to writer."""
        if audio is not None and len(audio) > 0:
            audio_sample = audio[0]
            self.writer.add_audio("audio", audio_sample, sample_rate=16000)

    def log_predictions(self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch):
        """Log predictions table with CER/WER."""
        if log_probs is None or log_probs_length is None:
            return

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[:int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)

            # Проверка на пустые предсказания
            if pred is None or len(pred) == 0:
                wer = 100.0
                cer = 100.0
            else:
                wer_val = calc_wer(target, pred)
                cer_val = calc_cer(target, pred)
                wer = (wer_val * 100) if wer_val is not None else 100.0
                cer = (cer_val * 100) if cer_val is not None else 100.0

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred if pred else "[empty]",
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
