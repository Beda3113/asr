import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriter:
    def __init__(self, logger, project_config, run_name="experiment", **kwargs):
        """
        Args:
            logger: логгер из setup_saving_and_logging
            project_config: конфиг проекта
            run_name: имя запуска (из конфига)
        """
        self.logger = logger
        self.project_config = project_config
        self.run_name = run_name
        
        self.log_dir = os.path.join("outputs", self.run_name, "tensorboard")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.step = 0
        self.mode = "train"
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_scalar(self, scalar_name, scalar):
        self.writer.add_scalar(f"{scalar_name}/{self.mode}", scalar, self.step)

    def add_scalars(self, scalars):
        for name, value in scalars.items():
            self.add_scalar(name, value)

    def add_image(self, image_name, image):
        self.writer.add_image(f"{image_name}/{self.mode}", image, self.step)

    def add_audio(self, audio_name, audio, sample_rate=None):
        self.writer.add_audio(f"{audio_name}/{self.mode}", audio, self.step, sample_rate)

    def add_text(self, text_name, text):
        self.writer.add_text(f"{text_name}/{self.mode}", text, self.step)

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        self.writer.add_histogram(f"{hist_name}/{self.mode}", values_for_hist, self.step, bins=bins)

    def add_table(self, table_name, table):
        self.writer.add_text(f"{table_name}/{self.mode}", table.to_string(), self.step)

    def add_checkpoint(self, checkpoint_path, save_dir):
        pass

    def close(self):
        self.writer.close()