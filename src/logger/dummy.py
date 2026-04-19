from datetime import datetime


class DummyWriter:
    def __init__(self, logger=None, project_config=None, run_name=None, id_length=8, loss_names=None, log_checkpoints=True, **kwargs):
        self.run_name = run_name
        self.id_length = id_length
        self.loss_names = loss_names or ["loss"]
        self.log_checkpoints = log_checkpoints
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_scalar(self, scalar_name, scalar):
        pass

    def add_scalars(self, scalars):
        pass

    def add_image(self, image_name, image):
        pass

    def add_audio(self, audio_name, audio, sample_rate=None):
        pass

    def add_text(self, text_name, text):
        pass

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        pass

    def add_table(self, table_name, table):
        pass

    def add_checkpoint(self, checkpoint_path, save_dir):
        pass