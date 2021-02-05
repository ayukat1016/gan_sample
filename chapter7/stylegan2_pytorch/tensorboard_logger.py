from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger():
    writer = SummaryWriter(log_dir='./logs')
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_step = 0
    tag = ''

    @classmethod
    def print_parameter(cls, model):
        for key in model.state_dict().keys():
            # print('name: {}, shape: {}'.format(key, model.state_dict()[key].shape))
            TensorboardLogger.writer.add_histogram(
                '{}/{}'.format(TensorboardLogger.now, key),
                model.state_dict()[key],
                TensorboardLogger.global_step)

    @classmethod
    def write_scalar(cls, metrics, value):
        metrics_path = '{}/{}'.format(TensorboardLogger.now, metrics)
        TensorboardLogger.writer.add_scalar(metrics_path, value, TensorboardLogger.global_step)

    @classmethod
    def write_histogram(cls, metrics, value):
        metrics_path = '{}/{}'.format(TensorboardLogger.now, metrics)
        TensorboardLogger.writer.add_histogram(metrics_path, value, TensorboardLogger.global_step)

    @classmethod
    def write_image(cls, metrics, value):
        metrics_path = '{}/{}'.format(TensorboardLogger.now, metrics)
        TensorboardLogger.writer.add_image(metrics_path, value, TensorboardLogger.global_step)
