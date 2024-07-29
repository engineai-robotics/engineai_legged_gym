import numpy as np
class EMAFilter:
    def __init__(self, sample_frequency, cutoff_frequency, start_step=5):
        self.sample_frequency = sample_frequency
        self.cutoff_frequency = cutoff_frequency  # [hz]
        self._init_EMA_coefficient()
        self.pre_value = 0
        self.count = 0
        self.ema_start_count = start_step

    def _init_EMA_coefficient(self):
        normalized_digital_radian_frequency = np.pi * self.cutoff_frequency / (self.sample_frequency / 2)
        a = np.cos(normalized_digital_radian_frequency)
        self.alpha = a - 1 + np.sqrt(a ** 2 - 4 * a + 3)

    def update_EMA_coefficient(self, sample_frequency, cutoff_frequency):
        new_normalized_frequency = np.pi * cutoff_frequency / (sample_frequency / 2)
        new_a = np.cos(new_normalized_frequency)
        self.alpha = new_a - 1 + np.sqrt(new_a ** 2 - 4 * new_a + 3)

    def ema_filter_out(self, input_data):
        """

        :param input_data: [1_time_step, n_feature_dim]
        :return: filtered data: [1_time_step, n_feature_dim]
        """

        if self.count < self.ema_start_count:  # use moving average to initialize the start value of EMA
            self.pre_value = (self.pre_value * self.count + input_data) / (self.count + 1)
            self.count += 1
        else:
            self.pre_value = (1 - self.alpha) * self.pre_value + self.alpha * input_data

        return self.pre_value
