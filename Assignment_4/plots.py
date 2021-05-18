import matplotlib.pyplot as plt
import numpy as np

acc_base = np.load('accuracy_base.npy')
acc_lstm = np.load('accuracy_lstm.npy')
val_base = np.load('val_accuracy_base.npy')
val_lstm = np.load('val_accuracy_lstm.npy')
los_base = np.load('loss_base.npy')
los_lstm = np.load('loss_lstm.npy')
steps_base = np.arange(acc_base.size) + 1
steps_lstm = np.arange(acc_lstm.size) + 1

# plt.subplot(311)
# plt.plot(steps_base, los_base, label="Baseline DNN")
# plt.plot(steps_lstm, los_lstm, label="LSTM")
# plt.ylabel("Training Loss")
# plt.grid()
# plt.legend()

alpha = 0.7
acc_base_m, acc_lstm_m = [acc_base[0]], [acc_lstm[0]]
for i in range(1, acc_base.size):
    acc_base_m.append((1 - alpha) * acc_base[i] + alpha * acc_base_m[-1])
for i in range(1, acc_base.size):
    acc_lstm_m.append((1 - alpha) * acc_lstm[i] + alpha * acc_lstm_m[-1])

val_base_m, val_lstm_m = [val_base[0]], [val_lstm[0]]
for i in range(1, val_base.size):
    val_base_m.append((1 - alpha) * val_base[i] + alpha * val_base_m[-1])
for i in range(1, val_base.size):
    val_lstm_m.append((1 - alpha) * val_lstm[i] + alpha * val_lstm_m[-1])

plt.subplot(211)
plt.plot(steps_base, np.ones(steps_base.size) * np.mean([0.8000, 0.8190, 0.8020, 0.8245, 0.7975]), label="Baseline SVM")
plt.plot(steps_base, acc_base_m, label="Baseline DNN")
plt.plot(steps_lstm, acc_lstm_m, label="LSTM")
plt.ylabel("Training Accuracy")
plt.ylim([0.3, 1.0])
plt.xlim([1, steps_base.size])
plt.grid()
plt.legend(loc='lower right')

plt.subplot(212)
plt.plot(steps_base, np.ones(steps_base.size) * np.mean([0.5710, 0.5870, 0.5625, 0.5750, 0.5515]), label="Baseline SVM")
plt.plot(steps_base, val_base_m, label="Baseline DNN")
plt.plot(steps_lstm, val_lstm_m, label="LSTM")
plt.ylabel("Testing Accuracy")
plt.ylim([0.3, 0.7])
plt.xlim([1, steps_base.size])
plt.grid()
plt.legend(loc='lower right')

plt.show()