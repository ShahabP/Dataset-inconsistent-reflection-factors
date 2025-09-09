import pickle
import matplotlib.pyplot as plt

# ---------- Load dataset ----------
with open('rir_dataset_walls.pkl', 'rb') as f:
    dataset = pickle.load(f)

fs = 16000  # Sampling rate
num_to_plot = 5
rirs_to_plot = [dataset[i]['rir'].numpy() for i in range(num_to_plot)]

# ---------- Plot ----------
plt.figure(figsize=(12, 6))
for i, rir in enumerate(rirs_to_plot):
    plt.plot(np.arange(len(rir)) / fs, rir, label=f'RIR {i+1}')

plt.title('Example Room Impulse Responses (RIRs)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
