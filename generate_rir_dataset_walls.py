import pickle
from utils import create_rir

# ---------- Parameters ----------
num_rooms = 5000

# ---------- Generate dataset ----------
dataset = []
for i in range(num_rooms):
    geom, rir = create_rir()
    dataset.append({'geom': geom, 'rir': rir})
    if (i+1) % 100 == 0:
        print(f'{i+1}/{num_rooms} rooms generated')

# ---------- Save dataset ----------
with open('rir_dataset_walls.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print('Dataset generation complete! Saved to rir_dataset_walls.pkl')
