
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

class SIRAugmenter:
    def __init__(self, param_noise=0.001, comp_noise=0.01,
                 n_param_aug=10, n_comp_aug=1):
        self.param_noise = param_noise
        self.comp_noise  = comp_noise
        self.n_param_aug = n_param_aug
        self.n_comp_aug  = n_comp_aug

    # This is the method your code expects
    def augment_simulation(self, sim):
        sims = [sim]
        for _ in range(self.n_param_aug):
            sims.append(self.augment_params(sim))
        for _ in range(self.n_comp_aug):
            sims.append(self.augment_compartments(sim))
        return sims

    def augment_params(self, sim):
        sim_new = deepcopy(sim)
        params = sim_new['params']
        for k in ['tau', 'gamma', 'rho']:
            params[k] *= 1 + np.random.normal(0, self.param_noise)
        return sim_new

    def augment_compartments(self, sim):
        sim_new = deepcopy(sim)
        out = sim_new['output']
        S, I, R = np.array(out['S']), np.array(out['I']), np.array(out['R'])
        S2 = np.maximum(S * (1 + np.random.normal(0, self.comp_noise, S.shape)), 0)
        I2 = np.maximum(I * (1 + np.random.normal(0, self.comp_noise, I.shape)), 0)
        R2 = np.maximum(R * (1 + np.random.normal(0, self.comp_noise, R.shape)), 0)
        N = S[0] + I[0] + R[0]
        factor = N / (S2 + I2 + R2 + 1e-8)
        sim_new['output'] = {'t': out['t'],
                             'S': (S2 * factor).tolist(),
                             'I': (I2 * factor).tolist(),
                             'R': (R2 * factor).tolist()}
        return sim_new


# %%
def augment_compartments(self, sim):
        sim_new = deepcopy(sim)
        out = sim["output"]

        # ensure arrays
        S, I, R = np.array(out["S"]), np.array(out["I"]), np.array(out["R"])

        S2 = S * (1 + np.random.normal(0, self.comp_noise, S.shape))
        I2 = I * (1 + np.random.normal(0, self.comp_noise, I.shape))
        R2 = R * (1 + np.random.normal(0, self.comp_noise, R.shape))

        S2 = np.maximum(S2, 0)
        I2 = np.maximum(I2, 0)
        R2 = np.maximum(R2, 0)

        N = S[0] + I[0] + R[0]
        total = S2 + I2 + R2
        factor = N / (total + 1e-8)

        S2 *= factor
        I2 *= factor
        R2 *= factor

        sim_new["output"] = {
            "t": out["t"],
            "S": S2.tolist(),
            "I": I2.tolist(),
            "R": R2.tolist()
        }

        return sim_new

# Apply augmentation


def augment_simulation(self, sim):

        sims = [deepcopy(sim)]

        for _ in range(self.n_param_aug):
            sims.append(self.augment_params(sim))

        for _ in range(self.n_comp_aug):
            sims.append(self.augment_compartments(sim))

        return sims



def augment_train_split(split_data, augmenter):

    augmented = deepcopy(split_data)

    original = split_data["train"]["simulations"]
    new_sims = []

    for sim in original:
        new_sims.extend(augmenter.augment_simulation(sim))

    augmented["train"]["simulations"] = new_sims

    return augmented


# Run augmentation

if __name__ == "__main__":

    with open("epidemic_data_age_adaptive_sobol_split.pkl", "rb") as f:
        data = pickle.load(f)

    augmenter = SIRAugmenter()

    augmented = augment_train_split(data, augmenter)

    with open("epidemic_data_age_adaptive_sobol_split_augmented.pkl", "wb") as f:
        pickle.dump(augmented, f)

    print("Augmentation complete.")


import pickle
import pandas as pd

# Load augmented data
with open("epidemic_data_age_adaptive_sobol_split_augmented.pkl", "rb") as f:
    split_data = pickle.load(f)

# Function to export CSV
def export_params_with_R0(split_data, csv_path, split_name="train"):
    sims = split_data[split_name]['simulations']
    rows = []
    for sim in sims:
        tau   = sim['params']['tau']
        gamma = sim['params']['gamma']
        rho   = sim['params']['rho']
        R0    = (tau / gamma) * 27.26
        rows.append({'tau': tau, 'gamma': gamma, 'rho': rho, 'R0': R0})

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f" CSV saved to: {csv_path}, shape={df.shape}")
    return df

# Call function
csv_file = "epidemic_data_age_adaptive_sobol_split_augmented.csv"
df_exported = export_params_with_R0(split_data, csv_file, split_name="train")

# Quick check
print(df_exported.head())



augmented_data=pd.read_csv('epidemic_data_age_adaptive_sobol_split_augmented.csv')

print(augmented_data.columns)
print(len(augmented_data))
print(augmented_data.isnull().sum()) # no missing values
augmented_data.describe(include='all')


total_samples=len(augmented_data)
print(total_samples)
greater=augmented_data[augmented_data['R0']>1.2]
pc1=(len(greater)/total_samples)*100
print(f"greater than 1.2: {len(greater),pc1}")
between = augmented_data[(augmented_data['R0'] >= 0.7) & (augmented_data['R0'] <= 1.2)]
pc2=(len(between)/total_samples)*100
print(f"between 0.7 and 1.2: {len(between),pc2}")

#print(pc2)

less_than=augmented_data[augmented_data['R0']<0.8]
pc3=(len(less_than)/total_samples)*100
print(f"less than 0.8: {len(less_than),pc3}")


#Plotting
import matplotlib.pyplot as plt
slope=1/34
plt.figure(figsize=(10,10))
plt.scatter(augmented_data['gamma'], augmented_data['tau'], alpha=0.1)

x_vals = np.linspace(augmented_data['gamma'].min(), augmented_data['gamma'].max(), 100)
y_vals = slope * x_vals

plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'slope={slope}')

plt.xlabel('gamma')
plt.ylabel('tau')
plt.title('Scatter plot of tau vs gamma')
plt.legend()
plt.grid(True)
plt.show()
    






