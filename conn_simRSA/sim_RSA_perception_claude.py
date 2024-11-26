from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
from scipy import spatial
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product


# N_CLASSES = 2
# ENCODING_PER_CLASS = 3
# FEATURES_PER_CLASS = 8
# CATEGORIES_PER_CLASS = 5

N_CLASSES = 1
ENCODING_PER_CLASS = 6
FEATURES_PER_CLASS = 16
CATEGORIES_PER_CLASS = 10

N_CLASSES = 1
ENCODING_PER_CLASS = 5
FEATURES_PER_CLASS = 30
CATEGORIES_PER_CLASS = 10

class FeatureGenerator_OLD:
    def __init__(self, n_classes=N_CLASSES, features_per_class=FEATURES_PER_CLASS,
                 correlation_strength=0.7):
        self.n_classes = n_classes
        self.features_per_class = features_per_class
        self.total_features = n_classes * features_per_class
        # self.correlation_strength = correlation_strength

        # Generate correlation matrices for each feature class
        self.correlation_matrices = []
        for _ in range(n_classes):
            # Create a base correlation matrix
            base = np.random.uniform(-0.6, 0.6, (features_per_class, features_per_class))
            base = np.sqrt(np.abs(base)) * np.sign(base)
            base = (base + base.T) / 2  # Make it symmetric
            # Set diagonal to 1
            np.fill_diagonal(base, 1)
            # Ensure it's positive definite
            base = base @ base.T
            # Normalize
            d = np.diag(1.0 / np.sqrt(np.diag(base)))
            base = d @ base @ d
            self.correlation_matrices.append(base)

    def generate_sample(self, n_samples=1):
        samples = np.zeros((n_samples, self.total_features))

        for class_idx in range(self.n_classes):
            # Generate correlated features for this class
            start_idx = class_idx * self.features_per_class
            end_idx = start_idx + self.features_per_class

            # Generate using multivariate normal distribution
            class_samples = np.random.multivariate_normal(
                mean=np.zeros(self.features_per_class),
                cov=self.correlation_matrices[class_idx],
                size=n_samples
            )

            samples[:, start_idx:end_idx] = class_samples

        return samples


def get_pattern(n_features):
    l = []
    for i in range(n_features):
        l.append(np.random.choice([-1, 1]))
    return l

class FeatureGenerator:
    def __init__(self, n_classes=N_CLASSES, categories_per_class=CATEGORIES_PER_CLASS,
                 features_per_class=FEATURES_PER_CLASS,
                 within_category_noise=0.1, between_category_dist=12.0):
        """
        Initialize the categorical feature generator.

        Args:
            n_classes: Number of distinct classes (default: 10)
            categories_per_class: Number of categories within each class (default: 20)
            features_per_class: Number of features defining each category (default: 5)
            within_category_noise: Standard deviation of noise within categories (default: 0.1)
            between_category_dist: Minimum distance between category centroids (default: 2.0)
        """
        self.n_classes = n_classes
        self.categories_per_class = categories_per_class
        self.features_per_category = features_per_class
        self.within_category_noise = within_category_noise
        self.between_category_dist = between_category_dist
        self.total_features = n_classes * features_per_class

        # Generate category prototypes for each class
        self.category_prototypes = self._generate_category_prototypes()

    def _generate_category_prototypes(self):
        """
        Generate well-separated category prototypes in feature space for each class.
        Returns a dictionary mapping class indices to arrays of category prototypes.
        """
        prototypes = {}

        # Create a set of possible category patterns
        # We'll use combinations of -1 and 1 to create distinct patterns
        # base_patterns = list(product([-1, 1], repeat=self.features_per_category))


        for class_idx in range(self.n_classes):
            # Randomly select patterns for this class
            class_patterns = []
            # available_patterns = base_patterns.copy()



            # print(len(available_patterns))
            # quit()

            for _ in range(self.categories_per_class):
                # if not available_patterns:
                #     raise ValueError("Not enough unique patterns available")

                # Select a random pattern and remove it from available patterns
                # pattern_idx = np.random.randint(len(available_patterns))
                # pattern = available_patterns.pop(pattern_idx)
                pattern = get_pattern(self.features_per_category)

                # Scale the pattern by the between_category_dist to ensure separation
                pattern = np.array(pattern) * self.between_category_dist
                class_patterns.append(pattern)

            prototypes[class_idx] = np.array(class_patterns)
        # print(np.array(prototypes).shape)
        # quit()

        return prototypes

    def _add_noise(self, prototype, n_samples=1):
        """Add Gaussian noise to a category prototype."""
        noise = np.random.normal(0, self.within_category_noise,
                                 (n_samples, len(prototype)))
        return prototype + noise

    def generate_sample(self, n_samples=1):
        """
        Generate samples with categorical structure.

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Array of shape (n_samples, total_features)
        """
        # Initialize output array
        samples = np.zeros((n_samples, self.total_features))

        # Generate samples for each class
        for class_idx in range(self.n_classes):
            # Calculate feature indices for this class
            start_idx = class_idx * self.features_per_category
            end_idx = start_idx + self.features_per_category

            # For each sample
            for sample_idx in range(n_samples):
                # Randomly select a category for this class
                category_idx = np.random.randint(self.categories_per_class)
                prototype = self.category_prototypes[class_idx][category_idx]

                # Add noise to the prototype
                noisy_features = self._add_noise(prototype, 1)

                # Assign to the appropriate feature slots
                samples[sample_idx, start_idx:end_idx] = noisy_features

        return samples

    def get_category_info(self):
        """
        Return information about the categories for visualization/analysis.
        """
        info = {
            'n_classes': self.n_classes,
            'categories_per_class': self.categories_per_class,
            'features_per_category': self.features_per_category,
            'prototypes': self.category_prototypes
        }
        return info


class VisualFeatureDataset(Dataset):
    def __init__(self, n_samples, generator):
        self.data = torch.FloatTensor(generator.generate_sample(n_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, input_size=N_CLASSES * FEATURES_PER_CLASS,
                 encoding_size=N_CLASSES * ENCODING_PER_CLASS):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, CATEGORIES_PER_CLASS * N_CLASSES),
            nn.BatchNorm1d(CATEGORIES_PER_CLASS * N_CLASSES),
            nn.LeakyReLU(0.2),
            # nn.Linear(30, 20),
            # nn.BatchNorm1d(20),
            # nn.LeakyReLU(0.2),
            # # nn.Linear(22, 12),
            # # nn.BatchNorm1d(12),
            # # nn.LeakyReLU(0.2),
            # nn.Linear(20, 10),
            # nn.BatchNorm1d(10),
            # nn.LeakyReLU(0.2),
            nn.Linear(CATEGORIES_PER_CLASS * N_CLASSES, encoding_size),
            nn.BatchNorm1d(encoding_size),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, CATEGORIES_PER_CLASS * N_CLASSES),
            nn.BatchNorm1d(CATEGORIES_PER_CLASS * N_CLASSES),
            nn.LeakyReLU(0.2),
            # nn.Linear(10, 20),
            # # nn.BatchNorm1d(12),
            # # nn.LeakyReLU(0.2),
            # # nn.Linear(12, 22),
            # nn.BatchNorm1d(20),
            # nn.LeakyReLU(0.2),
            # nn.Linear(20, 30),
            # nn.BatchNorm1d(30),
            # nn.LeakyReLU(0.2),
            nn.Linear(CATEGORIES_PER_CLASS * N_CLASSES, input_size),
            nn.Tanh()  # Using Tanh for output as features are normalized
        )

    def forward(self, x):
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # print(x.shape)
        return self.encoder(x)


def train_autoencoder(n_samples=1000, n_epochs=200, batch_size=32):
    # Initialize feature generator
    generator = FeatureGenerator()

    # Create dataset
    dataset = VisualFeatureDataset(n_samples, generator)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss / len(dataloader):.6f}')

    return model, generator


# Function to get encoding for new samples
def get_encoding(model, generator, n_samples=1):
    with torch.no_grad():
        new_samples = torch.FloatTensor(generator.generate_sample(n_samples))
        encodings = model.encode(new_samples)
    return encodings.numpy(), new_samples.numpy()

def make_RSMs(model, n_samples=1000):
    new_samples = torch.FloatTensor(generator.generate_sample(n_samples))
    if new_samples.dim() == 1:
        new_samples = new_samples.unsqueeze(0)
    # sensitivities = []
    # for feature in tqdm(range(new_samples.shape[1]), desc='testing features'):
    encodings_all = []
    for i in range(0, n_samples):
        # sample = deepcopy(new_samples[i:i+2])
        sample = np.array([deepcopy(new_samples[i]), deepcopy(new_samples[i])])
        with torch.no_grad():
            encoding0 = model.encode(torch.FloatTensor(sample)).numpy()
        encoding0 = encoding0[0, :]
        encodings_all.append(encoding0)
    encodings_all = np.array(encodings_all)
    encoding_mtxs = np.abs(encodings_all[:, None, :] - encodings_all[None, :, :])

    trils = np.tril_indices_from(encoding_mtxs[..., 0], k=-1)
    encoding_mtxs = encoding_mtxs[trils[0], trils[1], :]
    num_nans = np.sum(np.isnan(encoding_mtxs))
    print(f'Number of NaNs in encoding: {num_nans}')
    corr = np.corrcoef(encoding_mtxs.T)
    corr[np.diag_indices_from(corr)] = np.nan
    M_corr = np.nanmean(corr)
    SE_corr = np.nanstd(corr) / np.sqrt(corr.size)
    plt.imshow(corr, cmap='viridis')
    plt.title(f'RSM. {M_corr:.3f} +/- {SE_corr:.3f}')
    vabs = np.max(np.abs(corr))
    plt.clim(-vabs, vabs)
    plt.colorbar()
    plt.show()

    print(encoding_mtxs.shape)
    quit()

def sensitivity_toggle(model, n_samples=1000):

    make_RSMs(model, n_samples)

    new_samples = torch.FloatTensor(generator.generate_sample(n_samples))
    if new_samples.dim() == 1:
        new_samples = new_samples.unsqueeze(0)
    sensitivities = []
    for feature in tqdm(range(new_samples.shape[1]), desc='testing features'):
        all_difs = []
        for i in range(0, n_samples):
            # sample = deepcopy(new_samples[i:i+2])
            sample = np.array([deepcopy(new_samples[i]), deepcopy(new_samples[i])])
            sample[:, feature] = -1
            with torch.no_grad():
                encoding0 = model.encode(torch.FloatTensor(sample))
                sample[:, feature] = 1
                encoding1 = model.encode(torch.FloatTensor(sample))
            dif = encoding1 - encoding0
            # dif = np.mean(dif.numpy(), axis=0)
            # max_dif = torch.max(torch.abs(dif), dim=1)[0]
            # print(f'{max_dif=}')
            # print(dif.shape)
            # quit()
            all_difs.append(np.mean(dif.numpy(), axis=0))
        all_difs = np.array(all_difs)
        sensitivity = np.mean(np.abs(all_difs), axis=0)
        sensitivities.append(sensitivity)

    sensitivities = np.array(sensitivities)
    # print(sensitivities.shape)

    plt.imshow(sensitivities, aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()

    corr = np.corrcoef(sensitivities.T)
    corr[np.diag_indices_from(corr)] = np.nan
    plt.imshow(corr, cmap='magma')
    plt.colorbar()
    plt.show()


    quit()

# Example usage
if __name__ == "__main__":

    # x = np.random.normal(size=(10, 5))
    # y = np.random.normal(size=(30, 5))
    # print(x.shape)
    # print(y.shape)
    # corr = spatial.distance.cdist(x, y)
    # print(corr.shape)
    # quit()

    # Train the model
    model, generator = train_autoencoder()
    sensitivity_toggle(model)
    quit()

    # Generate and encode new samples
    new_encodings, originals = get_encoding(model, generator, n_samples=1000)
    # plt.imshow(originals, aspect='auto')
    # plt.colorbar()
    # plt.show()
    # quit()
    new_encodings_ = new_encodings.T
    originals = originals.T

    num_nans = np.sum(np.isnan(new_encodings_))
    print(f'Number of NaNs in encoding: {num_nans}')
    num_nans = np.sum(np.isnan(originals))
    print(f'Number of NaNs in original: {num_nans}')
    # quit()

    # print(new_encodings.shape)
    # print(originals.shape)
    # quit()

    new_encodings = stats.zscore(new_encodings_, axis=1)
    originals = stats.zscore(originals, axis=1)
    corr = new_encodings @ originals.T / new_encodings.shape[1]
    # print(corr.shape)
    # nan_idxs = np.argwhere(np.isnan(corr))
    # print(nan_idxs)
    # idx = nan_idxs[0][0]
    # print(new_encodings_[idx])


    # quit()


    # corr = new_encodings.T @ originals
    # print(corr.shape)
    #
    # # print(np.nanmean(new_encodings, axis=1).shape)
    # quit()

    # corr = spatial.distance.cdist(new_encodings.T, originals.T,
    #                               metric='correlation')
    # print(corr)
    # quit()
    plt.imshow(corr, cmap='viridis')#, vmax=1)
    plt.colorbar()
    plt.show()
    quit()

    print(f'{originals.shape=}')
    print("\nShape of encoded features:", new_encodings.shape)
    print("Sample encodings:\n", new_encodings[0])