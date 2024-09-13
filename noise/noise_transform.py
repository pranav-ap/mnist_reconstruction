from skimage.util import random_noise
import torchvision.transforms as T

class AddSpeckleNoise:
    def __init__(self, var=0.05):
        self.var = var

    def __call__(self, img):
        img = img.detach().numpy()
        noisy_img = random_noise(img, mode='speckle', var=self.var, clip=True)
        return torch.from_numpy(noisy_img) # .permute(2, 0, 1)  # Convert numpy array back to tensor

class AddSaltAndPepperNoise:
    def __init__(self, salt_vs_pepper=0.5):
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, img):
        img = img.detach().numpy()
        noisy_img = random_noise(img, mode='s&p', salt_vs_pepper=self.salt_vs_pepper, clip=True)
        return torch.from_numpy(noisy_img) # .permute(2, 0, 1)  # Convert numpy array back to tensor

class AddGaussianNoise:
    def __init__(self, var=0.05):
        self.var = var

    def __call__(self, img):
        img = img.detach().numpy()
        noisy_img = random_noise(img, mode='gaussian', var=self.var, clip=True)
        return torch.from_numpy(noisy_img) # .permute(2, 0, 1)  # Convert numpy array back to tensor

# Example transformation pipeline
noise_transform = T.Compose([
    T.ToTensor(),  # Convert to tensor before normalization
    T.RandomAdjustSharpness(sharpness_factor=2.0),
    T.GaussianBlur(kernel_size=(3, 3), sigma=1.0),
    T.RandomApply([AddSpeckleNoise(var=0.02)], p=0.7),
    T.RandomApply([AddSaltAndPepperNoise(salt_vs_pepper=0.2)], p=0.7),
    T.RandomApply([AddGaussianNoise(var=0.05)], p=0.7),
    # T.Normalize((0.5,), (0.5,)),  # Apply normalization if needed
])
