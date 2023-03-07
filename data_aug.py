import numpy as np
import cv2

def display_some_examples(examples, labels):
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5,i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")
    plt.show()
    
def color_normalize(train_images, test_images):
    mean = [np.mean(train_images[:, :, :, i]) for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i]) for i in range(3)]  # [62.9932, 62.0887, 66.7048]
    for i in range(3):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    return train_images, test_images

def images_augment(images):
    output = []
    for img in images:
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        if np.random.randint(0, 2):
            img = cv2.flip(img, 1)
        output.append(img[x: x+32, y:y+32, :])
    return np.ascontiguousarray(output, dtype=np.float32)
    
    