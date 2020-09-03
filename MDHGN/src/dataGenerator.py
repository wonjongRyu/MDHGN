from utils import *
from glob import glob


def main():

    phase = ["train", "valid", "test"]
    sz = 128
    dataset_path = "../dataset"
    load_root = os.path.join(dataset_path, "animal" +
                             str(sz) + '_' + str(100000))
    save_animal_root = os.path.join(dataset_path, "animal")
    save_phase_root = os.path.join(dataset_path, "phase")

    save_phase(load_root, save_animal_root, save_phase_root, phase, sz)


def gs_algorithm(img, random_phase, num_iter):
    img_rand = np.multiply(img, random_phase)
    hologram = np.fft.ifft2(img_rand)

    """Iteration"""
    for i in range(num_iter):
        reconimg = np.fft.fft2(np.exp(1j * np.angle(hologram)))
        hologram = np.fft.ifft2(np.multiply(
            img, np.exp(1j * np.angle(reconimg))))

    """Normalization"""
    return np.angle(hologram)/2/math.pi+0.5


def save_phase(load_root, save_animal_root, save_phase_root, phase, sz):
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(sz, sz))
    for k in range(len(phase)):
        print(phase[k], "folder has started")
        images = glob(os.path.join(load_root, mode[k], "*.*"))
        for i in range(len(images)):
            img = imread(images[i])
            phase = gs_algorithm(img, random_phase, 100)
            imwrite(img, os.path.join(
                save_animal_root, mode[k], str(1+i) + ".png"))
            imwrite(phase, os.path.join(
                save_phase_root, mode[k], str(1+i) + ".png"))
            if np.mod(i+1, 1000) == 0:
                print("{}th image complete".format(i+1))


if __name__ == "__main__":
    main()
