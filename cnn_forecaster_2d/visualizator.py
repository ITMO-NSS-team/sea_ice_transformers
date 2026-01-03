import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource

names_dict = {
    "kara": ((0, 140), (130, 250)),
    "laptev": ((50, 160), (210, 340)),
    "barents": ((20, 180), (50, 200)),
    "chukchi": ((180, 265), (260, 406)),
    "eastsib": ((100, 200), (260, 385)),
}


def full_name(sea_name):
    """
    Returns the full name of a sea given its short name.

        Args:
            sea_name: The short name of the sea (e.g., 'kara').

        Returns:
            str: The full name of the sea (e.g., 'Kara Sea').
    """
    names = {
        "kara": "Kara Sea",
        "laptev": "Laptev Sea",
        "barents": "Barents Sea",
        "chukchi": "Chukchi Sea",
        "eastsib": "East-Siberian Sea",
    }
    return names[sea_name]


def get_grid(sea_name):
    """
    Returns the latitude and longitude grid for a given sea.

        Loads latitude and longitude data, subsets it based on predefined indices
        for the specified sea, rotates the coordinates, and returns the resulting grid.

        Args:
            sea_name: The name of the sea to retrieve the grid for.  This name is used
                as a key to look up the appropriate indices in `names_dict`.

        Returns:
            tuple: A tuple containing two NumPy arrays:
                   - The latitude grid (lats).
                   - The longitude grid (lons).
    """
    module_path = "/path_to_data/"
    inds = names_dict[sea_name]
    lats = np.load(f"{module_path}/coordinates/Arctic_nav_lat.npy")
    lons = np.load(f"{module_path}/coordinates/Arctic_nav_lon.npy")
    lats = lats[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]]
    lons = lons[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]]
    lats = rotate_sea(lats, sea_name)
    lons = rotate_sea(lons, sea_name)
    return lats, lons


def get_topo_for_sea(sea_name):
    """
    Retrieves topography and mask data for a specified sea.

        Loads topography and coastline mask arrays, crops them to the region
        defined by the given sea name using pre-defined indices, rotates them
        to the correct orientation, and returns both arrays.

        Args:
            sea_name: The name of the sea to extract data for.  This is used as a key
                into `names_dict` to retrieve cropping indices.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the cropped and rotated
                topography array (topo) and coastline mask array (mask). Both arrays are of type float.
    """
    module_path = "/path_to_data/"
    inds = names_dict[sea_name]
    mask = np.load(f"{module_path}/coastline_masks/Arctic_mask.npy")
    mask = mask[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]]

    topo = np.load(f"{module_path}/coastline_masks/topo_bathy_nemo.npy").astype(float)
    topo = topo[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]]
    topo = rotate_sea(topo, sea_name)
    mask = rotate_sea(mask, sea_name)
    return topo, mask


def rotate_sea(image, sea_name):
    """
    Rotates a sea image for consistent orientation.

        Flips the image horizontally and rotates it 180 degrees.
        Additionally, performs an extra 270-degree rotation for 'chukchi' and 'eastsib' seas.

        Args:
            image: The input image as a NumPy array.
            sea_name: The name of the sea (string).  Used to determine if additional rotation is needed.

        Returns:
            NumPy array: The rotated image.
    """
    image = np.fliplr(image)
    image = np.rot90(image, 2)
    if sea_name in ["chukchi", "eastsib"]:
        image = np.rot90(image, 3)
    return image


def get_extent(image, sea_name):
    """
    Extracts a region of interest from an image based on predefined indices.

        Args:
            image: The input image (e.g., a NumPy array).
            sea_name:  The name identifying the desired region within the `names_dict`.

        Returns:
            A NumPy array representing the extracted region of the image.
    """
    inds = names_dict[sea_name]
    image = image[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]]
    return image


def return_sea_on_arctic(sea_name):
    """
    Returns a mask representing the specified sea on an Arctic projection.

        Args:
            sea_name: The name of the sea to extract.  This should correspond to a key in `names_dict`.

        Returns:
            numpy.ndarray: A 2D numpy array representing the sea's mask rotated for the Arctic projection.
    """
    module_path = "/path_to_data/"
    inds = names_dict[sea_name]
    mask = np.load(f"{module_path}/coastline_masks/Arctic_mask.npy")
    sea_mask = np.zeros(mask.shape)
    sea_mask[inds[0][0] : inds[0][1], inds[1][0] : inds[1][1]] = 1
    return rotate_sea(sea_mask, "Arctic")


def plot_comparison_map(
    prediction: np.ndarray, real: np.ndarray, sea_name: str, title: str, save=None
):
    """
    Plots a comparison map of prediction and real values for a given sea.

        Args:
            prediction: The predicted numpy array.
            real: The real numpy array.
            sea_name: The name of the sea region.
            title: The title of the plot.
            save: Optional path to save the figure, if None it saves in 'results/images'.

        Returns:
            None
    """
    lats, lons = get_grid(sea_name)
    topo, mask = get_topo_for_sea(sea_name)

    prediction = rotate_sea(prediction, sea_name)
    prediction[mask == 1] = None
    real = rotate_sea(real, sea_name)
    real[mask == 1] = None

    ls = LightSource(azdeg=315, altdeg=45)
    plt.rcParams["figure.figsize"] = (8, 4.7)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(ls.hillshade(topo, vert_exag=0.01), cmap="gray")
    ax1.imshow(prediction, cmap="Blues_r", vmin=0, vmax=1)
    CS = ax1.contour(prediction, [0.1, 0.4, 0.6, 0.8], cmap="Blues")
    ax1.clabel(CS, CS.levels, inline=True, fontsize=8)
    ax1.set_yticks(
        np.arange(mask.shape[0])[::25], np.rint(lats[:, 0][::25]).astype(int)
    )
    ax1.set_xticks(
        np.arange(mask.shape[1])[::25], np.rint(lons[-1, :][::25]).astype(int)
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.contour(lats, np.arange(70, 90, 5), colors="gray", linewidths=0.4)
    ax1.contour(lons, np.arange(-189, 180, 10), colors="gray", linewidths=0.4)
    ax1.contour(mask, [0], colors="black", linewidths=2)

    ax2.imshow(ls.hillshade(topo, vert_exag=0.01), cmap="gray")
    ax2.imshow(real, cmap="Blues_r", vmin=0, vmax=1)
    CS = ax2.contour(real, [0.1, 0.4, 0.6, 0.8], cmap="Blues")
    ax2.clabel(CS, CS.levels, inline=True, fontsize=8)
    ax2.set_yticks(
        np.arange(mask.shape[0])[::25], np.rint(lats[:, 0][::25]).astype(int)
    )
    ax2.set_xticks(
        np.arange(mask.shape[1])[::25], np.rint(lons[-1, :][::25]).astype(int)
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.contour(lats, np.arange(70, 90, 5), colors="gray", linewidths=0.4)
    ax2.contour(lons, np.arange(-189, 180, 10), colors="gray", linewidths=0.4)
    ax2.contour(mask, [0], colors="black", linewidths=2)

    fig.suptitle(title)

    ax1.set_title(f"Prediction")
    ax2.set_title(f"Real")

    if save is None:
        if not os.path.exists(f"results/images/{sea_name}"):
            os.makedirs(f"results/images/{sea_name}")
        plt.savefig(
            f'results/images/{sea_name}/{title.split(",")[0].replace("/", "")}.png',
            dpi=150,
        )
        print(f'results/images/{sea_name}/{title.split(",")[0].replace("/", "")}.png')
        plt.close()
    else:
        if not os.path.exists(f"{save}/{sea_name}"):
            os.makedirs(f"{save}/{sea_name}")
        plt.savefig(
            f'{save}/{sea_name}/{title.split(",")[0].replace("/", "")}.png', dpi=150
        )
        print(f'{save}/{sea_name}/{title.split(",")[0].replace("/", "")}.png')
        plt.close()
