# RADTrack

**RADTrack** is the modified version of the [RADDet](https://github.com/ZhangAoCanada/RADDet) dataset which incorporates the Multi-Object Tracking (MOT) use case. 
**RADTrack** transforms the [RADDet](https://github.com/ZhangAoCanada/RADDet) data into sequences showing different times and locations where the data was recorded. 
Labeled object IDs allow for tracking of these objects using Range-Azimuth-Doppler (RAD) data.

With the **RADTrack** dataset, you can:
- Train and evaluate object detection models on Range-Azimuth, Range-Doppler, and Cartesian Radar data
- Train and evaluate object classification models on Range-Azimuth, Range-Doppler, and Cartesian Radar data
- Train and evaluate object tracking models on Range-Azimuth, Range-Doppler, and Cartesian Radar data

## Paper

- This dataset is introduced and used by [RadarMOTR](https://github.com/madeit07/RadarMOTR) which is accepted in 2024 International Radar Conference (RADAR).
- [Accepted Preprint](https://bwsyncandshare.kit.edu/s/zCgc5o89L44oN5a)
- [IEEE Xplore Version](https://ieeexplore.ieee.org/document/10994166)

## Radar(stationary) Dataset for Dynamic Road Users

### Dataset link

- [Nextcloud](https://nx47801.your-storageshare.de/s/7JDez5BXZEE5gPr)

### Dataset format

The folder structure and additional metadata in RADTrack are designed to mirror the [MOT20](https://arxiv.org/abs/2003.09003) format. 
This similarity enables seamless integration with standard evaluation tools like [TrackEval](https://github.com/JonathonLuiten/TrackEval), 
requiring only minor adjustments to the ground truth (GT) data format.

```
|-- {RADTrack ROOT}
|   |-- radtrack-train
|   |   |-- radtrack0001
|   |   |   |-- RAD
|   |   |   |   |-- 000001.npy
|   |   |   |   |-- 000002.npy
|   |   |   |   |-- ...
|   |   |   |-- stereo_image
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- 000002.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.json
|   |   |   |-- frame_mapping.txt
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- radtrack-val
|   |   |-- ...
|   |-- radtrack-test
|   |   |-- ...
|   |-- seqmaps
|   |   |-- radtrack-train.txt
|   |   |-- radtrack-val.txt
|   |   |-- radtrack-test.txt
|   |-- sensors_para
|   |   |-- registration_matrix
|   |   |   |-- ...
|   |   |-- stereo_para
|   |   |   |-- ...
|   |   |-- radar_config.json
```

### Dataset details

The RADTrack dataset comprises **10,158 frames**, organized into **24 sequences**. 
To ensure consistency, the same radar configuration throughout the entire data collection process was used. 
The specifics of the data capture settings are summarized below and can also be found in the `sensors_para/radar_config.json` file.

```jsonc
"designed_frequency":  76.8, // Hz
"config_frequency":    77, // Hz
"maximum_range":       50, // m
"range_size":          256,
"azimuth_size":        256,
"doppler_size":        64,
"range_resolution":    0.1953125, // m/bin
"angular_resolution":  0.006135923, // radian/bin
"velocity_resolution": 0.41968030701528203, // (m/s)/bin
```

The dataset consists of 6 classes and various input and ground truth formats. Below is a summary of the information stored in the dataset.

- **RAD:** 3D-FFT radar data stored as a matrix of `complex64` numbers with dimensions (256, 256, 64) in NumPy format.
- **stereo_image:**	Two rectified stereo images.
- **gt:** A JSON file containing a list of dictionaries. Each dictionary corresponds to a specific frame and includes a list of entries, where the number of entries is equal to the number of radar objects present in that frame. The dictionary has the following keys: 
  - `classes`: A list of class labels. There are 6 classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.
  - `boxes`: A list of bounding box coordinates in the format `[x_center, y_center, z_center, w, h, d]`, where `x` represents Range, `y` represents Angle, and `z` represents Doppler axis.
  - `cart_boxes`: A list of Cartesian bounding box coordinates in the format `[y_center, x_center, h, w]`.
  - `ids`: A list of object IDs.
- **sensors_para:** Includes `stereo_para` for stereo depth estimation and `registration_matrix` for cross-sensor registration.
- **seqmaps:** A CSV file for each split, indicating which sequences are included.
- **frame_mapping.txt:** A CSV file containing a mapping of matching frames between the RADDet and RADTrack datasets for backwards compatibility.
- **seqinfo.ini:** A file containing meta information about each sequence, such as its length.

> [!NOTE]
> The `stereo_para` includes `left_maps.npy` and `right_maps.npy`, which are derived from `cv2.initUndistortRectifyMap(...)` and contain maps in both `x` and `y` directions. All other matrices are derived from `cv2.stereoRectify(...)`.

### Dataset splits

The dataset is divided into three splits: **70%** for training, **20%** for validation, and **10%** for testing. 
If you need to modify these splits, you can merge them by copying the sequence directories into the desired split folder and updating the corresponding `seqmaps` files.

### Statistics

| Statistics                        | Test  | Train | Validation | Overall |
| --------------------------------- | :---: | :---: | :--------: | :-----: |
| Frames                            |  585  | 7528  |    2045    |  10158  |
| Objects                           |  88   | 1313  |    295     |  1696   |
| Frames with one object            |  104  | 1273  |    499     |  1876   |
| Frames with two objects           |  166  | 1918  |    758     |  2842   |
| Frames with three objects         |  116  | 2071  |    441     |  2628   |
| Frames with four objects          |  87   | 1249  |    238     |  1574   |
| Frames with five objects          |  85   |  694  |     71     |   850   |
| Frames with six objects           |  26   |  252  |     35     |   313   |
| Frames with seven or more objects |   1   |  71   |     3      |   75    |

| Classes     | Test  | Train | Validation | Overall |
| ----------- | :---: | :---: | :--------: | :-----: |
| cars        |  64   |  926  |    203     |  1193   |
| trucks      |  13   |  202  |     45     |   260   |
| buses       |   0   |  10   |     1      |   11    |
| bicycles    |   2   |  24   |     9      |   35    |
| persons     |   9   |  153  |     36     |   198   |
| motorcycles |   0   |   4   |     1      |    5    |

### Dataset license

The tools and code are licensed under [MIT](./LICENSE). The dataset is licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## RADTrack Labeling/Viewing Tool

The labeling tool for the annotation of the object IDs and class can be located in the `labeling_tool` folder. It also can be used as a dataset viewer.

### Installation

**Requirements:**

* Dataset
* [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) used as virtual environment (Skip if python venv is used)

1. Clone repository
    ```sh
    git clone https://github.com/madeit07/RADTrack.git
    cd RADTrack/labeling_tool
    ```

2. Create virtual environment (e.g. using conda)
    ```sh
    conda create -n radtrack python=3.11
    conda activate radtrack
    ```

3. Install other requirements
    ```sh
    pip install -r requirements.txt
    ```

4. *(Optional)* If you want to use the scripts in the `tools` folder, install those requirements as well:
    ```sh
    pip install -r ../tools/requirements.txt
    conda install ffmpeg # Only for tools/visualize.py
    ```

### Run Labeling Tool

Execute Python script in console:
```sh
python radtrack_labeler.py
```

Under File->New... create a new project.

After the project is opened, all RAD data is processed and cached so that scrubbing through the images is responsive. This is done only once and can a long time.

### Default project

If you don't want to load the project every time you start the tool, you can set a default project which gets automatically loaded on startup.
Create a new project and configure it. Name it `Default` and save it under `labeling_tool\projects\`.

### Keybinds

**Playback Controls:**  

| Action              |   Keybind    |   Alternative Keybind   |
| ------------------- | :----------: | :---------------------: |
| Next frame          | <kbd>D</kbd> | <kbd>&rightarrow;</kbd> |
| Previous frame      | <kbd>A</kbd> | <kbd>&leftarrow;</kbd>  |
| Jump to first frame | <kbd>Q</kbd> |     <kbd>Home</kbd>     |
| Jump to last frame  | <kbd>E</kbd> |     <kbd>End</kbd>      |
| Next sequence       | <kbd>S</kbd> | <kbd>&downarrow;</kbd>  |
| Previous sequence   | <kbd>W</kbd> |  <kbd>&uparrow;</kbd>   |

**Labeling:**

> [!WARNING]
> Alternative keybinds for playback control do not work while an object is selected. They are used to control the ID field.

| Action                                     |                       Keybind                       |
| ------------------------------------------ | :-------------------------------------------------: |
| Select next object                         |                    <kbd>F</kbd>                     |
| Select previous object                     |           <kbd>Ctrl</kbd> + <kbd>F</kbd>            |
| Deselect object                            |                    <kbd>R</kbd>                     |
| Set ID in current frame                    |                  <kbd>Enter</kbd>                   |
| Set ID in all frames                       |         <kbd>Ctrl</kbd> + <kbd>Enter</kbd>          |
| Set ID in current and all following frames |          <kbd>Alt</kbd> + <kbd>Enter</kbd>          |
| Set ID in current and all previous frames  | <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>Enter</kbd> |

**Utilities:**

| Action                    |            Keybind             |
| ------------------------- | :----------------------------: |
| Backup ground truth files | <kbd>Ctrl</kbd> + <kbd>B</kbd> |
| Reorder IDs in sequence   | <kbd>Ctrl</kbd> + <kbd>R</kbd> |

> [!NOTE]
> To be able to activate the <kbd>Enter</kbd> keybinds the focus must be in the ID field.

## Credits

The original dataset [RADDet](https://github.com/ZhangAoCanada/RADDet) which RADTrack is based on, was created by Ao Zhang, Farzan Erlik Nowruzi and Robert Laganiere from University of Ottawa and Sensorcortek Inc.

## Citation

Please use the following citation when using the dataset:

```bib
@inproceedings{RadarMOTR2024,
    author = {Dell, Martin and Bradfisch, Wolfgang and Schober, Steffen and Klöck, Clemens},
    title = {{RadarMOTR: Multi-Object Tracking with Transformers on Range-Doppler Maps}},
    booktitle = {2024 International Radar Conference (RADAR)},
    year = {2024},
    pages = {1-6},
    doi = {10.1109/RADAR58436.2024.10994166}
}
```
