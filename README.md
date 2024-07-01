# RADTrack

**RADTrack** is the modified version of the [RADDet](https://github.com/ZhangAoCanada/RADDet) dataset which incorporates the Multi-Object Tracking (MOT) use case. **RADTrack** transforms the RADDet data into sequences showing different times and locations where the data was recorded. Labeled object IDs allow for tracking of these objects using Range-Azimuth-Doppler (RAD) data. 

With **RADTrack** you can:
- Train and evaluate an object detector on Range-Azimuth, Range-Doppler and Cartesian Radar data
- Train and evaluate an object classifier on Range-Azimuth, Range-Doppler and Cartesian Radar data
- Train and evaluate a tracker on Range-Azimuth, Range-Doppler and Cartesian Radar data

## Radar(stationary) Dataset for Dynamic Road Users

### Dataset link

- TBA

### Dataset format

The folder structure and additional meta data resemble the [MOT20](https://arxiv.org/abs/2003.09003) format allowing to use standard evaluation tools like [TrackEval](https://github.com/JonathonLuiten/TrackEval) only after minimal change of the GT data format.

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

The dataset contains totally **10158 frames** distributed through **24 sequences**. 
For the data capture, the same radar configuration through the entire research was used. 
The details of the data capture is shown below and available in the `sensors_para/radar_config.json` file.
```json
"designed_frequency":       76.8 Hz,
"config_frequency":         77 Hz,
"maximum_range":            50 m,
"range_size":               256,
"azimuth_size":             256,
"doppler_size":             64,
"range_resolution":         0.1953125 m/bin,
"angular_resolution":       0.006135923 radian/bin,
"velocity_resolution":      0.41968030701528203 (m/s)/bin
```

The dataset has 6 classes, different input formats and ground truth formats. All the information that stored in the dataset can be concluded as follow.
- **RAD:** 3D-FFT radar data as a matrix of `complex64` numbers with size (256, 256, 64) saved in numpy format.
- **stereo_image:**	2 rectified stereo images.
- **gt:** Ground truth as JSON file. It includes a list of `{ "classes", "boxes", "cart_boxes", "ids" }` dictionaries where each one represents one frame. Each field in the dictionary contains a list of size `n` where `n` is the number of radar objects in the frame.
- **sensors_para:** `stereo_para` for stereo depth estimation, and `registration_matrix` for cross-sensor registration.
- **seqmaps:** Contains a sequence mapping file (csv) for each split. This tells other programs which splits exist and what sequences are included.
- **frame_mapping.txt:** Contains a mapping of matching frames in [RADDet](https://github.com/ZhangAoCanada/RADDet) and RADTrack datasets as a CSV file. It is included to support backwards compatibility to the [RADDet](https://github.com/ZhangAoCanada/RADDet) dataset.
- **seqinfo.ini:** Contains meta information about a sequence. E.g. the length of the sequence.

**Note:** for `classes`, they are `["person", "bicycle", "car", "motorcycle", "bus", "truck"]`.  
**Also Note:** for `boxes`, the format is `[x_center, y_center, z_center, w, h, d]` where `x` is Range, `y` is Angle and `z` is Doppler axis.  
**Also Note:** for `cart_box`, the format is `[y_center, x_center, h, w]`.  
**Also Note:** for `stereo_para`, `left_maps.npy` and `right_maps.npy` are derived from `cv2.initUndistortRectifyMap(...)` and include the maps in both `x` and `y` directions; all other matrices are derived from `cv2.stereoRectify(...)`.  

### Dataset splits

The dataset is split as follows: **70%** training data, **20%** validation data, **10%** test data.

If one wished to, the splits can be merged. Simply copy each sequence directory into the desired split folder and adjust the `seqmaps` files accordingly.

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

The dataset is licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

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
@inproceedings{RADTrack2024,
    author = {Dell, Martin and Bradfisch, Wolfgang and Schober, Steffen and Klöck, Clemens},
    title = {{RadarMOTR: Multi-Object Tracking with Transformers on Range-Doppler Maps}},
    booktitle = {International Conference Radar 2024 (RADAR2024)},
    year = {2024}
}
```
