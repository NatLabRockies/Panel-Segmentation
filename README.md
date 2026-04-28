![Panel Segmentation Icon](docs/_static/panel_segmentation_cropped_icon.png)

This repo contains the scripts for automated metadata extraction of solar PV installations, 
using satellite imagery coupled with computer vision techniques. In this package, the user
can perform the following actions:
- Automatically generate a satellite image using a set of lat-long coordinates, and a Google 
Maps API key. To get a Google Maps API key, go to the following site and set up an account:
https://developers.google.com/maps/documentation/maps-static/get-api-key
- Determine the presence of a solar array in the satellite image (boolean True/False), using a 
classification model (VGG16_classification_model.h5).
- Perform image segmentation on the satellite image, to locate the solar array(s) in the 
image on a pixel-by-pixel basis, using an image segmentation model (VGG16Net_ConvTranpose_complete.h5).
- Using connected components clustering, isolate individual solar arrays in the masked image.
- Perform azimuth estimation on each solar array cluster in the masked image.
- Using an object detection model (Faster R-CNN Resnet 50 trained via transfer learning), detect
and classify mounting type and configuration of solar installations in satellite imagery. This includes
classification of fixed tilt and single-axis trackers, as well as the rooftop, 
ground, and carport mounting configurations.
- Detect solar panels and get its latitude, longitude, and address within a geographic bounding box through the SOL-Searcher Pipeline.
- Detect and calculate hurricane damage on solar installations given pre-hurricane and post-hurricane satellite imagery through the Hurricane Detection Pipeline.
- Detect and calculate hail damage on solar installations given satellite imagery through the Hail Detection pipeline.
- Convert NOAA MESH (Maximum Estimated Size of Hail) grib2 files into kml or geojson files.
- Estimate tilt and azimuth of a solar array by processing USGS LiDAR data for the array's location.
- Automatically estimate the solar array's tilt and azimuth by pulling and processing USGS LiDAR data
within a shapely polygon outlining the solar array.

To install Panel-Segmentation, perform the following steps:

1. You must have Git large file storage (lfs) on your computer in order to download the deep learning models in this package. Go to the following site to download Git lfs: 

https://git-lfs.github.com/

2. Once git lfs is installed, you can now install Panel-Segmentation on your computer. We are still working on making panel-segmentation available via PyPi, so entering the following in the command line will install the package locally on your computer:

git clone https://github.com/NatLabRockies/Panel-Segmentation.git
cd Panel-Segmentation
pip install .

3. Panel-Segmentation requires the MMCV package, which can be tricky to install for CPU-only, and needs to be installed from source. To install MMCV for source, run the following in the command line:

pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0

Please note that installation will likely take several minutes, but is necessary for running any of the storm-related CV models.

3. When initiating the PanelDetection() class, be sure to point your file paths to the model paths in your local Panel-Segmentation folder!




