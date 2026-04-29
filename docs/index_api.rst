.. currentmodule:: panel_segmentation

API Reference
=============
The following gives a description of the classes and functions used in the Panel-Segmentation package.
The main panel segmentation pipeline is used to detect solar panels in satellite imagery, using the pre-trained models provided in the Panel-Segmentation package.


Utilities
---------
Helper functions and utilities.

.. autosummary::
   :toctree: generated/
   :caption: Utilities

   utils.downloadModel
   utils.generateSatelliteImage
   utils.generateAddress
   utils.generateSatelliteImageryGrid
   utils.visualizeSatelliteImageryGrid
   utils.splitTifToPngs
   utils.locateLatLonGeotiff
   utils.translateLatLongCoordinates
   utils.getInferenceBoxLatLonCoordinates
   utils.binaryMaskToPolygon
   utils.convertMaskToLatLonPolygon
   utils.convertPolygonToGeojson
   utils.detectAzimuth
   utils.plotEdgeAz
   utils.getRectangleDimensions
   utils.standardizeRectangleWidth

LiDAR
-----
LiDAR data processing utilities.

.. TODO: ADD LIDAR FUNCTIONS TO THE BELOW AFTER LIDAR PR IS MERGED

.. Point Cloud Data (PCD) Processing
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. Processing and filtering functions for Point Cloud Data (PCD) files.

.. .. autosummary::
..    :toctree: generated/
..    :caption: Point Cloud Data (PCD) Processing  

..    lidar.pcd_data.PCD

.. Plane Segmentation
.. ^^^^^^^^^^^^^^^^^^
.. Plane segmentation functions that segments planes from point cloud data and calculates its tilt and azimuth.

.. .. autosummary::
..    :toctree: generated/
..    :caption: Plane Segmentation

..    lidar.plane_segmentation.planeSegmentation

.. USGS LiDAR API
.. ^^^^^^^^^^^^^^
.. API for downloading USGS LiDAR data.

.. .. autosummary::
..    :toctree: generated/
..    :caption: USGS LiDAR API

..    lidar.usgs_lidar_api.USGSLidarAPI


Models
======
The following deep learning models are included in the Panel-Segmentation package:

Panel Detection Models
----------------------
* **panel_detection_model.pth**: This is the configuration file for the DL instance segmentation panel detection model.
* **panel_detection_model.py**: This is the DL isntance segmentation model, which detects and classifies solar array mounting configuration.
* **sol_searcher_config.py**: This is the configuration file for the DL object detection sol-searcher model.
* **sol_searcher_model.pth**: This is the checkpoint file for the DL object detection sol-searcher model, which searches for solar panels in satellite imagery.This model is trained on 3783 images from Google Maps imagery of the Austin, TX area from November 2023 and Denver, CO area from June 2023 with a resolution of 0.2986 meters per pixel (Google Maps zoom level 19). The architecture of the model is RTMDet-X with a mAP-50 score of 0.884.

Extreme Weather: Hail Models
----------------------------
* **hail_config.py**: This is the configuration file for the DL instance segmentation hail model.
* **hail_model.pth**: This is the checkpoint file for the DL instance segmentation hail model, which detects hail on solar arrays in satellite imagery. This model is trained on 1883 images from Google Maps imagery of the Austin, TX area from November 2023 with a resolution of 0.0746 meters per pixel (Google Maps zoom level 21). The architecture of the model is RTMDet-Ins-X with a mAP-50 score of 0.859.

Extreme Weather: Hurricane Models
---------------------------------
* **pre_hurricane_config.py**: This is the configuration file for the DL instance segmentation pre-hurricane model.
* **pre_hurricane_model.pth**: This is the checkpoint file for the DL instance segmentation pre-hurricane model, which detects solar arrays in pre-hurricane satellite imagery. This model is trained on 1883 images from Google Maps imagery of various areas before hurricane impact with a resolution of 0.0746 meters per pixel (Google Maps zoom level 21). Many of these images includes Puerto Rico. The architecture of this model is RTMDet-Ins-l with a mAP-50 score of 0.942.
* **post_hurricane_config.py**: This is the configuration file for the DL instance segmentation post-hurricane model.
* **post_hurricane_model.pth**: This is the checkpoint file for the DL instance segmentation post-hurricane model, which detects solar arrays in post-hurricane satellite imagery. This model is trained on 863 images from NOAA post-Hurricane Maria satellite imagery of the Puerto Rico area with a resolution of 0.0746 meters per pixel (Google Maps zoom level 21). The architecture of this model is Mask-RCNN X-101-64x4d-FPN with a mAP-50 score of 0.844.

Automated Geotagging Models
---------------------------
* **automated_geotagging_config.py**: This is the configuration file for the DL instance segmentation automated geotagging model.
* **automated_geotagging_model.pth**: This is the checkpoint file for the DL instance segmentation hail model, which detects site equipment in satellite imagery. This model is trained on utility PV sites with a resolution of 0.0746 meters per pixel (Google Maps zoom level 21). The architecture of the model is RTMDet-Ins-X with a mAP-50 score of 0.892.
