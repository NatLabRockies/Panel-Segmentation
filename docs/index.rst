
.. Panel-Segmentation documentation master file, created by
   sphinx-quickstart on Mon Nov 2 00:00:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. currentmodule:: panel_segmentation
   
.. image:: _static/panel_segmentation_cropped_icon.png
   :alt: Panel Segmentation Icon
   :align: center
   :scale: 50%
   
Introduction
============
This repo contains the scripts for automated metadata extraction of solar PV installations, using satellite imagery coupled with computer vision techniques. In this package, the user can perform the following actions:

- Automatically generate a satellite image using a set of lat-long coordinates, and a Google Maps API key. Users would need to set up a Google Cloud account and get a Maps Static API key. Please refer to :ref:`google-api-key-setup` section for this process.  
- Determine the presence of a solar array in the satellite image (boolean True/False), using a classification model (VGG16_classification_model.h5).
- Perform image segmentation on the satellite image, to locate the solar array(s) in the image on a pixel-by-pixel basis, using an image segmentation model (VGG16Net_ConvTranpose_complete.h5).
- Using connected components clustering, isolate individual solar arrays in the masked image.
- Perform azimuth estimation on each solar array cluster in the masked image.
- Using an object detection model (Faster R-CNN Resnet 50 trained via transfer learning), detect and classify mounting type and configuration of solar installations in satellite imagery. This includes classification of fixed tilt and single-axis trackers, as well as the rooftop, ground, and carport mounting configurations.
- Detect solar panels and get its latitude, longitude, and address within a geographic bounding box through the SOL-Searcher Pipeline.
- Detect and calculate hurricane damage on solar installations given pre-hurricane and post-hurricane satellite imagery through the Hurricane Detection Pipeline.
- Detect and calculate hail damage on solar installations given satellite imagery through the Hail Detection pipeline.
- Convert NOAA MESH (Maximum Estimated Size of Hail) grib2 files into kml or geojson files.
- Estimate tilt and azimuth of a solar array by processing USGS LiDAR data for the array's location.

Documentation Contents
======================

.. include a toctree entry here so that the index page appears in the top navigation bar

.. toctree::
   :maxdepth: 2

   self
   index_getting_started
   index_api
   examples/index_examples
   changelog/index_changelog
   

Navigation
==========
* :ref:`genindex` - Function index
* :ref:`search` - Search the documentation
