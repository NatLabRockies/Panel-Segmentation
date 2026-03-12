"""
Test suite for panel segmentation code.
"""
import os
import pytest
import numpy as np
from panel_segmentation import panel_detection as pan_det
from tensorflow.keras.preprocessing import image as imagex
import PIL

img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"


def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'


@pytest.fixture()
def panelDetectionClass():
    '''Generate an instance of the PanelDetection() class to run unit
    tests on.'''
    # Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection()
    return pc


def testHasPanels(panelDetectionClass, satelliteImg):
    # Assert that the returned value is a boolean
    panel_loc = panelDetectionClass.hasPanels(satelliteImg)
    assert_isinstance(panel_loc, bool)
    assert panel_loc


def testCropPanels(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = panelDetectionClass.cropPanels(satelliteImg, res)
    # Assert that the 'new_res' variable is a numpy array and the dimensions.
    assert_isinstance(new_res, np.ndarray)
    assert (new_res.shape == (1, 640, 640, 3))


def testDetectAzimuth(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = panelDetectionClass.cropPanels(satelliteImg, res)
    az = panelDetectionClass.detectAzimuth(new_res)
    # Assert that the azimuth returned is a float instance
    assert_isinstance(az, float)


def testClassifyMountingConfiguration(panelDetectionClass,
                                      satelliteImg):
    (scores, labels, boxes) = \
        panelDetectionClass.classifyMountingConfiguration(
            img_file,
            acc_cutoff=.65,
            file_name_save=None)
    # Verify that we return 4 different labels, each
    # one associated with a carport installation
    assert (len(labels) == 4) & (len(scores) == 4) & (len(boxes) == 4)
    assert (all([label == 'carport-fixed' for label in labels]))
    # Assert that all scores associated with the labels are above .65
    assert (all([score > 0.65 for score in scores]))


def testPlotEdgeAz(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = panelDetectionClass.cropPanels(satelliteImg, res)
    panelDetectionClass.plotEdgeAz(
        new_res, 10, 1,
        save_img_file_path="./panel_segmentation/tests/",
        plot_show=True)
    # Open the image and assert that it exists
    im = PIL.Image.open("./panel_segmentation/tests/crop_mask_az_0.png")
    assert_isinstance(im, PIL.PngImagePlugin.PngImageFile)


def testRunSiteAnalysisPipeline(panelDetectionClass):
    site_analysis_dict = panelDetectionClass.runSiteAnalysisPipeline(
        file_name_save_img=img_file,
        file_name_save_mount=None,
        file_path_save_azimuth=None,
        generate_image=False)
    # Assert that a dictionary is returned with specific
    # attributes
    assert isinstance(site_analysis_dict, dict)
    assert (all([label == 'carport-fixed' for label in
                 site_analysis_dict["mounting_type"]]))
    assert (sorted(site_analysis_dict['associated_azimuths']) ==
            [90.0, 91.0, 161.0, 179.0])
