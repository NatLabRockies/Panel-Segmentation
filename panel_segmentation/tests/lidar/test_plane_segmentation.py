from panel_segmentation.lidar.plane_segmentation import PlaneSegmentation
import pytest
import os
import laspy
import numpy as np
import open3d as o3d
import pandas as pd
import pyproj
import os
os.environ["OMP_NUM_THREADS"] = "1"

example_path = os.path.join("panel_segmentation", "examples",
                            "lidar_tilt_azimuth_detection_examples")


@pytest.fixture
def planeSegmentationClass():
    """
    A PlaneSegmentation class with an example point cloud data (pcd).
    """
    # Get the example laz file
    laz_file_path = os.path.join(
        example_path,
        "cropped_USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_3097513c1" +
        "_LAS_2019.laz")
    # Read in the laz file and create a laspy.LasData object
    with laspy.open(laz_file_path) as laz_reader:
        laz_data = laz_reader.read()
    # Create a o3d.geometry.PointCloud object from laz data
    point_clouds = np.column_stack(
        [laz_data["X"], laz_data["Y"], laz_data["Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    # Initialize PlaneSegmentation class with pcd
    ps = PlaneSegmentation(pcd)
    return ps


@pytest.fixture
def pcdInfoParams():
    """
    Contains a sample information of the pcd, specifically its source crs,
    scales, and offsets.
    """
    source_crs = pyproj.CRS.from_epsg(6343)
    scales = (0.01, 0.01, 0.01)
    offsets = (0, 0, 0)
    return source_crs, scales, offsets


@pytest.fixture
def planeLatLonParams():
    """
    Contains a sample plane lat, lon coordinates.
    """
    lat = 30.1478
    lon = -97.74105
    return lat, lon


def testSegmentPlanesTypeErrors(planeSegmentationClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if distance_threshold is correct
    with pytest.raises(
            TypeError,
            match="distance_threshold variable must be of type float."):
        planeSegmentationClass.segmentPlanes(distance_threshold=1234)
    # Test if ransac_n is correct
    with pytest.raises(TypeError,
                       match="ransac_n variable must be of type int."):
        planeSegmentationClass.segmentPlanes(ransac_n=0.2)
    # Test if num_ransac_iterations is correct
    with pytest.raises(
            TypeError,
            match="num_ransac_iterations variable must be of type int."):
        planeSegmentationClass.segmentPlanes(num_ransac_iterations="873")
    # Test if min_plane_points is correct
    with pytest.raises(TypeError,
                       match="min_plane_points variable must be of type int."):
        planeSegmentationClass.segmentPlanes(min_plane_points=9.0)
    # Test if max_num_planes is correct
    with pytest.raises(TypeError,
                       match="max_num_planes variable must be of type int."):
        planeSegmentationClass.segmentPlanes(max_num_planes="10")


def testSegmentPlanesResult(planeSegmentationClass):
    """
    Tests if a .plane_list attribute is created and a None value is returned
    after running segmentPlanes.
    """
    # Test segmentPlanes function
    result = planeSegmentationClass.segmentPlanes()
    # Assert that a laneSegmentationClass.plane_list attribute is created and
    # is a list
    assert isinstance(planeSegmentationClass.plane_list, list)
    # Assert that it is a list of dictionaries
    assert all(isinstance(plane, dict)
               for plane in planeSegmentationClass.plane_list)
    # Assert that the desired keys are in the dictionary
    expected_dict_keys = ["plane_id", "normal_plane_vectors", "tilt",
                          "azimuth", "num_points", "pcd", "color"]
    for plane in planeSegmentationClass.plane_list:
        assert all(key in plane for key in expected_dict_keys)
    # Assert that the result is None
    assert result is None


def testMergeSimilarPlanesTypeErrors(planeSegmentationClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if tilt_diff_threshold is correct
    with pytest.raises(
            TypeError,
            match="tilt_diff_threshold variable must be of type float."):
        planeSegmentationClass.mergeSimilarPlanes(tilt_diff_threshold=7645)
    # Test if azimuth_diff_threshold is correct
    with pytest.raises(
            TypeError,
            match="azimuth_diff_threshold variable must be of type float."):
        planeSegmentationClass.mergeSimilarPlanes(
            azimuth_diff_threshold="dsf4")


def testMergeSimilarPlanesResult(planeSegmentationClass):
    """
    Tests if a new .plane_list attribute (with the merged planes) is created
    and a None value is returned after running mergeSimilarPlanes.
    """
    # Run segmentPlanes function to get plane_list attribute
    planeSegmentationClass.segmentPlanes()
    # Test mergeSimilarPlanes function
    result = planeSegmentationClass.mergeSimilarPlanes()
    # Assert that a laneSegmentationClass.plane_list attribute is created and
    # is a list
    assert isinstance(planeSegmentationClass.plane_list, list)
    # Assert that it is a list of dictionaries
    assert all(isinstance(plane, dict)
               for plane in planeSegmentationClass.plane_list)
    # Assert that the desired keys are in the dictionary
    expected_dict_keys = ["plane_id", "normal_plane_vectors", "tilt",
                          "azimuth", "num_points", "pcd", "color",
                          "combined_from"]
    for plane in planeSegmentationClass.plane_list:
        assert all(key in plane for key in expected_dict_keys)
    # Assert that the result is None
    assert result is None


def testVisualizePlanesResult(planeSegmentationClass):
    """
    Tests if a list of o3d.geometry.TriangleMesh objects is returned after
    running visualizePlanes.
    """
    # Run segmentPlanes function to get plane_list attribute
    planeSegmentationClass.segmentPlanes()
    # Test visualizePlanes function
    result_mesh = planeSegmentationClass.visualizePlanes()
    # Assert that the result is a list
    assert isinstance(result_mesh, list)
    # Assert that the result is a list of dictionaries
    assert all(isinstance(plane, dict)
               for plane in result_mesh)
    # Assert that the desired keys are in the dictionary
    expected_dict_keys = ["plane_id", "pcd", "plane_mesh", "color"]
    for plane in result_mesh:
        assert all(key in plane for key in expected_dict_keys)


def testCreateSummaryPlaneDataframeTypeErrors(planeSegmentationClass,
                                              pcdInfoParams):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Get a sample source crs, scales, and offsets from the pcd
    source_crs, scales, offsets = pcdInfoParams
    # Test if source_crs is correct
    with pytest.raises(
            TypeError,
            match="source_crs variable must be of a pyproj.crs.CRS object."):
        planeSegmentationClass.createSummaryPlaneDataframe(
            source_crs="EPSG:4326", scales=scales, offsets=offsets)
    # Test if scales is correct
    with pytest.raises(TypeError,
                       match="scales variable must be of type tuple, list, " +
                       "or numpy.ndarray."):
        planeSegmentationClass.createSummaryPlaneDataframe(
            source_crs=source_crs, scales="qwe", offsets=offsets)
    # Test if offsets is correct
    with pytest.raises(TypeError,
                       match="offsets variable must be of type tuple, list, " +
                       "or numpy.ndarray."):
        planeSegmentationClass.createSummaryPlaneDataframe(
            source_crs=source_crs, scales=scales, offsets="[124, 5678, 91011]")


def testCreateSummaryPlaneDataframeResult(planeSegmentationClass,
                                          pcdInfoParams):
    """
    Tests if a pandas.DataFrame object is returned after running
    createSummaryPlaneDataframe.
    """
    # Run segmentPlanes function to get plane_list attribute
    planeSegmentationClass.segmentPlanes()
    # Get a sample source crs, scales, and offsets from the pcd
    source_crs, scales, offsets = pcdInfoParams
    # Test createSummaryPlaneDataframe function
    result = planeSegmentationClass.createSummaryPlaneDataframe(
        source_crs=source_crs, scales=scales, offsets=offsets)
    # Assert that the result is a pandas.DataFrame object
    assert isinstance(result, pd.DataFrame)
    # Assert that the desired columns are in the df
    expected_df_columns = ["plane_id", "tilt", "azimuth", "num_points",
                           "center_lat", "center_lon"]
    assert all(col in result.columns for col in expected_df_columns)


def testGetPlaneCentersTypeErrors(planeSegmentationClass, pcdInfoParams,
                                  planeLatLonParams):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Get a sample source crs, scales, and offsets from the pcd
    source_crs, scales, offsets = pcdInfoParams
    # Get a sample lat, lon coordinates of the plane
    lat, lon = planeLatLonParams
    # Test if source_crs is correct
    with pytest.raises(
            TypeError,
            match="source_crs variable must be of type pyproj.crs.CRS."):
        planeSegmentationClass.getPlaneCenters(
            source_crs="EPSG:4326", scales=scales, offsets=offsets,
            center_x=lat, center_y=lon)
    # Test if scales is correct
    with pytest.raises(TypeError,
                       match="scales variable must be of type tuple, list, " +
                       "or numpy.ndarray."):
        planeSegmentationClass.getPlaneCenters(
            source_crs=source_crs, scales="qwe", offsets=offsets,
            center_x=lat, center_y=lon)
    # Test if offsets is correct
    with pytest.raises(TypeError,
                       match="offsets variable must be of type tuple, list, " +
                       "or numpy.ndarray."):
        planeSegmentationClass.getPlaneCenters(
            source_crs=source_crs, scales=scales, offsets="[124, 5678, 91011]",
            center_x=lat, center_y=lon)
    # Test if center_x is correct
    with pytest.raises(TypeError,
                       match="center_x variable must be of type float."):
        planeSegmentationClass.getPlaneCenters(
            source_crs=source_crs, scales=scales, offsets=offsets,
            center_x=10, center_y=lon)
    # Test if center_y is correct
    with pytest.raises(TypeError,
                       match="center_y variable must be of type float."):
        planeSegmentationClass.getPlaneCenters(
            source_crs=source_crs, scales=scales, offsets=offsets,
            center_x=lat, center_y="1")


def testGetPlaneCentersResult(planeSegmentationClass, pcdInfoParams,
                              planeLatLonParams):
    """
    Tests if a tuple of floats of the format (center_lat, center_lon) is
    returned after running getPlaneCenters.
    """
    # Get a sample source crs, scales, and offsets from the pcd
    source_crs, scales, offsets = pcdInfoParams
    # Get a sample lat, lon coordinates of the plane
    lat, lon = planeLatLonParams
    # Test getPlaneCenters function
    result = planeSegmentationClass.getPlaneCenters(
        source_crs=source_crs, scales=scales, offsets=offsets,
        center_x=lat, center_y=lon)
    # Assert that the result is a tuple
    assert isinstance(result, tuple)
    # Assert that the (tile, azimuth) result is a tuple of floats
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def testGetBestPlaneResult(planeSegmentationClass):
    """
    Tests if a dictionary of the best plane and a boolean flag to is found
    after running getBestPlane.
    """
    # Run segmentPlanes function to get plane_list attribute
    planeSegmentationClass.segmentPlanes()
    # Test getBestPlane function
    result = planeSegmentationClass.getBestPlane()
    # Assert that the result is a dictionary
    assert isinstance(result[0], dict)
    # Assert that the result is a boolean flag
    assert isinstance(result[1], bool)


def testCalculatePlaneTiltAzimuthTypeErrors(planeSegmentationClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if plane_normal_vector is correct
    with pytest.raises(TypeError,
                       match=("plane_normal_vector variable must be of type " +
                              "tuple, list, or numpy.ndarray.")):
        planeSegmentationClass.calculatePlaneTiltAzimuth(
            plane_normal_vector="(1,2)")


def testCalculatePlaneTiltAzimuthNormalization(planeSegmentationClass, capsys):
    """
    Tests if the plane normal vector is normalized before calculating the tilt
    and azimuth.
    """
    # Make a sample non-normalized vector
    plane_normal_vector = (10, 10, 10)
    # Test calculatePlaneTiltAzimuth function
    planeSegmentationClass.calculatePlaneTiltAzimuth(
        plane_normal_vector=plane_normal_vector)
    # Test the the correct print statement is printed
    captured = capsys.readouterr()
    print_msg = "Plane vectors are not normalized. Normalizing them now.\n"
    assert print_msg == captured.out


def testCalculatePlaneTiltAzimuthOrientation(planeSegmentationClass):
    """
    Tests if the plane normal vector is in the correct orientation
    (z vector is facing upwards) before calculating the tilt and azimuth.
    """
    # Make a sample vector where z vector is facing downwards
    # The tilt and azimuth should be 45 degrees
    plane_normal_vector = (-1, -1, -(np.sqrt(2)/2))
    # Test calculatePlaneTiltAzimuth function
    result = planeSegmentationClass.calculatePlaneTiltAzimuth(
        plane_normal_vector=plane_normal_vector)
    # Assert that the result is a tuple
    assert isinstance(result, tuple)
    tilt, azimuth = result
    # Assert that the result is a tuple of floats
    assert isinstance(tilt, float)
    assert isinstance(azimuth, float)
    # Check if the tilt and azimuth returned the expected result
    assert tilt == 45.0
    assert azimuth == 45.0


def testCalculatePlaneTiltAzimuthResult(planeSegmentationClass):
    """
    Tests if a tuple of floats of the format (tilt, azimuth) is returned
    after running calculatePlaneTiltAzimuth.
    """
    # Make a sample normal vector where tilt and azimuth should be 45 degrees
    plane_normal_vector = (1, 1, (np.sqrt(2)/2))
    # Test calculatePlaneTiltAzimuth function
    result = planeSegmentationClass.calculatePlaneTiltAzimuth(
        plane_normal_vector=plane_normal_vector)
    # Assert that the result is a tuple
    assert isinstance(result, tuple)
    tilt, azimuth = result
    # Assert that the result is a tuple of floats
    assert isinstance(tilt, float)
    assert isinstance(azimuth, float)
    # Check if the tilt and azimuth returned the expected result
    assert tilt == 45.0
    assert azimuth == 45.0
