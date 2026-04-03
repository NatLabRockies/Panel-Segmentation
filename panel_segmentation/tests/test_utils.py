"""
Test suite for utils code.
"""

from panel_segmentation import utils
import os
from PIL import Image
import requests
import pytest
import numpy as np
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import json
from shapely.geometry import Polygon
import matplotlib
matplotlib.use("Agg")

example_path = os.path.join("panel_segmentation", "examples", "utils_examples")


@pytest.fixture
def satelliteImageParams():
    """
    Contains the proper variable types for running GenerateSatelliteImage
    and generateAddress functions.
    """
    # Proper variable types
    lat = 37.18385
    lon = -113.70663
    file_name_save = os.path.join(example_path,
                                  "satellite_img")
    api_key = "123456789ABC"
    zoom = 19
    return lat, lon, file_name_save, api_key, zoom


@pytest.fixture
def satelliteImageGridParams():
    """
    Contains the proper variable types for running
    generateSatelliteImageryGrid function.
    """
    # Proper variable types
    nw_lat = 37.18385
    nw_lon = -113.70663
    se_lat = 37.17385
    se_lon = -113.69663
    api_key = "123456789ABC"
    file_save_folder = example_path
    zoom = 19
    lat_lon_dist = 0.01
    num_allowed_img = 2
    return nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img


@pytest.fixture
def visSatelliteImgGridParams():
    """
    Contains the proper variable types for running
    visualizeSatelliteImageryGrid function.
    """
    # Proper variable types
    grid_location_list = [
        {'file_name': '37.18385_-113.70663.png', 'latitude': 37.18385,
            'lon': -113.70663, 'grid_x': 0, 'grid_y': 0},
        {'file_name': '37.17385_-113.70663.png', 'latitude': 37.17385,
            'lon': -113.70663, 'grid_x': 1, 'grid_y': 0},
        {'file_name': '37.18385_-113.69663.png', 'latitude': 37.18385,
            'lon': -113.69663, 'grid_x': 0, 'grid_y': 1},
        {'file_name': '37.18385_-113.68663.png', 'latitude': 37.18385,
            'lon': -113.68663, 'grid_x': 0, 'grid_y': 2}
    ]
    file_save_folder = os.path.join(example_path,
                                    "satellite_grid")
    return grid_location_list, file_save_folder


@pytest.fixture
def splitTifToPngsParams():
    """
    Contains the proper variable types for running splitTifToPngs
    function
    """
    geotiff_file = os.path.join(example_path,
                                "40.1072_-75.0137.tif")
    meters_per_pixel = 0.152401
    meters_png_image = 2500.0
    file_save_folder = example_path
    return geotiff_file, meters_per_pixel, meters_png_image, file_save_folder


@pytest.fixture
def locateLatLonGeotiffParams():
    """
    Contains the proper variable types for running locateLatLonGeotiff
    function
    """
    geotiff_file = os.path.join(example_path, "40.1072_-75.0137.tif")
    latitude = 40.1072
    longitude = -75.0137
    file_name_save = os.path.join(example_path, "40.1072_-75.0137.png")
    pixel_resolution = 600
    return geotiff_file, latitude, longitude, file_name_save, pixel_resolution


@pytest.fixture
def translateLatLongCoordsParams():
    """
    Contains the proper variable types for running
    translateLatLongCoordinates function
    """
    latitude = 40.1072
    longitude = -75.0137
    lat_translation_meters = 100.0
    long_translation_meters = -200.0
    return latitude, longitude, lat_translation_meters, long_translation_meters


@pytest.fixture
def getInferenceBoxLatLonCoordsParams():
    """
    Contains the proper variable types for running
    getInferenceBoxLatLonCoordinates function.
    """
    box = [15.0, 0.0, 485.0, 500.0]
    img_center_lat = 40.1072
    img_center_lon = -75.0137
    image_x_pixels = 600
    image_y_pixels = 600
    zoom_level = 18
    return box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level


@pytest.fixture
def convertMaskToLatLonPolygonParams():
    """
    Contains the proper variable types for running
    convertMaskToLatLonPolygon function.
    """
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1
    img_center_lat = 40.1072
    img_center_lon = -75.0137
    image_x_pixels = 600
    image_y_pixels = 600
    zoom_level = 18
    return mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level


@pytest.fixture
def detectAzimuthParams():
    """
    Contains the proper variable types for running detectAzimuth function.
    """
    mask = _make_horizontal_bar_mask()
    number_lines = 4
    return mask, number_lines


@pytest.fixture
def plotEdgeAzParams():
    """
    Contains the proper variable types for running plotEdgeAz function.
    """
    mask = _make_horizontal_bar_mask()
    no_lines = 4
    save_img_file_path = None
    plot_show = False
    return mask, no_lines, save_img_file_path, plot_show


@pytest.fixture
def getRectangleDimensionsParams():
    """
    Contains the proper variable types for running getRectangleDimensions
    function.
    """
    polygon = _make_rectangle_polygon()
    return (polygon,)


@pytest.fixture
def standardizeRectangleWidthParams():
    """
    Contains the proper variable types for running standardizeRectangleWidth
    function.
    """
    polygon = _make_rectangle_polygon()
    target_width = 1.0
    return polygon, target_width


def testGenerateSatelliteImageTypeErrors(satelliteImageParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    lat, lon, file_name_save, api_key, zoom = satelliteImageParams
    # Tests for float latitude type
    with pytest.raises(TypeError,
                       match="latitude variable must be of type float."):
        utils.generateSatelliteImage(1, lon, file_name_save, api_key, zoom)
    # Tests for float longitude type
    with pytest.raises(TypeError,
                       match="longitude variable must be of type float."):
        utils.generateSatelliteImage(
            lat, "-12.34", file_name_save, api_key, zoom)
    # Tests for int zoom level type
    with pytest.raises(TypeError,
                       match="zoom_level variable must be of type int."):
        utils.generateSatelliteImage(lat, lon, file_name_save, api_key, True)
    # Tests for str file_name_save type
    with pytest.raises(
            TypeError,
            match="file_name_save variable must be of type string."):
        utils.generateSatelliteImage(lat, lon, 19.3, api_key, zoom)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generateSatelliteImage(lat, lon, file_name_save, 4, zoom)


def testGenerateSatelliteImageErrorResponse(satelliteImageParams, mocker):
    """
    Tests if requests response returns the correct output for non-200
    status code.
    """
    lat, lon, file_name_save, api_key, zoom = satelliteImageParams
    # Simulate mocked status code response of 404
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 404
    mocked_requests_get.return_value = mocked_response
    error_string = ("Response status code 404: "
                    "Image not pulled successfully from API.")
    # Test is ValueErroor is raised if status code is 404
    with pytest.raises(ValueError, match=error_string):
        utils.generateSatelliteImage(lat, lon, file_name_save, api_key, zoom)


def testGenerateSatelliteImage200Response(satelliteImageParams, mocker):
    """
    Tests if requests response returns the correct output for 200 status code.
    """
    lat, lon, file_name_save, api_key, zoom = satelliteImageParams
    # Simulate mocked response of code 200 and returned image data
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 200
    mocked_response.content = b"satellite_img"
    mocked_requests_get.return_value = mocked_response
    # Mock opened file when open() is called
    mocked_file = mocker.mock_open()
    mocker.patch("panel_segmentation.utils.open", mocked_file)
    # Mock opened PIL Image
    mocked_img = mocker.Mock(spec=Image.Image)
    mocker.patch("PIL.Image.open", return_value=mocked_img)
    # Test function with mocked parameters
    actual_pulled_img = utils.generateSatelliteImage(
        lat, lon, file_name_save, api_key, zoom)
    # Aseert that the file is correctly written with mocked content
    mocked_file().write.assert_called_once_with(b"satellite_img")
    # Assert the pulled image returned the mocked image
    assert actual_pulled_img == mocked_img


def testGenerateAddressTypeErrors(satelliteImageParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    lat, lon, _, api_key, _ = satelliteImageParams
    # Tests for float latitude type
    with pytest.raises(TypeError,
                       match="latitude variable must be of type float."):
        utils.generateAddress(1, lon, api_key)
    # Tests for float longitude type
    with pytest.raises(TypeError,
                       match="longitude variable must be of type float."):
        utils.generateAddress(lat, "-12.34", api_key)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generateAddress(lat, lon, 4)


def testGenerateAddressErrorResponse(satelliteImageParams, mocker):
    """
    Tests if requests response returns the correct output for non-200
    status code.
    """
    lat, lon, _, api_key, _ = satelliteImageParams
    # Simulate mocked status code response of 404
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 404
    mocked_requests_get.return_value = mocked_response
    error_string = ("Response status code 404: "
                    "Address not pulled successfully from API.")
    # Test is ValueErroor is raised if status code is 404
    with pytest.raises(ValueError, match=error_string):
        utils.generateAddress(lat, lon, api_key)


def testGenerateAddressImage200Response(satelliteImageParams, mocker):
    """
    Tests if requests response returns the correct output for 200 status code.
    """
    lat, lon, _, api_key, _ = satelliteImageParams
    expected_address = "875 Coyote Gulch Ct, Ivins, UT 84738, USA"
    # Simulate mocked response of code 200 and returned str address
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 200
    mocked_response.json.return_value = {"results": [
        {"formatted_address": expected_address}
    ]}
    mocked_requests_get.return_value = mocked_response
    # Test function with mocked parameters
    actual_address = utils.generateAddress(lat, lon, api_key)
    assert actual_address == expected_address
    # Assert address is a string
    assert isinstance(actual_address, str)


def testGenerateSatelliteImageryGridTypeErrors(satelliteImageGridParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img = satelliteImageGridParams
    # Tests for float northwest_latitude type
    with pytest.raises(
            TypeError,
            match="northwest_latitude variable must be of type float."):
        utils.generateSatelliteImageryGrid(1, nw_lon, se_lat, se_lon,
                                           api_key, zoom, file_save_folder,
                                           lat_lon_dist, num_allowed_img)
    # Tests for float northwest_longitude type
    with pytest.raises(
            TypeError,
            match="northwest_longitude variable must be of type float."):
        utils.generateSatelliteImageryGrid(nw_lat, "13", se_lat, se_lon,
                                           api_key, zoom, file_save_folder,
                                           lat_lon_dist, num_allowed_img)
    # Tests for float southeast_latitude type
    with pytest.raises(
            TypeError,
            match="southeast_latitude variable must be of type float."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, True, se_lon,
                                           api_key, file_save_folder, zoom,
                                           lat_lon_dist, num_allowed_img)
    # Tests for float southeast_longitude type
    with pytest.raises(
            TypeError,
            match="southeast_longitude variable must be of type float."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, 21,
                                           api_key, file_save_folder, zoom,
                                           lat_lon_dist, num_allowed_img)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, se_lon,
                                           123, file_save_folder, zoom,
                                           lat_lon_dist, num_allowed_img)
    # Tests for str file_save_folder type
    with pytest.raises(
            TypeError,
            match="file_save_folder variable must be of type string."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, se_lon,
                                           api_key, 902, zoom,
                                           lat_lon_dist, num_allowed_img)
    # Tests for int zoom_level type
    with pytest.raises(
            TypeError,
            match="zoom_level variable must be of type int."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, se_lon,
                                           api_key, file_save_folder, False,
                                           lat_lon_dist, num_allowed_img)
    # Tests for float lat_lon_distance type
    with pytest.raises(
            TypeError,
            match="lat_lon_distance variable must be of type float."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, se_lon,
                                           api_key, file_save_folder, zoom,
                                           "lat_lon_dist", num_allowed_img)
    # Tests for int number_allowed_images_taken type
    with pytest.raises(
            TypeError,
            match="number_allowed_images_taken variable must be of type int."):
        utils.generateSatelliteImageryGrid(nw_lat, nw_lon, se_lat, se_lon,
                                           api_key, file_save_folder, zoom,
                                           lat_lon_dist, 9.1)


def testGenerateSatelliteImageryGridOutput(satelliteImageGridParams, mocker):
    """
    Tests if generateSatelliteImageryGrid returns the correct output for
    sateliite images.
    """
    nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img = satelliteImageGridParams

    # Mock satellite file pull from generateSatelliteImage function
    mocked_pull = mocker.patch(
        "panel_segmentation.utils.generateSatelliteImage")

    actual_output = utils.generateSatelliteImageryGrid(
        nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom,
        lat_lon_dist, num_allowed_img
    )
    expected_output = [
        {'file_name': '37.18385_-113.70663.png', 'latitude': 37.18385,
            'longitude': -113.70663, 'grid_x': 0, 'grid_y': 0},
        {'file_name': '37.17385_-113.70663.png', 'latitude': 37.17385,
            'longitude': -113.70663, 'grid_x': 1, 'grid_y': 0},
        {'file_name': '37.18385_-113.69663.png', 'latitude': 37.18385,
            'longitude': -113.69663, 'grid_x': 0, 'grid_y': 1},
        {'file_name': '37.18385_-113.68663.png', 'latitude': 37.18385,
            'longitude': -113.68663, 'grid_x': 0, 'grid_y': 2}
    ]
    # Assert that mocked expected output equals to actual output
    assert actual_output == expected_output
    # Assert that the output is a list
    assert isinstance(actual_output, list)
    # Assert that the call count is equal to the numbers of files pulled
    assert mocked_pull.call_count == len(expected_output)


def testVisualizeSatelliteImageryGridTypeErrors(visSatelliteImgGridParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    grid_location_list, file_save_folder = visSatelliteImgGridParams
    # Tests for grid_location_list list type
    with pytest.raises(
            TypeError,
            match="grid_location_list variable must be of type list."):
        utils.visualizeSatelliteImageryGrid(
            {'file_name': '37.17385_-113.70663.png'}, file_save_folder)
    # Tests for values within grid_location_list are dict type
    with pytest.raises(
            TypeError,
            match="grid_location_list must be a list of dictionaries."):
        utils.visualizeSatelliteImageryGrid([1, 2, 3, 4],
                                            file_save_folder)
    # Tests for file_str_folder str type
    with pytest.raises(
            TypeError, match="file_save_folder variable must be of type str."):
        utils.visualizeSatelliteImageryGrid(grid_location_list,
                                            False)


def testVisualizeSatelliteImageryGridOutput(visSatelliteImgGridParams):
    """
    Tests if visualizeSatelliteImageryGrid returns the correct output.
    """
    grid_location_list, file_save_folder = visSatelliteImgGridParams
    grid = utils.visualizeSatelliteImageryGrid(grid_location_list,
                                               file_save_folder)
    # Assert grid is of ImageGrid (plot of gridded satellite images)
    assert isinstance(grid, ImageGrid)


def testSplitTifToPngsTypeErrors(splitTifToPngsParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    geotiff_file, meters_per_pixel, meters_png_image, \
        file_save_folder = splitTifToPngsParams
    # Tests for geotiff_file str type
    with pytest.raises(
            TypeError, match="geotiff_file variable must be of type str."):
        utils.splitTifToPngs(["a_geotiff_file.tif"], meters_per_pixel,
                             meters_png_image, file_save_folder)
    # Tests for meters_per_pixel float type
    with pytest.raises(
            TypeError,
            match="meters_per_pixel variable must be of type float."):
        utils.splitTifToPngs(geotiff_file, 871,
                             meters_png_image, file_save_folder)
    # Tests for meters_png_image float type
    with pytest.raises(
            TypeError,
            match="meters_png_image variable must be of type float."):
        utils.splitTifToPngs(geotiff_file, meters_per_pixel,
                             "201", file_save_folder)
    # Tests for file_save_folder str type
    with pytest.raises(
            TypeError, match="file_save_folder variable must be of type str."):
        utils.splitTifToPngs(geotiff_file, meters_per_pixel,
                             meters_png_image, False)


def testSplitTifToPngsTypeOutput(splitTifToPngsParams, mocker):
    """
    Tests if splitTifToPngs function returns None.
    """
    geotiff_file, meters_per_pixel, meters_png_image, \
        file_save_folder = splitTifToPngsParams
    actual_output = utils.splitTifToPngs(
        geotiff_file, meters_per_pixel,
        meters_png_image, file_save_folder)
    # Delete the produced image
    os.remove(os.path.join(file_save_folder,
                           "40.0990931_-75.0039022.png"))
    assert actual_output is None


def testLocateLatLonGeotiffImageTypeErrors(locateLatLonGeotiffParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    geotiff_file, latitude, longitude, file_name_save, \
        pixel_resolution = locateLatLonGeotiffParams
    # Tests for geotiff_file str type
    with pytest.raises(
            TypeError, match="geotiff_file variable must be of type str."):
        utils.locateLatLonGeotiff(-231, latitude, longitude,
                                  file_name_save, pixel_resolution)
    # Tests for latitude float type
    with pytest.raises(
            TypeError, match="latitude variable must be of type float."):
        utils.locateLatLonGeotiff(geotiff_file, [3219, 31], longitude,
                                  file_name_save, pixel_resolution)
    # Tests for longitude float type
    with pytest.raises(
            TypeError, match="longitude variable must be of type float."):
        utils.locateLatLonGeotiff(geotiff_file, latitude, "1093",
                                  file_name_save, pixel_resolution)
    # Tests for file_name_save str type
    with pytest.raises(
            TypeError, match="file_name_save variable must be of type str."):
        utils.locateLatLonGeotiff(geotiff_file, latitude, longitude,
                                  True, pixel_resolution)
    # Tests for pixel_resolution int type
    with pytest.raises(
            TypeError, match="pixel_resolution variable must be of type int."):
        utils.locateLatLonGeotiff(geotiff_file, latitude, longitude,
                                  file_name_save, 678.1)


def testLocateLatLonGeotiffImageOutput(locateLatLonGeotiffParams):
    """
    Tests if the output returns an image output for when latitude and longitude
    is within bounds to capture the png image.
    """
    geotiff_file, latitude, longitude, file_name_save, \
        pixel_resolution = locateLatLonGeotiffParams
    actual_output = utils.locateLatLonGeotiff(
        geotiff_file, latitude, longitude, file_name_save, pixel_resolution)
    # Assert that actual_output returns an image
    assert isinstance(actual_output, Image.Image)


def testLocateLatLonGeotiffNoneOutput(locateLatLonGeotiffParams, capsys):
    """
    Tests if the output returns None for when latitude and longitude is out of
    bounds.
    """
    geotiff_file, _, _, file_name_save, \
        pixel_resolution = locateLatLonGeotiffParams
    # Make the latitude and longitude out of image bounds
    latitude = 39.0662
    longitude = -121.1606
    actual_output = utils.locateLatLonGeotiff(
        geotiff_file, latitude, longitude, file_name_save, pixel_resolution)
    captured = capsys.readouterr()
    print_msg = ("""Latitude-longitude coordinates are not within bounds of
                the image, no PNG captured...\n""")
    # Assert that print_msg gets printed
    assert captured.out == print_msg
    # Assert that actual output returns None
    assert actual_output is None


def testTranslateLatLongCoordinatesTypeErrors(translateLatLongCoordsParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    latitude, longitude, lat_translation_meters, \
        long_translation_meters = translateLatLongCoordsParams
    # Tests for latitude float type
    with pytest.raises(
            TypeError, match="latitude variable must be of type float."):
        utils.translateLatLongCoordinates(91, longitude,
                                          lat_translation_meters,
                                          long_translation_meters)
    # Tests for longitude float type
    with pytest.raises(
            TypeError, match="longitude variable must be of type float."):
        utils.translateLatLongCoordinates(latitude, "191",
                                          lat_translation_meters,
                                          long_translation_meters)
    # Tests for lat_translation_meters float type
    with pytest.raises(
            TypeError,
            match="lat_translation_meters variable must be of type float."):
        utils.translateLatLongCoordinates(latitude, longitude,
                                          True, long_translation_meters)
    # Tests for long_translation_meters float type
    with pytest.raises(
            TypeError,
            match="long_translation_meters variable must be of type float."):
        utils.translateLatLongCoordinates(latitude, longitude,
                                          lat_translation_meters, ["5"])


def testTranslateLatLongCoordinatesTypeOutput(translateLatLongCoordsParams):
    """
    Tests the if the output returns float values.
    """
    latitude, longitude, lat_translation_meters, \
        long_translation_meters = translateLatLongCoordsParams
    lat_lon_coords = utils.translateLatLongCoordinates(
        latitude, longitude, lat_translation_meters, long_translation_meters)
    # Assert actual_lat is a float
    assert isinstance(lat_lon_coords[0][0], float)
    # Assert actual_lon is a float
    assert isinstance(lat_lon_coords[0][1], float)


def testGetInferenceBoxLatLonCoordinatesTypeErrors(
        getInferenceBoxLatLonCoordsParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = getInferenceBoxLatLonCoordsParams
    # Tests for box list type
    with pytest.raises(
            TypeError, match="box variable must be of type list."):
        utils.getInferenceBoxLatLonCoordinates(
            124, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lat float type
    with pytest.raises(
            TypeError, match="img_center_lat variable must be of type float."):
        utils.getInferenceBoxLatLonCoordinates(
            box, "341", img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lon float type
    with pytest.raises(
            TypeError, match="img_center_lon variable must be of type float."):
        utils.getInferenceBoxLatLonCoordinates(
            box, img_center_lat, 82, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for image_x_pixels int type
    with pytest.raises(
            TypeError, match="image_x_pixels variable must be of type int."):
        utils.getInferenceBoxLatLonCoordinates(
            box, img_center_lat, img_center_lon, 51.0,
            image_y_pixels, zoom_level)
    # Tests for image_y_pixels int type
    with pytest.raises(
            TypeError, match="image_y_pixels variable must be of type int."):
        utils.getInferenceBoxLatLonCoordinates(
            box, img_center_lat, img_center_lon, image_x_pixels,
            "152", zoom_level)
    # Tests for zoom_level int type
    with pytest.raises(
            TypeError, match="zoom_level variable must be of type int."):
        utils.getInferenceBoxLatLonCoordinates(
            box, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, [51])


def testGetInferenceBoxLatLonCoordinatesOutput(
        getInferenceBoxLatLonCoordsParams):
    """
    Tests if the output returns a tuple type with float values.
    """
    box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = getInferenceBoxLatLonCoordsParams
    actual_output = utils.getInferenceBoxLatLonCoordinates(
        box, img_center_lat, img_center_lon, image_x_pixels,
        image_y_pixels, zoom_level)
    # Assert actual output is a tuple
    assert isinstance(actual_output, tuple)
    # Asert that the length of actual_output is 2 values (latitude, longitude)
    assert len(actual_output) == 2
    # Assert that the values in actual_output are floats
    for val in actual_output:
        assert isinstance(val, float)


def testBinaryMaskToPolygonTypeErrors():
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    mask = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]
    # Tests for mask np.ndarray type
    with pytest.raises(
            TypeError, match="mask variable must be of type numpy.ndarray"):
        utils.binaryMaskToPolygon(mask)


def testBinaryMaskToPolygonOutput():
    """
    Tests if the output returns a list of coordinates for a polygon.
    """
    # Generate a square binary mask of 2x2 block (1,1) to (2,2) in a 4x4 matrix
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1
    polygon_contours = utils.binaryMaskToPolygon(mask)
    # Assert that polygon_contours is a list
    assert isinstance(polygon_contours, np.ndarray)
    # Assert that polygon_contours is of length 4 (a square) from mask
    assert len(polygon_contours) == 4
    for coordinates in polygon_contours:
        # Assert that the coordinates inside polygon_coutours is a tuple
        assert isinstance(coordinates,  np.ndarray)
        # Assert that the coordinates is a tuple is length 2 (x,y)
        assert len(coordinates) == 2


def testConvertMaskToLatLonPolygonTypeErrors(convertMaskToLatLonPolygonParams):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = convertMaskToLatLonPolygonParams
    # Tests for mask variable np.ndarray type
    with pytest.raises(
            TypeError, match="mask variable must be of type numpy.ndarray"):
        utils.convertMaskToLatLonPolygon(
            [(0, 0, 0), (1, 1, 0)], img_center_lat, img_center_lon,
            image_x_pixels, image_y_pixels, zoom_level)
    # Tests for img_center_lat float type
    with pytest.raises(
            TypeError, match="img_center_lat variable must be of type float."):
        utils.convertMaskToLatLonPolygon(
            mask, [4231], img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lon float type
    with pytest.raises(
            TypeError, match="img_center_lon variable must be of type float."):
        utils.convertMaskToLatLonPolygon(
            mask, img_center_lat, False, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for image_x_pixels int type
    with pytest.raises(
            TypeError, match="image_x_pixels variable must be of type int."):
        utils.convertMaskToLatLonPolygon(
            mask, img_center_lat, img_center_lon, True,
            image_y_pixels, zoom_level)
    # Tests for image_y_pixels int type
    with pytest.raises(
            TypeError, match="image_y_pixels variable must be of type int."):
        utils.convertMaskToLatLonPolygon(
            mask, img_center_lat, img_center_lon, image_x_pixels,
            -2.0, zoom_level)
    # Tests for zoom_level int type
    with pytest.raises(
            TypeError, match="zoom_level variable must be of type int."):
        utils.convertMaskToLatLonPolygon(
            mask, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, "18")


def testConvertMaskToLatLonPolygonOutput(convertMaskToLatLonPolygonParams):
    """
    Tests if the output returns a list of coordinates for a polygon.
    """
    mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = convertMaskToLatLonPolygonParams
    polygon_coord_list = utils.convertMaskToLatLonPolygon(
        mask, img_center_lat, img_center_lon, image_x_pixels,
        image_y_pixels, zoom_level)
    # Assert that polygon_coord_list is a list
    assert isinstance(polygon_coord_list, list)
    # Assert that polygon_contours is of length 4 (a square) from mask
    assert len(polygon_coord_list) == 4
    for coordinates in polygon_coord_list:
        # Assert that the coordinates inside polygon_coutours is a tuple
        assert isinstance(coordinates, tuple)
        # Assert that the coordinates is a tuple is length 2 (x,y)
        assert len(coordinates) == 2


def testConvertPolygonToGeojsonTypeErrors():
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    polygon_coord_list = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    # Tests for polygon_coord_list list type
    with pytest.raises(
            TypeError, match="polygon_coord_list variable must be of type" +
            " list"):
        utils.convertPolygonToGeojson(polygon_coord_list)


def testConvertPolygonToGeojsonOutput():
    """
    Tests if the output returns a GeoJSON string.
    """
    polygon_coord_list = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    geojson_poly_coord_list = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                [0.0, 1.0], [0.0, 0.0]]]
    geojson_poly = utils.convertPolygonToGeojson(polygon_coord_list)
    geojson_dict = json.loads(geojson_poly)
    # Assert that geojson_poly is a string
    assert isinstance(geojson_poly, str)
    # Assert that type is a FeatureCollection
    assert geojson_dict["type"] == "FeatureCollection"
    # Assert that the feature is a Polygon
    feature = geojson_dict["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    # Assert that the feature returns the polygon_coord_list in geojson format
    assert feature["geometry"]["coordinates"] == geojson_poly_coord_list


def _make_horizontal_bar_mask(size=100):
    """
    Creates a binary mask with a clear horizontal bar, giving the Hough
    transform a dominant horizontal line to detect.
    """
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 2 - 2: size // 2 + 2, 10: size - 10] = True
    return mask


def _make_rectangle_polygon():
    """
    Creates a simple axis-aligned 4x2 rectangle Shapely polygon. Width = 2,
    length = 4, so angles are 0 and pi/2.
    """
    return Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])


def _make_non_rectangle_polygon():
    """
    Creates a triangle (3-sided) polygon, which getRectangleDimensions
    should return None for.
    """
    return Polygon([(0, 0), (1, 0), (0.5, 1)])


def testDetectAzimuthTypeErrors(detectAzimuthParams):
    """
    Tests if TypeErrors is raised for incorrect variable types.
    """
    mask, number_lines = detectAzimuthParams
    # Tests for mask np.ndarray type
    with pytest.raises(
            TypeError,
            match="Variable mask must be of type Numpy ndarray."):
        utils.detectAzimuth([[0, 1], [1, 0]], number_lines)
    # Tests for number_lines int type
    with pytest.raises(
            TypeError,
            match="Variable number_lines must be of type int."):
        utils.detectAzimuth(mask, "4")


def testDetectAzimuthOutput(detectAzimuthParams):
    """
    Tests if detectAzimuth returns an integer azimuth value in the
    expected range [0, 360).
    """
    mask, number_lines = detectAzimuthParams
    azimuth = utils.detectAzimuth(mask, number_lines)
    # Assert azimuth is an int
    assert isinstance(azimuth, int)
    # Assert azimuth is within the valid compass bearing range
    assert 0 <= azimuth < 360


def testDetectAzimuthDefaultNumberLines(detectAzimuthParams):
    """
    Tests that detectAzimuth runs correctly with the default number_lines
    parameter.
    """
    mask, _ = detectAzimuthParams
    azimuth = utils.detectAzimuth(mask)
    assert isinstance(azimuth, int)
    assert 0 <= azimuth < 360


def testPlotEdgeAzTypeErrors(plotEdgeAzParams):
    """
    Tests if TypeErrors is raised for incorrect variable types.
    """
    mask, no_lines, save_img_file_path, plot_show = plotEdgeAzParams
    # Tests for mask np.ndarray type
    with pytest.raises(
            TypeError,
            match="Variable mask must be of type Numpy ndarray."):
        utils.plotEdgeAz([[0, 1], [1, 0]], no_lines,
                         save_img_file_path, plot_show)
    # Tests for no_lines int type
    with pytest.raises(
            TypeError,
            match="Variable no_lines must be of type int."):
        utils.plotEdgeAz(mask, 4.0, save_img_file_path, plot_show)
    # Tests for plot_show bool type
    with pytest.raises(
            TypeError,
            match="Variable no_figs must be of type boolean."):
        utils.plotEdgeAz(mask, no_lines, save_img_file_path, "False")


def testPlotEdgeAzReturnsNone(plotEdgeAzParams):
    """
    Tests that plotEdgeAz returns None (output is a plot, not a value).
    """
    mask, no_lines, save_img_file_path, plot_show = plotEdgeAzParams
    result = utils.plotEdgeAz(mask, no_lines, save_img_file_path, plot_show)
    assert result is None


def testGetRectangleDimensionsTypeErrors(getRectangleDimensionsParams):
    """
    Tests if TypeErrors is raised for incorrect variable types.
    """
    (polygon,) = getRectangleDimensionsParams
    # Tests for polygon Shapely Polygon type
    with pytest.raises(
            TypeError,
            match="Variable polygon must be of type Shapely polygon."):
        utils.getRectangleDimensions([(0, 0), (1, 0), (1, 1), (0, 1)])


def testGetRectangleDimensionsOutput(getRectangleDimensionsParams):
    """
    Tests that getRectangleDimensions returns a dictionary with the
    expected keys and correct value types for a rectangular polygon.
    """
    (polygon,) = getRectangleDimensionsParams
    result = utils.getRectangleDimensions(polygon)
    # Assert result is a dictionary
    assert isinstance(result, dict)
    # Assert all expected keys are present
    for key in ("width", "length", "width_angle", "length_angle"):
        assert key in result
    # Assert width <= length (width is the shorter dimension)
    assert result["width"] <= result["length"]
    # Assert all values are floats
    for key in ("width", "length", "width_angle", "length_angle"):
        assert isinstance(result[key], float)


def testGetRectangleDimensionsCorrectValues(getRectangleDimensionsParams):
    """
    Tests that getRectangleDimensions returns correct width and length
    values for a known 4x2 rectangle.
    """
    (polygon,) = getRectangleDimensionsParams
    result = utils.getRectangleDimensions(polygon)
    # For a 4x2 rectangle: width=2, length=4
    assert result["width"] == pytest.approx(2.0, abs=1e-6)
    assert result["length"] == pytest.approx(4.0, abs=1e-6)


def testGetRectangleDimensionsNonRectangleReturnsNone():
    """
    Tests that getRectangleDimensions returns None for a non-4-sided polygon.
    """
    triangle = _make_non_rectangle_polygon()
    result = utils.getRectangleDimensions(triangle)
    assert result is None


def testStandardizeRectangleWidthTypeErrors(standardizeRectangleWidthParams):
    """
    Tests if TypeErrors is raised for incorrect variable types.
    """
    polygon, target_width = standardizeRectangleWidthParams
    # Tests for polygon Shapely Polygon type
    with pytest.raises(
            TypeError,
            match="Variable polygon must be of type Shapely polygon."):
        utils.standardizeRectangleWidth(
            [(0, 0), (4, 0), (4, 2), (0, 2)], target_width)
    # Tests for target_width float type
    with pytest.raises(
            TypeError,
            match="Variable targrt_width must be of type float."):
        utils.standardizeRectangleWidth(polygon, 1)


def testStandardizeRectangleWidthOutput(standardizeRectangleWidthParams):
    """
    Tests that standardizeRectangleWidth returns a Shapely Polygon.
    """
    polygon, target_width = standardizeRectangleWidthParams
    result = utils.standardizeRectangleWidth(polygon, target_width)
    assert isinstance(result, Polygon)


def testStandardizeRectangleWidthCorrectWidth(standardizeRectangleWidthParams):
    """
    Tests that the output polygon has approximately the target width.
    """
    polygon, target_width = standardizeRectangleWidthParams
    result = utils.standardizeRectangleWidth(polygon, target_width)
    result_dims = utils.getRectangleDimensions(result)
    # The standardized width should match the target width
    assert result_dims["width"] == pytest.approx(target_width, abs=1e-6)


def testStandardizeRectangleWidthPreservesLength(
        standardizeRectangleWidthParams):
    """
    Tests that standardizeRectangleWidth preserves the original length
    of the polygon.
    """
    polygon, target_width = standardizeRectangleWidthParams
    original_dims = utils.getRectangleDimensions(polygon)
    result = utils.standardizeRectangleWidth(polygon, target_width)
    result_dims = utils.getRectangleDimensions(result)
    # The length should be preserved after standardizing width
    assert result_dims["length"] == pytest.approx(
        original_dims["length"], abs=1e-6)


def testStandardizeRectangleWidthPreservesCentroid(
        standardizeRectangleWidthParams):
    """
    Tests that standardizeRectangleWidth preserves the centroid of the
    original polygon.
    """
    polygon, target_width = standardizeRectangleWidthParams
    result = utils.standardizeRectangleWidth(polygon, target_width)
    # Assert centroid x is preserved
    assert result.centroid.x == pytest.approx(polygon.centroid.x, abs=1e-6)
    # Assert centroid y is preserved
    assert result.centroid.y == pytest.approx(polygon.centroid.y, abs=1e-6)


def testStandardizeRectangleWidthNonRectanglePassthrough():
    """
    Tests that standardizeRectangleWidth returns the original polygon
    unchanged if getRectangleDimensions returns None (non-4-sided polygon).
    """
    triangle = _make_non_rectangle_polygon()
    target_width = 1.0
    result = utils.standardizeRectangleWidth(triangle, target_width)
    # Triangle is passed through unchanged
    assert result.equals(triangle)
