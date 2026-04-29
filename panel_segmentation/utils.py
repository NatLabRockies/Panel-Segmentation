"""
Utility functions that are used when dealing with deep learning models.
"""

import requests
from PIL import Image
import numpy as np
import os
import time
import random
import rasterio
from rasterio.warp import transform
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas
from pyproj import Transformer
from skimage.transform import hough_line, hough_line_peaks
from huggingface_hub import hf_hub_download


def downloadModel(filename,
                  repo_id="kperrynrel/panel-segmentation-models"):
    """
    Download a model file from a Hugging Face repository and save it
    to the panel_segmentation/models folder.

    Parameters
    ----------
    filename : str
        The name of the model file to download from the repository.
        Example: 'panel_detection_model.pth'
    repo_id : str
        The Hugging Face repository ID in the format 'owner/repo-name'.
        Default is 'kperrynrel/panel-segmentation-models', which is where
        the models are located.

    Returns
    -------
    model_path : str
        The full file path of the downloaded model.
    """
    # Check input variable types
    if not isinstance(repo_id, str):
        raise TypeError("repo_id variable must be of type string.")
    if not isinstance(filename, str):
        raise TypeError("filename variable must be of type string.")
    # models folder is panel_segmentation/models relative to utils.py
    models_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )
    # Build the destination path
    model_path = os.path.join(models_folder, filename)
    # Skip download if the file already exists locally
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}, skipping download.")
        return model_path
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_folder
        )
    except Exception as e:
        raise ValueError(
            f"Failed to download '{filename}' from '{repo_id}': {e}"
        )
    print(f"Model downloaded to {downloaded_path}")
    return downloaded_path


def generateSatelliteImage(latitude, longitude,
                           file_name_save, google_maps_api_key,
                           zoom_level=18):
    """
    Generates satellite image via Google Maps, using a set of lat-long
    coordinates.

    Parameters
    -----------
    latitude: float
        Latitude coordinate of the site.
    longitude: float
        Longitude coordinate of the site.
    file_name_save: string
        File path that we want to save
        the image to, where the image is saved as a PNG file.
    google_maps_api_key: string
        Google Maps API Key for automatically pulling satellite images. For
        more information, see here:
        https://developers.google.com/maps/documentation/maps-static/start
    zoom_level: int, default 18
        Zoom level of the image. Set to 18 as default, as that's what's used
        for the original panel-segmentation models.

    Returns
    -------
    Figure
        Figure of the satellite image
    """
    # Check input variable for types
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    if not isinstance(file_name_save, str):
        raise TypeError("file_name_save variable must be of type string.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    # Build up the lat_long string from the latitude-longitude coordinates
    lat_long = str(latitude) + ", " + str(longitude)
    # get method of requests module
    # return response object
    r = requests.get(
        "https://maps.googleapis.com/maps/api/staticmap?maptype"
        "=satellite&center=" + lat_long +
        "&zoom=" + str(zoom_level) + "&size=40000x40000&key=" +
        google_maps_api_key,
        verify=False)
    # Raise an exception if image is not successfully returned
    if r.status_code != 200:
        raise ValueError("Response status code " +
                         str(r.status_code) +
                         ": Image not pulled successfully from API.")
    # wb mode is stand for write binary mode
    with open(file_name_save, 'wb') as f:
        f.write(r.content)
        # close method of file object
        # save and close the file
        f.close()
    # Read in the image and return it via the console
    return Image.open(file_name_save)


def generateAddress(latitude, longitude, google_maps_api_key):
    """
    Gets the address of a latitude, longitude coordinates using Google
    Geocoding API. Please note rates for running geocoding checks here:
        https://developers.google.com/maps/billing-and-pricing/pricing

    Parameters
    -----------
    latitude: float
        Latitude coordinate of the site.
    longitude: float
        Longitude coordinate of the site.
    google_maps_api_key: string
        Google Maps API Key for geocoding a site. For further information,
        see here:
        https://developers.google.com/maps/documentation/geocoding/overview

    Returns
    -----------
    address: str
        Address of given latitude, longitude coordinates.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    # return response object
    r = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json?latlng=" +
        str(latitude) + "," + str(longitude) + "&key=" + google_maps_api_key,
        verify=True)
    # Raise an exception if address is not successfully returned
    if r.status_code != 200:
        raise ValueError("Response status code " +
                         str(r.status_code) +
                         ": Address not pulled successfully from API.")
    data = r.json()
    address = data["results"][0]["formatted_address"]
    return address


def generateSatelliteImageryGrid(northwest_latitude, northwest_longitude,
                                 southeast_latitude, southeast_longitude,
                                 google_maps_api_key,
                                 file_save_folder,
                                 zoom_level=18,
                                 lat_lon_distance=0.00145,
                                 number_allowed_images_taken=6000):
    """
    Take satellite images via the Google Maps API in a grid fashion for a large
    area, and save the associated images to a folder. The associated images
    can then be used to feed into different models to assess a larger area.

    Please note rates for running the Google Maps API here:
        https://developers.google.com/maps/billing-and-pricing/pricing

    Parameters
    ----------
    northwest_latitude : float
        Latitude coordinate on the northwest corner of the area we wish
        to scan.
    northwest_longitude : float
        Longitude coordinate on the northwest corner of the area we wish
        to scan.
    southeast_latitude : float
        Latitude coordinate on the southeast corner of the area
        we wish to scan.
    southeast_longitude : float
        Longitude coordinate on the southeast corner of the area
        we wish to scan.
    google_maps_api_key : str
        API key for Google Maps API
    file_save_folder : str
        Folder path for which to save all of the images
    zoom_level: int, default 18
        Zoom level of the image.
    lat_lon_distance : float, default 0.00145
        Distance to traverse between images, in terms of lat-long
        degree distance. For example, default distance of 0.00145 degrees
        equates to approx. 161 meters.
    number_allowed_images_taken : int, default 6000
        Number of allowed images for the Google Maps API to take before
        stopping. If we pull too many images in one go, google may flag
        this as webscraping so it's advised to not pull too many images
        at once.

    Returns
    -------
    grid_location_list: list of dictionaries
        A list of dictionaries with metadata information about each grid
        location in the image with the keys "file_name", "latitude",
        "longitude", "grid_x", and "grid_y".

    """
    # Ensure that the inputs are of the correct type
    if not isinstance(northwest_latitude, float):
        raise TypeError("northwest_latitude variable must be of type float.")
    if not isinstance(northwest_longitude, float):
        raise TypeError("northwest_longitude variable must be of type float.")
    if not isinstance(southeast_latitude, float):
        raise TypeError("southeast_latitude variable must be of type float.")
    if not isinstance(southeast_longitude, float):
        raise TypeError("southeast_longitude variable must be of type float.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be "
                        "of type string.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    if not isinstance(lat_lon_distance, float):
        raise TypeError("lat_lon_distance variable must be "
                        "of type float.")
    if not isinstance(number_allowed_images_taken, int) or \
            isinstance(number_allowed_images_taken, bool):
        raise TypeError("number_allowed_images_taken variable must be "
                        "of type int.")
    # Build the grid out
    start_lat, start_lon = northwest_latitude, northwest_longitude
    lat_list, lon_list = [start_lat], [start_lon]
    # Latitude list
    while (start_lat >= southeast_latitude):
        start_lat = start_lat - lat_lon_distance
        lat_list.append(start_lat)
    # Longitude list
    while (start_lon <= southeast_longitude):
        start_lon = start_lon + lat_lon_distance
        lon_list.append(start_lon)
    counter = 0
    coord_list = list()
    grid_location_list = list()
    grid_y = 0
    for lon in lon_list:
        grid_x = 0
        for lat in lat_list:
            coord_list.append((lat, lon))
            # For every coordinate, take a satellite image and save it
            file_name = (str(round(lat, 7)) + "_" + str(
                         round(lon, 7)) + ".png")
            file_save = os.path.join(file_save_folder,
                                     file_name)
            grid_location_list.append({"file_name": file_name,
                                       "latitude": lat,
                                       "longitude": lon,
                                       "grid_x": grid_x,
                                       "grid_y": grid_y})
            grid_x += 1
            if not os.path.exists(file_save):
                generateSatelliteImage(lat, lon,
                                       file_save,
                                       google_maps_api_key,
                                       zoom_level)
            else:
                print("File already pulled!")
                continue
            counter += 1
            time.sleep(random.randint(1, 5))
            if counter >= number_allowed_images_taken:
                break
        grid_y += 1
    return grid_location_list


def visualizeSatelliteImageryGrid(grid_location_list, file_save_folder):
    """
    Using the grid_location_list output from the
    generateSatelliteImageryGrid() function, visualize all of the images
    taken in a grid.

    Parameters
    ----------
    grid_location_list: List of dictionaries
        List of dictionaries directly outputed from the
        generateSatelliteImageryGrid() function.
    file_save_folder: Str
        Folder path where all of the outputed satellite images from the
        generateSatelliteImageryGrid() function are stored.

    Returns
    -------
    grid: Plot Object
        Plot of gridded satellite images.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(grid_location_list, list):
        raise TypeError("grid_location_list variable must be of type list.")
    if not isinstance(grid_location_list[0], dict):
        raise TypeError("grid_location_list must be a list of dictionaries.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be of type str.")
    # Get the max grid coordinates so we can build the appropriate matplotlib
    # gridded graphic
    x_max = max([x['grid_x'] for x in grid_location_list]) + 1
    y_max = max([x['grid_y'] for x in grid_location_list]) + 1
    fig = plt.figure(figsize=(x_max*4, y_max*4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(x_max, y_max),
                     axes_pad=0.1,
                     )
    # Read in all of the imagery into the grid
    for file in grid_location_list:
        file_name = file['file_name']
        print(file_name)
        x_loc = file['grid_x']
        y_loc = file['grid_y']
        img = Image.open(os.path.join(file_save_folder, file_name))
        grid[(x_loc * y_max) + y_loc].imshow(img)
    return grid


def splitTifToPngs(geotiff_file, meters_per_pixel,
                   meters_png_image, file_save_folder):
    """
    Take a master GEOTIFF file, grid it, and convert it to a series of PNG
    files.

    Parameters
    ----------
    geotiff_file: str
        File name of the TIF file we want to grid images from.
    meters_per_pixel: float
        TIF file resolution in meters/pixel. This needs to be previously known
        for the TIF file, so we can grid images based on the number of meters
        each image represents (ie zoom level)
    meters_png_image: float
        Number of meters we want an individual image output to represent, in
        both the x- and y-direction
    file_save_folder: str
        Folder path where we write all of the gridded images from the master
        TIF file.

    Returns
    -------
    None.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(geotiff_file, str):
        raise TypeError("geotiff_file variable must be of type str.")
    if not isinstance(meters_per_pixel, float):
        raise TypeError("meters_per_pixel variable must be of type float.")
    if not isinstance(meters_png_image, float):
        raise TypeError("meters_png_image variable must be of type float.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be of type str.")
    # Ignore aux.xml files when creating the png
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    with rasterio.open(geotiff_file) as img:
        # All tif measurements were similar so divide the measured meters
        # from ArcGIS by its pixels
        pixel_meter_conversion = int(meters_png_image/meters_per_pixel)
        # Get image dimensions
        img_width, img_height = img.width, img.height
        # Loop over dimensions to crop images in a grid
        # Start from top left and stop at bottom right
        for top in range(0, img_height, pixel_meter_conversion):
            for left in range(0, img_width, pixel_meter_conversion):

                # Create a Window and calculate the transform
                window = rasterio.windows.Window(left, top,
                                                 pixel_meter_conversion,
                                                 pixel_meter_conversion)
                transform = img.window_transform(window)

                # Skip images where all pixels are black
                if np.all(img.read(window=window) == 0):
                    continue

                # Create a new cropped raster to write to
                profile = img.profile
                profile.update({'height': pixel_meter_conversion,
                                'width': pixel_meter_conversion,
                                'transform': transform,
                                'driver': "PNG"})
                # Get center of cropped image
                center = pixel_meter_conversion/2
                # Check if crs is in  "EPSG:4326" lat-lon coordinates.
                # If it's not, transform it to "EPSG:4326" coordinates.
                # Otherwise, get coordinates directly
                lon, lat = rasterio.transform.xy(transform, center, center)
                if img.crs and not img.crs.is_geographic:
                    transformer = Transformer.from_crs(img.crs, "EPSG:4326",
                                                       always_xy=True)
                    lon, lat = transformer.transform(lon, lat)

                # Put coordinats in lat_lon format for file name
                lat_lon = f"{lat:.7f}_{lon:.7f}"
                file_path = os.path.join(file_save_folder, f"{lat_lon}.png")
                with rasterio.open(file_path, 'w',
                                   **profile) as png:
                    # Read the data from the window and write it as a
                    # png output
                    png.write(img.read(window=window))
    return


def locateLatLonGeotiff(geotiff_file, latitude, longitude,
                        file_name_save, pixel_resolution=300):
    """
    Locate a lat-lon coordinate in a GEOTIFF image, and then box that area
    and capture a PNG image of it.

    Parameters
    -----------
    geotiff_file: str
        File name of the TIF file we want to scan for a particular latitude-
        longitude coordinate.
    latitude: float
        Target latitude coordinate we want to find in the TIFF file.
    longitude: float
        Target longitude coordinate we want to find in the TIFF file.
    file_name_save: str
        Name of the file of the image taken of the target lat-lon
        coordinate and its surrounding area.
    pixel_resolution : int, default 300
        Number of pixels in the x- and y-direction of the resulting image.

    Returns
    -------
    image or None
        The cropped image is returned as a PIL Image Object if designated
        lat-lon coordinates can be located in the GEOTIFF file with the image
        center set as designated lat-lon. Otherwise, returns None if the input
        coordinates are outside the image bounds or if all regions of the
        image are all black pixels.

    """
    # Ensure that the inputs are of the correct type
    if not isinstance(geotiff_file, str):
        raise TypeError("geotiff_file variable must be of type str.")
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(file_name_save, str):
        raise TypeError("file_name_save variable must be of type str.")
    if not isinstance(pixel_resolution, int) or \
            isinstance(pixel_resolution, bool):
        raise TypeError("pixel_resolution variable must be of type int.")
    # Open the file and get its boundaries
    with rasterio.open(geotiff_file) as dat:
        bounds = dat.bounds
        # Check if crs is in "EPSG:4326" lat-lon coordinates.
        # If not, transform it to "EPSG:4326" coordinates.
        if dat.crs and not dat.crs.is_geographic:
            lon, lat = transform("EPSG:4326", dat.crs,
                                 [longitude], [latitude])
            longitude, latitude = lon[0], lat[0]
        # Check that the lat-lon is within the image. If so, go to it
        if ((longitude >= bounds.left) & (longitude <= bounds.right) &
                (latitude <= bounds.top) & (latitude >= bounds.bottom)):
            # Convert lon, lat to pixel row, col coords
            rows, cols = dat.index(longitude, latitude)
            # Get top left corner of window
            top, left = (rows - pixel_resolution //
                         2), (cols - pixel_resolution//2)
            # Create a cropping window
            window = rasterio.windows.Window(
                col_off=left,
                row_off=top,
                width=pixel_resolution,
                height=pixel_resolution
            )
            # Read data in the window
            clip = dat.read(window=window)
            # Skip image if all pixels are black
            if np.all(clip == 0):
                print("All pixels are black in area, skipping...")
                return None
            # Save metadata
            meta = dat.meta
            meta['width'], meta['height'] = pixel_resolution, pixel_resolution
            meta['transform'] = rasterio.windows.transform(
                window, dat.transform)
            with rasterio.open(file_name_save, 'w', **meta) as dst:
                # Read the data from the window and write it as a png output
                dst.write(clip)
            # Open the file again and re-write via pillow so it doesn't have
            # any strange format issues
            image = Image.open(file_name_save)
            image.save(file_name_save)
            return image
        else:
            print("""Latitude-longitude coordinates are not within bounds of
                the image, no PNG captured...""")
            return None


def translateLatLongCoordinates(latitude, longitude,
                                lat_translation_meters,
                                long_translation_meters):
    """
    Vectorized translation - handles arrays of translations at once.

    Parameters
    ----------
    latitude, longitude : float
        Center point coordinates
    lat_translation_meters : array-like or float
        Translation(s) in latitude direction
    long_translation_meters : array-like or float
        Translation(s) in longitude direction

    Returns
    -------
    coords : ndarray
        (N, 2) array of (latitude, longitude) pairs
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not (isinstance(lat_translation_meters, float) |
            isinstance(lat_translation_meters, np.ndarray)):
        raise TypeError("lat_translation_meters variable must be of" +
                        " type float.")
    if not (isinstance(long_translation_meters, float) |
            isinstance(long_translation_meters, np.ndarray)):
        raise TypeError("long_translation_meters variable must be of" +
                        " type float.")
    # Convert scalars to arrays for consistent handling
    lat_trans = np.atleast_1d(lat_translation_meters)
    lon_trans = np.atleast_1d(long_translation_meters)

    # Create transformer ONCE for all points
    local_crs = (f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} +x_0=0 "
                 f"+y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    latlon_to_local = Transformer.from_crs(
        "EPSG:4326", local_crs, always_xy=True)
    local_to_latlon = Transformer.from_crs(
        local_crs, "EPSG:4326", always_xy=True)

    # Transform center point to local CRS
    m_lon, m_lat = latlon_to_local.transform(longitude, latitude)

    # Apply translations (vectorized)
    m_lons = m_lon + lon_trans
    m_lats = m_lat + lat_trans

    # Transform all points back at once
    lons_new, lats_new = local_to_latlon.transform(m_lons, m_lats)

    # Return as (N, 2) array
    return np.column_stack([lats_new, lons_new])


def getInferenceBoxLatLonCoordinates(box, img_center_lat, img_center_lon,
                                     image_x_pixels, image_y_pixels,
                                     zoom_level):
    """
    Get the latitude-longitude coordinates of the centroid of a box output
    from model inference, based on the image center location & zoom level.

    Parameters
    -----------
    box : list
        A list of float pixel values containing the coordinates of a bounding
        box in the format [xmin, ymin, xmax, ymax].
    img_center_lat : float
        Latitude coordinate of the image center.
    img_center_lon : float
        Longitude coordinate of the image center.
    image_x_pixels : int
        The x width of the image in pixels.
    image_y_pixels : int
        The y height of the image in pixels.
    zoom_level : int
        Zoom level of the image.

    Returns
    -------
    (box_lat, box_lon) : tuple
        The (latitude, longitude) coordinates of the centroid of a box.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(box, list):
        raise TypeError("box variable must be of type list.")
    if not isinstance(img_center_lat, float):
        raise TypeError("img_center_lat variable must be of type float.")
    if not isinstance(img_center_lon, float):
        raise TypeError("img_center_lon variable must be of type float.")
    if not isinstance(image_x_pixels, int) or isinstance(image_x_pixels, bool):
        raise TypeError("image_x_pixels variable must be of type int.")
    if not isinstance(image_y_pixels, int) or isinstance(image_y_pixels, bool):
        raise TypeError("image_y_pixels variable must be of type int.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    image_center_pixels_x, image_center_pixels_y = (image_x_pixels/2,
                                                    image_y_pixels/2)
    xmin, ymin, xmax, ymax = (float(box[0]), float(box[1]),
                              float(box[2]), float(box[3]))
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    # Get the difference in meters between the main centroid and
    # label centroid, based on the image zoom level
    meter_pixel_conversion = 156543.03392 * \
        np.cos(np.radians(img_center_lat)) / (2**zoom_level)
    lon_translation_meters = ((cx - image_center_pixels_x) *
                              meter_pixel_conversion)
    lat_translation_meters = ((image_center_pixels_y - cy) *
                              meter_pixel_conversion)
    polygon_lat_lon_coords = translateLatLongCoordinates(
        latitude=img_center_lat,
        longitude=img_center_lon,
        lat_translation_meters=lat_translation_meters,
        long_translation_meters=lon_translation_meters)
    return polygon_lat_lon_coords[0]


def binaryMaskToPolygon(mask):
    """
    Convert a binary mask output (from a deep learning model) to a list of
    polygon coordinates, which can later be converted to latitude-longitude
    coordinates.

    Parameters
    -----------
    mask : nparray
        A binary mask output from a deep learning model, which can be converted
        to a polygon.

    Returns
    -------
    contours_new : nparray
        Array of (x,y) image pixel-coordinate arrays for a polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask variable must be of type numpy.ndarray")
    # Ensure the mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(binary_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # Fast concatenation using vstack instead of loop
    coords = np.vstack(contours).reshape(-1, 2)
    return coords


def convertMaskToLatLonPolygon(mask, img_center_lat,
                               img_center_lon,
                               image_x_pixels, image_y_pixels,
                               zoom_level):
    """
    Take an inference mask output from a model, and convert it to a polygon
    with listed latitude-longitude coordinates.

    Parameters
    -----------
    mask : nparray
        A binary mask output from a deep learning model, which can be converted
        to a polygon.
    img_center_lat : float
        Latitude coordinate of the image center.
    img_center_lon : float
        Longitude coordinate of the image center.
    image_x_pixels : int
        The x width of the image in pixels.
    image_y_pixels : int
        The y height of the image in pixels.
    zoom_level : int
        The zoom level of the image.

    Returns
    -------
    polygon_coord_list : list
        A list of (latitude, longitude) coordinates for a polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask variable must be of type numpy.ndarray")
    if not isinstance(img_center_lat, float):
        raise TypeError("img_center_lat variable must be of type float.")
    if not isinstance(img_center_lon, float):
        raise TypeError("img_center_lon variable must be of type float.")
    if not isinstance(image_x_pixels, int) or isinstance(image_x_pixels, bool):
        raise TypeError("image_x_pixels variable must be of type int.")
    if not isinstance(image_y_pixels, int) or isinstance(image_y_pixels, bool):
        raise TypeError("image_y_pixels variable must be of type int.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    # First convert the mask to a polygon (in pixel coordinates)
    polygon_coords = binaryMaskToPolygon(mask)
    # Vectorized calculations
    x_center = image_x_pixels / 2
    y_center = image_y_pixels / 2
    dx = -(x_center - polygon_coords[:, 0])
    dy = (y_center - polygon_coords[:, 1])
    meter_pixel_conversion = (156543.03392 *
                              np.cos(np.radians(img_center_lat)) /
                              (2 ** zoom_level))
    dx_meters = dx * meter_pixel_conversion
    dy_meters = dy * meter_pixel_conversion
    # Return numpy array directly
    polygon_lat_lon_coords = translateLatLongCoordinates(
        img_center_lat, img_center_lon,
        dy_meters, dx_meters
    )
    return polygon_lat_lon_coords


def convertPolygonToGeojson(polygon_coord_list):
    """
    Take a list of lat-lon coordinates for a polygon and convert
    to GeoJSON format.

    Parameters
    -----------
    polygon_coord_list : list
        A list of (latitude, longitude) coordinates for a polygon.

    Returns
    -------
    geojson_poly: str
        A GeoJSON string representation of the polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(polygon_coord_list, list):
        raise TypeError("polygon_coord_list variable must be of type list")
    shapely_poly = Polygon(polygon_coord_list)
    geojson_poly = geopandas.GeoSeries(shapely_poly).to_json()
    return geojson_poly


def detectAzimuth(mask, number_lines=4):
    """
    This function uses a Hough transform to detect the most dominant lines in
    an image mask and the azimuth of the longest direction.

    Parameters
    -----------
    mask: (nparray bool)
        The mask containing the extracted solar panels set True,
        and other pixels set to False.
        Dimension: (640, 640)
    number_lines: (int)
        This variable tells the function the number of dominant
        lines it should examine. Default set to 4.

    Returns
    -----------
    azimuth: (int)
        The azimuth of the panel in the image.
    """
    # Check that the input variables are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("Variable mask must be of type Numpy ndarray.")
    if not isinstance(number_lines, int):
        raise TypeError("Variable number_lines must be of type int.")
    # Run through the function
    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150, apertureSize=3)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(edges, theta=tested_angles)
    origin = np.array((0, edges.shape[1]))
    ind = 0
    azimuth = 0
    az = np.zeros((number_lines))
    for _, angle, dist in zip(*hough_line_peaks(
            h, theta, d, num_peaks=number_lines,
            threshold=0.25 * np.max(h))):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)

        deg_ang = int(np.rad2deg(angle))
        if deg_ang >= 0:
            az[ind] = 90+deg_ang
        else:
            az[ind] = 270 + deg_ang
        ind = ind+1
    unique_elements, counts_elements = np.unique(az, return_counts=True)
    for _, angle, dist in zip(*hough_line_peaks(
            h, theta, d, num_peaks=1,
            threshold=0.25 * np.max(h))):
        deg_ang = int(np.rad2deg(angle))
        if deg_ang >= 0:
            azimuth = 90 + deg_ang
        else:
            azimuth = 270 + deg_ang
    return azimuth


def plotEdgeAz(mask, no_lines=4,
               save_img_file_path=None, plot_show=False):
    """
    Draws the Hough line in the dominant (longest) direction on a
    segmentation mask.

    Parameters
    -----------
    mask: (nparray bool)
        The mask containing the extracted solar panels set True, and other
        pixels set to False.  Dimension: (640, 640)
    number_lines: (int)
        This variable tells the function the number of dominant lines it
        should examine. We currently inspect the top 4 lines.
    save_img_file_path: (string or None)
        Optional field where, if set, the assocaited output image is saved
        under the given path
    plot_show: (boolean)
        Show the generated output plot in the console.

    Returns
    -----------
    Plot is generated in console or saved, based on params passed
    """
    # Check that the input variables are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("Variable mask must be of type Numpy ndarray.")
    if not isinstance(no_lines, int):
        raise TypeError("Variable no_lines must be of type int.")
    if not isinstance(plot_show, bool):
        raise TypeError("Variable no_figs must be of type boolean.")
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Edge detection
    edges = cv2.Canny(mask_uint8, 50, 150, apertureSize=3)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(edges, theta=tested_angles)
    az = np.zeros((no_lines))
    origin = np.array((0, edges.shape[1]))
    ind = 0
    # Generating figure 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(mask)  # cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(
            h, theta, d, num_peaks=no_lines,
            threshold=0.25 * np.max(h))):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        deg_ang = int(np.rad2deg(angle))
        if deg_ang >= 0:
            az[ind] = 90+deg_ang
        else:
            az[ind] = 270 + deg_ang
        ind = ind+1
        ax.plot(origin, (y0, y1), '-r')
    ax.set_xlim(origin)
    ax.set_ylim((edges.shape[0], 0))
    ax.set_axis_off()
    unique_elements, counts_elements = np.unique(az, return_counts=True)
    for _, angle, dist in zip(*hough_line_peaks(
            h, theta, d, num_peaks=1,
            threshold=0.25 * np.max(h))):
        deg_ang = int(np.rad2deg(angle))
        if deg_ang >= 0:
            azimuth = 90 + deg_ang
        else:
            azimuth = 270 + deg_ang
    # print(np.asarray((unique_elements, counts_elements)))
    ax.set_title('Azimuth = %i' % azimuth)
    # save the image
    if save_img_file_path is not None:
        plt.savefig(save_img_file_path + '/crop_mask_az',
                    dpi=300)
    # Show the plot if plot_show is True
    if plot_show is True:
        plt.tight_layout()
        plt.show()
    return


def getRectangleDimensions(polygon):
    """
    Calculate the width and length of a rectangular polygon.
    Returns the two edge lengths and their orientations.

    Parameters
    -----------
    polygon: (shapely.geometry.Polygon)
        Rectangular polygon that we want to calculate the width, length, and
        associated angles for

    Returns
    -----------
    dictionary or None:
        Dictionary object with fields for the width, length, and
        associated angles of the mask. If not a rectangle, returns None
    """
    # Check that the input variables are of the correct type
    if not isinstance(polygon, Polygon):
        raise TypeError("Variable polygon must be of type Shapely polygon.")
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate last point

    # Calculate distances between consecutive vertices
    edges = []
    for i in range(len(coords)):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % len(coords)])

        # Calculate edge vector and length
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        edge_angle = np.arctan2(edge_vector[1], edge_vector[0])

        edges.append({
            'length': edge_length,
            'angle': edge_angle,
            'vector': edge_vector,
            'start': p1,
            'end': p2
        })

    # Group opposite edges (should be parallel and equal length in a rectangle)
    # For a 4-sided polygon, opposite edges are at indices 0-2 and 1-3
    if len(edges) == 4:
        edge1_length = edges[0]['length']
        edge2_length = edges[1]['length']

        # Determine which is width (shorter) and which is length (longer)
        if edge1_length < edge2_length:
            width = edge1_length
            length = edge2_length
            width_angle = edges[0]['angle']
            length_angle = edges[1]['angle']
        else:
            width = edge2_length
            length = edge1_length
            width_angle = edges[1]['angle']
            length_angle = edges[0]['angle']

        return {
            'width': width,
            'length': length,
            'width_angle': width_angle,
            'length_angle': length_angle
        }
    else:
        return None


def standardizeRectangleWidth(polygon, target_width):
    """
    Adjust a rectangular polygon to have a standardized width while
    maintaining its center position, orientation, and length.

    Parameters
    -----------
    polygon: (shapely.geometry.Polygon)
        Rectangular polygon that has a non-standardized width (tracker row or
        fixed-tilt row)
    target_width: (float)
        target width of the row. Normally determined via finding the median
        row width across all masks.
    Returns
    -----------
    Shapely polygon representing the newly standardized rectangle mask
    """
    if not isinstance(polygon, Polygon):
        raise TypeError("Variable polygon must be of type Shapely polygon.")
    if not isinstance(target_width, float):
        raise TypeError("Variable targrt_width must be of type float.")

    dims = getRectangleDimensions(polygon)
    if dims is None:
        return polygon

    # Get the center of the polygon
    centroid = polygon.centroid
    center = np.array([centroid.x, centroid.y])

    # Calculate the length direction (perpendicular to width)
    length_angle = dims['length_angle']
    width_angle = dims['width_angle']

    # Unit vectors for length and width directions
    length_dir = np.array([np.cos(length_angle), np.sin(length_angle)])
    width_dir = np.array([np.cos(width_angle), np.sin(width_angle)])

    # Half dimensions
    half_length = dims['length'] / 2.0
    half_width = target_width / 2.0

    # Create new rectangle centered at the centroid
    # The four corners are: center ± half_length * length_dir ±
    # half_width * width_dir
    corners = [
        center + half_length * length_dir + half_width * width_dir,
        center + half_length * length_dir - half_width * width_dir,
        center - half_length * length_dir - half_width * width_dir,
        center - half_length * length_dir + half_width * width_dir,
        center + half_length * length_dir + half_width * width_dir,
    ]
    return Polygon(corners)
