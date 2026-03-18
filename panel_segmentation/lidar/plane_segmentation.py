import numpy as np
import pandas as pd
import pyproj
from sklearn.cluster import DBSCAN
import open3d as o3d
from shapely.ops import transform as shapely_transform
from shapely.geometry import MultiPoint

class PlaneSegmentation:
    '''
    A class that segments planes from a pcd (point cloud data).
    '''

    def __init__(self, pcd):
        # Point cloud data that is an o3d.geometry.PointCloud object
        self.pcd = pcd

    def segmentPlanes(self, distance_threshold=1.0, ransac_n=3,
                      num_ransac_iterations=5000, min_plane_points=10,
                      max_num_planes=10):
        """
        Segment planes from point cloud data using RANSAC algorithm.

        Parameters:
        -----------
        distance_threshold: float
            The maximum distance a point can be from the generated plane.
            The points included in the plane are inliers.
            Defaulted to 1.0.
        ransac_n: int
            The minimum number of points needed to form a plane.
            Defaulted to 10.
        num_ransac_iterations: int
            The number of iterations to run the RANSAC algorithm.
            Defaulted to 5000.
        min_plane_points: int
            The minimum number of points to form a plane.
            Defaulted to 3.
        max_num_planes: int
            The maximum number of planes to created from the point cloud
            data using the RANSAC algorithm. Lesser number of planes can
            be created if the RANSAC algorithm used up all the points
            to create a plane or if it cannot find a plane with
            the requested minimum number of points.
            Defaulted to 10.

        Returns:
        --------
        None.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(distance_threshold, float):
            raise TypeError("distance_threshold variable must be of type " +
                            "float.")
        if not isinstance(ransac_n, int) or \
                isinstance(ransac_n, bool):
            raise TypeError("ransac_n variable must be of type int.")
        if not isinstance(num_ransac_iterations, int) or \
                isinstance(num_ransac_iterations, bool):
            raise TypeError("num_ransac_iterations variable must be of " +
                            "type int.")
        if not isinstance(min_plane_points, int) or \
                isinstance(min_plane_points, bool):
            raise TypeError("min_plane_points variable must be of type int.")
        if not isinstance(max_num_planes, int) or \
                isinstance(max_num_planes, bool):
            raise TypeError("max_num_planes variable must be of type int.")
        # A master list of dictionaries with info from all generated planes
        self.plane_list = []
        # Initialize while loop
        current_pcd = self.pcd
        plane_count = 0
        # Create planes up to max number of planes
        max_iterations = max_num_planes * 5
        total_iterations = 0
        while plane_count <= max_num_planes and total_iterations < max_iterations:
            total_iterations += 1
            # Stop loop if there's not enough points to create a plane with
            # the requested minimum points
            if len(current_pcd.points) < max(min_plane_points, ransac_n + 1):
                break
            # Use RANSAC algorithm to detect planes
            plane_model, inliers = current_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_ransac_iterations)
            # Get pcd that generates a plane
            plane_pcd = current_pcd.select_by_index(inliers)
            # Get plane's x, y, z normal vectors
            plane_normal_vectors = np.array(plane_model[:3])
            # Calculate tilt and azimuth
            tilt, az = self.calculatePlaneTiltAzimuth(plane_normal_vectors)
            # Filter for logical rooftop tilt values(any tilt greater
            # than 85 can be assumed to be walls/ non roof structures)
            if tilt > 85:
                # Remove the current pcd from the remaining pcd for the
                # next plane segmentation
                current_pcd = current_pcd.select_by_index(inliers, invert=True)
            else:
                # Generate a random plane color for visualization later
                color = np.random.rand(3)
                # Store plane information
                plane_info_dict = {
                    "plane_id": plane_count,
                    "tilt": tilt,
                    "azimuth": az,
                    "num_points": len(inliers),
                    "pcd": plane_pcd,
                    "color": color
                }
                self.plane_list.append(plane_info_dict)
                # Remove the current pcd from the remaining pcd for the
                # next plane segmentation
                current_pcd = current_pcd.select_by_index(inliers, invert=True)
                plane_count += 1
        return

    def mergeSimilarPlanes(self, az_weight=1.0, tilt_weight=1.0, eps=0.75, min_samples=1):
        """
        Merge planes that are within similar tilt and azimuth threshold.
        Gets the mean tilt and azimuth of the combined planes.

        Parameters:
        -----------
        tilt_diff_threshold: float
            The maximum difference in tilt between the merged planes.
            Defaulted to 5.0.
        azimuth_diff_threshold: float
            The maximum difference in azimuth between the merged planes.
            Defaulted to 10.0.

        Returns:
        --------
        None.
        """
        # Iterate through each plane in the list
        az_rad = np.array([np.radians(p['azimuth']) for p in self.plane_list])
        tilts  = np.array([p['tilt'] for p in self.plane_list])
    
        # Normalize tilt to [0, 1] range so it's comparable to az components
        tilt_norm = (tilts - tilts.min()) / (tilts.max() - tilts.min() + 1e-9)
    
        features = np.column_stack([
            np.sin(az_rad) * az_weight,
            np.cos(az_rad) * az_weight,
            tilt_norm      * tilt_weight,
        ])
    
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean'
                        ).fit_predict(features)
        clusters = {}
        for plane, label in zip(self.plane_list, labels):
            plane['cluster_id'] = int(label)
            clusters.setdefault(int(label), []).append(plane)
        
        # For each cluster, create a summary plane
        cluster_planes_merged = list()
        for cluster in clusters:
            cluster_merged = dict()
            cluster_planes = clusters[cluster]
            cluster_merged['plane_id'] = cluster
            cluster_merged['azimuth'] = np.median([x['azimuth'] for x in cluster_planes])
            cluster_merged['tilt'] = np.median([x['tilt'] for x in cluster_planes])
            cluster_merged['num_points'] = sum([x['num_points'] for x in cluster_planes])
            merged_point_cloud = o3d.geometry.PointCloud()
            point_clouds = [x['pcd'] for x in cluster_planes]
            for pcd in point_clouds:
                merged_point_cloud += pcd
            cluster_merged['pcd'] = merged_point_cloud
            cluster_merged['color'] = np.random.rand(3)
            cluster_planes_merged.append(cluster_merged)
        # save the sumamry planes to the main plane_list
        self.plane_list = cluster_planes_merged
        return 
    
    def visualizePlanes(self):
        """
        Creates a mesh for each plane to create a surface model
        for visualization.

        Parameters:
        -----------
        None.

        Returns:
        --------
        pcd_plane_mesh_list: list
            A list of dictionaries with pcd and its associated mesh.
            The dictionaries has the following "plane_id",
            "pcd", "mesh", and "color" keys.
        """
        # A list to store the all the plane mesh
        pcd_plane_mesh_list = []
        # Create a mesh for each plane in the list
        for plane in self.plane_list:
            # Make a dict to store the pcd and its assocaited mesh
            pcd_plane_mesh_dict = {
                "plane_id": plane["plane_id"],
                # open3d.geometry.PointCloud object
                "pcd": plane["pcd"],
                # open3d.geometry.TriangleMesh object
                "plane_mesh": None,
                "color": plane["color"],
            }
            # Needs at least 3 points to create a mesh from convex hull
            if plane["num_points"] > 3:
                # Create mesh/surface model of plane from convex hull
                hull, _ = plane["pcd"].compute_convex_hull(joggle_inputs=True)
                hull.compute_vertex_normals()
                hull.paint_uniform_color(plane["color"])
                pcd_plane_mesh_dict["plane_mesh"] = hull
            pcd_plane_mesh_list.append(pcd_plane_mesh_dict)
        return pcd_plane_mesh_list

    def createSummaryPlaneDataframe(self, source_crs, scales, offsets):
        """
        Creates dataframe of all the generated planes.

        Parameters:
        -----------
        source_crs: pyproj.crs.CRS
            The source coordinate reference system (crs) of the original
            LiDAR point cloud data.
        scales: tuple, list, or numpy.ndarray
            The scales of the original LiDAR point cloud data.
            The scales contains the x, y, and z scale components in
            the tuple format (x_scale, y_scale, z_scale), list format
            [x_scale, y_scale, z_scale], or numpy array format
            [x_scale y_scale z_scale].
        offsets: tuple, list, or numpy.ndarray
            The offsets of the original LiDAR point cloud data.
            The offsets contains the x, y, and z offset components in
            the tuple format (x_offset, y_offset, z_offset), list format
            [x_offset, y_offset, z_offset], or numpy array format
            [x_offset y_offset z_offset].

        Returns:
        --------
        resultant_df: pandas.DataFrame
            Pandas dataframe of all generated planes. This dataframe
            contains "plane_id", "tilt", "azimuth", "num_points",
            "center_lat", and "center_lon" columns.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(source_crs, pyproj.crs.CRS):
            raise TypeError("source_crs variable must be of a " +
                            "pyproj.crs.CRS object.")
        if not isinstance(scales, (tuple, list, np.ndarray)):
            raise TypeError("scales variable must be of type tuple, list, " +
                            "or numpy.ndarray.")
        if not isinstance(offsets, (tuple, list, np.ndarray)):
            raise TypeError("offsets variable must be of type tuple, list, " +
                            "or numpy.ndarray.")
        # A list to store the metadata of all the resultant planes
        plane_metadata_list = []
        # Iterate through each plane in the list
        for plane in self.plane_list:
            # Get x, y center coordinates from plane and convert those into
            # EPSG:4326 lat, lon center coordinates
            points = np.asarray(plane["pcd"].points)
            center_points = np.mean(points, axis=0)
            lat, lon = self.getPlaneCenters(
                source_crs, scales, offsets,
                center_points[0], center_points[1])
            # Get the plane boundary in lat-lon form
            # generate a concave hull via alphashape
            xy = points[:, :2] 
            boundary_polygon = MultiPoint(xy).convex_hull
            plane_poly = self.getPlaneBoundaryLatLon(source_crs, 
                                                     scales, offsets,
                                                     boundary_polygon)
            
            # Get only important metadata
            plane_metadata_list.append({"plane_id": plane["plane_id"],
                                        "tilt": plane["tilt"],
                                        "azimuth": plane["azimuth"],
                                        "num_points": plane["num_points"],
                                        "center_lat": lat,
                                        "center_lon": lon,
                                        "plane_polygon": plane_poly})
        resultant_df = pd.DataFrame(plane_metadata_list)
        return resultant_df

    def getPlaneCenters(self, source_crs, scales, offsets,
                        center_x, center_y):
        """
        Gets the center of the plane in EPSG:4326 lat, lon format.

        Parameters:
        -----------
        source_crs: pyproj.crs.CRS
            The source coordinate reference system (crs) of the original
            LiDAR point cloud data.
         scales: tuple, list, or numpy.ndarray
            The scales of the original LiDAR point cloud data.
            The scales contains the x, y, and z scale components in
            the tuple format (x_scale, y_scale, z_scale), list format
            [x_scale, y_scale, z_scale], or numpy array format
            [x_scale y_scale z_scale].
        offsets: tuple, list, or numpy.ndarray
            The offsets of the original LiDAR point cloud data.
            The offsets contains the x, y, and z offset components in
            the tuple format (x_offset, y_offset, z_offset), list format
            [x_offset, y_offset, z_offset], or numpy array format
            [x_offset y_offset z_offset].
        center_x: float
            The center x coordinate of the plane.
        center_y: float
            The center y coordinate of the plane.

        Returns:
        --------
        center_lat: float
            The center latitude of the plane.
        center_lon: float
            The center longitude of the plane.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(source_crs, pyproj.crs.CRS):
            raise TypeError("source_crs variable must be of type " +
                            "pyproj.crs.CRS.")
        if not isinstance(scales, (tuple, list, np.ndarray)):
            raise TypeError("scales variable must be of type tuple, list, " +
                            "or numpy.ndarray.")
        if not isinstance(offsets, (tuple, list, np.ndarray)):
            raise TypeError("offsets variable must be of type tuple, list, " +
                            "or numpy.ndarray.")
        if not isinstance(center_x, float):
            raise TypeError("center_x variable must be of type float.")
        if not isinstance(center_y, float):
            raise TypeError("center_y variable must be of type float.")
        # Scale x,y to match data
        scaled_x = center_x * scales[0] + offsets[0]
        scaled_y = center_y * scales[1] + offsets[1]
        # Create projection transformer
        if source_crs.is_compound:
            # For componded crs, get its horizontal crs component
            horizontal_crs = source_crs.sub_crs_list[0]
            transformer = pyproj.Transformer.from_crs(
                horizontal_crs, "EPSG:4326",  always_xy=True)
        else:
            transformer = pyproj.Transformer.from_crs(
                source_crs, "EPSG:4326",  always_xy=True)
        # Project lidar source crs onto lat, lon "EPSG:4326" crs
        center_lon, center_lat = transformer.transform(scaled_x, scaled_y)
        return center_lat, center_lon
    
    def getPlaneBoundaryLatLon(self, source_crs, scales, offsets, boundary_polygon):
        """
        Reprojects a plane boundary polygon from LiDAR source CRS to EPSG:4326.
    
        Parameters:
        -----------
        source_crs: pyproj.crs.CRS
            The source coordinate reference system of the original LiDAR data.
        scales: tuple, list, or numpy.ndarray
            The x, y, z scale components (x_scale, y_scale, z_scale).
        offsets: tuple, list, or numpy.ndarray
            The x, y, z offset components (x_offset, y_offset, z_offset).
        boundary_polygon: shapely.geometry.Polygon or MultiPolygon
            The plane boundary in raw LiDAR coordinate space.
    
        Returns:
        --------
        shapely.geometry.Polygon or MultiPolygon in EPSG:4326 (lat/lon)
        """
        if not isinstance(source_crs, pyproj.crs.CRS):
            raise TypeError("source_crs must be of type pyproj.crs.CRS.")
        if not isinstance(scales, (tuple, list, np.ndarray)):
            raise TypeError("scales must be of type tuple, list, or numpy.ndarray.")
        if not isinstance(offsets, (tuple, list, np.ndarray)):
            raise TypeError("offsets must be of type tuple, list, or numpy.ndarray.")
    
        # Build transformer
        if source_crs.is_compound:
            horizontal_crs = source_crs.sub_crs_list[0]
            transformer = pyproj.Transformer.from_crs(
                horizontal_crs, "EPSG:4326", always_xy=True)
        else:
            transformer = pyproj.Transformer.from_crs(
                source_crs, "EPSG:4326", always_xy=True)
    
        def _transform_coords(xs, ys):
            # Apply scale + offset
            scaled_x = np.asarray(xs) * scales[0] + offsets[0]
            scaled_y = np.asarray(ys) * scales[1] + offsets[1]
            lons, lats = transformer.transform(scaled_x, scaled_y)
            # shapely_transform expects (x, y) = (lon, lat) for EPSG:4326
            return lons, lats
        
        return shapely_transform(_transform_coords, boundary_polygon)
    
    def getBestPlane(self):
        """
        Gets the best plane from the number of points.
        The plane with the largest number of points is the best plane.

        Parameters:
        -----------
        None.

        Returns:
        --------
        best_plane: dict
            A dictionary containing the metadata of the best plane.
        found_best_plane: bool
            A boolean flag to indicate if a best plane can be found in the
            planes list.
        """
        # Keep track of the largest number of points in the plane
        largest_num_points = -float("inf")
        # Best plane found
        best_plane = None
        # Flag if a best plane can be found in the list
        found_best_plane = False
        # Iterate through each plane in the list to get the plane
        # with the largest number of points
        for plane in self.plane_list:
            num_points = plane["num_points"]
            # Update the best plane if the current plane has more points
            if largest_num_points < num_points:
                largest_num_points = num_points
                best_plane = plane
                found_best_plane = True
        return best_plane, found_best_plane

    def calculatePlaneTiltAzimuth(self, plane_normal_vector):
        """
        Calculates a plane's tilt and azimuth from a plane's normal vectors.

        Parameters:
        -----------
        plane_normal_vector: tuple, list, or numpy.ndarray
            A tuple, list, or numpy array containing the x, y, and z components
            of the plane's normal vector in a tuple format (normal_x, normal_y,
            normal_z), list format [normal_x, normal_y, normal_z], or numpy
            array format[normal_x normal_y normal_z].

        Returns:
        --------
        tilt: float
            The calculated tilt angle of the plane in degrees.
        azimuth: float
            The calculated azimuth angle of the plane in degrees.
        """
        # Ensure that the input is of the correct type
        if not isinstance(plane_normal_vector, (tuple, list, np.ndarray)):
            raise TypeError("plane_normal_vector variable must be of type " +
                            "tuple, list, or numpy.ndarray.")
        normal_x, normal_y, normal_z = plane_normal_vector
        if abs(normal_x) > 1 or abs(normal_y) > 1 or abs(normal_z) > 1:
            print("Plane vectors are not normalized. Normalizing them now.")
            # Normalize the plane's vector if they are not normalized
            magnitude = np.linalg.norm(plane_normal_vector)
            normal_x, normal_y, normal_z = (normal_x/magnitude,
                                            normal_y/magnitude,
                                            normal_z/magnitude)
        # Make sure that the vectors are in the correct orientation
        if normal_z < 0:
            normal_x, normal_y, normal_z = -normal_x, -normal_y, -normal_z
        # Get tilt angle from horizontal
        tilt = np.degrees(np.arccos(normal_z))
        # Get azimuth angle in degrees
        azimuth = np.degrees(np.arctan2(normal_x, normal_y))
        # Make sure azimuth is in 0-360 range
        if azimuth < 0:
            azimuth += 360
        return float(tilt), float(azimuth)
