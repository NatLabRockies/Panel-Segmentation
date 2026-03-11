import numpy as np
import pandas as pd
import pyproj


class PlaneSegmentation:
    '''
    A class that segments planes from a pcd (point cloud data).
    '''

    def __init__(self, pcd):
        # Point cloud data that is an o3d.geometry.PointCloud object
        self.pcd = pcd

    def segmentPlanes(self, distance_threshold=1.0, ransac_n=3,
                      num_ransac_iterations=5000, min_plane_points=3,
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
            Defaulted to 3.
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
        while plane_count <= max_num_planes:
            # Stop loop if there's not enough points to create a plane with
            # the requested minimum points
            if len(current_pcd.points) < min_plane_points:
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
                    "normal_plane_vectors": plane_normal_vectors,
                    "tilt": tilt,
                    "azimuth": az,
                    "num_points": len(inliers),
                    "pcd": np.asarray(plane_pcd.points),
                    "color": color
                }
                self.plane_list.append(plane_info_dict)
                # Remove the current pcd from the remaining pcd for the
                # next plane segmentation
                current_pcd = current_pcd.select_by_index(inliers, invert=True)
                plane_count += 1

    def mergeSimilarPlanes(self, tilt_diff_threshold=5.0,
                           azimuth_diff_threshold=10.0):
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
        # Ensure that the inputs are of the correct type
        if not isinstance(tilt_diff_threshold, float):
            raise TypeError("tilt_diff_threshold variable must be of type " +
                            "float.")
        if not isinstance(azimuth_diff_threshold, float):
            raise TypeError("azimuth_diff_threshold variable must be of " +
                            "type float.")
        # A master list of dictionaries with info from all merged planes
        merged_plane_list = []
        merged_plane_id = set()
        # Initialize new plane id
        new_idx = 0
        # Iterate through each plane in the list
        for idx_1, plane_1 in enumerate(self.plane_list):
            # Skip plane idx if it is already merged
            if idx_1 in merged_plane_id:
                continue
            # Group similar planes together
            grouped_planes = []
            grouped_planes.append(plane_1)
            merged_plane_id.add(idx_1)
            # Compare plane_1 with plane_2
            for idx_2, plane_2 in enumerate(self.plane_list):
                if idx_2 in merged_plane_id:
                    continue
                # Calculate the difference in tilt and azimuth
                tilt_diff = abs(plane_1["tilt"] - plane_2["tilt"])
                # Find the smallest angle between the difference in azimuth
                az_diff = abs(plane_1["azimuth"] - plane_2["azimuth"]) % 360
                az_diff = min(az_diff, 360 - az_diff)
                # Combine plane if the tilt and azimuth is within the threshold
                if tilt_diff <= tilt_diff_threshold:
                    if az_diff <= azimuth_diff_threshold:
                        grouped_planes.append(plane_2)
                        merged_plane_id.add(idx_2)
            # Add plane_1 to the master list if no similarities are found
            if len(grouped_planes) <= 1:
                plane_1["plane_id"] = new_idx
                plane_1["combined_from"] = "None"
                merged_plane_list.append(plane_1)
                new_idx += 1
            # Combine metadata of similar planes
            else:
                # Find the mean tilt, azimuth, and normal vectors
                mean_tilt = np.mean([plane["tilt"]
                                    for plane in grouped_planes])
                mean_az = np.mean([plane["azimuth"]
                                  for plane in grouped_planes])
                mean_normal_vectors = np.mean(
                    [plane["normal_plane_vectors"]
                     for plane in grouped_planes], axis=0)
                # Get other combined metadata
                num_points = sum(plane["num_points"]
                                 for plane in grouped_planes)
                combined_from = [plane["plane_id"]
                                 for plane in grouped_planes]
                combined_pcd = sum(
                    [plane["pcd"]for plane in grouped_planes[1:]],
                    grouped_planes[0]["pcd"])
                # Generate a random plane color for visualization later
                color = np.random.rand(3)
                # Add everything to a dict
                new_plane_info_dict = {
                    "plane_id": new_idx,
                    "normal_plane_vectors": mean_normal_vectors,
                    "tilt": mean_tilt,
                    "azimuth": mean_az,
                    "num_points": num_points,
                    "pcd": combined_pcd,
                    "color": color,
                    "combined_from": combined_from
                }
                # Add the merged plane and its metadata to the master list
                merged_plane_list.append(new_plane_info_dict)
                new_idx += 1
        self.plane_list = merged_plane_list

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
            # Get only important metadata
            plane_metadata_list.append({"plane_id": plane["plane_id"],
                                        "tilt": plane["tilt"],
                                        "azimuth": plane["azimuth"],
                                        "num_points": plane["num_points"],
                                        "center_lat": lat,
                                        "center_lon": lon})
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
