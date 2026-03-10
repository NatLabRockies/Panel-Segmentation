"""
Utility functions specifically for auto-geotagging algorithm.
"""

import pandas as pd
import glob
import geopandas as gpd
from shapely.geometry import Polygon
import os
import numpy as np
import math

def get_rectangle_dimensions(polygon):
    """
    Calculate the width and length of a rectangular polygon.
    Returns the two edge lengths and their orientations.
    """
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
            width_edge_idx = 0
        else:
            width = edge2_length
            length = edge1_length
            width_angle = edges[1]['angle']
            length_angle = edges[0]['angle']
            width_edge_idx = 1
            
        return {
            'width': width,
            'length': length,
            'width_angle': width_angle,
            'length_angle': length_angle,
            'edges': edges,
            'width_edge_idx': width_edge_idx
        }
    
    return None

def standardize_rectangle_width(polygon, target_width):
    """
    Adjust a rectangular polygon to have a standardized width while
    maintaining its center position, orientation, and length.

    Parameters
    -----------
    polygon: Shapely polygon representing a tracker row
        Latitude coordinate of the site.
    target_width: float
        Longitude coordinate of the site.

    Returns
    -------
    Figure
        Figure of the satellite image
    """
    dims = get_rectangle_dimensions(polygon)
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
    # The four corners are: center ± half_length * length_dir ± half_width * width_dir
    corners = [
        center + half_length * length_dir + half_width * width_dir,
        center + half_length * length_dir - half_width * width_dir,
        center - half_length * length_dir - half_width * width_dir,
        center - half_length * length_dir + half_width * width_dir,
        center + half_length * length_dir + half_width * width_dir,
    ]
    
    return Polygon(corners)


def remove_false_positives(class_counts, df):
    """
    """
    class_counts_rows = class_counts[class_counts['class'].isin([1, 2, 3])]
    class_counts_rows['total_pct'] = class_counts_rows['count'] / class_counts_rows['count'].sum()
    # if one of the cases is more than 98% represented, then delete the other cases as an error
    classes_to_omit = list(class_counts_rows[class_counts_rows['total_pct']<= 0.02]['class'])
    df = df[~df['class'].isin(classes_to_omit)]
    
    
def remove_anomalous_area_cases(df):
    """
    """
    df['area'] = [x.area for x in list(df['geometry'])]
    df['avg_class_area'] = df.groupby("class")['area'].transform('mean')    
    df = df[df['area'] >= df['avg_class_area']*.05]