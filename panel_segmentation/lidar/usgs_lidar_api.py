import requests
import os
import pandas as pd
import zipfile
from pathlib import Path
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class USGSLidarAPI:
    '''
    A class that pulls LiDAR data from USGS.
    
    An interactive explorer is found at:
    https://apps.nationalmap.gov/lidar-explorer/#/
    '''

    def __init__(self, output_folder="data"):
        # Output folder to save the pulled LiDAR data
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # Base url where all dataset files are located
        self.base_url =  "https://tnmaccess.nationalmap.gov/api/v1/products"
        
    def getLazFile(self, polygon, max_retries=5, backoff_factor=3):
        """
        Based on the polygon boundaries, pull down the associated LiDAR Laz file.
        """
        minx, miny, maxx, maxy = list(polygon.bounds)
        bbox_str = f"{minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy}"
    
        params = {
            "polygon": bbox_str,
            "datasets": "Lidar Point Cloud (LPC)",
            "prodFormats": "LAS,LAZ",
            "outputFormat": "JSON",
            "max": 50,
            "offset": 0,
        }
    
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,  # waits 3, 6, 12, 24, 48 seconds between retries
            status_forcelist=[500, 502, 503, 504],  # retry on these HTTP errors
            allowed_methods=["GET"]
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
    
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(
                    self.base_url,
                    params=params,
                    verify=False,
                    timeout=180
                )
                response.raise_for_status()
                data = response.json()
                print(f"Success on attempt {attempt}")
                return data
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt}")
                if attempt == max_retries:
                    raise
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
    
    def bbox_area_deg2(self, bbox):
        """
        Returns area in degrees² adjusted for latitude distortion.
        Comparable across bounding boxes — no absolute units needed.
        """
        lat_center = (bbox['minY'] + bbox['maxY']) / 2
        width  = (bbox['maxX'] - bbox['minX']) * np.cos(np.radians(lat_center))
        height = (bbox['maxY'] - bbox['minY'])
        return width * height

        
    def selectBestLazFile(self, data):
        """
        Select the best Laz file to download based on its recency and resolution.
        """
        df = pd.DataFrame(data['items'])
        # 1. Drop duplicate download URLs — same file registered multiple times
        df_unique = df.drop_duplicates(subset='downloadURL')
        
        # 2. Compare file size against bounding box size to get
        # relative point density
        df_unique['relative_pts_per_size'] = [self.bbox_area_deg2(x)/y for x, y in 
                                              zip(df_unique['boundingBox'],
                                                  df_unique['sizeInBytes'])]
        
        # 3. Also prefer newer acquisition dates
        df_unique['publicationDate'] = pd.to_datetime(df_unique['publicationDate'])
        df_unique = df_unique.sort_values(['relative_pts_per_size', 'publicationDate'], 
                                           ascending=[False, False])
        # Take the first case for download
        return df_unique['downloadURL'].iloc[0]


    def download_and_extract(self, url, output_dir):
        filename = Path(url).name
        local_path = Path(output_dir) / filename
        
        # Check if the file already exists. Don't download it if it does
        if os.path.exists(local_path):
            print(f"File {local_path} already exists, skipping...")
        else:
            # Download it if it doesn't exist
            print(f"\nDownloading {filename}...")
            with requests.get(url, stream=True, verify=False) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    
            # If zip, extract LAZ only and delete zip
            if filename.endswith(".zip"):
                with zipfile.ZipFile(local_path, "r") as z:
                    laz_files = [f for f in z.namelist() if f.endswith((".laz", ".las"))]
                    for laz in laz_files:
                        out = output_dir / Path(laz).name
                        with z.open(laz) as src, open(out, "wb") as dst:
                            dst.write(src.read())
                        print(f"  Extracted: {out}")
                local_path.unlink()
            else:
                print(f"  Saved: {local_path}")
        return local_path