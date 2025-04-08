import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
import argparse


def create_bbox(npz_file, output_file):
    # Load .npz file
    data = np.load(npz_file)
    meta = data['metadata']

    # Extract latitude and longitude
    lats= np.array(meta[:,1])
    lons = np.array(meta[:,2])

    # Compute the bbox
    minx, maxx = lons.min(), lons.max()
    miny, maxy = lats.min(), lats.max()
    bbox = box(minx, miny, maxx, maxy)

    # Create gdf and save
    ext = os.path.splitext(output_file)[1].lower()

    if ext == ".geojson" or ext == ".json":
        gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs = CRS.from_epsg(4326))
        gdf.to_file(output_file, driver = "GeoJSON")
    elif ext == ".shp":
        gdf.to_file(output_file, driver="ESRI Shapefile")
    else:
        raise ValueError("Unsupported file format. Use .geojson, .json, or .shp")

    print(f"Bounding box saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create bounding box from .npz metadata and export as GeoJSON or Shapefile.")
    parser.add_argument("npz_file", help="Path to input file containing metadata")
    parser.add_argument("output_file", help="Output file path, must have a .geojson or .shp extension")
    args = parser.parse_args()

    create_bbox(args.npz_file, args.output_file)