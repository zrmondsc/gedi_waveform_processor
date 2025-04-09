import os
os.environ['PROJ_LIB'] = r"C:\Users\Zachary\anaconda3\envs\gedi_pro_env\Library\share\proj"
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
import argparse

def create_points(npz_file, output_file):
    # Load .npz file
    data = np.load(npz_file)
    meta = data['metadata']
    shot_index = data['shot_index']

    # Extract lat/lon
    lats = np.array(meta[:, 0])
    lons = np.array(meta[:, 1])

    # Extract shot numbers
    shot_number = shot_index

    # Create a list of geometries
    points = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    gdf = gpd.GeoDataFrame({'index': np.arange(len(points)), 'shot_num': shot_number}, geometry=points, crs=CRS.from_epsg(4326))

    # Save based on file extension
    ext = os.path.splitext(output_file)[1].lower()
    if ext in [".geojson", ".json"]:
        gdf.to_file(output_file, driver="GeoJSON")
    elif ext == ".shp":
        gdf.to_file(output_file, driver="ESRI Shapefile")
    else:
        raise ValueError("Unsupported file format. Use .geojson, .json, or .shp")

    print(f"Saved {len(gdf)} points to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GEDI point geometries from .npz metadata.")
    parser.add_argument("npz_file", help="Path to input .npz file")
    parser.add_argument("output_file", help="Path to output .geojson or .shp file")
    args = parser.parse_args()

    create_points(args.npz_file, args.output_file)