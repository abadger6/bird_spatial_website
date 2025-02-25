import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
from pyproj import Transformer
import matplotlib.pyplot as plt
from rasterio.plot import show

# Define a bounding box in lat/lon (EPSG:4326) for North America.
min_lon, min_lat = -170, 15
max_lon, max_lat = -50, 75
bbox = box(min_lon, min_lat, max_lon, max_lat)

# Transform the bounding box from EPSG:4326 to ESRI:54012.
transformer = Transformer.from_crs("EPSG:4326", "ESRI:54012", always_xy=True)
minx, miny = transformer.transform(min_lon, min_lat)
maxx, maxy = transformer.transform(max_lon, max_lat)
bbox_transformed = box(minx, miny, maxx, maxy)

# Open the TIF file and crop it using the transformed bounding box.
tif_path = "bird_spatial_website/birds_ml/amekes/amekes_abundance_seasonal_breeding_mean_2022.tif"
with rasterio.open(tif_path) as src:
    print("CRS:", src.crs)
    print("Dimensions:", src.width, "x", src.height)
    print("Number of bands:", src.count)
    # Read the first band.
    band1 = src.read(1)


with rasterio.open(tif_path) as src:
    out_image, out_transform = mask(src, [mapping(bbox_transformed)], crop=True)
    out_meta = src.meta.copy()

# Update the metadata with new dimensions and transform.
out_meta.update({
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

# Save the cropped image to a new file.
cropped_tif_path = "amekes_abundance_seasonal_breeding_mean_2022_cropped.tif"
with rasterio.open(cropped_tif_path, "w", **out_meta) as dest:
    dest.write(out_image)

print("Cropped TIF saved as", cropped_tif_path)

import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show

# Path to the cropped TIF file.
cropped_tif_path = "amekes_abundance_seasonal_breeding_mean_2022_cropped.tif"

with rasterio.open(cropped_tif_path) as src:
    # Read the first band.
    band1 = src.read(1)
    print("CRS:", src.crs)
    print("Dimensions:", src.width, "x", src.height)

# Plot the cropped TIF.
plt.figure(figsize=(10, 8))
show(band1, cmap='viridis')
plt.title("Cropped TIF: Amekes Abundance Seasonal Breeding Mean 2022")
#plt.show()
