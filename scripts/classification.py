# Example: Land cover classification with rasterio + sklearn
import rasterio
from sklearn.cluster import KMeans

with rasterio.open("data/maharashtra_satellite.tif") as src:
    img = src.read().reshape((src.count, -1)).T

kmeans = KMeans(n_clusters=5, random_state=42).fit(img)
classified = kmeans.labels_.reshape(src.height, src.width)

print("Classification done. Shape:", classified.shape)
