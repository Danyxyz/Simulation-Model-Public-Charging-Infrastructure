import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# List preparation
# Reads a subset of the actual list with 1165 datapoints - testing purposes
# List had to be changed due to different coordinate system on the full list (LV95 but WGS85 is used here)
# All Columns =  ADR_EGAID / ZIP_LABEL / BDG_NAME / COM_FOSNR / ADR_STATUS / ADR_OFFICIAL / ADR_MODIFIED / ADR_EDID / STN_LABEL / ADR_VALID / ADR_NUMBER / STR_ESID / BDG_EGID / ADR_MODIFIED
# Relevant columns for this exercise COM_CANTON / BDG_CATEGORY / ADR_EASTING / ADR_NORTHING

# Read in data
df = pd.read_csv("TestSet_Pfeffingen.csv", delimiter=";",encoding="ISO-8859-1",engine='python', on_bad_lines="skip", usecols=["COM_CANTON","BDG_CATEGORY",
 "ADR_EASTING", "ADR_NORTHING"])

#Create lists to store x-coordiantes and y-coordinate separately
x_List = []
y_List = []

# Loop through dateset and append x and y coordinates
for i in df:
 x_List.append(df["ADR_EASTING"])
 y_List.append(df["ADR_NORTHING"])

# Create a seperate list only x- and y-axis for kmeans algorithm and put into pandas Dataframe
df_coordinates = df.drop(["COM_CANTON", "BDG_CATEGORY"], axis=1)
df_coordinates = pd.DataFrame(df_coordinates)

# Convert df to numpy array
np_array = df_coordinates.to_numpy()

# Perform kmeans

# Set number of cluster
k = 10
kmeans = KMeans(n_clusters=k, random_state=0).fit(np_array)

# what are labels ?
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#Store centroids of x and y individually
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

print("Data volume = "+str(df.count()))
print("Number of Cluster = "+str(k))
print("Clustercentroids = "+str(centroids))

#plot the results
plt.scatter(x_List, y_List, color="blue")
plt.scatter(centroids_x, centroids_y, s=50, color="red")
plt.show()

