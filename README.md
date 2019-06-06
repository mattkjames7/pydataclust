# pydataclust

```pydataclust``` is currently a pair of unsupervised data clustering 
routines: k-means and DBSCAN.

## k-means

To test the k-means routine using:

```python
import pydataclust as pdc
pdc.Test()
```

This will set up some data then attempt to cluster it.

To use this routine properly:

```python
#by default, only k needs to be provided and data will automatically be scaled
km = pdc.kmeans(k)
km.InsertData(data,Rescale=True)

#otherwise we could do this in one step
km = pdc.kmeans(k,data=data,Rescale=True)
```
where ```k``` is the number of desired clusters. ```data``` is a matrix, 
shape ```(m,n)```, containing ```m``` sample points in a parameter space 
of ```n``` dimensions. In the above code, a ```kmeans``` object is 
created, where data are scaled automatically by default - by setting 
```Rescale=False```,this behaviour can be suppressed. 

We can then train the clusters using:

```python
km.Train(nSteps=10,Reset=False)
```
where setting ```Reset=True``` will randomize the cluster centroids 
again. To access the cluster centroids rescaled to the original data scale:

```python
cent = km.UnscaledCentroids()
```

where ```cent``` has the shape ```(k,n)```.

To obtain the classifications for each point, we can access 
```km.clusters```, which is an array of indices of shape ```(m,)```, the
value of each corresponding to the first dimension of ```cent``` (i.e.
if ```km.clusters[i] == j```, the point defined by ```data[i,:]``` is a
part of the cluster ```j``` with the centroid ```cent[j,:]``` .

## DBSCAN

This routine is slightly different to the k-means clustering routine in
that the number of clusters is not predefined, instead it effectively
uses the density of points to define the clusters. 

Test DBSCAN using:

```python
import pydataclust as pdc
pdc.TestDBSCAN()
```

Usage for real data:

```python
#no need to supply anything to create the initial object
db = pdc.dbscan()
db.InsertData(data,Rescale=True)

#otherwise we could do this in one step
db = pdc.dbscan(data,Rescale=True)
```

where ```data``` has the shape ```(m,n)```, ```m``` being the number of 
sample points, and ```n``` being the number of dimensions. As with
```kmeans```, we can opt out of the automatic rescaling using 
```Rescale=False```.

To cluster the data:

```python
#cluster the data supplying epsilon and minpts
db.ClusterData(epsilon,minpts)

#obtain the cluster IDS for each data point
clustID = db.clusters
```

where ```epsilon``` is the radial distance from a given point within
which any other point must lie in parameter space in order to be part of
the same cluster. ```minpts``` defines the minimum number of neighbours
a point must have in order to start creating a cluster. ```clustID``` is
an integer array, where each element corresponds to a point
in ```data``` and the value correponds to the cluster it has been 
assigned to. A positive integer corresponds to a cluster, -1 corresponds 
to noise.
