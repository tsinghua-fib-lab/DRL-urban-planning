#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString
from geopandas import GeoSeries, GeoDataFrame

get_ipython().run_line_magic('matplotlib', 'inline')
ox.__version__


# ### retrieve HLG streets

# In[28]:


# define a bounding box in Beijing
north, south, east, west = 40.0887, 40.0698, 116.3444, 116.31852


# create network from that bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
G_projected = ox.project_graph(G)
gdf = ox.graph_to_gdfs(G_projected)


# In[29]:


fig, ax = ox.plot_graph(G_projected)


# In[30]:


road_types = gdf[1]['highway'].to_list()
set([var for var in road_types if not isinstance(var, list)])


# In[31]:


gdf_main = gdf[1][
    (gdf[1]['highway']=='primary') |
    (gdf[1]['highway']=='secondary') |
    (gdf[1]['highway']=='tertiary')]


# In[32]:


gdf_main.explore()


# In[33]:


intersections = gdf[0][gdf[0].intersects(gdf_main.unary_union)]
intersections.explore()


# In[42]:


points = MultiPoint([var.centroid for var in intersections.unary_union.buffer(30).geoms])
intersections_slim = GeoDataFrame({'id': list(range(len(points))), 'geometry': list(points.geoms)})
intersections_slim = intersections_slim.set_crs(intersections.crs)
intersections_slim['x'] = intersections_slim['geometry'].x
intersections_slim['y'] = intersections_slim['geometry'].y
intersections_slim.explore()


# In[43]:


intersections_slim['geometry'] = intersections_slim['geometry'].apply(lambda p: Point(20*int(p.x/20), 20*int(p.y/20)))
intersections_slim['x'] = intersections_slim['geometry'].x
intersections_slim['y'] = intersections_slim['geometry'].y
intersections_slim.explore()


# ### retrieve DHM streets

# In[2]:


# define a bounding box in Beijing
north, south, east, west = 39.8558, 39.8305, 116.4173, 116.3939


# create network from that bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
G_projected = ox.project_graph(G)
gdf = ox.graph_to_gdfs(G_projected)


# In[3]:


fig, ax = ox.plot_graph(G_projected)


# In[4]:


road_types = gdf[1]['highway'].to_list()
set([var for var in road_types if not isinstance(var, list)])

# In[6]:


gdf_main = gdf[1][
    (gdf[1]['highway']=='trunk') |
    (gdf[1]['highway']=='primary') |
    (gdf[1]['highway']=='secondary') |
    (gdf[1]['highway']=='tertiary')]


# In[7]:


gdf_main.explore()


# In[8]:


intersections = gdf[0][gdf[0].intersects(gdf_main.unary_union)]
intersections.explore()


# In[9]:


points = MultiPoint([var.centroid for var in intersections.unary_union.buffer(30).geoms])
intersections_slim = GeoDataFrame({'id': list(range(len(points))), 'geometry': list(points.geoms)})
intersections_slim = intersections_slim.set_crs(intersections.crs)
intersections_slim['x'] = intersections_slim['geometry'].x
intersections_slim['y'] = intersections_slim['geometry'].y
intersections_slim.explore()


# In[10]:


intersections_slim['geometry'] = intersections_slim['geometry'].apply(lambda p: Point(int(p.x), int(p.y)))
intersections_slim['x'] = intersections_slim['geometry'].x
intersections_slim['y'] = intersections_slim['geometry'].y
intersections_slim.explore()

