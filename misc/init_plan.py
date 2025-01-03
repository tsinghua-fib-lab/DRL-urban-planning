#!/usr/bin/env python
# coding: utf-8

# ### import

# In[1]:


from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, box
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from shapely.ops import split, unary_union, polygonize
import pickle
import osmnx as ox


# ### aggregate function from main roads

# In[2]:


def aggregate(lines):
    line = lines[0]
    for var in range(1, len(lines)):
        line = split(line, lines[var])
        if len(line.geoms) > 1:
            line = MultiLineString(line)
        else:
            line = line.geoms[0]
        new_line = split(lines[var], line)
        if len(new_line.geoms) > 1:
            new_line = MultiLineString(new_line)
        else:
            new_line = new_line.geoms[0]
        line = line.union(new_line)
    roads = list(line.geoms)
    roads_type = [2 for _ in range(len(roads))]
    intersections = []
    for road in roads:
        intersections.append(Point(road.coords[0]))
        intersections.append(Point(road.coords[1]))
    intersections = list(unary_union(intersections).geoms)
    intersections_type = [13 for _ in range(len(intersections))]
    feasibles = list(polygonize(roads))
    feasibles_type = [1 for _ in range(len(feasibles))]
    types = roads_type + intersections_type + feasibles_type
    geometries = roads + intersections + feasibles
    gdf = GeoDataFrame({'id': list(range(len(types))),
                        'type': types,
                        'existence': [True for _ in range(len(types))],
                        'geometry': geometries}).set_index('id')
    return gdf


# ## synthetic

# ### grid

# In[26]:


z = [LineString([(0, 0), (0, 240)]),
     LineString([(0, 240), (240, 240)]),
     LineString([(240, 240), (240, 0)]),
     LineString([(240, 0), (0, 0)]),
     LineString([(0, 120), (240, 120)]),
     LineString([(120, 0), (120, 240)]),
     LineString([(60, 0), (60, 240)]),
     LineString([(190, 0), (190, 240)]),
     LineString([(0, 50), (240, 50)]),
     LineString([(0, 180), (240, 180)])
    ]


# In[27]:


gdf = aggregate(z)


# In[28]:


gdf


# In[29]:


gdf[gdf.geom_type!='Polygon'].plot()


# In[30]:


d = dict()
d['gdf'] = gdf
with open('../cfg/test_data/synthetic/init_plan_grid.pickle', 'wb') as f:
    pickle.dump(d, f)



# ## real

# ### hlg

# #### road + land_use

# In[138]:


z = [LineString([(442340, 4435700), (442720, 4435700)]), # 同成街
     LineString([(442720, 4435700), (443180, 4435700)]), # 同成街
     LineString([(443180, 4435700), (443600, 4435700)]), # 同成街
     LineString([(443600, 4435700), (444060, 4435700)]), # 同成街
     
     LineString([(441900, 4437560), (442000, 4437240)]), # 育知路
     LineString([(442000, 4437240), (442120, 4436880)]), # 育知路
     LineString([(442120, 4436880), (442240, 4436480)]), # 育知路
     LineString([(442240, 4436480), (442300, 4436160)]), # 育知路
     LineString([(442300, 4436160), (442340, 4435700)]), # 育知路
     
     LineString([(441900, 4437560), (442440, 4437650)]), # 回南北路
     LineString([(442440, 4437650), (442980, 4437740)]), # 回南北路
     LineString([(442980, 4437740), (443500, 4437740)]), # 回南北路
     LineString([(443500, 4437740), (443960, 4437740)]), # 回南北路
     
     LineString([(443960, 4437740), (443980, 4437400)]), # 文华东路
     LineString([(443980, 4437400), (444000, 4437060)]), # 文华东路
     LineString([(444000, 4437060), (444060, 4436720)]), # 文华东路
     LineString([(444060, 4436720), (444060, 4436200)]), # 文华东路
     LineString([(444060, 4436200), (444060, 4435700)]), # 文华东路
     
     LineString([(442240, 4436480), (442650, 4436630)]), # 回龙观西大街
     LineString([(442650, 4436630), (443100, 4436720)]), # 回龙观西大街
     LineString([(443100, 4436720), (443560, 4436720)]), # 回龙观西大街
     LineString([(443560, 4436720), (444060, 4436720)]), # 回龙观东大街
     
     LineString([(442440, 4437650), (442510, 4437310)]), # 育知东路
     LineString([(442510, 4437310), (442580, 4436970)]), # 育知东路
     LineString([(442580, 4436970), (442650, 4436630)]), # 育知东路
     LineString([(442650, 4436630), (442720, 4436200)]), # 育知东路
     LineString([(442720, 4436200), (442720, 4435700)]), # 育知东路
     
     LineString([(442980, 4437740), (443020, 4437400)]), # 文华西路
     LineString([(443020, 4437400), (443060, 4437060)]), # 文华西路
     LineString([(443060, 4437060), (443100, 4436720)]), # 文华西路
     LineString([(443100, 4436720), (443180, 4436200)]), # 文华西路
     LineString([(443180, 4436200), (443180, 4435700)]), # 文华西路
     
     LineString([(443500, 4437740), (443520, 4437400)]), # 文华路
     LineString([(443520, 4437400), (443540, 4437060)]), # 文华路
     LineString([(443540, 4437060), (443560, 4436720)]), # 文华路
     LineString([(443560, 4436720), (443600, 4436200)]), # 文华路
     LineString([(443600, 4436200), (443600, 4435700)]), # 文华路
     
     LineString([(442300, 4436160), (442720, 4436200)]), # 龙跃街
     LineString([(442720, 4436200), (443180, 4436200)]), # 龙跃街
     LineString([(443180, 4436200), (443600, 4436200)]), # 龙跃街
     LineString([(443600, 4436200), (444060, 4436200)]), # 龙跃街
     
     LineString([(442000, 4437240), (442510, 4437310)]), # 龙禧二街
     LineString([(442510, 4437310), (443020, 4437400)]), # 龙禧二街
     LineString([(443020, 4437400), (443520, 4437400)]), # 龙禧二街
     LineString([(443520, 4437400), (443980, 4437400)]), # 龙禧二街
     
     LineString([(442120, 4436880), (442580, 4436970)]), # 龙禧三街
     LineString([(442580, 4436970), (443060, 4437060)]), # 龙禧三街
     LineString([(443060, 4437060), (443540, 4437060)]), # 龙禧三街
     LineString([(443540, 4437060), (444000, 4437060)]), # 龙禧三街
     
     LineString([(442060, 4437060), (442302, 4437100)]), # 回龙园东侧空地
     LineString([(442255, 4437275), (442302, 4437100)]), # 云趣园二期西侧道路
     LineString([(442302, 4437100), (442350, 4436925)]), # 云趣园二期西侧道路
     
     LineString([(443400, 4437740), (443400, 4437400)]), # 禧乐汇
     
     LineString([(443525, 4437315), (443760, 4437315)]), # 回龙观镇医院
     LineString([(443760, 4437400), (443760, 4437315)]), # 回龙观镇医院
     
     LineString([(443535, 4437145), (443995, 4437145)]), # 爱蕾幼儿园
     
     LineString([(443050, 4437145), (443535, 4437145)]), # 幸福童心幼儿园
     
     LineString([(443555, 4436805), (444045, 4436805)]), # 龙泽园街道办事处
     
     LineString([(443090, 4436805), (443555, 4436805)]), # 回龙观法治文化公园
     
     LineString([(443568, 4436616), (444060, 4436616)]), # 腾讯众创空间
     
     LineString([(443830, 4436440), (444060, 4436440)]), # 昌平二中
     LineString([(443830, 4436440), (443830, 4436200)]), # 昌平二中
     
     LineString([(443400, 4436200), (443400, 4435950)]), # 龙腾苑四区改造
     LineString([(443400, 4435950), (443600, 4435950)]), # 龙腾苑四区改造
     
     LineString([(442720, 4435950), (443180, 4435950)]), # BHG华联购物中心
     
     LineString([(442320, 4435930), (442405, 4435930)]), # 幸福童年幼儿园
     LineString([(442405, 4436170), (442405, 4435930)]), # 幸福童年幼儿园
     
     LineString([(442255, 4436400), (442615, 4436400)]), # 港龙商业中心
     LineString([(442615, 4436400), (442615, 4436190)]), # 港龙商业中心
     LineString([(442615, 4436400), (442685, 4436415)]), # 北京安达医院
     
     LineString([(442685, 4436415), (442835, 4436415)]), # 昌平第二实验小学
     LineString([(442835, 4436415), (442835, 4436200)]), # 昌平第二实验小学
     
     LineString([(442664, 4436544), (443116, 4436616)]), # 上品折扣
     
     LineString([(442820, 4437015), (442859.5, 4436768.5)]), # 文华市场
     LineString([(442629, 4436732), (443090, 4436805)]), # 北店时代广场
     
     LineString([(442488, 4437140), (442545, 4437140)]), # 云趣园东南角
     LineString([(442488, 4437140), (442488, 4436952)]), # 云趣园东南角
     
    ]


# In[139]:


gdf = GeoDataFrame({'id': list(range(len(z))), 'geometry': z})
# define a bounding box in Beijing
north, south, east, west = 40.0887, 40.0698, 116.3444, 116.31852


# create network from that bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
G_projected = ox.project_graph(G)

gdf = gdf.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf.explore()


# In[140]:


gdf = aggregate(z)
gdf[gdf.geom_type!='Polygon'].plot()


# In[141]:


print(len(gdf[gdf.geom_type=='LineString']))
print(len(gdf[gdf.geom_type=='Point']))
print(len(gdf[gdf.geom_type=='Polygon']))


# In[142]:


gdf_roads = gdf[gdf.geom_type!='Polygon']
gdf_roads = gdf_roads.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf_roads.explore()


# In[143]:


gdf = gdf.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf.explore()


# In[144]:


residential_ids = [187, 194, 195, 197, 189, 209, 210, 217, 198, 190, 212, 220, 200, 192, 213, 203, 183, 216, 185, 186]
green_l_ids = [207]
gdf.loc[residential_ids, 'type'] = 4
gdf.loc[green_l_ids, 'type'] = 7


# In[145]:


minx, miny, _, _ = gdf.unary_union.bounds
print(minx, miny)
gdf['geometry'] = gdf['geometry'].translate(-minx, -miny)


# In[146]:


print(gdf.area.sum())
print(gdf.unary_union.bounds)


# In[147]:


gdf[gdf.geom_type=='Polygon'].plot('type')


# In[148]:


d = dict()
d['gdf'] = gdf
with open('../cfg/test_data/real/hlg/init_plan_hlg.pickle', 'wb') as f:
    pickle.dump(d, f)


# ### dhm

# #### road + land_use

# In[3]:


z = [LineString([(448167, 4409142), (450034, 4409142)]), # 南四环中路
     LineString([(448167, 4411910), (448167, 4409142)]), # 南苑路
     LineString([(448167, 4411910), (450034, 4411910)]), # 南三环中路
     LineString([(450034, 4411910), (450034, 4409142)]), # 榴乡路
     
     LineString([(449391, 4411910), (449391, 4410264)]), # 光彩路
     LineString([(449391, 4410264), (449391, 4409142)]), # 光彩路
     
     LineString([(448167, 4410664), (450034, 4410664)]), # 时村大街
     
     LineString([(448167, 4411556), (448326, 4411556)]), # 丰海北街
     LineString([(448326, 4411556), (449391, 4411556)]), # 丰海北街延长线及光彩北路
     
     LineString([(448167, 4411074), (448515, 4411074)]), # 丰海南街
     LineString([(448515, 4411074), (449391, 4411074)]), # 丰海南街延长线
     
     LineString([(448574, 4410927), (449391, 4410927)]), # 时村内部道路
     LineString([(449391, 4410927), (449712, 4410927)]), # 时村内部道路延长道路
     LineString([(449712, 4410927), (449785, 4410836)]), # 时村内部道路延长道路
     LineString([(449785, 4410836), (450034, 4410836)]), # 时村内部道路延长道路
     
     LineString([(449535, 4410927), (449535, 4410664)]), # 石榴园北里内部道路
     LineString([(449785, 4410836), (449785, 4410664)]), # 石榴庄菜市场西侧道路
     
     LineString([(449624, 4411085), (449624, 4410927)]), # 东铁匠营二中
     LineString([(449712, 4411085), (449712, 4410927)]), # 东铁匠营二中
     
     LineString([(449712, 4410957), (449830, 4410957)]), # 丰台区时光小学
     LineString([(449830, 4410957), (449830, 4410836)]), # 丰台区时光小学
     
     LineString([(449535, 4410664), (449535, 4410445)]), # 石榴园南里西侧道路
     LineString([(449391, 4410445), (450034, 4410445)]), # 石榴园南里南侧道路
     
     LineString([(449785, 4410664), (449785, 4410445)]), # 东铁匠营第一小学分校东侧道路
     
     LineString([(449680, 4410560), (449680, 4410445)]), # 东铁匠营第一小学分校
     LineString([(449680, 4410560), (449785, 4410560)]), # 东铁匠营第一小学分校
     
     LineString([(449009, 4411910), (449009, 4410664)]), # 光彩体育场西侧道路
     
     LineString([(448379, 4411426), (449009, 4411426)]), # 内部道路
     
     LineString([(448167, 4411910), (448326, 4411556)]), # 大红门路
     LineString([(448326, 4411556), (448379, 4411426)]), # 大红门路
     LineString([(448379, 4411426), (448515, 4411074)]), # 大红门路
     LineString([(448515, 4411074), (448574, 4410927)]), # 大红门路
     LineString([(448574, 4410927), (448740, 4410664)]), # 大红门路
     LineString([(448740, 4410664), (448740, 4410015)]), # 大红门路
     LineString([(448740, 4410015), (448599, 4409651)]), # 大红门路
     LineString([(448599, 4409651), (448167, 4409651)]), # 大红门路
     
     LineString([(448599, 4409651), (448599, 4409142)]), # 大红门东前街
     
     LineString([(449200, 4410664), (449200, 4410264)]), # 南顶小区东侧道路
     
     LineString([(449100, 4410264), (449100, 4409950)]), # 彩虹城四区北侧
     LineString([(449100, 4409950), (449391, 4409950)]), # 彩虹城四区北侧
     
     LineString([(448400, 4410927), (448574, 4410927)]), # 福海公园
     LineString([(448400, 4410927), (448400, 4410664)]), # 福海公园
     
     LineString([(448167, 4410015), (448740, 4410015)]), # 红门鞋城
     
     LineString([(448167, 4410664), (448740, 4410015)]), # 凉水河
     LineString([(448740, 4410015), (449391, 4409373)]), # 凉水河
     LineString([(449391, 4409373), (450034, 4409373)]), # 凉水河
     
     
     LineString([(449391, 4411638), (449862, 4411237)]), # 沙子口路
     LineString([(449862, 4411237), (450034, 4411085)]), # 沙子口路
     
     LineString([(449806, 4411910), (449862, 4411638)]), # 同仁东路
     LineString([(449862, 4411638), (449862, 4411237)]), # 同仁东路
     
     LineString([(449862, 4411538), (450034, 4411538)]), # 东铁匠营第二小学
     
     LineString([(449862, 4411638), (450034, 4411638)]), # 顺三条
     
     LineString([(449391, 4411085), (450034, 4411085)]), # 贾家花园南侧道路
     
     LineString([(448740, 4410264), (450034, 4410264)]), # 南顶路
     
     LineString([(448740, 4410464), (448920, 4410464)]), # 佟麟阁中学
     LineString([(448920, 4410464), (448920, 4410264)]), # 佟麟阁中学
     
     LineString([(448920, 4410464), (449100, 4410464)]), # 城市时尚家园
     LineString([(449100, 4410464), (449100, 4410264)]), # 城市时尚家园
     
     LineString([(449100, 4410664), (449100, 4410464)]), # 城市时尚家园东侧道路
     
     LineString([(449391, 4409632), (450034, 4409632)]), # 金桥西街
     
     LineString([(449391, 4410084), (450034, 4410084)]), # 京深海鲜市场
     
     LineString([(449391, 4409837), (449540, 4409837)]), # 安榴南街
     LineString([(449540, 4409837), (449630, 4409900)]), # 安榴南街
     LineString([(449630, 4409900), (450034, 4409900)]), # 安榴南街
     
     LineString([(449630, 4410084), (449630, 4409900)]), # 彩虹城一区东侧道路
     
     LineString([(449702, 4409900), (449702, 4409632)]), # 丰彩南路
    ]


# In[4]:


gdf = GeoDataFrame({'id': list(range(len(z))), 'geometry': z})
# define a bounding box in Beijing
north, south, east, west = 39.8558, 39.8305, 116.4173, 116.3939


# create network from that bounding box
G = ox.graph_from_bbox(north, south, east, west, network_type="drive")
G_projected = ox.project_graph(G)

gdf = gdf.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf.explore()


# In[5]:


gdf = aggregate(z)
gdf[gdf.geom_type!='Polygon'].plot()


# In[6]:


print(len(gdf[gdf.geom_type=='LineString']))
print(len(gdf[gdf.geom_type=='Point']))
print(len(gdf[gdf.geom_type=='Polygon']))


# In[7]:


gdf_roads = gdf[gdf.geom_type!='Polygon']
gdf_roads = gdf_roads.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf_roads.explore()


# In[8]:


gdf = gdf.set_crs(ox.graph_to_gdfs(G_projected)[1].crs)
gdf.explore()




# In[9]:

feasible_ids = [223, 232, 224, 261, 233, 255, 257, 262, 245, 264, 265, 235, 267, 268, 250, 248, 227, 249, 266, 238, 239]
residential_ids = [228, 229, 231, 263, 242, 246, 244, 247, 260, 234, 226, 254, 256, 252, 258, 220, 259, 236, 237, 251, 253, 240, 241]
school_ids = [243]
office_ids = [230, 221]
green_l_ids = [225, 222]
gdf.loc[residential_ids, 'type'] = 4
gdf.loc[green_l_ids, 'type'] = 7
gdf.loc[school_ids, 'type'] = 9
gdf.loc[office_ids, 'type'] = 6


# In[10]:


minx, miny, _, _ = gdf.unary_union.bounds
print(minx, miny)
gdf['geometry'] = gdf['geometry'].translate(-minx, -miny)


# In[11]:


print(gdf.area.sum())
print(gdf.unary_union.bounds)


# In[12]:


gdf[gdf.geom_type=='Polygon'].plot('type')


# In[13]:


d = dict()
d['gdf'] = gdf
with open('../../cfg/test_data/real/dhm/init_plan_dhm_v2.pickle', 'wb') as f:
    pickle.dump(d, f)


