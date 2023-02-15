from typing import Union, Text, Tuple

import numpy as np
from geopandas import GeoSeries, GeoDataFrame
from shapely.geometry import Polygon, MultiLineString, LineString, Point, MultiPoint
from shapely.ops import snap, substring, nearest_points


def get_boundary_edges(polygon: Polygon, return_type: Text) -> Union[MultiLineString, GeoSeries]:
    """
    Get the boundary edges of a polygon.

    Args:
        polygon: Polygon.
        return_type: 'MultiLineString' or 'GeoSeries'.

    Returns:
        MultiLineString.
    """
    intersection_sequence = polygon.exterior.coords
    if return_type == 'MultiLineString':
        boundary_edges = MultiLineString(list(map(LineString, zip(intersection_sequence[:-1], intersection_sequence[1:]))))
    elif return_type == 'GeoSeries':
        boundary_edges = GeoSeries(list(map(LineString, zip(intersection_sequence[:-1], intersection_sequence[1:]))))
    else:
        raise ValueError('return_type must be "MultiLineString" or "GeoSeries"')
    return boundary_edges


def get_angles(vec_1: np.ndarray, vec_2: np.ndarray) -> np.ndarray:
    """
    Compute the angle, in degrees, between two vectors

    Args:
        vec_1: np.ndarray of shape (2,).
        vec_2: np.ndarray of shape (2,).

    Returns:
        angle between vec_1 and vec_2 in degrees: np.ndarray of shape ().
    """

    dot = np.dot(vec_1, vec_2)
    det = np.cross(vec_1, vec_2)
    angle_in_rad = np.arctan2(det, dot)
    return np.degrees(angle_in_rad)


def simplify_by_angle(poly_in: Polygon, deg_tol: float = 1) -> Polygon:
    """
    Try to remove persistent coordinate points that remain after
    simplify, convex hull, or something, etc. with some trig instead

    Args:
        poly_in: Polygon.
        deg_tol: tolerance for angle between successive vectors.

    Returns:
        Polygon with persistent coordinate points removes.
    """
    ext_poly_coords = poly_in.exterior.coords[:]
    vector_rep = np.diff(ext_poly_coords, axis=0)
    num_vectors = len(vector_rep)
    angles_list = []
    for i in range(0, num_vectors):
        angles_list.append(np.abs(get_angles(vector_rep[i], vector_rep[(i + 1) % num_vectors])))

    #   get mask satisfying tolerance
    thresh_vals_by_deg = np.where(np.array(angles_list) > deg_tol)

    new_idx = list(thresh_vals_by_deg[0] + 1)
    new_vertices = [ext_poly_coords[idx] for idx in new_idx]

    return Polygon(new_vertices)


def simplify_by_distance(poly_in: Polygon, distance_tol: float = 1) -> Polygon:
    """
    Simplify polygon by distance tolerance. Try to remove points that are too close to each other.

    Args:
        poly_in: Polygon.
        distance_tol: distance tolerance.

    Returns:
        Simplified polygon.
    """
    ext_poly_coords = poly_in.exterior.coords[:]
    vector_rep = np.diff(ext_poly_coords, axis=0)
    vector_lengths = np.linalg.norm(vector_rep, axis=1)
    thresh_vals_by_length = np.where(vector_lengths >= distance_tol)

    new_idx = list(thresh_vals_by_length[0] + 1)
    new_vertices = [ext_poly_coords[idx] for idx in new_idx]

    return Polygon(new_vertices)


def check_horizontal_vertical(edge: LineString, epsilon: float) -> bool:
    """Check if edge is horizontal or vertical."""
    bounds = edge.bounds
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    min_range = min([x_range, y_range])
    if min_range < epsilon:
        return True
    else:
        return False


def check_interval_angle(polygon: Polygon, p_c: Point, p_1: Point, p_2: Point,
                         epsilon: float, deg_thres: float = 150) -> Text:
    """Compute the angle (convex or concave) of p_1-p_c-p_2 with respect to polygon"""
    p_t = LineString([p_1, p_2]).centroid
    test_line = substring(LineString([p_c, p_t]), start_dist=0, end_dist=epsilon)
    if polygon.contains(test_line):
        angle = np.abs(get_angles(
            (np.array(p_1.coords) - np.array(p_c.coords)).squeeze(),
            (np.array(p_2.coords) - np.array(p_c.coords)).squeeze()))
        if angle > deg_thres - epsilon:
            return 'concave'
        else:
            return 'convex'
    else:
        return 'concave'


def get_the_other_edge(polygon_boundary: GeoSeries, p_c: Point, p_1: Point, epsilon: float) -> LineString:
    """Get the edge from p_c other than p_c-p_1 from the polygon boundary.

    Args:
        polygon_boundary: the boundary edges of the polygon.
        p_c: the point to get the other edge from.
        p_1: the end point of the known edge.
        epsilon: the tolerance for the distance between the point and the edge.

    Returns:
        The other edge from p_c.
    """
    edge = polygon_boundary[(polygon_boundary.intersects(p_c)) & (polygon_boundary.distance(p_1) >= epsilon)]
    if len(edge) != 1:
        error_msg = 'Polygon boundary: {}'.format(polygon_boundary)
        error_msg += '\np_c: {}'.format(p_c)
        error_msg += '\np_1: {}'.format(p_1)
        error_msg += '\nNumber of edges: {}'.format(len(edge))
        error_msg += '\nEdge: {}'.format(edge)
        raise ValueError(error_msg + '\nThe number of the other edge from p_c is not 1.')
    edge = edge.iloc[0]
    return edge


def rectify_slice_edge_length(search_max_length: float, min_edge_length: float, max_edge_length: float,
                              search_max_area: float, search_min_area:float, cell_edge_length: float,
                              edge: LineString) -> Tuple[float, float, float]:
    """
    Rectify the length for slicing edge.

    Args:
        search_max_length: maximum search length of the edge.
        min_edge_length: minimum length of the edge.
        max_edge_length: maximum length of the edge.
        search_max_area: maximum area of the polygon.
        search_min_area: minimum area of the polygon.
        cell_edge_length: length of the cell edge.
        edge: the existing edge.

    Returns:
        rectified search_max_length.
        rectified max_edge_length.
        rectified min_edge_length.
    """
    common_min_edge_length = search_max_length - max_edge_length
    rectified_min_edge_length = max(min_edge_length, search_min_area/(edge.length*cell_edge_length))
    rectified_max_edge_length = max(rectified_min_edge_length,
                                    min(max_edge_length, search_max_area/(edge.length*cell_edge_length)))
    rectified_search_max_length = rectified_max_edge_length + common_min_edge_length

    return rectified_search_max_length, rectified_min_edge_length, rectified_max_edge_length


def slice_edge(edge: LineString,
               point: Point,
               intersections: GeoDataFrame,
               epsilon: float,
               cell_edge_length: float,
               min_edge_length: float,
               max_edge_length: float,
               search_max_length: float) -> Tuple[LineString, bool]:
    """Slice edge from an end point."""
    if edge.length*cell_edge_length <= search_max_length:
        return edge, True
    candidate_intersections = intersections[intersections.distance(edge) < epsilon]
    distances = candidate_intersections.distance(point)
    feasible_intersections = candidate_intersections[(distances*cell_edge_length >= min_edge_length) &
                                                     (distances*cell_edge_length <= max_edge_length)]
    if len(feasible_intersections) > 0:
        point_2 = feasible_intersections.iloc[feasible_intersections.distance(point).argmax()]['geometry']
        sliced_edge = LineString([point, point_2])
    else:
        sliced_edge = substring(edge, start_dist=0, end_dist=max_edge_length/cell_edge_length)
    return sliced_edge, False


def slice_from_u_shape(edge_c: LineString, edge_1: LineString, edge_2: LineString,
                       epsilon: float, thres_deg: float = 150) -> Polygon:
    """Slice a polygon from three U-shape edges.

    Args:
        edge_c: the center edge.
        edge_1: the first edge.
        edge_2: the second edge.
        epsilon: the tolerance for rotation.
        thres_deg: the threshold for the angle.

    Returns:
        The sliced polygon.
    """
    edge_c_hv = check_horizontal_vertical(edge_c, epsilon)
    edge_1_hv = check_horizontal_vertical(edge_1, epsilon)
    edge_2_hv = check_horizontal_vertical(edge_2, epsilon)
    if not edge_c_hv and not edge_1_hv and not edge_2_hv:
        polygon = MultiLineString([edge_c, edge_1, edge_2]).minimum_rotated_rectangle
    elif (edge_c_hv and edge_1_hv and not edge_2_hv) or (edge_c_hv and not edge_1_hv and edge_2_hv):
        p_c_1 = edge_c.intersection(edge_1)
        p_c_2 = edge_c.intersection(edge_2)
        p_1 = MultiPoint(edge_1.coords).difference(p_c_1)
        p_2 = MultiPoint(edge_2.coords).difference(p_c_2)
        if edge_1_hv:
            angle = np.abs(get_angles(
                (np.array(p_2.coords) - np.array(p_c_2.coords)).squeeze(),
                (np.array(p_c_1.coords) - np.array(p_c_2.coords)).squeeze()))
            if angle > thres_deg:
                polygon = MultiLineString([edge_c, edge_1, edge_2]).envelope
            else:
                foot, _ = nearest_points(edge_1, p_2)
                scale_count = 0
                while epsilon < foot.distance(p_1) and scale_count < 3:
                    p_2 = Point(p_2.x + p_2.x - p_c_2.x, p_2.y + p_2.y - p_c_2.y)
                    foot, _ = nearest_points(edge_1, p_2)
                    scale_count += 1
                polygon = MultiLineString([edge_c, edge_1, LineString([p_c_2, p_2])]).envelope
        else:
            angle = np.abs(get_angles(
                (np.array(p_1.coords) - np.array(p_c_1.coords)).squeeze(),
                (np.array(p_c_2.coords) - np.array(p_c_1.coords)).squeeze()))
            if angle > thres_deg:
                polygon = MultiLineString([edge_c, edge_1, edge_2]).envelope
            else:
                foot, _ = nearest_points(edge_2, p_1)
                scale_count = 0
                while epsilon < foot.distance(p_2) and scale_count < 3:
                    p_1 = Point(p_1.x + p_1.x - p_c_1.x, p_1.y + p_1.y - p_c_1.y)
                    foot, _ = nearest_points(edge_2, p_1)
                    scale_count += 1
                polygon = MultiLineString([edge_c, edge_2, LineString([p_c_1, p_1])]).envelope
    else:
        polygon = MultiLineString([edge_c, edge_1, edge_2]).envelope
    return polygon


def slice_from_angle(edge_1: LineString,
                     edge_2: LineString,
                     p_c: Point,
                     p_1: Point,
                     p_2: Point,
                     epsilon: float) -> Polygon:
    """Slice a polygon from an angle.

    Args:
        edge_1: the first angle edge LineString.
        edge_2: the second angle edge LineString.
        p_c: the center point of the angle.
        p_1: the end point of the first angle edge.
        p_2: the end point of the second angle edge.
        epsilon: tolerance for rotation.

    Returns:
        The sliced Polygon.
    """
    edge_1_hv = check_horizontal_vertical(edge_1, epsilon)
    edge_2_hv = check_horizontal_vertical(edge_2, epsilon)
    if edge_1_hv or edge_2_hv:
        polygon = MultiPoint([p_c, p_1, p_2]).envelope
    else:
        p_t = Point(p_2.x + p_1.x - p_c.x, p_2.y + p_1.y - p_c.y)
        polygon = MultiPoint([p_c, p_1, p_t, p_2]).minimum_rotated_rectangle
    return polygon


def slice_from_angle_rect_tri(edge_1: LineString,
                              edge_2: LineString,
                              p_c: Point,
                              p_1: Point,
                              p_2: Point,
                              epsilon: float,
                              thres_dis: float,
                              thres_deg: float = 60) -> Polygon:
    """Slice a polygon from an angle. The sliced polygon can be a triangle or a rectangle.

    Args:
        edge_1: the first angle edge LineString.
        edge_2: the second angle edge LineString.
        p_c: the center point of the angle.
        p_1: the end point of the first angle edge.
        p_2: the end point of the second angle edge.
        epsilon: tolerance for rotation.
        thres_dis: the threshold distance for slicing triangle instead of rectangle.

    Returns:
        The sliced Polygon.
    """
    edge_1_hv = check_horizontal_vertical(edge_1, epsilon)
    edge_2_hv = check_horizontal_vertical(edge_2, epsilon)
    if edge_1_hv and edge_2_hv:
        polygon = MultiPoint([p_c, p_1, p_2]).envelope
    elif edge_1_hv or edge_2_hv:
        angle = np.abs(get_angles(
            (np.array(p_1.coords) - np.array(p_c.coords)).squeeze(),
            (np.array(p_2.coords) - np.array(p_c.coords)).squeeze()))
        if angle > thres_deg:
            polygon = MultiPoint([p_c, p_1, p_2]).envelope
        else:
            if edge_1_hv:
                foot, _ = nearest_points(edge_1, p_2)
                scale_count = 0
                while epsilon < foot.distance(p_1) < thres_dis and scale_count < 3:
                    p_2 = Point(p_2.x + p_2.x - p_c.x, p_2.y + p_2.y - p_c.y)
                    foot, _ = nearest_points(edge_1, p_2)
                    scale_count += 1
            elif edge_2_hv:
                foot, _ = nearest_points(edge_2, p_1)
                scale_count = 0
                while epsilon < foot.distance(p_2) < thres_dis and scale_count < 3:
                    p_1 = Point(p_1.x + p_1.x - p_c.x, p_1.y + p_1.y - p_c.y)
                    foot, _ = nearest_points(edge_2, p_1)
                    scale_count += 1
            polygon = MultiPoint([p_c, p_1, p_2]).envelope
    else:
        p_t = Point(p_2.x + p_1.x - p_c.x, p_2.y + p_1.y - p_c.y)
        polygon = MultiPoint([p_c, p_1, p_t, p_2]).minimum_rotated_rectangle
    return polygon


def slice_from_part_edge(polygon: Polygon,
                         edge: LineString,
                         epsilon: float,
                         cell_edge_length: float,
                         max_edge_length: float,
                         thres_dis: float) -> Polygon:
    """Slice a polygon from a part edge.

    Args:
        polygon: the polygon to be sliced.
        edge: the part edge LineString.
        epsilon: tolerance for test small polygon.
        cell_edge_length: the length of the cell edge.
        max_edge_length: the maximum length of the sliced polygon.
        thres_dis: the left threshold distance for buffer.

    Returns:
        The sliced Polygon.
    """
    temp_polygon = snap(polygon, edge, epsilon)
    left_b = edge.buffer(epsilon, single_sided=True)
    right_b = edge.buffer(-epsilon, single_sided=True)
    left_intersection_area = temp_polygon.intersection(left_b).area
    right_intersection_area = temp_polygon.intersection(right_b).area
    if left_intersection_area > right_intersection_area:
        probe_sliced_polygon = edge.buffer((max_edge_length + thres_dis) / cell_edge_length, single_sided=True)
        if temp_polygon.difference(probe_sliced_polygon).geom_type == 'Polygon':
            sliced_polygon = edge.buffer(max_edge_length/cell_edge_length, single_sided=True)
        else:
            sliced_polygon = probe_sliced_polygon
    elif left_intersection_area < right_intersection_area:
        probe_sliced_polygon = edge.buffer(-(max_edge_length + thres_dis) / cell_edge_length, single_sided=True)
        if temp_polygon.difference(probe_sliced_polygon).geom_type == 'Polygon':
            sliced_polygon = edge.buffer(-max_edge_length/cell_edge_length, single_sided=True)
        else:
            sliced_polygon = probe_sliced_polygon
    else:
        error_msg = 'temp polygon: {}'.format(temp_polygon)
        error_msg += '\nedge: {}'.format(edge)
        raise ValueError(error_msg + '\nLeft and right side both not within polygon.')
    return sliced_polygon


def slice_from_l_shape(polygon: Polygon,
                       polygon_boundary: GeoSeries,
                       edge_1: LineString,
                       edge_2: LineString,
                       p_c: Point,
                       p_1: Point,
                       p_2: Point,
                       all_intersections: GeoDataFrame,
                       epsilon: float,
                       cell_edge_length: float,
                       min_edge_length: float,
                       max_edge_length: float,
                       search_max_length: float,
                       search_max_area: float,
                       search_min_area: float) -> Polygon:
    """Slice a polygon from L-shape.

    Args:
        polygon: the polygon to be sliced.
        polygon_boundary: the polygon boundary.
        edge_1: the first angle edge LineString of L-shape.
        edge_2: the second angle edge LineString of L-shape.
        p_c: the center point of the L-shape.
        p_1: the end point of the first angle edge.
        p_2: the end point of the second angle edge.
        all_intersections: the intersections of the polygon boundary.
        epsilon: tolerance.
        cell_edge_length: the length of the cell edge.
        min_edge_length: the minimum length of the sliced polygon.
        max_edge_length: the maximum length of the sliced polygon.
        search_max_length: the maximum length of the search range.
        search_max_area: the maximum area of the search range.
        search_min_area: the minimum area of the search range.

    Returns:
        The sliced Polygon.
    """
    edge_3 = get_the_other_edge(polygon_boundary, p_1, p_c, epsilon)
    p_3 = MultiPoint(edge_3.coords).difference(p_1)
    if check_interval_angle(polygon, p_1, p_c, p_3, epsilon) == 'concave':
        land_use_polygon = slice_from_angle(edge_1, edge_2, p_c, p_1, p_2, epsilon)
        area = land_use_polygon.area*cell_edge_length**2
        angle = np.abs(get_angles(
            (np.array(p_1.coords) - np.array(p_c.coords)).squeeze(),
            (np.array(p_2.coords) - np.array(p_c.coords)).squeeze()))
        if area < search_min_area and np.abs(angle - 90) < epsilon:
            thres_dis = search_max_length - max_edge_length
            land_use_polygon = slice_from_part_edge(polygon, edge_2, epsilon,
                                                    cell_edge_length, max_edge_length, thres_dis)
    else:
        rectified_search_max_length, rectified_min_edge_length, rectified_max_edge_length = \
            rectify_slice_edge_length(search_max_length, min_edge_length, max_edge_length,
                                      search_max_area, search_min_area, cell_edge_length, edge_1)
        slice_edge_3, _ = slice_edge(LineString([p_1, p_3]), p_1, all_intersections, epsilon, cell_edge_length,
                                     rectified_min_edge_length, rectified_max_edge_length,
                                     rectified_search_max_length)
        land_use_polygon = slice_from_u_shape(edge_1, edge_2, slice_edge_3, epsilon)
    return land_use_polygon


def slice_from_half_edge(polygon: Polygon,
                         polygon_boundary: GeoSeries,
                         half_edge: LineString,
                         p_c: Point,
                         p_1: Point,
                         intersections: GeoDataFrame,
                         epsilon: float,
                         cell_edge_length: float,
                         min_edge_length: float,
                         max_edge_length: float,
                         search_max_length: float,
                         search_max_area: float,
                         search_min_area: float) -> Polygon:
    """Slice a polygon from half edge.

    Args:
        polygon: the polygon to be sliced.
        polygon_boundary: the polygon boundary.
        half_edge: the half edge LineString.
        p_c: the center point of the half edge.
        p_1: the end point of the half edge.
        intersections: the intersections of the polygon boundary.
        epsilon: tolerance.
        cell_edge_length: the length of the cell edge.
        min_edge_length: the minimum length of the sliced polygon.
        max_edge_length: the maximum length of the sliced polygon.
        search_max_length: the maximum length of the search range.
        search_max_area: the maximum area of the search range.
        search_min_area: the minimum area of the search range.

    Returns:
        The sliced Polygon.
    """
    edge_2 = get_the_other_edge(polygon_boundary, p_c, p_1, epsilon)
    p_2 = MultiPoint(edge_2.coords).difference(p_c)
    if check_interval_angle(polygon, p_c, p_1, p_2, epsilon) == 'concave':
        max_buffer_length = max(max_edge_length, search_max_area/(half_edge.length*cell_edge_length))
        thres_dis = search_max_length - max_edge_length
        sliced_polygon = slice_from_part_edge(polygon, half_edge, epsilon,
                                              cell_edge_length, max_buffer_length, thres_dis)
    else:
        rectified_search_max_length, rectified_min_edge_length, rectified_max_edge_length = \
            rectify_slice_edge_length(search_max_length, min_edge_length, max_edge_length,
                                      search_max_area, search_min_area, cell_edge_length, half_edge)
        slice_edge_2, whole = slice_edge(LineString([p_c, p_2]), p_c, intersections, epsilon, cell_edge_length,
                                         rectified_min_edge_length, rectified_max_edge_length,
                                         rectified_search_max_length)
        if not whole:
            common_min_edge_length = search_max_length - max_edge_length
            thres_distance = common_min_edge_length / cell_edge_length
            sliced_polygon = slice_from_angle_rect_tri(
                half_edge, slice_edge_2, p_c, p_1, Point(slice_edge_2.coords[1]), epsilon, thres_distance)
        else:
            sliced_polygon = slice_from_l_shape(polygon, polygon_boundary, slice_edge_2, half_edge, p_c, p_2, p_1,
                                                intersections, epsilon, cell_edge_length,
                                                min_edge_length, max_edge_length, search_max_length,
                                                search_max_area, search_min_area)
    return sliced_polygon


def slice_polygon_from_half_or_part_edge(polygon: Polygon,
                                         polygon_boundary: GeoSeries,
                                         edge: LineString,
                                         intersection: Point,
                                         corner: Point,
                                         all_intersections: GeoDataFrame,
                                         epsilon: float,
                                         cell_edge_length: float,
                                         min_edge_length: float,
                                         max_edge_length: float,
                                         search_max_length: float,
                                         search_max_area: float,
                                         search_min_area: float) -> Polygon:
    """
    Slice a polygon from a half or part edge.

    Args:
        polygon: Polygon to be sliced.
        polygon_boundary: GeoSeries of polygon boundary edges.
        edge: LineString to slice from.
        intersection: intersection point.
        corner: corner point.
        all_intersections: GeoDataFrame of all intersections.
        epsilon: tolerance for distance between intersection and edge.
        cell_edge_length: length of cell edge.
        min_edge_length: minimum length of edge.
        max_edge_length: maximum length of edge.
        search_max_length: maximum length of search.
        search_max_area: maximum area of search.
        search_min_area: minimum area of search.

    Returns:
        The sliced Polygon.
    """
    sliced_edge, whole = slice_edge(edge, intersection, all_intersections, epsilon,
                                    cell_edge_length, min_edge_length, max_edge_length, search_max_length)
    if whole:
        polygon = slice_from_half_edge(polygon, polygon_boundary, sliced_edge, corner, intersection, all_intersections,
                                       epsilon, cell_edge_length, min_edge_length, max_edge_length, search_max_length,
                                       search_max_area, search_min_area)
    else:
        max_buffer_length = max(max_edge_length, search_max_area/(sliced_edge.length*cell_edge_length))
        thres_dis = search_max_length - max_edge_length
        polygon = slice_from_part_edge(polygon, sliced_edge, epsilon, cell_edge_length, max_buffer_length, thres_dis)
    return polygon


def slice_from_whole_edge(polygon: Polygon,
                          polygon_boundary: GeoSeries,
                          edge: LineString,
                          all_intersections: GeoDataFrame,
                          epsilon: float,
                          cell_edge_length: float,
                          min_edge_length: float,
                          max_edge_length: float,
                          search_max_length: float,
                          search_max_area: float,
                          search_min_area: float) -> Polygon:
    """Slice a polygon from a whole edge.

    Args:
        polygon: the polygon to be sliced.
        polygon_boundary: the polygon boundary.
        edge: the whole edge LineString.
        all_intersections: the intersections of the polygon boundary.
        epsilon: tolerance.
        cell_edge_length: the length of the cell edge.
        min_edge_length: the minimum length of the sliced polygon.
        max_edge_length: the maximum length of the sliced polygon.
        search_max_length: the maximum length of the search range.
        search_max_area: the maximum area of the search range.
        search_min_area: the minimum area of the search range.

    Returns:
        The sliced Polygon.
    """
    p_c_1 = Point(edge.coords[0])
    p_c_2 = Point(edge.coords[1])
    edge_1 = get_the_other_edge(polygon_boundary, p_c_1, p_c_2, epsilon)
    p_1 = MultiPoint(edge_1.coords).difference(p_c_1)
    edge_2 = get_the_other_edge(polygon_boundary, p_c_2, p_c_1, epsilon)
    p_2 = MultiPoint(edge_2.coords).difference(p_c_2)
    angle_1 = check_interval_angle(polygon, p_c_1, p_1, p_c_2, epsilon)
    angle_2 = check_interval_angle(polygon, p_c_2, p_2, p_c_1, epsilon)
    if angle_1 == 'concave' and angle_2 == 'concave':
        max_buffer_length = max(max_edge_length, search_max_area/(edge.length*cell_edge_length))
        thres_dis = search_max_length - max_edge_length
        sliced_polygon = slice_from_part_edge(polygon, edge, epsilon, cell_edge_length, max_buffer_length, thres_dis)
    else:
        rectified_search_max_length, rectified_min_edge_length, rectified_max_edge_length = \
            rectify_slice_edge_length(search_max_length, min_edge_length, max_edge_length,
                                      search_max_area, search_min_area, cell_edge_length, edge)
        if angle_1 == 'convex' and angle_2 == 'convex':
            slice_edge_1, _ = slice_edge(LineString([p_c_1, p_1]), p_c_1, all_intersections, epsilon, cell_edge_length,
                                         rectified_min_edge_length, rectified_max_edge_length,
                                         rectified_search_max_length)
            slice_edge_2, _ = slice_edge(LineString([p_c_2, p_2]), p_c_2, all_intersections, epsilon, cell_edge_length,
                                         rectified_min_edge_length, rectified_max_edge_length,
                                         rectified_search_max_length)
            sliced_polygon = slice_from_u_shape(edge, slice_edge_1, slice_edge_2, epsilon)
        elif angle_1 == 'convex':
            slice_edge_1, whole = slice_edge(LineString([p_c_1, p_1]), p_c_1, all_intersections, epsilon,
                                             cell_edge_length, rectified_min_edge_length, rectified_max_edge_length,
                                             rectified_search_max_length)
            if not whole:
                sliced_polygon = slice_from_angle(LineString([p_c_1, p_c_2]), slice_edge_1,
                                                  p_c_1, p_c_2, Point(slice_edge_1.coords[1]), epsilon)
            else:
                sliced_polygon = slice_from_l_shape(polygon, polygon_boundary, slice_edge_1, LineString([p_c_1, p_c_2]),
                                                    p_c_1, p_1, p_c_2, all_intersections, epsilon, cell_edge_length,
                                                    min_edge_length, max_edge_length, search_max_length,
                                                    search_max_area, search_min_area)
        else:
            slice_edge_2, whole = slice_edge(LineString([p_c_2, p_2]), p_c_2, all_intersections, epsilon,
                                             cell_edge_length, rectified_min_edge_length, rectified_max_edge_length,
                                             rectified_search_max_length)
            if not whole:
                sliced_polygon = slice_from_angle(LineString([p_c_2, p_c_1]), slice_edge_2,
                                                  p_c_2, p_c_1, Point(slice_edge_2.coords[1]), epsilon)
            else:
                sliced_polygon = slice_from_l_shape(polygon, polygon_boundary, slice_edge_2, LineString([p_c_2, p_c_1]),
                                                    p_c_2, p_2, p_c_1, all_intersections, epsilon, cell_edge_length,
                                                    min_edge_length, max_edge_length, search_max_length,
                                                    search_max_area, search_min_area)
    return sliced_polygon


def slice_polygon_from_edge(polygon: Polygon,
                            polygon_boundary: GeoSeries,
                            edge: LineString,
                            intersection: Point,
                            all_intersections: GeoDataFrame,
                            distance: float,
                            epsilon: float,
                            cell_edge_length: float,
                            min_edge_length: float,
                            max_edge_length: float,
                            search_max_length: float,
                            search_max_area: float,
                            search_min_area: float) -> Polygon:

    """
    Slice a polygon from an edge.

    Args:
        polygon: Polygon to be sliced.
        polygon_boundary: GeoSeries of polygon boundary edges.
        edge: LineString to slice from.
        intersection: intersection point.
        all_intersections: GeoDataFrame of all intersections.
        distance: the distance between the intersection and the edge.
        epsilon: tolerance.
        cell_edge_length: length of cell edge.
        min_edge_length: minimum length of edge.
        max_edge_length: maximum length of edge.
        search_max_length: maximum length of search.
        search_max_area: maximum area of search.
        search_min_area: minimum area of search.

    Returns:
        The sliced Polygon.
    """
    if edge.length * cell_edge_length <= search_max_length:
        polygon = slice_from_whole_edge(polygon, polygon_boundary, edge, all_intersections, epsilon,
                                        cell_edge_length, min_edge_length, max_edge_length,
                                        search_max_length, search_max_area, search_min_area)
    else:
        polygon = snap(polygon, intersection, distance + epsilon)
        edge_coords = list(edge.coords)
        edge_1 = LineString([intersection, edge_coords[0]])
        edge_2 = LineString([intersection, edge_coords[1]])
        polygon_boundary = get_boundary_edges(polygon, 'GeoSeries')
        if edge_1.length >= edge_2.length:
            polygon = slice_polygon_from_half_or_part_edge(
                polygon, polygon_boundary, edge_1, intersection, Point(edge_coords[0]), all_intersections, epsilon,
                cell_edge_length, min_edge_length, max_edge_length, search_max_length, search_max_area, search_min_area)
        else:
            polygon = slice_polygon_from_half_or_part_edge(
                polygon, polygon_boundary, edge_2, intersection, Point(edge_coords[1]), all_intersections, epsilon,
                cell_edge_length, min_edge_length, max_edge_length, search_max_length, search_max_area, search_min_area)
    return polygon


def slice_polygon_from_corner(polygon: Polygon,
                              polygon_boundary: GeoSeries,
                              corner: Point,
                              edge_1: LineString,
                              p_1: Point,
                              edge_2: LineString,
                              p_2: Point,
                              all_intersections: GeoDataFrame,
                              epsilon: float,
                              cell_edge_length: float,
                              min_edge_length: float,
                              max_edge_length: float,
                              search_max_length: float,
                              search_max_area: float,
                              search_min_area: float) -> Polygon:
    """
    Slice a polygon from a corner.

    Args:
        polygon: Polygon to be sliced.
        polygon_boundary: GeoSeries of polygon boundary edges.
        corner: Point to slice from.
        edge_1: LineString of the first corner edge.
        p_1: Point of the first corner edge.
        edge_2: LineString of the second corner edge.
        p_2: Point of the second corner edge.
        all_intersections: GeoDataFrame of all intersections.
        epsilon: tolerance.
        cell_edge_length: length of cell edge.
        min_edge_length: minimum length of edge.
        max_edge_length: maximum length of edge.
        search_max_length: maximum length of search.
        search_max_area: maximum area of search.
        search_min_area: minimum area of search.

    Returns:
        The sliced Polygon.
    """
    if check_interval_angle(polygon, corner, p_1, p_2, epsilon) == 'convex':
        slice_edge_1, whole1 = slice_edge(edge_1, corner, all_intersections, epsilon,
                                          cell_edge_length, min_edge_length, max_edge_length, search_max_length)
        slice_edge_2, whole2 = slice_edge(edge_2, corner, all_intersections, epsilon,
                                          cell_edge_length, min_edge_length, max_edge_length, search_max_length)
        if not whole1 and not whole2:
            common_min_edge_length = search_max_length - max_edge_length
            thres_distance = common_min_edge_length / cell_edge_length
            land_use_polygon = slice_from_angle_rect_tri(
                slice_edge_1, slice_edge_2, corner, Point(slice_edge_1.coords[1]), Point(slice_edge_2.coords[1]),
                epsilon, thres_distance)
        elif whole1:
            land_use_polygon = slice_from_l_shape(polygon, polygon_boundary, slice_edge_1, slice_edge_2,
                                                  corner, p_1, Point(slice_edge_2.coords[1]), all_intersections,
                                                  epsilon, cell_edge_length,
                                                  min_edge_length, max_edge_length, search_max_length,
                                                  search_max_area, search_min_area)
        else:
            land_use_polygon = slice_from_l_shape(polygon, polygon_boundary, slice_edge_2, slice_edge_1,
                                                  corner, p_2, Point(slice_edge_1.coords[1]), all_intersections,
                                                  epsilon, cell_edge_length,
                                                  min_edge_length, max_edge_length, search_max_length,
                                                  search_max_area, search_min_area)
    else:
        if edge_1.length >= edge_2.length:
            land_use_polygon = slice_polygon_from_half_or_part_edge(
                polygon, polygon_boundary, edge_1, corner, p_1, all_intersections, epsilon, cell_edge_length,
                min_edge_length, max_edge_length, search_max_length, search_max_area, search_min_area)
        else:
            land_use_polygon = slice_polygon_from_half_or_part_edge(
                polygon, polygon_boundary, edge_2, corner, p_2, all_intersections, epsilon, cell_edge_length,
                min_edge_length, max_edge_length, search_max_length, search_max_area, search_min_area)
    return land_use_polygon


def get_intersection_polygon_with_maximum_area(polygon_s: Polygon, polygon: Polygon) -> Polygon:
    """
    Get the intersection polygon with maximum area.

    Args:
        polygon_s: Polygon to be sliced.
        polygon: Polygon to be sliced from.

    Returns:
        The intersection polygon with maximum area.
    """
    polygon_s = polygon.intersection(polygon_s)
    if polygon_s.geom_type == 'Polygon':
        return polygon_s
    else:
        if polygon_s.geom_type in ['MultiPolygon', 'GeometryCollection']:
            candidates = [var for var in list(polygon_s.geoms) if var.geom_type == 'Polygon']
            if len(candidates) > 0:
                max_area = max([var.area for var in candidates])
                polygon_s = [var for var in candidates if var.area == max_area][0]
                return polygon_s
    error_msg = 'polygon: {}'.format(polygon)
    error_msg += '\nsliced polygon: {}'.format(polygon_s)
    raise ValueError(error_msg + '\nSliced polygon is not a polygon.')
