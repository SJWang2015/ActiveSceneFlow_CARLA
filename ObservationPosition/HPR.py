# import open3d as o3d
# from open3d import *
import numpy as np

# from pyhull.delaunay import DelaunayTri
# from pyhull.voronoi import VoronoiTess
from pyhull.convex_hull import ConvexHull
from pyhull.simplex import Simplex
from scipy.fftpack import diff


def draw(vis, vis_list, use_mesh_frame=True, param_file='test.json', save_fov=False):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 1.4])
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # ctr = vis.get_view_control()
        # ctr.set_zoom(0.5)
        vis.get_render_option().light_on = True
        vis.get_render_option().point_size = 3.0
        vis.get_render_option().background_color = [1, 1, 1]
        # vis.get_render_option().background_color = [0, 0, 0]
    if use_mesh_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 1.4])
        vis.add_geometry(mesh_frame)

    for item in vis_list:
        vis.add_geometry(item)

    param = o3d.io.read_pinhole_camera_parameters(param_file)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.clear_geometries()

    if save_fov:
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('test.json', param)


def HPR(p, C, param):
    '''
    % p - NxD D dimensional point cloud.
    % C - 1xD D dimensional viewpoint.
    % param - parameter for the algorithm. Indirectly sets the radius.
    %
    % Output:
    % visiblePtInds - indices of p that are visible from C.
    %
    % This code is adapted form the matlab code written by Sagi Katz
    % For more information, see "Direct Visibility of Point Sets", Katz S., Tal
    % A. and Basri R., SIGGRAPH 2007, ACM Transactions on Graphics, Volume 26, Issue 3, August 2007.
    % This method is patent pending.
    '''
    numPts, dim = p.shape
    # Move C to the origin
    p = p - np.tile(C, (numPts, 1))
    # Calculate ||p||
    normp = np.linalg.norm(p, axis=1)
    # Sphere radius
    R = np.max(normp)
    R = np.tile(R * (10 ** param), (numPts, 1))
    # Spherical flipping
    P = p + 2 * np.tile(R - normp.reshape([numPts, 1]), (1, dim)) * p / np.tile(normp.reshape([numPts, 1]), (1, dim))
    # convex hull
    aug_P = np.vstack((P, np.zeros((1, dim))))
    d = ConvexHull(aug_P)
    visiblePtInds = np.unique(d.vertices)
    inds = np.where(visiblePtInds == numPts)
    visiblePtInds = np.delete(visiblePtInds, inds)

    return visiblePtInds


def in_convex_polyhedron(convex_hull, points, vis=None):
    in_bool = np.zeros([points.shape[0]])
    # ori_set = np.asarray(convex_hull.points)
    ori_set = np.asarray(convex_hull[:, :2])
    d = ConvexHull(ori_set, False)
    ori_edge_index = np.sort(np.unique(d.vertices))
    # a = np.array([1,2,3])
    for i in range(points.shape[0]):
        new_set = np.row_stack([ori_set, points[i, :2]])
        d = ConvexHull(new_set)
        new_edge_index = np.sort(np.unique(d.vertices))
        if ori_edge_index.shape[0] == new_edge_index.shape[0]:
            in_bool[i] = (ori_edge_index == new_edge_index).all()
        else:
            in_bool[i] = False

    # sphere = o3d.geometry.TriangleMesh.create_sphere(1).translate(points[0])
    # if in_bool[i]:
    #     sphere.paint_uniform_color([0.2,0.2,1.0])
    # else:
    #     sphere.paint_uniform_color([1,0,0])

    # pcd_cur2 = o3d.geometry.PointCloud()
    # pcd_cur2.points = o3d.utility.Vector3dVector(convex_hull[ori_edge_index,:])
    # pcd_cur2.paint_uniform_color([0,0,1.0])
    # vis_list = [sphere, pcd_cur2]
    # for item in vis_list:
    #     vis.add_geometry(item)
    # vis.run()

    return bool(in_bool.all())


def calcNormVector(a, b):
    return [a[1] * b[2] - b[1] * a[2], a[2] * b[0] - b[2] * a[0], a[0] * b[1] - b[0] * a[1]]


def validDrivableArea(bbx_pts, points, vis=None):
    '''
    Calculate the distances between the points and all planes of bounding box.
    bbx: [8,3] (corners, pos)
          0 *----* 1
            |\  4  |\5
            | \----|-\
          2 *-|---*3|
             \|   \ |
            6 \----\|7
    ref_pt: [x,y,z]
    '''
    # is_ground = np.logical_not(points[:, 2] <= -2.5)
    # points = points[is_ground, :]
    # Front view:
    fvec_norm = calcNormVector(
        [bbx_pts[2, 0] - bbx_pts[3, 0], bbx_pts[2, 1] - bbx_pts[3, 1], bbx_pts[2, 2] - bbx_pts[3, 2]],
        [bbx_pts[2, 0] - bbx_pts[6, 0], bbx_pts[2, 1] - bbx_pts[6, 1], bbx_pts[2, 2] - bbx_pts[6, 2]])
    # Side View
    svec_norm = calcNormVector(
        [bbx_pts[2, 0] - bbx_pts[0, 0], bbx_pts[2, 1] - bbx_pts[0, 1], bbx_pts[2, 2] - bbx_pts[0, 2]],
        [bbx_pts[2, 0] - bbx_pts[6, 0], bbx_pts[2, 1] - bbx_pts[6, 1], bbx_pts[2, 2] - bbx_pts[6, 2]])
    # BEV View
    bvec_norm = calcNormVector(
        [bbx_pts[2, 0] - bbx_pts[3, 0], bbx_pts[2, 1] - bbx_pts[3, 1], bbx_pts[2, 2] - bbx_pts[3, 2]],
        [bbx_pts[2, 0] - bbx_pts[0, 0], bbx_pts[2, 1] - bbx_pts[0, 1], bbx_pts[2, 2] - bbx_pts[0, 2]])

    X = np.array([points[:, 0] - bbx_pts[2, 0], points[:, 0] - bbx_pts[0, 0],  # Front, Back
                  points[:, 0] - bbx_pts[2, 0], points[:, 0] - bbx_pts[3, 0],  # Left, Right
                  points[:, 0] - bbx_pts[2, 0], points[:, 0] - bbx_pts[6, 0]])  # Top, Down
    Y = np.array([points[:, 1] - bbx_pts[2, 1], points[:, 1] - bbx_pts[0, 1],  # Front, Back
                  points[:, 1] - bbx_pts[2, 1], points[:, 1] - bbx_pts[3, 1],  # Left, Right
                  points[:, 1] - bbx_pts[2, 1], points[:, 1] - bbx_pts[6, 1]])  # Top, Down
    Z = np.array([points[:, 2] - bbx_pts[2, 2], points[:, 2] - bbx_pts[0, 2],  # Front, Back
                  points[:, 2] - bbx_pts[2, 2], points[:, 2] - bbx_pts[3, 2],  # Left, Right
                  points[:, 2] - bbx_pts[2, 2], points[:, 2] - bbx_pts[6, 2]])  # Top, Down
    A = np.array([fvec_norm[0], fvec_norm[0], svec_norm[0], svec_norm[0], bvec_norm[0], bvec_norm[0]]).reshape(-1, 1)
    B = np.array([fvec_norm[1], fvec_norm[1], svec_norm[1], svec_norm[1], bvec_norm[1], bvec_norm[1]]).reshape(-1, 1)
    C = np.array([fvec_norm[2], fvec_norm[2], svec_norm[2], svec_norm[2], bvec_norm[2], bvec_norm[2]]).reshape(-1, 1)
    abc_norms = [np.linalg.norm(fvec_norm), np.linalg.norm(svec_norm), np.linalg.norm(bvec_norm)]
    D = np.array([abc_norms[0], abc_norms[0], abc_norms[1], abc_norms[1], abc_norms[2], abc_norms[2]]).reshape(-1, 1)

    bbx_length = np.array(
        [bbx_pts[2, 0] - bbx_pts[0, 0], bbx_pts[2, 1] - bbx_pts[0, 1], bbx_pts[2, 2] - bbx_pts[0, 2]]) @ np.array(
        fvec_norm) / abc_norms[0]
    bbx_width = np.array(
        [bbx_pts[2, 0] - bbx_pts[3, 0], bbx_pts[2, 1] - bbx_pts[3, 1], bbx_pts[2, 2] - bbx_pts[3, 2]]) @ np.array(
        svec_norm) / abc_norms[1]
    bbx_height = np.array(
        [bbx_pts[2, 0] - bbx_pts[6, 0], bbx_pts[2, 1] - bbx_pts[6, 1], bbx_pts[2, 2] - bbx_pts[6, 2]]) @ np.array(
        bvec_norm) / abc_norms[2]

    bbx_length = np.abs(bbx_height)
    bbx_width = np.abs(bbx_width)
    bbx_height = np.abs(bbx_height)

    dists = np.abs(A * X + B * Y + C * Z) / (D + 1e-5)
    diff_dists = np.array([bbx_length, bbx_length, bbx_width, bbx_width, bbx_height, bbx_height]).reshape(-1, 1)
    diff_dists = np.repeat(diff_dists, points.shape[0], axis=1) - dists
    diff_flags = np.zeros(diff_dists.shape)
    diff_flags[diff_dists > 0] = 1

    # diff_flags[diff_dists <= 0] = -1
    diff_flags2 = diff_flags[0, :]
    for i in range(1, diff_flags.shape[0]):
        diff_flags2 = np.logical_and(diff_flags2, diff_flags[i, :])
    diff_flags3 = np.ones(diff_flags2.shape)

    # new_dist = dists[:, diff_flags2].T
    # new_diff_dists = diff_dists[:, diff_flags2].T
    # new_diff_flags = diff_flags[:, diff_flags2].T
    # new_points = points[diff_flags2, :]
    # new_diff_flags3 = diff_flags3[diff_flags2]
    # diff_flags4 = new_diff_flags[0, :]
    # for i in range(1, diff_flags.shape[0]):
    #     print(i)
    #     diff_flags4 = np.logical_and(diff_flags4, new_diff_flags[i, :])

    if np.sum(diff_flags3[diff_flags2]) > 0:
        return False, diff_flags2
    else:
        return True, diff_flags2

# import open3d as o3d
def validateDrivableArea2(bbx, pts, strict=False):
    inds = np.linspace(0, pts.shape[0], pts.shape[0])
    min_x = np.min(bbx[:, 0])
    min_y = np.min(bbx[:, 1])
    min_z = np.min(bbx[:, 2])

    max_x = np.max(bbx[:, 0])
    max_y = np.max(bbx[:, 1])
    max_z = np.max(bbx[:, 2])

    x_inds = np.logical_and(pts[:, 0] > min_x, pts[:, 0] < max_x)
    pts1 = pts[x_inds, :]
    inds = inds[x_inds]
    y_inds = np.logical_and(pts1[:, 1] > min_y, pts1[:, 1] < max_y)
    pts2 = pts1[y_inds, :]
    inds = inds[y_inds]
    # z_inds = np.logical_and(pts[:, 2] >= min_z, pts[:, 2] <= max_z)
    # inds = inds[z_inds]

    # np_pc2 = pts1[:,:]
    # np_pc3 = pts2[:,:]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(bbx)
    # pcd.paint_uniform_color([0,1,0])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(np_pc2)
    # pcd2.paint_uniform_color([1,0,0])
    # pcd3 = o3d.geometry.PointCloud()
    # pcd3.points = o3d.utility.Vector3dVector(np_pc3)
    # pcd3.paint_uniform_color([0,0,1])
    # pcd4 = o3d.geometry.PointCloud()
    # pcd4.points = o3d.utility.Vector3dVector(pts)
    # pcd4.paint_uniform_color([0,1,1])
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    # vis_list = [pcd,mesh]
    # vis_list += [pcd2, pcd3, pcd4]
    # o3d.visualization.draw_geometries(vis_list)

    if strict:
        if sum(y_inds) > 0:
            return False, inds
        else:
            return True, inds
    else:
        if sum(y_inds) > 3:
            return False, inds
        else:
            return True, inds


def readPointCloud(filename):
    """
    reads bin file and returns
    as m*4 np array
    all points are in meters
    you can filter out points beyond(in x y plane)
    50m for ease of computation
    and above or below 10m
    """
    pcl = np.fromfile(filename, dtype=np.float32, count=-1)
    pcl = pcl.reshape([-1, 4])
    return pcl


def main():
    # pc1 = np.load('./data/pc1.npy')
    # pc2 = np.load('./data/pc2.npy')
    pc1 = readPointCloud('./dataset/0000000789.bin')[:, :3]
    pc2 = readPointCloud('./dataset/0000000790.bin')[:, :3]
    rm_ground = True
    if rm_ground:
        is_ground = np.logical_not(pc1[:, 2] <= -2.4)
        pc1 = pc1[is_ground, :]
        is_ground = np.logical_not(pc2[:, 2] <= -2.4)
        pc2 = pc2[is_ground, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc1)
    pcd.paint_uniform_color([0, 1, 0])
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.paint_uniform_color([1, 0, 0])
    # vis_list = [pcd, pcd2]
    # draw(None, vis_list, save_fov=False)
    pcd3 = pcd + pcd2
    diameter = np.linalg.norm(np.asarray(pcd3.get_max_bound()) - np.asarray(pcd3.get_min_bound()))
    print("Define parameters used for hidden_point_removal")

    camera = [0, 0, -2.5]
    # camera = [-2.0, -1.4, 3]
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0).translate(camera)
    print("Get all points that are visible from given view point")
    radius = diameter * 10
    _, pt_map = pcd3.hidden_point_removal(camera, radius)
    # pc = np.vstack([pc1, pc2])

    # pt_map = HPR(pc, camera, 3)
    print("Visualize result")
    pcd3 = pcd3.select_by_index(pt_map)

    bbx = np.array(
        [[0, 0, 0], [4, 0, 0], [0, -1, 0], [4, -1, 0], [0, 0, -2.75], [4, 0, -2.75], [0, -1, -2.75], [4, -1, -2.75]])
    bbx = np.array(
        [[0, 0, 0], [1, 0, 0], [0, -1, 0], [1, -1, 0], [0, 0, -2.5], [1, 0, -2.5], [0, -1, -2.5], [1, -1, -2.5]]) * 0.2
    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(bbx)
    pcd4.paint_uniform_color([0, 0, 1])
    # pts = np.array([[0.5,-0.5,0.5], [0.5,-0.5,0.2]])

    # bool_in, diff_flags2 = validDrivableArea(bbx, pc1)
    # new_pc1 = pc1[diff_flags2, :]
    # pcd5 = o3d.geometry.PointCloud()
    # pcd5.points = o3d.utility.Vector3dVector(new_pc1)
    # pcd5.paint_uniform_color([1,0,1])
    bool_in, _ = validateDrivableArea2(bbx, pc1)

    print(bool_in)

    sphere = o3d.geometry.TriangleMesh.create_sphere(1.0).translate(np.array([0, 0, -2.5]))
    if bool_in:
        sphere.paint_uniform_color([0.2, 0.2, 1.0])
    else:
        sphere.paint_uniform_color([1, 0, 0])

    # bool_in = in_convex_polyhedron(pcd3,np.array([camera]))
    # sphere = o3d.geometry.TriangleMesh.create_sphere(1.0).translate(camera)
    # bool_in = False
    # if bool_in:
    #     sphere.paint_uniform_color([0.8,0.2,1.0])
    # else:
    #     sphere.paint_uniform_color([1,0,0]) , mesh, sphere
    o3d.visualization.draw_geometries([pcd3, sphere, pcd4])


if __name__ == '__main__':
    # vec = calcNormVector([1,2,0], [2,5,0])
    # print(vec)
    # vec = calcNormVector([1,0,2], [2,0,5])
    # print(vec)
    # np_vec = np.abs(np.array(vec))
    # print(np_vec)
    main()


