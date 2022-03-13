import os
from scipy.io import loadmat

def save_scene(scene, root_set, test_summary_file, img_id, sample_rate):
	root_set_scene = os.path.join(root_set, scene)
	try:
		data_files = os.listdir(root_set_scene)
	except OSError:
		return
	meta_path = os.path.join(root_set_scene,'metadata.mat')
	meta = loadmat(meta_path)
	color_files = sorted([fl for fl in data_files if fl.endswith('color.png')])
	depth_files = sorted([fl for fl in data_files if fl.endswith('depth.png')])
	label_files = sorted([fl for fl in data_files if fl.endswith('label.png')])
	box_files = sorted([fl for fl in data_files if fl.endswith('box.txt')])
	for i, fl in enumerate(color_files):
		if i % sample_rate != 0:
			continue
		intid = fl.split('-')[0]
		obj_ids, _, depth_factor, cam_pose, obj_poses, _, obj_boxes = meta[intid][0][0]
		obj_ids = obj_ids.flatten()
		num_objs = len(obj_ids)
		test_summary_file.write("{},{},{},{}\n".format(img_id, root_set_scene, intid, num_objs))
		img_id += 1
	return img_id, test_summary_file

def standard_test(root, train_sets):
	test_summary_file = open('./data/standard_test.csv', 'w')
	img_id = 0
	for sett in train_sets:
		root_set = os.path.join(root, sett)
		if sett in train_sets:
			scenes = [scn for scn in os.listdir(root_set) if scn.startswith('scene')][-1:]
		for scene in scenes:
			img_id, test_summary_file = save_scene(scene, root_set, test_summary_file, img_id, sample_rate=21*400)
	test_summary_file.close()


def occlusion_test(root):
	test_summary_file = open('./data/occlusion_test.csv', 'w')
	img_id = 0
	for sett in ['set2', 'set3']:
		root_set = os.path.join(root, sett)
		if sett in ['set2', 'set3']:
			scenes = [scn for scn in os.listdir(root_set) if scn.startswith('scene')]
		for scene in scenes:
			img_id, test_summary_file = save_scene(scene, root_set, test_summary_file, img_id, sample_rate=38*400)
	test_summary_file.close()

def wou_test(root):# water opaque unseen background
	scenes = ["scene1", "scene2", "scene3"]
	test_summary_file = open('./data/wou_test.csv', 'w')
	img_id = 0
	root_set = os.path.join(root, 'set8')
	for scene in scenes:
		img_id, test_summary_file = save_scene(scene, root_set, test_summary_file, img_id, sample_rate=1200)
	test_summary_file.close()

def color_test(root):
	scene = ['set8/scene6', 'set9/scene9', 'set9/scene10']
	test_summary_file = open('./data/color_test.csv', 'w')
	img_id = 0
	for s in scene:
		set_nam, scene_nam = s.split('/')
		root_set = os.path.join(root, set_nam)
		img_id, test_summary_file = save_scene(scene_nam, root_set, test_summary_file, img_id, sample_rate=800)
	test_summary_file.close()

def covered_test(root):
	scene = ['set8/scene4', 'set9/scene7', 'set9/scene8']
	test_summary_file = open('./data/covered_test.csv', 'w')
	img_id = 0
	for s in scene:
		set_nam, scene_nam = s.split('/')
		root_set = os.path.join(root, set_nam)
		img_id, test_summary_file = save_scene(scene_nam, root_set, test_summary_file, img_id, sample_rate=800)
	test_summary_file.close()

def non_planner_test(root):
	scene = ['set8/scene5', 'set9/scene11', 'set9/scene12']
	test_summary_file = open('./data/non-planner_test.csv', 'w')
	img_id = 0
	for s in scene:
		set_nam, scene_nam = s.split('/')
		root_set = os.path.join(root, set_nam)
		img_id, test_summary_file = save_scene(scene_nam, root_set, test_summary_file, img_id, sample_rate=800)
	test_summary_file.close()


def train_set(root, train_sets):
	train_summary_file = open('./data/train_images.csv', 'w')
	sets = [sett for sett in os.listdir(root) if sett.startswith('set')]
	sets = [sett for sett in sets if sett in train_sets]
	img_id = 0
	for sett in sets:
		root_set = os.path.join(root, sett)
		scenes = [scn for scn in os.listdir(root_set) if scn.startswith('scene')][:-1]

		for scene in scenes:
			img_id, train_summary_file = save_scene(scene, root_set, train_summary_file, img_id, sample_rate=400)

	train_summary_file.close()

def set4_train(root):# water opaque unseen background
	scenes = ["scene1", "scene2", "scene3", "scene4", "scene5"]
	train_summary_file = open('./data/set4_train.csv', 'w')
	test_summary_file = open('./data/set4_test.csv', 'w')
	train_id = 0
	test_id = 0
	root_set = os.path.join(root, 'set4')
	for scene in scenes:
		root_set_scene = os.path.join(root_set, scene)
		try:
			data_files = os.listdir(root_set_scene)
		except OSError:
			return
		meta_path = os.path.join(root_set_scene,'metadata.mat')
		meta = loadmat(meta_path)
		color_files = sorted([fl for fl in data_files if fl.endswith('color.png')])
		depth_files = sorted([fl for fl in data_files if fl.endswith('depth.png')])
		label_files = sorted([fl for fl in data_files if fl.endswith('label.png')])
		box_files = sorted([fl for fl in data_files if fl.endswith('box.txt')])
		for i, fl in enumerate(color_files):
			if i % 1 != 0:
				continue
			if i % (36) != 0:
				intid = fl.split('-')[0]
				obj_ids, _, depth_factor, cam_pose, obj_poses, _, obj_boxes = meta[intid][0][0]
				obj_ids = obj_ids.flatten()
				num_objs = len(obj_ids)
				train_summary_file.write("{},{},{},{}\n".format(train_id, root_set_scene, intid, num_objs))
				train_id += 1
			else:
				intid = fl.split('-')[0]
				obj_ids, _, depth_factor, cam_pose, obj_poses, _, obj_boxes = meta[intid][0][0]
				obj_ids = obj_ids.flatten()
				num_objs = len(obj_ids)
				test_summary_file.write("{},{},{},{}\n".format(test_id, root_set_scene, intid, num_objs))
				test_id += 1
	train_summary_file.close()
	test_summary_file.close()

def main():

	train_sets = ['set1','set4','set5','set6','set7']
	root = "/home/yuzeren/data/downsample/clearpose"
	# train_set(root, train_sets)


	root = "/media/yuzeren/6358C6357FEBD1E6/"
	sets = [sett for sett in os.listdir(root) if sett.startswith('set')]
	standard_test(root, train_sets)
	occlusion_test(root)
	wou_test(root)
	color_test(root)
	covered_test(root)
	non_planner_test(root)
	# set4_train(root)



if __name__=="__main__":
	main()