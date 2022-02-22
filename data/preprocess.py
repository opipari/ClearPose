import os
from scipy.io import loadmat

def main():

	train_summary_file = open('./data/train_images.csv', 'w')
	train_sets = ['set1','set4','set5','set6','set7']

	root = "/media/mytre/6358C6357FEBD1E6/"
	sets = [sett for sett in sorted(os.listdir(root)) if sett.startswith('set')]
	sets = [sett for sett in sets if sett in train_sets]
	img_id = 0
	for sett in sets:
		root_set = os.path.join(root, sett)
		scenes = [scn for scn in sorted(os.listdir(root_set)) if scn.startswith('scene')][:-1]

		for scene in scenes:
			root_set_scene = os.path.join(root_set, scene)
			data_files = sorted(os.listdir(root_set_scene))

			meta_path = os.path.join(root_set_scene,'metadata.mat')
			meta = loadmat(meta_path)

			color_files = sorted([fl for fl in data_files if fl.endswith('color.png')])
			depth_files = sorted([fl for fl in data_files if fl.endswith('depth.png')])
			label_files = sorted([fl for fl in data_files if fl.endswith('label.png')])
			box_files = sorted([fl for fl in data_files if fl.endswith('box.txt')])

			for fl in color_files:
				intid = fl.split('-')[0]

				obj_ids, _, depth_factor, cam_pose, obj_poses, _, obj_boxes = meta[intid][0][0]
				obj_ids = obj_ids.flatten()
				num_objs = len(obj_ids)

				train_summary_file.write("{},{},{},{}\n".format(img_id, root_set_scene, intid, num_objs))
				img_id += 1

	train_summary_file.close()



	test_summary_file = open('./data/test_images.csv', 'w')
	val_summary_file = open('./data/val_images.csv', 'w')

	sets = [sett for sett in sorted(os.listdir(root)) if sett.startswith('set')]

	img_id = 0
	for sett in sets:
		root_set = os.path.join(root, sett)
		
		if sett in train_sets:
			scenes = [scn for scn in sorted(os.listdir(root_set)) if scn.startswith('scene')][-1:]
		else:
			scenes = [scn for scn in sorted(os.listdir(root_set)) if scn.startswith('scene')]


		for scene in scenes:
			root_set_scene = os.path.join(root_set, scene)
			data_files = sorted(os.listdir(root_set_scene))

			meta_path = os.path.join(root_set_scene,'metadata.mat')
			meta = loadmat(meta_path)

			color_files = sorted([fl for fl in data_files if fl.endswith('color.png')])
			depth_files = sorted([fl for fl in data_files if fl.endswith('depth.png')])
			label_files = sorted([fl for fl in data_files if fl.endswith('label.png')])
			box_files = sorted([fl for fl in data_files if fl.endswith('box.txt')])

			for fl in color_files:
				intid = fl.split('-')[0]

				obj_ids, _, depth_factor, cam_pose, obj_poses, _, obj_boxes = meta[intid][0][0]
				obj_ids = obj_ids.flatten()
				num_objs = len(obj_ids)

				if img_id%100==0:
					val_summary_file.write("{},{},{},{}\n".format(img_id, root_set_scene, intid, num_objs))
				else:
					test_summary_file.write("{},{},{},{}\n".format(img_id, root_set_scene, intid, num_objs))
				img_id += 1
	
	val_summary_file.close()
	test_summary_file.close()

	



if __name__=="__main__":
	main()