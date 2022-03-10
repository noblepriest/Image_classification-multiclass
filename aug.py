import Augmentor
p = Augmentor.Pipeline("dataset/New_data/Swimming")
p.ground_truth("dataset/New_data/Swimming")
p.rotate(probability=1, max_left_rotation=2, max_right_rotation=2)
#p.zoom_random(probability=0.5, percentage_area=0.5)
p.sample(500,multi_threaded=False)