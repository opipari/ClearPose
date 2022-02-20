
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def build_model(config):
	
	mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True, 
									  progress=True, 
									  num_classes=91, 
									  pretrained_backbone=True, 
									  trainable_backbone_layers=None)

	num_classes = config["num_classes"]
	in_features = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
	mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	in_features_mask = mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
													   hidden_layer,
													   num_classes)

	return mask_rcnn
