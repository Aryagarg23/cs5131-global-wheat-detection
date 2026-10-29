from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# The densest training image has 116 boxes; torchvision's default of 100 detections would cap recall.
MAX_DETECTIONS = 300


def build_model(pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, box_detections_per_img=MAX_DETECTIONS)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # background + wheat head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model
