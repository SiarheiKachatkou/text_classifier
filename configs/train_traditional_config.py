from .base.base_dataset import *
from .base.base_train import *
from .base.base_runtime import *

is_debug = False
classifier_head_str = 'RandomForestClassifierHead(n_estimators=500, n_jobs=-1)'
feature_extractor_str = 'TFIDFFeatureExtractor(max_features=100)'
work_dir = 'artifacts/traditional'