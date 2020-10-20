from .base.base_dataset import *
from .base.base_train import *
from .base.base_runtime import *

classifier_head_str = 'BoostingClassifierHead(n_estimators=500, n_jobs=-1)'
feature_extractor_str = 'BERTFeatureExtractor()'

work_dir = 'artifacts/modern'

is_debug=False
