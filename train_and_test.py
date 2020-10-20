import pickle
import os
from model import build_model


def train_and_test(cfg, dataset, fold):
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    train_dataset, val_dataset = dataset.get_fold(fold)
    model = build_model(cfg)
    model.fit(train_dataset)

    with open(os.path.join(cfg.work_dir,f'model{fold}.pkl'),'wb') as file:
        pickle.dump(model, file)

    predictions=model.predict(val_dataset)

    return predictions, val_dataset.get_labels()



