from model import build_model

def train_and_test(cfg, dataset, fold):

    train_dataset, val_dataset = dataset.get_fold(fold)
    model = build_model(cfg)
    model.fit(train_dataset)
    predictions=model.predict(val_dataset)
    return predictions, val_dataset.get_labels()



