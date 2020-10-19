

def train_and_test(build_model_fn, dataset, fold):

    train_dataset, val_dataset = dataset.get_fold(fold)
    model = build_model_fn()
    model.fit(train_dataset)
    predictions=model.predict(val_dataset)
    return predictions, val_dataset.get_labels()



