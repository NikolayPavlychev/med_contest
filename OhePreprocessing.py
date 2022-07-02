#Function OnePreprocessing() convert dataset, contain categorical features and transform them into OHE form.

def OhePreprocessing(dataset, train_bool=True, feature_list=None,train_cols_order=None):
    import pandas as pd

    cols=list(set(list(dataset.columns))-set(feature_list))
    print(cols)
    dataset_ = dataset[cols]

    for col in feature_list:
        df = pd.get_dummies(dataset[col], prefix=str(col), prefix_sep="__",
                              columns=dataset[col])

        dataset_ = pd.concat((dataset_,df),axis=1)

    if train_bool:
        train_cols_order = list(dataset_.columns)
    else:
        for col in dataset_.columns:
            if ("__" in col) and (col.split("__")[0] in cols) and col not in train_cols_order:
                print("Removing additional feature {}".format(col))
                dataset_.drop(col, axis=1, inplace=True)

        for col in train_cols_order:
            if col not in dataset_.columns:
                print("Adding missing feature {}".format(col))
                dataset_[col] = 0

    ohe_cols_order = list(dataset_.drop(cols,axis=1).columns)

    if train_bool:
        return dataset_, train_cols_order, ohe_cols_order
    else:
        return dataset_[train_cols_order]

