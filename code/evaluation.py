

accuracies = {}

for model in models:
    model.fit(X_train, y_train)
    train_acc = round(model.score(X_train, y_train) * 100, 2)

    val_acc = round(model.score(X_valid, y_valid) * 100, 2)
    accuracies[model.__class__.__name__] = {"train_acc": train_acc, "val_acc": val_acc}

classification_preds = lgm.predict(X_valid)
acc_df = pd.DataFrame(accuracies).transpose().sort_values("val_acc", ascending=False)
acc_df