EXCLUDED_WORDS = ["REMOTE_BC_KEY", "args", "kwargs"]

class RemoteSession():
    def __init__(self):
        pass
    def __getattribute__(self, key):
        return RemoteObject(key)

def parameterize_remote(object):
    if issubclass(type(object), RemoteObject):
        if not hasattr(object, "args") or not hasattr(object, "kwargs"):
            print("Warning: RemoteObject has no args or kwargs")
            print(f"Object is {object.REMOTE_BC_KEY}")
        return {
            **{"BC_class_name": object.REMOTE_BC_KEY.split(".")[-1]},
            **{"args": [parameterize_remote(arg) for arg in object.args]},
            **{"kwargs": {k: parameterize_remote(v) for k, v in object.kwargs.items()}},
        }
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
        or issubclass(type(object), dict)
    ):
        return object
    elif issubclass(type(object), list):
        return [parameterize_remote(x) for x in object]
    else:
        raise ValueError(f"Cannot parameterize {object} of type {type(object)}")

class RemoteObject():
    def __init__(self, key, *args, **kwargs) -> None:
        self.REMOTE_BC_KEY = key

    def __getattribute__(self, key):
        if key in EXCLUDED_WORDS:
            return object.__getattribute__(self, key)
        else:
            return RemoteObject(self.REMOTE_BC_KEY + "." + key)

    def __repr__(self) -> str:
        if hasattr(self, "args") or hasattr(self, "kwargs"):
            return parameterize_remote(self)

    def __call__(self, *args, **kwargs):
        if self.REMOTE_BC_KEY.split('.')[-1][0].isupper(): # instantiating an object
            self.args = args
            self.kwargs = kwargs
            return self
        else:
            return self # TODO: handle calling a function (like train)




# TODO: track operators on RemoteObjects

if __name__ == "__main__":
    oce = RemoteSession()
    dataset = oce.ExampleDataset()
    sub_model1 = oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000)
    model = oce.ensemble.BaseBoosting([
            sub_model1,
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)])
    # only go to api when function is called!
    print(parameterize_remote(model))
    # model.fit(*dataset.train_dataset)
    # oce.save(model, "model.oce")