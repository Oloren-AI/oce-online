
def _is_object(key):
    return key.split('.')[-1][0].isupper()



class RemoteSession():
    def __init__(self):
        pass

    def construct(self, object):
        if hasattr(object, "REMOTE_ID"):
            return object

        if _is_object(object.REMOTE_BC_KEY):
            object.REMOTE_ID = generate_random_uuid()
            print(f"CONSTRUCT {object.REMOTE_ID} => {parameterize_remote(object)}")
            return object
        else:
            raise NotImplementedError("Cannot construct a method or attribute yet!")

    def run_method(self, object, method_name, args, kwargs):
        if not hasattr(object, "REMOTE_ID"):
            object = self.construct(object)
        print(f"RUN {object.REMOTE_ID}.{method_name}")

    def __getattribute__(self, key):
        EXCLUDED_WORDS = ["run_method", "construct"]
        if key in EXCLUDED_WORDS:
            return object.__getattribute__(self, key)
        return RemoteObject(key, "oce", self)

def generate_random_uuid():
    import uuid
    return str(uuid.uuid4())

def parameterize_remote(object):
    if issubclass(type(object), RemoteObject):
        if not hasattr(object, "args") or not hasattr(object, "kwargs"):
            print("Warning: RemoteObject has no args or kwargs")
            print(f"Object is {object.REMOTE_BC_KEY}")
        return {
            **({'BC_remote_id': object.REMOTE_ID} if hasattr(object, "REMOTE_ID") else {}),
            **{"BC_class_name": object.REMOTE_BC_KEY.split(".")[-1]},
            **{"args": [parameterize_remote(arg) for arg in object.args]},
            **{"kwargs": {k: parameterize_remote(v) for k, v in object.kwargs.items()}},
        }
    elif (
        object is None
        or issubclass(type(object), int)
        or issubclass(type(object), float)
        or issubclass(type(object), str)
    ):
        return object
    elif issubclass(type(object), list):
        return [parameterize_remote(x) for x in object]
    else:
        raise ValueError(f"Cannot parameterize {object} of type {type(object)}")

class RemoteObject():

    def __init__(self, key, parent, session) -> None:
        self.REMOTE_BC_KEY = key
        self.REMOTE_PARENT = parent
        self.REMOTE_SESSION = session

    def __getattribute__(self, key):
        EXCLUDED_WORDS = ["REMOTE_BC_KEY", "REMOTE_SESSION", "REMOTE_PARENT", "REMOTE_ID", "args", "kwargs"]
        if key in EXCLUDED_WORDS:
            return object.__getattribute__(self, key)
        else:
            return RemoteObject(self.REMOTE_BC_KEY + "." + key, self, self.REMOTE_SESSION)

    def __repr__(self) -> str:
        if hasattr(self, "args") or hasattr(self, "kwargs"):
            return parameterize_remote(self)

    def __call__(self, *args, **kwargs):
        if self.REMOTE_BC_KEY.split('.')[-1][0].isupper(): # instantiating an object
            self.args = args
            self.kwargs = kwargs
            return self
        else:
            return self.REMOTE_SESSION.run_method(self.REMOTE_PARENT, self.REMOTE_BC_KEY.split('.')[-1], args, kwargs)
            # parameterized_kwargs = {parameterize_remote(k): parameterize_remote(v) for k, v in kwargs.items()}
            # print(f"Calling function {self.REMOTE_BC_KEY.split('.')[-1]} with: args: {[parameterize_remote(a) for a in args]}, kwargs: {parameterized_kwargs} on {parameterize_remote(self.REMOTE_PARENT)}" )
            # return self # TODO: handle calling a function (like train)

    def __iter__(self):
        return iter([0, 1])




# TODO: track operators on RemoteObjects

# We are basically making a Python "Compiler" that compiles down to a JSON dictionary - but we can use python
# in the compiler lol


if __name__ == "__main__":
    oce = RemoteSession()
    dataset = oce.ExampleDataset()
    sub_model1 = oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000)
    model = oce.ensemble.BaseBoosting([
            sub_model1,
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000)])
    model.fit(*dataset.train_dataset)
    # oce.save(model, "model.oce")


    # CONSTRUCT model FROM JSON_PARAMS
    # CONSTRUCT dataset FROM JSON_PARAMS
    # CONSTRUCT train_dataset FROM dataset.train_dataset
    # RUN fit on model with train_dataset