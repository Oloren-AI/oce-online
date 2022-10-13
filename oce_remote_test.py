import olorenchemengine as oce

# Train a model and predict with it
dataset = oce.ExampleDataset()
with oce.Remote("http://api.oloren.ai:5000", debug=False) as sid:
    model = oce.BaseBoosting(
        [
            oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),
        ]
    )
    model.fit(*dataset.train_dataset)
    model.predict(["CC(=O)Nc1ccc(O)cc1"])
    oce.save(model, "demo.oce")

# TODO: Save the above model locally and load it back in (IN ANOTHER SESSION!!!)

# Generate a visualization on a dataset
with oce.Remote("http://api.oloren.ai:5000", debug=False) as sid:
    dataset = oce.ExampleDataset()
    vis = oce.ChemicalSpacePlot(dataset, oce.DescriptastorusDescriptor("morgan3counts"), opacity=0.4, dim_reduction="tsne")
    vis.render_data_url()