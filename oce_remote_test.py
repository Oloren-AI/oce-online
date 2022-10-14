import olorenchemengine as oce

# # Train a model and predict with it
# dataset = oce.ExampleDataset()
# with oce.Remote("http://api.oloren.ai:5000", debug=True) as sid:
#     df = oce.ExampleDataFrame()
#     # model = oce.BaseTorchGeometricModel(oce.TLFromCheckpoint("default"), batch_size=8, epochs=1, preinitialized=True)

#     model = oce.BaseBoosting(
#         [
#             oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1000),
#             oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1000),
#         ]
#     )
#     model.fit(df["Smiles"], df["pChEMBL Value"])
#     x = model.predict(["CC(=O)Nc1ccc(O)cc1"])
#     print(x)
#     while True: pass
#     # x = model.predict(["CC(=O)Nc1ccc(O)cc1"])
#     # print("Saving model")
#     # oce.save(model, "demo.oce")
#     # print("Done saving model")

# Load a model and predict with it (in another session!!)
# with oce.Remote("http://api.oloren.ai:5000", debug=True) as sid:
#     print("Loading model")
#     model = oce.load("demo.oce")
#     print("Done loading model")
#     x = model.predict(["CC(=O)Nc1ccc(O)cc1"])
#     print(x)
#     print(type(x))

# # Generate a visualization on a dataset
# with oce.Remote("http://api.oloren.ai:5000", debug=False) as sid:
#     dataset = oce.ExampleDataset()
#     vis = oce.ChemicalSpacePlot(dataset, oce.DescriptastorusDescriptor("morgan3counts"), opacity=0.4, dim_reduction="tsne")
#     vis.render_data_url()


with oce.Remote("http://api.oloren.ai:5000", debug=True) as sid:
    df = oce.ExampleDataFrame()

    model = oce.BaseBoosting(
        [
            oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators=1),
            oce.BaseTorchGeometricModel(oce.TLFromCheckpoint("default"), batch_size=8, epochs=1, preinitialized=True),
            oce.RandomForestModel(oce.OlorenCheckpoint("default"), n_estimators=1),
        ]
    )

    model.fit(df["Smiles"], df["pChEMBL Value"])
    _ = model.predict(["CC(=O)OC1=CC=CC=C1C(=O)O"])
    oce.save(model, "model.oce")
    _ = oce.load("model.oce")
    import os
    os.remove("model.oce")