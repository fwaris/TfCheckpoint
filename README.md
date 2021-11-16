# TfCheckpoint
A library to extract tensor data from a Tensorflow checkpoint folder. The main use it to load pre-trained weights into model structures e.g. for PyTorch models.

## Usage Example
```F#

let idxFile = @"C:\s\hack\uncased_L-2_H-128_A-2\bert_model.ckpt.index"

let tensors = CheckpointReader.readCheckpoint (Path.GetDirectoryName idxFile) |> Seq.toArray

let (tensorName,tensorData) = tensors.[0]
```
### Output:
```
val tensorName: string = "bert/embeddings/LayerNorm/beta"
val tensorData: CheckpointReader.ShapedTensor =
  { Shape = [|128L|]
    Tensor =
     TdFloat
       [|0.1427177936f; 0.1417384148f; 0.1129989177f; 0.008431605063f;
         -0.3839171827f; -0.04579306394f; -0.009391464293f; 0.2562615871f;
         0.02031775191f; -0.1169935018f; 0.04341379181f; -0.03693608567f;
         -0.1498966217f; -0.04671567678f; -0.05263318121f; -0.1550539136f;
         ...
```

## Build Instructions

The build relies on Tensorflow Protobuf definitions. The Tensorflow repo should be copied into the folder for the TfProto project such that the .proto file references in the .fsproj file are valid.
