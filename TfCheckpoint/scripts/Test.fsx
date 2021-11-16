#load "Packages.fsx"
open System.IO
open TfCheckpoint
open System
open Google.Protobuf
open System.Text
open System.Runtime.InteropServices

let idxFile = @"C:\s\hack\uncased_L-2_H-128_A-2\bert_model.ckpt.index"
//let idxFile = @"C:\s\hack\vision_resnet-rs_resnet-rs-350-i320\model.ckpt.index"
//let idxFile = @"C:\s\hack\uncased_L-12_H-768_A-12\bert_model.ckpt.index"
//let idxFile = @"C:\Users\fwaris1\Downloads\4\variables\variables.index"
//let idxFile = @"C:\s\hack\1\variables\variables.index"


let tensors = CheckpointReader.readCheckpoint (Path.GetDirectoryName idxFile) |> Seq.toArray

let (tensorName,tensorData) = tensors.[0]

match tensorData.Tensor with
| CheckpointReader.TdFloat ds  -> printfn "%A" ds

let header,tensorMeta = CheckpointIndex.readFromFile idxFile
//#r "nuget: TorchSharp"
//#r "nuget: Microsoft.ML"
//open TorchSharp

let testM() =
    let xs = [|for i in 0 .. 10-1 -> i|]
    let rs = Span(xs)
    let x = MemoryMarshal.Cast<int,float32>(rs)
    x.ToArray()