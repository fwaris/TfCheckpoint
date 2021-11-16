module Tests

open System
open Xunit
open TfCheckpoint
open System.IO

[<Fact>]
let ``Read Checkpoint`` () =
    let idxFile = @"C:\s\hack\uncased_L-2_H-128_A-2\bert_model.ckpt.index"
    let ds = CheckpointReader.readCheckpoint (Path.GetDirectoryName idxFile) |> Seq.toArray
    Assert.True(true)
