namespace TfCheckpoint 
open System
open System.IO
open Google.Protobuf
open System.Text
open System.Runtime.InteropServices

module CheckpointReader =

    open System.Runtime.CompilerServices
    open Tensorflow

    type TensorData =
    | TdFloat       of float32[]
    | TdDouble      of float[]
    ///Half is available in .Net 5.0 - not  .Net Standard
    | TdBfloat16    of Half[]
    | TdBool        of bool[]
    | TdInt8        of int8[]
    | TdInt16       of int16[]
    | TdInt32       of int[]
    | TdInt64       of int64[]
    | TdUint8       of uint8[]
    | TdUint16      of uint16[]
    | TdUint32      of uint[]
    | TdUint64      of uint64[]
    | TdRaw of Tensorflow.DataType * byte[]

    type ShapedTensor = {Shape:int64[]; Tensor:TensorData}

    let toShapedTensor (bndl:Tensorflow.BundleEntryProto) (data:Span<byte>) =
        let shape = bndl.Shape.Value.Dim |> Seq.map (fun x->x.Size.Value) |> Seq.toArray
        let tnsrData =
            match bndl.Dtype.Value with
            | DataType.DtFloat    -> MemoryMarshal.Cast<byte,float32>(data).ToArray()   |> TdFloat
            | DataType.DtDouble   -> MemoryMarshal.Cast<byte,float>(data).ToArray()     |> TdDouble
            | DataType.DtBfloat16 -> MemoryMarshal.Cast<byte,Half>(data).ToArray()      |> TdBfloat16
            | DataType.DtBool     -> MemoryMarshal.Cast<byte,bool>(data).ToArray()      |> TdBool
            | DataType.DtInt8     -> MemoryMarshal.Cast<byte,int8>(data).ToArray()      |> TdInt8
            | DataType.DtInt16    -> MemoryMarshal.Cast<byte,int16>(data).ToArray()     |> TdInt16
            | DataType.DtInt32    -> MemoryMarshal.Cast<byte,int32>(data).ToArray()     |> TdInt32
            | DataType.DtInt64    -> MemoryMarshal.Cast<byte,int64>(data).ToArray()     |> TdInt64
            | DataType.DtUint8    -> MemoryMarshal.Cast<byte,uint8>(data).ToArray()     |> TdUint8
            | DataType.DtUint16   -> MemoryMarshal.Cast<byte,uint16>(data).ToArray()    |> TdUint16
            | DataType.DtUint32   -> MemoryMarshal.Cast<byte,uint32>(data).ToArray()    |> TdUint32
            | DataType.DtUint64   -> MemoryMarshal.Cast<byte,uint64>(data).ToArray()    |> TdUint64
            | _                   -> TdRaw (bndl.Dtype.Value,data.ToArray())
        {Shape=shape; Tensor=tnsrData}

    ///Reads tensorflow checkpoint from folder.
    //Folder is expected to contain the *.index file and the data 'shards'
    let readCheckpoint folderPath =
        if Directory.Exists folderPath |> not then failwithf $"Not a folder {folderPath}"
        let idxFile = Directory.GetFiles(folderPath,"*.index")
        if Seq.isEmpty idxFile then failwith ".index file not found"
        if Seq.length idxFile > 1 then failwith "multiple .index files not handled - separate each checkpoint into its own folder"
        let idxFile = Seq.head idxFile
        let hdr,bundles = 
            try
                CheckpointIndex.readFromFile idxFile
            with ex -> 
                failwithf $"unable to parse index file {idxFile}: {ex.Message}"
        if hdr.NumShards.IsNone || hdr.NumShards.Value <= 0 then failwith "number of shards not found in header"
        let totalShards = hdr.NumShards.Value
        let pathPrefx = Path.GetFileNameWithoutExtension(idxFile)
        
        bundles
        |> Seq.map (fun (name,bndl) -> 
            let shardId = match bndl.ShardId with ValueNone -> 0 | ValueSome s -> s //assume shard 0 if none
            let shard = $"{folderPath}/{pathPrefx}.data-%05d{shardId}-of-%05d{totalShards}"
            let offset = match bndl.Offset with ValueNone -> 0L | ValueSome s -> s
            use str = File.OpenRead shard                                                   //trading lower memory for performance
            str.Seek(offset, SeekOrigin.Begin) |> ignore
            let tensorSizeInBytes = int bndl.Size.Value
            let buffr = Array.zeroCreate tensorSizeInBytes
            let span = Span<byte>(buffr)
            let bytesRead = str.Read(span)
            if bytesRead <> tensorSizeInBytes then failwith $"unable to load tensor data" 
            let shapedTensor = toShapedTensor bndl span
            name,shapedTensor)
       