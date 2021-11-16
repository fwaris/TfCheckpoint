
//#r "nuget: Microsoft.Data.SqlClient"

let TRCH_PATH = @"c:\s\libtorch\lib"
System.Runtime.InteropServices.NativeLibrary.Load($"{TRCH_PATH}/torch_cpu.dll")
let path = System.Environment.GetEnvironmentVariable("path")
let path' = $"{path};{TRCH_PATH}"
System.Environment.SetEnvironmentVariable("path",path')

#r "nuget: SQLProvider"
#r "nuget: Microsoft.ML.AutoML"
#r "nuget: Plotly.NET, 2.0.0-preview.10"
#r "nuget: TorchSharp, Version=0.93.6" 
#r "nuget: FsPickler"
#r "nuget: Parquet.Net"
#r "nuget: FSharp.Control.AsyncSeq"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: FSharp.Data"
#r "nuget: MessagePack"
#r "nuget: Microsoft.ML.TensorFlow"
#r "nuget: SciSharp.TensorFlow.Redist"
#r "nuget: NumSharp"
#I @"../"
//#load "MathUtils.fs"
//#load "MLUtils.fs"
//#load "ignored/CN.fs"
//#load "DataAccess.fs"
//#load "TorchSharp.Fun.fs"
//#load "FsJson.fs"
//#load "ParquetUtils.fs"
//#load "UC4Schema.fs"

#if RT
printfn "RT"
#endif



