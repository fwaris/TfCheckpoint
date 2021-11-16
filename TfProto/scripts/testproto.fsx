#r "nuget: TfProto"
#r "nuget: Google.protobuf"
#r "nuget: IronSnappy"
open System
open System.IO
open Google.Protobuf
open Google.Protobuf.FSharp
open System.Text

let fn =  @"C:\s\hack\uncased_L-2_H-128_A-2\bert_model.ckpt.index"

let magic = [|87uy; 251uy; 128uy; 139uy; 36uy; 117uy; 71uy; 219uy|]

let bytes = File.ReadAllBytes fn

let footer = bytes.[^7..]

footer = magic

type ReadState = {Marker : int; PrefixCache:byte[]}

let FOOTER_SIZE = 48

let isAtEnd (bytes:byte[]) mark = mark >= bytes.Length - FOOTER_SIZE - 1

let readKeyHead (bytes:byte[]) mark =
    use str = new MemoryStream(bytes, mark, bytes.Length - mark)
    use cs = new CodedInputStream(str)
    let sharedBytes     = cs.ReadInt32()
    let unsharedBytes   = cs.ReadInt32()
    let valueLength     = cs.ReadInt32()
    let mark = mark + int cs.Position 
    (sharedBytes,unsharedBytes,valueLength),mark

let readValue (bytes:byte[]) valueLength mark =
    let bundle = Tensorflow.BundleEntryProto.Parser.ParseFrom(bytes,mark,valueLength)
    bundle,(mark+valueLength)

let readKeyValue bytes st =
    let (sharedBytes,unsharedBytes,valueLength),mark = readKeyHead bytes st.Marker
    let haveData = sharedBytes + unsharedBytes + valueLength > 0
    if haveData then
        let prefixData = Array.append st.PrefixCache.[0..sharedBytes-1] bytes.[mark .. mark+unsharedBytes-1]
        let key = UTF8Encoding.Default.GetString(prefixData)
        let value,mark = readValue bytes valueLength (mark + unsharedBytes)
        let st = {st with Marker=mark; PrefixCache=prefixData}
        ((key,value),st) |> Some
    else
        None

let read (bytes:byte[]) =
    use str = new CodedInputStream(bytes)
    let b0 = str.ReadInt32()
    if b0 <> 0 then failwithf "expect 0 as the first byte. Data may be snappy compressed "
    let _       = str.ReadInt32()
    let hdrLen  = str.ReadInt32()
    let mark = int str.Position
    let hdr = Tensorflow.BundleHeaderProto.Parser.ParseFrom(bytes,mark,hdrLen)
    let mark = mark + hdrLen
    let initState = {Marker=mark; PrefixCache=[||]}
    let bundles = initState |> Seq.unfold (readKeyValue bytes)
    hdr,bundles
    
let h,bds = read bytes
bds |> Seq.toArray

