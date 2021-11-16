namespace TfCheckpoint 
open System
open System.IO
open Google.Protobuf
open System.Text

module CheckpointIndex =
    type ReadState = {Marker : int; PrefixCache:byte[]}
    let MAGIC = [|87uy; 251uy; 128uy; 139uy; 36uy; 117uy; 71uy; 219uy|]
    let MAGIC_OFFSET_END  = 7
    let FOOTER_SIZE       = 40
    let FOOTER_OFFSET_END = 47

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

    //data block expected to be formatted as per this post:
    //https://chromium.googlesource.com/external/leveldb/+/HEAD/doc/table_format.md
    let readDataBlock (bytes:byte[]) =
        let (_,unshared,valueLen),mark = readKeyHead bytes 0
        let hdr = Tensorflow.BundleHeaderProto.Parser.ParseFrom(bytes,mark+unshared,valueLen)
        let mark = mark + unshared + valueLen
        let initState = {Marker=mark; PrefixCache=[||]}
        let bundles = initState |> Seq.unfold (readKeyValue bytes)
        hdr,bundles

    ///Gets handle to the data index block from the footer
    let indexHandle (footerBytes:byte[]) =
        use cs = new CodedInputStream(footerBytes)
        let metaOff,metaLen,idxOff,idxLen = cs.ReadInt32(),cs.ReadInt32(),cs.ReadInt32(),cs.ReadInt32()
        idxOff,idxLen        
            
    ///Assumes there is only one data block. Does not handle multiple data blocks
    let readIndexBlock (indexBlock:byte[]) =        
        let ((us,sh,vl),mark) = readKeyHead indexBlock 0
        use cs = new CodedInputStream(indexBlock.[mark + sh ..])
        let dataBlockOff = cs.ReadInt32()
        let dataBlockLen = cs.ReadInt32()
        dataBlockOff,dataBlockLen

    let readFromBytes (indexFileBytes:byte[]) =
        let magic = indexFileBytes.[^MAGIC_OFFSET_END..]
        if magic <> MAGIC then failwithf "expected 'magic' bytes not found"
        let footer = indexFileBytes.[^FOOTER_OFFSET_END .. ^(MAGIC_OFFSET_END + 1)]
        assert (footer.Length = FOOTER_SIZE)
        let idxOff,idxLen = indexHandle footer
        let indexBlock = indexFileBytes.[idxOff .. idxOff + idxLen - 1]
        let dataBlockOffs,dataBlockLen = readIndexBlock indexBlock
        let data = indexFileBytes.[dataBlockOffs .. dataBlockOffs + dataBlockLen - 1] //extra allocation but index files are small
        let data = 
            if data.[0] = 0uy then
               data 
            else 
                IronSnappy.Snappy.Decode(data)
        readDataBlock data
        
    ///Read the checkpoint .index file (there should only be one in the checkpoint folder)   
    let readFromFile path = 
        let indexFileBytes = File.ReadAllBytes path
        readFromBytes indexFileBytes
