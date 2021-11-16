#load "Packages.fsx"
open TorchSharp
open type TorchSharp.torch


//let NUM_EMB_WORD = 30522L
//let NUM_EMB_POS = 512l
//let NUM_EMB_TKN = 2L
//let H = 768L

type Config = 
    {
      attention_probs_dropout_prob: float //0.1,
      gradient_checkpointing: bool// false,
      hidden_act: torch.nn.Activations// gelu,
      hidden_dropout_prob: float// 0.1,
      hidden_size: int64 //768,
      initializer_range: float// 0.02,
      intermediate_size: int64 //3072,
      layer_norm_eps: float // 1e-12,
      max_position_embeddings: int64// 512,
      model_type: string //bert,
      num_attention_heads: int64 // 12,
      num_hidden_layers: int64 //12,
      pad_token_id: int64 //0,
      position_embedding_type: string// absolute,
      transformers_version: string //4.6.0.dev0,
      type_vocab_size: int64 //2,
      use_cache: bool //true,
      vocab_size: int64 //30522    
    }

let cfg = 
    {
      attention_probs_dropout_prob  = 0.1
      gradient_checkpointing        = false
      hidden_act                    = torch.nn.Activations.GELU// gelu,
      hidden_dropout_prob           = 0.1
      hidden_size                   = 768L
      initializer_range             = 0.02
      intermediate_size             = 3072L
      layer_norm_eps                = 1e-12
      max_position_embeddings       = 512
      model_type                    = "bert"
      num_attention_heads           = 12L
      num_hidden_layers             = 12L
      pad_token_id                  = 0L
      position_embedding_type       = "absolute"
      transformers_version          = "4.6.0.dev0"
      type_vocab_size               = 2L
      use_cache                     = true
      vocab_size                    = 30522L
    }


type BertEmbeddings(cfg:Config) as this =
    inherit nn.Module("embeddings") 

    let word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size,padding_idx=0L)
    let position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
    let token_type_embeddigns = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
    let LayerNorm = nn.LayerNorm([|cfg.hidden_size|], eps=cfg.layer_norm_eps)
    let position_ids = torch.arange(cfg.max_position_embeddings).expand([|1L;-1L|])
    let token_type_ids =  torch.zeros(position_ids.size(),dtype=torch.long,device=position_ids.device)
    do   
        this.RegisterComponents()
        
    member _.forward(?input_ids,?token_type_ids',?position_ids',?input_embeds:torch.Tensor,?past_key_values_length) =
        let past_key_values_length = defaultArg past_key_values_length 0L
        let input_shape = 
            input_ids 
            |> Option.map(fun (t:torch:Tensor) -> t.size()) 
            |> Option.defaultValue(input_embeds.Value.size().[..^1])

        let seq_length = input_shape.[1]
           
        let position_ids = 
            defaultArg 
                position_ids' 
                (position_ids.index(torch.TensorIndex.c,torch.TensorIndex.Slice(start=past_key_values_length, stop=seq_length + past_key_values_length)))

        




        //f position_ids is None:
        //position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

