use std::cell::RefCell;

use half::f16;
use serde::Deserialize;

use mlx_derive::Module;
use mlx_nn::activations::Activation;
use mlx_nn::embedding::Embedding;
use mlx_nn::layer_norm::LayerNorm;
use mlx_nn::linear::Linear;
use mlx_nn::mlp::MLP;
use mlx_rust::fast::{fast_RoPE, fast_scaled_dot_product_attention};
use mlx_rust::MLXArray;
use mlx_rust::module::Module;
use mlx_rust::r#type::MlxType;
use mlx_rust::stream::get_default_stream;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f64,
    pub qk_layernorm: bool,
}

impl Config {
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Clone, Debug, Module)]
pub struct Model {
    #[param(rename = "model.embed_tokens")]
    embed_tokens: Embedding,
    #[param(rename = "model.layers")]
    layers: Vec<DecoderLayer>,
    #[param(rename = "model.final_layernorm")]
    final_layernorm: LayerNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new<T: MlxType>(cfg: &Config) -> Self {
        let embed_tokens = Embedding::new::<T>(cfg.vocab_size, cfg.hidden_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayer::new::<T>(cfg))
            .collect();
        let final_layernorm =
            LayerNorm::new::<T>(cfg.hidden_size as i32, true, cfg.layer_norm_eps as f32);
        let lm_head = Linear::new::<T>(cfg.hidden_size, cfg.vocab_size, true);
        Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
        }
    }
}

impl Model {
    pub fn fwd(&self, x: MLXArray) -> MLXArray {
        let (_b_size, seq_len) = (x.dim(0), x.dim(1));
        let mut xs = self.embed_tokens.forward(x);
        // println!("xs embed_tokens: {}", xs);
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::get_mask(seq_len as usize))
        };
        // println!("mask: {:?}", mask);
        // xs.eval();
        for layer in &self.layers {
            let start = std::time::Instant::now();
            xs = layer.forward((xs, mask.clone()));
            // xs.eval();
            // println!("decode time: {:?}", start.elapsed())
        }
        let xs = self.final_layernorm.forward(xs);
        self.lm_head.forward(xs)
    }

    fn get_mask(size: usize) -> MLXArray {
        let mask = MLXArray::arange::<f32>(0.0..(size as f64), 1.0, get_default_stream());
        let mask = mask.reshape(&[size as  i32, 1]).less_with_stream(
            &mask.reshape(&[1, size as i32]),
            get_default_stream()
        );
        mask.as_type::<i32>() * (-1e9)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (MLXArray, Option<MLXArray>))]
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
    pub fn new<T: MlxType>(cfg: &Config) -> Self {
        let self_attn = Attention::new::<T>(cfg);
        let mlp = MLP::new::<T>(
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
            cfg.hidden_act.clone(),
        );
        let input_layernorm =
            LayerNorm::new::<T>(cfg.hidden_size as i32, true, cfg.layer_norm_eps as f32);
        Self {
            self_attn,
            mlp,
            input_layernorm,
        }
    }
    pub fn fwd(&self, (x, mask): (MLXArray, Option<MLXArray>)) -> MLXArray {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let attn_outputs = self.self_attn.forward((x.clone(), mask));
        // x.eval();
        // mask.clone().map(|m|m.eval());
        // let instant = std::time::Instant::now();
        let x = x.as_type::<f16>();
        let feed_forward_hidden_states = self.mlp.forward(x);
        // attn_outputs.eval();
        // feed_forward_hidden_states.eval();
        // println!("attention time: {:?}", instant.elapsed());
        attn_outputs + feed_forward_hidden_states + residual
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (MLXArray, Option<MLXArray>))]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    // q_layernorm: Option<LayerNorm>,
    // k_layernorm: Option<LayerNorm>,
    #[param(skip)]
    rotary_emb: RotaryEmbedding,
    #[param(skip)]
    softmax_scale: f64,
    #[param(skip)]
    num_heads: usize,
    #[param(skip)]
    num_kv_heads: usize,
    #[param(skip)]
    head_dim: usize,
    #[param(skip)]
    kv_cache: RefCell<Option<(MLXArray, MLXArray)>>,
}

impl Attention {
    pub fn new<T: MlxType>(cfg: &Config) -> Self {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let q_proj = Linear::new::<T>(cfg.hidden_size, num_heads * head_dim, true);
        let k_proj = Linear::new::<T>(cfg.hidden_size, num_kv_heads * head_dim, true);
        let v_proj = Linear::new::<T>(cfg.hidden_size, num_kv_heads * head_dim, true);
        let dense = Linear::new::<T>(num_heads * head_dim, cfg.hidden_size, true);
        // Alternative rope scaling are not supported.
        let dim = (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize;
        let rotary_emb = RotaryEmbedding::new(dim);
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            // q_layernorm,
            // k_layernorm,
            rotary_emb,
            softmax_scale,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: RefCell::new(None),
        }
    }

    pub fn fwd(&self, (x, mask): (MLXArray, Option<MLXArray>)) -> MLXArray {
        // x.eval();

        // let start_gen = std::time::Instant::now();
        let query = self.q_proj.forward(x.clone());
        // query.eval();
        // println!("attention time: {:?}", start_gen.elapsed());
        let key = self.k_proj.forward(x.clone());
        let value = self.v_proj.forward(x.clone());

        let (b_size, seq_len) = (x.dim(0), x.dim(1));


        let query_states = query
            .reshape(&[
                b_size,
                seq_len,
                self.num_heads as i32,
                self.head_dim as i32
            ])
            .transpose(&[0, 2, 1, 3]);

        let key_states = key
            .reshape(&[
                b_size,
                seq_len,
                self.num_kv_heads as i32,
                self.head_dim as i32,
            ])
            .transpose(&[0, 2, 1, 3]);

        let value_states = value
            .reshape(&[
                b_size,
                seq_len,
                self.num_kv_heads as i32,
                self.head_dim as i32,
            ])
            .transpose(&[0, 2, 1, 3]);


        let kv_cache = self.kv_cache.borrow();
        let (query_states, key_states, value_states) = match &*kv_cache {
            None => {
                let query_states = self.rotary_emb.forward((query_states, 0));
                let key_states = self.rotary_emb.forward((key_states, 0));
                (query_states, key_states, value_states)
            }
            Some((key_cache, value_cache)) => {
                let offset = key_cache.dim(2) as usize;
                let query_states = self.rotary_emb.forward((query_states, offset));
                let key_states = self.rotary_emb.forward((key_states, offset));
                let k = MLXArray::cat(
                    (key_cache.clone(), key_states.clone()),
                    2,
                    get_default_stream(),
                );
                let v = MLXArray::cat(
                    (value_cache.clone(), value_states.clone()),
                    2,
                    get_default_stream(),
                );
                (query_states, k, v)
            }
        };


        // let seqlen_offset = match &*kv_cache {
        //     None => 0,
        //     Some((prev_k, _)) => prev_k.dim(2) as usize,
        // };
        // let query_states = self.rotary_emb.forward((query_states, seqlen_offset));
        // let key_states = self.rotary_emb.forward((key_states, seqlen_offset));

        // // KV cache.
        // let (key_states, value_states) = match &*kv_cache {
        //     None => (key_states, value_states),
        //     Some((prev_k, prev_v)) => {
        //         let k = MLXArray::cat(
        //             (prev_k.clone(), key_states.clone()),
        //             2,
        //             get_default_stream(),
        //         );
        //         let v = MLXArray::cat(
        //             (prev_v.clone(), value_states.clone()),
        //             2,
        //             get_default_stream(),
        //         );
        //         (k, v)
        //     }
        // };
        drop(kv_cache);
        self.kv_cache
            .replace(Some((key_states.clone(), value_states.clone())));

        // query_states.eval();
        // key_states.eval();
        // value_states.eval();


        // Finally perform the attention computation
        let scale = f32::sqrt(1.0 / query_states.dim(-1) as f32);

        // let query_states = query_states.as_type::<f32>();
        // let score = (query_states * scale).matmul(key_states.swap_axes(-1, -2), get_default_stream());
        // let score = match mask {
        //     None => { score}
        //     Some(mask) => {
        //         score + mask
        //     }
        // };
        // let score = soft_max(score, &[-1]);
        // let output = score.matmul(value_states, get_default_stream())
        //     .transpose(&[0, 2, 1, 3])
        //     .reshape(&[b_size, seq_len, -1]);


        let output = fast_scaled_dot_product_attention(
            // query_states,
            query_states.as_type::<f32>(),
            key_states,
            value_states,
            scale,
            mask,
            get_default_stream(),
        )
            .as_type::<f16>()
            .transpose(&[0, 2, 1, 3])
            .reshape(&[b_size, seq_len, -1]);
        let r = self.dense.forward(output);
        r
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (MLXArray, usize), trainable = false)]
struct RotaryEmbedding {
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl RotaryEmbedding {
    pub fn fwd(&self, (x, offset): (MLXArray, usize)) -> MLXArray {
        let shape = x.shape();
        let x = x.reshape(&[-1, x.dim(-2), x.dim(-1)]);
        let x = fast_RoPE(
            x,
            self.dim.clone() as i32,
            false,
            10_000.0,
            1.0,
            offset as i32,
            get_default_stream(),
        );
        return x.reshape(shape);
    }
}
