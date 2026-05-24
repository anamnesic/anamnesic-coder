use std::collections::HashMap;
use crate::llm::infer::gguf::{GgufReader, GgmlType};

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (h & 0x03FF) as u32;
    if exp <= 0 {
        let m = (mant | 0x0400) >> (1 - exp);
        let u = sign | (m << 13);
        return f32::from_le_bytes(u.to_le_bytes());
    }
    if exp >= 255 {
        let u = sign | 0x7F800000 | (mant << 13);
        return f32::from_le_bytes(u.to_le_bytes());
    }
    let u = sign | ((exp as u32) << 23) | (mant << 13);
    f32::from_le_bytes(u.to_le_bytes())
}

fn dequantize_q4_0_row(block: &[u8], out: &mut [f32], n: i64) {
    let num_blocks = (n + 31) / 32;
    for b in 0..num_blocks {
        let d_bits = u16::from_le_bytes(block[b as usize * 18..b as usize * 18 + 2].try_into().unwrap());
        let d = half_to_f32(d_bits);
        for i in 0..16 {
            let q = block[b as usize * 18 + 2 + i as usize];
            let idx_base = b * 32 + i * 2;
            let q0 = (((q & 0x0F) as i8) << 4) as f32 * 0.0625f32;
            let q1 = (((q & 0xF0) as i8)) as f32 * 0.0625f32;
            if idx_base < n { out[idx_base as usize] = q0 * d; }
            if idx_base + 1 < n { out[(idx_base + 1) as usize] = q1 * d; }
        }
    }
}

fn dequantize_q8_0_row(block: &[u8], out: &mut [f32], n: i64) {
    let num_blocks = (n + 31) / 32;
    for b in 0..num_blocks {
        let d_bits = u16::from_le_bytes(block[b as usize * 34..b as usize * 34 + 2].try_into().unwrap());
        let d = half_to_f32(d_bits);
        for i in 0..32 {
            let q = block[b as usize * 34 + 2 + i as usize] as i8;
            if b * 32 + i as i64 < n {
                out[(b * 32 + i as i64) as usize] = (q as f32) * d;
            }
        }
    }
}

fn dequantize_f16_row(block: &[u8], out: &mut [f32], n: i64) {
    for i in 0..n as usize {
        let h = u16::from_le_bytes(block[i * 2..i * 2 + 2].try_into().unwrap());
        out[i] = half_to_f32(h);
    }
}

pub struct Tensor {
    pub name: String,
    pub ty: GgmlType,
    pub dims: Vec<i64>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn nelements(&self) -> i64 {
        let mut n = 1;
        for &d in &self.dims { n *= d; }
        n
    }

    pub fn dequantize_to_f32(&self, out: &mut [f32]) {
        let n = self.nelements();
        match self.ty {
            GgmlType::F32 => {
                let src = bytemuck::cast_slice::<u8, f32>(&self.data);
                out[..n as usize].copy_from_slice(&src[..n as usize]);
            },
            GgmlType::F16 => dequantize_f16_row(&self.data, out, n),
            GgmlType::Q4_0 => {
                for row in (0..n).step_by(self.dims[0] as usize) {
                    let row_size = self.dims[0];
                    let src = &self.data[(row / self.dims[0] as i64) as usize * ((row_size + 31) / 32) as usize * 18..];
                    dequantize_q4_0_row(src, &mut out[row as usize..], row_size);
                }
            },
            GgmlType::Q8_0 => {
                for row in (0..n).step_by(self.dims[0] as usize) {
                    let row_size = self.dims[0];
                    let src = &self.data[(row / self.dims[0] as i64) as usize * ((row_size + 31) / 32) as usize * 34..];
                    dequantize_q8_0_row(src, &mut out[row as usize..], row_size);
                }
            },
            _ => {
                log::warn!("Unsupported type for dequantization: {:?}", self.ty);
                for o in out.iter_mut().take(n as usize) { *o = 0.0; }
            }
        }
    }
}

pub struct Model {
    pub n_vocab: i64,
    pub n_embd: i64,
    pub n_mult: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub n_layer: i64,
    pub n_ff: i64,
    pub norm_eps: f32,
    pub n_embd_head_k: i64,
    pub n_embd_head_v: i64,
    pub n_expert: i64,
    pub n_expert_used: i64,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub tensors: HashMap<String, Tensor>,
}

impl Model {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let reader = GgufReader::load(path)?;

        let n_vocab = reader.get_metadata_int("llama.vocab_size",
            reader.get_metadata_int("tokenizer.ggml.vocab_size", 32000));
        let n_embd = reader.get_metadata_int("llama.embedding_length",
            reader.get_metadata_int("llama.n_embd", 4096));
        let n_mult = reader.get_metadata_int("llama.feed_forward_length",
            reader.get_metadata_int("llama.n_mult", 256));
        let n_head = reader.get_metadata_int("llama.attention.head_count",
            reader.get_metadata_int("llama.n_head", 32));
        let n_head_kv = reader.get_metadata_int("llama.attention.head_count_kv",
            reader.get_metadata_int("llama.n_head_kv", n_head));
        let n_layer = reader.get_metadata_int("llama.block_count",
            reader.get_metadata_int("llama.n_layer", 32));
        let n_ff = reader.get_metadata_int("llama.feed_forward_length",
            reader.get_metadata_int("llama.n_ff", 4 * n_embd));
        let norm_eps = reader.get_metadata_float("llama.attention.layer_norm_rms_epsilon",
            reader.get_metadata_float("llama.norm_eps", 1e-5)) as f32;
        let n_expert = reader.get_metadata_int("llama.expert_count", 0);
        let n_expert_used = reader.get_metadata_int("llama.expert_used_count", 0);
        let rope_freq_base = reader.get_metadata_float("llama.rope.freq_base", 10000.0) as f32;
        let rope_freq_scale = reader.get_metadata_float("llama.rope.freq_scale", 1.0) as f32;

        let n_embd_head_k = n_embd / n_head;
        let n_embd_head_v = n_embd_head_k;

        let mut tensors = HashMap::new();
        for (name, info) in &reader.tensors {
            if let Some(data) = reader.tensor_data(name) {
                let t = Tensor {
                    name: name.clone(),
                    ty: info.ty,
                    dims: info.dims.clone(),
                    data: data.to_vec(),
                };
                tensors.insert(name.clone(), t);
            }
        }

        log::info!("Model: vocab={} embd={} head={} layers={} ff={} n_head_kv={}",
            n_vocab, n_embd, n_head, n_layer, n_ff, n_head_kv);
        log::info!("Loaded {} tensors", tensors.len());

        Ok(Model {
            n_vocab, n_embd, n_mult, n_head, n_head_kv, n_layer, n_ff,
            norm_eps, n_embd_head_k, n_embd_head_v, n_expert, n_expert_used,
            rope_freq_base, rope_freq_scale, tensors,
        })
    }
}
