use anyhow::Result;
use rand::Rng;
use crate::llm::infer::model::Tensor;
use crate::llm::infer::tokenizer::Tokenizer;
use crate::llm::infer::{ops, gguf};

pub struct InferenceEngine {
    model: super::model::Model,
    tokenizer: Tokenizer,
    kv_cache: Vec<f32>,
    n_past: usize,
    max_seq_len: usize,
    act: Vec<f32>,
    weights: Vec<f32>,
}

fn tn(layer: i64, base: &str) -> String {
    format!("blk.{}.{}.weight", layer, base)
}

fn dequant_tensor(t: &Tensor, weights: &mut [f32]) {
    t.dequantize_to_f32(weights);
}

fn forward_layer(
    model: &super::model::Model,
    kv_cache: &mut [f32],
    weights: &mut [f32],
    n_past: usize,
    max_seq_len: usize,
    layer: i64,
    hidden: &mut [f32], scores: &mut [f32], attn_out: &mut [f32],
    residual: &mut [f32], q_buf: &mut [f32], k_buf: &mut [f32],
    v_buf: &mut [f32], gate_buf: &mut [f32], up_buf: &mut [f32],
) {
    let n_embd = model.n_embd as usize;
    let n_head = model.n_head as usize;
    let n_kv_head = model.n_head_kv as usize;
    let head_dim = model.n_embd_head_k as usize;
    let n_ff = model.n_ff as usize;
    let q_size = n_head * head_dim;
    let kv_size = n_kv_head * head_dim;

    residual.copy_from_slice(&hidden[..n_embd]);

    let name = tn(layer, "attn_norm");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::rms_norm_inplace(hidden, &weights[..n_embd], n_embd, 1, model.norm_eps);
    }

    let name = tn(layer, "attn_q");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::matmul_nt(q_buf, hidden, weights, 1, q_size, n_embd);
    }
    let name = tn(layer, "attn_k");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::matmul_nt(k_buf, hidden, weights, 1, kv_size, n_embd);
    }
    let name = tn(layer, "attn_v");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::matmul_nt(v_buf, hidden, weights, 1, kv_size, n_embd);
    }

    ops::rope(q_buf, q_size, n_head, n_past, 1, model.rope_freq_base);
    ops::rope(k_buf, kv_size, n_kv_head, n_past, 1, model.rope_freq_base);

    let k_slot_start = layer as usize * 2 * max_seq_len * n_embd;
    let (k_slot, v_slot) = kv_cache[k_slot_start..].split_at_mut(max_seq_len * n_embd);

    k_slot[n_past * n_embd..n_past * n_embd + kv_size].copy_from_slice(&k_buf[..kv_size]);
    v_slot[n_past * n_embd..n_past * n_embd + kv_size].copy_from_slice(&v_buf[..kv_size]);

    let s = n_past + 1;
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let q_per_kv = n_head / n_kv_head;

    for h in 0..n_head {
        let h_kv = h / q_per_kv;
        for ss in 0..s {
            let mut sum = 0.0;
            for d in 0..head_dim {
                sum += q_buf[h * head_dim + d] * k_slot[ss * n_embd + h_kv * head_dim + d];
            }
            scores[h * s + ss] = sum * inv_scale;
        }
    }

    for h in 0..n_head {
        let offset = h * s;
        let mut maxv = scores[offset];
        for ss in 0..s {
            if ss > n_past { scores[offset + ss] = f32::NEG_INFINITY; }
            else if scores[offset + ss] > maxv { maxv = scores[offset + ss]; }
        }
        let mut sum = 0.0;
        for ss in 0..=n_past {
            scores[offset + ss] = (scores[offset + ss] - maxv).exp();
            sum += scores[offset + ss];
        }
        let inv = 1.0 / sum;
        for ss in 0..s { scores[offset + ss] *= inv; }
    }

    attn_out.fill(0.0);
    for h in 0..n_head {
        let h_kv = h / q_per_kv;
        for ss in 0..=n_past {
            let w = scores[h * s + ss];
            for d in 0..head_dim {
                attn_out[h * head_dim + d] += w * v_slot[ss * n_embd + h_kv * head_dim + d];
            }
        }
    }

    let name = tn(layer, "attn_output");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::matmul_nt(gate_buf, attn_out, weights, 1, n_embd, n_embd);
        attn_out.copy_from_slice(&gate_buf[..n_embd]);
    }

    for i in 0..n_embd {
        hidden[i] += attn_out[i];
    }
    residual.copy_from_slice(&hidden[..n_embd]);

    let name = tn(layer, "ffn_norm");
    if let Some(t) = model.tensors.get(&name) {
        dequant_tensor(t, weights);
        ops::rms_norm_inplace(hidden, &weights[..n_embd], n_embd, 1, model.norm_eps);
    }

    let gw_exists = model.tensors.contains_key(&tn(layer, "ffn_gate"));
    let uw_exists = model.tensors.contains_key(&tn(layer, "ffn_up"));
    if gw_exists && uw_exists {
        let name = tn(layer, "ffn_gate");
        if let Some(t) = model.tensors.get(&name) {
            dequant_tensor(t, weights);
        }
        ops::matmul_nt(gate_buf, hidden, weights, 1, n_ff, n_embd);
        let name = tn(layer, "ffn_up");
        if let Some(t) = model.tensors.get(&name) {
            dequant_tensor(t, weights);
        }
        ops::matmul_nt(up_buf, hidden, weights, 1, n_ff, n_embd);
        ops::silu_inplace(gate_buf, n_ff);
        for i in 0..n_ff { gate_buf[i] *= up_buf[i]; }

        let name = tn(layer, "ffn_down");
        if let Some(t) = model.tensors.get(&name) {
            dequant_tensor(t, weights);
            ops::matmul_nt(hidden, gate_buf, weights, 1, n_embd, n_ff);
        }
        for i in 0..n_embd {
            hidden[i] += residual[i];
        }
    }
}

impl InferenceEngine {
    pub fn new(model: super::model::Model, tokenizer: Tokenizer, max_seq_len: usize) -> Self {
        let scratch = (model.n_embd * 3 + model.n_head * model.n_embd_head_k * 2
            + model.n_ff * 2 + model.n_head * max_seq_len as i64 + model.n_embd) as usize;
        let act = vec![0.0f32; scratch];

        let max_tensor = model.tensors.values().map(|t| t.nelements() as usize).max().unwrap_or(1);
        let weights = vec![0.0f32; max_tensor];

        let kv_cache_size = (model.n_layer * 2 * max_seq_len as i64 * model.n_embd) as usize;
        let kv_cache = vec![0.0f32; kv_cache_size];

        InferenceEngine { model, tokenizer, kv_cache, n_past: 0, max_seq_len, act, weights }
    }

    fn forward(&mut self, token_id: u32, logits: &mut [f32]) -> Result<()> {
        let n_embd = self.model.n_embd as usize;
        let n_layers = self.model.n_layer as usize;
        let s = self.n_past + 1;

        anyhow::ensure!(s <= self.max_seq_len, "Max sequence length exceeded");

        let emb = {
            let emb = self.model.tensors.get("token_embd.weight")
                .or_else(|| self.model.tensors.get("tok_embeddings.weight"))
                .or_else(|| self.model.tensors.get("gpt.embd.weight"))
                .ok_or_else(|| anyhow::anyhow!("No token embedding tensor found"))?;

            let emb_row_size = emb.dims[0] as usize;
            match emb.ty {
                gguf::GgmlType::F32 => {
                    let src = bytemuck::cast_slice::<u8, f32>(&emb.data);
                    src[token_id as usize * emb_row_size..(token_id as usize + 1) * emb_row_size].to_vec()
                },
                _ => {
                    emb.dequantize_to_f32(&mut self.weights);
                    self.weights[token_id as usize * emb_row_size..(token_id as usize + 1) * emb_row_size].to_vec()
                }
            }
        };

        let n_head = self.model.n_head as usize;
        let n_kv_head = self.model.n_head_kv as usize;
        let head_dim = self.model.n_embd_head_k as usize;
        let q_size = n_head * head_dim;
        let kv_size = n_kv_head * head_dim;
        let n_ff = self.model.n_ff as usize;

        let (act_head, rest) = self.act.split_at_mut(n_embd);
        let (residual, rest) = rest.split_at_mut(n_embd);
        let (q_buf, rest) = rest.split_at_mut(q_size);
        let (k_buf, rest) = rest.split_at_mut(kv_size);
        let (v_buf, rest) = rest.split_at_mut(kv_size);
        let (gate_buf, rest) = rest.split_at_mut(n_ff);
        let (up_buf, rest) = rest.split_at_mut(n_ff);
        let (scores_h, attn_out_h) = rest.split_at_mut(n_head * self.max_seq_len);

        act_head.copy_from_slice(&emb);

        for layer in 0..n_layers {
            forward_layer(
                &self.model, &mut self.kv_cache, &mut self.weights,
                self.n_past, self.max_seq_len,
                layer as i64,
                act_head, scores_h, attn_out_h, residual,
                q_buf, k_buf, v_buf, gate_buf, up_buf,
            );
        }

        {
            let norm_w = self.model.tensors.get("output_norm.weight")
                .or_else(|| self.model.tensors.get("norm.weight"));
            if let Some(nw) = norm_w {
                nw.dequantize_to_f32(&mut self.weights);
                ops::rms_norm_inplace(act_head, &self.weights[..n_embd], n_embd, 1, self.model.norm_eps);
            }
        }

        {
            let output_w = self.model.tensors.get("output.weight")
                .or_else(|| self.model.tensors.get("token_embd.weight"));
            if let Some(ow) = output_w {
                ow.dequantize_to_f32(&mut self.weights);
                let n_out = ow.dims[1] as usize;
                let k_dim = ow.dims[0] as usize;
                ops::matmul_nt(logits, act_head, &self.weights, 1, n_out, k_dim);
            }
        }

        self.n_past += 1;
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize, temperature: f32, top_k: usize) -> Result<String> {
        let input_tokens = self.tokenizer.encode(prompt, 512);
        if input_tokens.is_empty() {
            anyhow::bail!("Failed to tokenize prompt");
        }

        let n_vocab = self.model.n_vocab as usize;
        let mut logits = vec![0.0f32; n_vocab];
        let mut output = String::new();

        for &tok in &input_tokens {
            self.forward(tok, &mut logits)?;
        }

        let mut rng = rand::thread_rng();

        for _ in 0..max_tokens {
            let mut last_token;
            if temperature < 0.01 {
                let idx = logits.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                last_token = idx as u32;
            } else {
                for l in logits.iter_mut() { *l /= temperature; }
                if top_k > 0 && top_k < n_vocab {
                    let mut scored: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                    scored.select_nth_unstable_by(top_k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                    let threshold = scored[top_k - 1].0;
                    for (_i, v) in logits.iter_mut().enumerate() {
                        if *v < threshold { *v = f32::NEG_INFINITY; }
                    }
                }
                let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for l in logits.iter_mut() { *l = (*l - maxv).exp(); sum += *l; }
                let inv = 1.0 / sum;
                for l in logits.iter_mut() { *l *= inv; }

                let r: f32 = rng.gen();
                let mut cum = 0.0;
                last_token = (n_vocab - 1) as u32;
                for (j, &v) in logits.iter().enumerate() {
                    cum += v;
                    if r < cum { last_token = j as u32; break; }
                }
            }

            if last_token == self.tokenizer.eos_id { break; }
            let piece = self.tokenizer.decode(&[last_token]);
            output.push_str(&piece);
            print!("{}", piece);
            use std::io::Write;
            std::io::stdout().flush().ok();

            self.forward(last_token, &mut logits)?;
        }
        println!();
        Ok(output)
    }
}
