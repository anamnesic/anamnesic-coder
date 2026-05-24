use std::f32::consts::E;

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], n: usize, rows: usize, eps: f32) {
    for r in 0..rows {
        let offset = r * n;
        let mut ss = 0.0f32;
        for i in 0..n { ss += x[offset + i] * x[offset + i]; }
        let s = 1.0 / (ss / n as f32 + eps).sqrt();
        for i in 0..n { out[offset + i] = x[offset + i] * s * weight[i]; }
    }
}

pub fn silu(out: &mut [f32], x: &[f32], n: usize) {
    for i in 0..n { out[i] = x[i] / (1.0 + (-x[i]).exp()); }
}

pub fn matmul(dst: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            dst[i * n + j] = sum;
        }
    }
}

pub fn matmul_nt(dst: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[j * k + kk];
            }
            dst[i * n + j] = sum;
        }
    }
}

pub fn rope(x: &mut [f32], n_embd: usize, n_head: usize, pos: usize, n_tokens: usize, freq_base: f32) {
    let head_dim = n_embd / n_head;
    for t in 0..n_tokens {
        for h in 0..n_head {
            for hh in 0..head_dim / 2 {
                let theta = pos as f32 * freq_base.powf(-2.0 * hh as f32 / head_dim as f32);
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let row = &mut x[t * n_embd + h * head_dim..];
                let v0 = row[hh];
                let v1 = row[hh + head_dim / 2];
                row[hh] = v0 * cos_t - v1 * sin_t;
                row[hh + head_dim / 2] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

pub fn softmax(x: &mut [f32], n: usize, rows: usize) {
    for r in 0..rows {
        let row = &mut x[r * n..(r + 1) * n];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for i in 0..n { row[i] = (row[i] - maxv).exp(); sum += row[i]; }
        let inv = 1.0 / sum;
        for i in 0..n { row[i] *= inv; }
    }
}

pub fn add(dst: &mut [f32], a: &[f32], b: &[f32], n: usize) {
    for i in 0..n { dst[i] = a[i] + b[i]; }
}
