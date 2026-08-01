use std::collections::HashMap;
use std::io::Read;
use anyhow::{Result, Context};

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
    IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27,
    F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 31, TQ2_0 = 32, NVFP4 = 33,
}

pub fn ggml_blck_size(t: GgmlType) -> usize {
    match t {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I8
        | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 | GgmlType::F64 => 1,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1
        | GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::IQ4_NL => 32,
        GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K
        | GgmlType::Q6_K | GgmlType::Q8_K | GgmlType::IQ2_XXS | GgmlType::IQ2_XS
        | GgmlType::IQ3_XXS | GgmlType::IQ1_S | GgmlType::IQ3_S | GgmlType::IQ2_S
        | GgmlType::IQ4_XS | GgmlType::IQ1_M | GgmlType::TQ1_0 | GgmlType::TQ2_0 => 256,
        GgmlType::NVFP4 => 64,
    }
}

pub fn ggml_type_size(t: GgmlType) -> usize {
    match t {
        GgmlType::F32 => 4, GgmlType::F16 => 2, GgmlType::BF16 => 2, GgmlType::I8 => 1,
        GgmlType::I16 => 2, GgmlType::I32 => 4, GgmlType::I64 => 8, GgmlType::F64 => 8,
        GgmlType::Q4_0 => 18, GgmlType::Q4_1 => 20, GgmlType::Q5_0 => 22, GgmlType::Q5_1 => 24,
        GgmlType::Q8_0 => 34, GgmlType::Q8_1 => 36,
        GgmlType::Q2_K => 2 + 64 + 2, GgmlType::Q3_K => 2 + 32 + 2 + 64,
        GgmlType::Q4_K => 2 + 2 + 12 + 128, GgmlType::Q5_K => 2 + 2 + 12 + 32 + 128,
        GgmlType::Q6_K => 128 + 64 + 16 + 2, GgmlType::Q8_K => 4 + 256 + 32,
        GgmlType::IQ2_XXS | GgmlType::IQ2_XS => 2 + 4 + 32 + 64,
        GgmlType::IQ3_XXS => 2 + 4 + 64 + 32, GgmlType::IQ1_S => 2 + 4 + 32 + 32,
        GgmlType::IQ3_S => 2 + 4 + 64 + 32, GgmlType::IQ2_S => 2 + 4 + 32 + 64,
        GgmlType::IQ4_XS => 2 + 4 + 128 + 64, GgmlType::IQ1_M => 2 + 4 + 32 + 32,
        GgmlType::IQ4_NL => 2 + 16,
        GgmlType::NVFP4 => 4 + 32,
        _ => 0,
    }
}

pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<i64>,
    pub ty: GgmlType,
    pub offset: u64,
}

pub struct GgufReader {
    pub data: Vec<u8>,
    pub alignment: u64,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub metadata_str: HashMap<String, String>,
    pub metadata_int: HashMap<String, i64>,
    pub metadata_float: HashMap<String, f64>,
    pub tensor_data_offset: usize,
}

impl GgufReader {
    pub fn load(path: &str) -> Result<Self> {
        let mut file = std::fs::File::open(path).context("Failed to open GGUF file")?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut pos = 0usize;
        let mut rdr = GgufReader { data, alignment: 32, tensors: HashMap::new(), metadata_str: HashMap::new(), metadata_int: HashMap::new(), metadata_float: HashMap::new(), tensor_data_offset: 0 };

        let magic = rdr.read_u32(&mut pos);
        anyhow::ensure!(magic == GGUF_MAGIC, "Not a valid GGUF file");
        let _version = rdr.read_u32(&mut pos);

        let tensor_count = rdr.read_u64(&mut pos) as usize;
        let kv_count = rdr.read_u64(&mut pos) as usize;

        for _ in 0..kv_count {
            let key = rdr.read_string(&mut pos);
            let val_type_i = rdr.read_i32(&mut pos);
            match val_type_i {
                0 => { rdr.metadata_uint32_insert(&key, rdr.read_u8(&mut pos) as u32); },
                1 => { rdr.metadata_int.insert(key.clone(), rdr.read_i8(&mut pos) as i64); },
                2 => { rdr.metadata_uint32_insert(&key, rdr.read_u16(&mut pos) as u32); },
                3 => { rdr.metadata_int.insert(key.clone(), rdr.read_i16(&mut pos) as i64); },
                4 => { rdr.metadata_uint32_insert(&key, rdr.read_u32(&mut pos)); },
                5 => { rdr.metadata_int.insert(key.clone(), rdr.read_i32(&mut pos) as i64); },
                6 => { rdr.metadata_float.insert(key.clone(), rdr.read_f32(&mut pos) as f64); },
                7 => { rdr.metadata_int.insert(key.clone(), rdr.read_u8(&mut pos) as i64); },
                8 => { rdr.metadata_str.insert(key.clone(), rdr.read_string(&mut pos)); },
                10 | 11 => { rdr.metadata_int.insert(key.clone(), rdr.read_i64(&mut pos)); },
                12 => { rdr.metadata_float.insert(key.clone(), rdr.read_f64(&mut pos)); },
                9 => {
                    let arr_type = rdr.read_i32(&mut pos);
                    let arr_n = rdr.read_u64(&mut pos) as usize;
                    for i in 0..arr_n {
                        match arr_type {
                            0 | 1 | 7 => { rdr.read_u8(&mut pos); },
                            2 | 3     => { rdr.read_u16(&mut pos); },
                            4 | 5 | 6 => {
                                let v = rdr.read_u32(&mut pos);
                                rdr.metadata_int.insert(format!("{}_{}", key, i), v as i64);
                            },
                            8 => {
                                let s = rdr.read_string(&mut pos);
                                rdr.metadata_str.insert(format!("{}_{}", key, i), s);
                            },
                            10 | 11 | 12 => {
                                let v = rdr.read_u64(&mut pos);
                                rdr.metadata_int.insert(format!("{}_{}", key, i), v as i64);
                            },
                            _ => {},
                        }
                    }
                },
                _ => {},
            }
            if key == "general.alignment" {
                rdr.alignment = rdr.metadata_int.get("general.alignment").copied().unwrap_or(32) as u64;
            }
        }

        for _ in 0..tensor_count {
            let name = rdr.read_string(&mut pos);
            let n_dims = rdr.read_u32(&mut pos) as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(rdr.read_i64(&mut pos));
            }
            let ty_i = rdr.read_i32(&mut pos);
            let ty = unsafe { std::mem::transmute::<i32, GgmlType>(ty_i) };
            let offset = rdr.read_u64(&mut pos);
            rdr.tensors.insert(name.clone(), GgufTensorInfo { name, dims, ty, offset });
        }

        let align = rdr.alignment as usize;
        rdr.tensor_data_offset = (pos + align - 1) & !(align - 1);

        Ok(rdr)
    }

    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.tensor_data_offset + info.offset as usize;
        let blck = ggml_blck_size(info.ty);
        let ts = ggml_type_size(info.ty);
        let n_blocks = (info.nelements() + blck as i64 - 1) / blck as i64;
        let nbytes = n_blocks as usize * ts;
        Some(&self.data[start..start + nbytes])
    }

    fn read_u8(&self, pos: &mut usize) -> u8 { let v = self.data[*pos]; *pos += 1; v }
    fn read_i8(&self, pos: &mut usize) -> i8 { let v = self.data[*pos] as i8; *pos += 1; v }
    fn read_u16(&self, pos: &mut usize) -> u16 { let v = u16::from_le_bytes(self.data[*pos..*pos+2].try_into().unwrap()); *pos += 2; v }
    fn read_i16(&self, pos: &mut usize) -> i16 { let v = i16::from_le_bytes(self.data[*pos..*pos+2].try_into().unwrap()); *pos += 2; v }
    fn read_u32(&self, pos: &mut usize) -> u32 { let v = u32::from_le_bytes(self.data[*pos..*pos+4].try_into().unwrap()); *pos += 4; v }
    fn read_i32(&self, pos: &mut usize) -> i32 { let v = i32::from_le_bytes(self.data[*pos..*pos+4].try_into().unwrap()); *pos += 4; v }
    fn read_u64(&self, pos: &mut usize) -> u64 { let v = u64::from_le_bytes(self.data[*pos..*pos+8].try_into().unwrap()); *pos += 8; v }
    fn read_i64(&self, pos: &mut usize) -> i64 { let v = i64::from_le_bytes(self.data[*pos..*pos+8].try_into().unwrap()); *pos += 8; v }
    fn read_f32(&self, pos: &mut usize) -> f32 { let v = f32::from_le_bytes(self.data[*pos..*pos+4].try_into().unwrap()); *pos += 4; v }
    fn read_f64(&self, pos: &mut usize) -> f64 { let v = f64::from_le_bytes(self.data[*pos..*pos+8].try_into().unwrap()); *pos += 8; v }

    fn read_string(&self, pos: &mut usize) -> String {
        let len = self.read_u64(pos) as usize;
        let s = String::from_utf8_lossy(&self.data[*pos..*pos+len]).to_string();
        *pos += len;
        s
    }

    fn metadata_uint32_insert(&mut self, key: &str, val: u32) {
        self.metadata_int.insert(key.to_string(), val as i64);
    }

    pub fn get_metadata_int(&self, key: &str, default: i64) -> i64 {
        self.metadata_int.get(key).copied().unwrap_or(default)
    }

    pub fn get_metadata_float(&self, key: &str, default: f64) -> f64 {
        self.metadata_float.get(key).copied().unwrap_or(default)
    }

    pub fn get_metadata_str(&self, key: &str, default: &str) -> String {
        self.metadata_str.get(key).cloned().unwrap_or(default.to_string())
    }
}

impl GgufTensorInfo {
    pub fn nelements(&self) -> i64 {
        let mut n = 1i64;
        for &d in &self.dims { n *= d; }
        n
    }
}
