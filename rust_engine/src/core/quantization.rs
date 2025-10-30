use crate::core::types::*;
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Price quantization utilities for performance optimization
pub struct PriceQuantizer {
    scale_factor: i64,
}

impl Default for PriceQuantizer {
    fn default() -> Self {
        Self { scale_factor: 10_000 } // 4 decimal places
    }
}

impl PriceQuantizer {
    /// Create new quantizer with custom scale
    pub fn new(scale: i64) -> Self {
        Self { scale_factor: scale }
    }

    /// Quantize price to integer for fast computation
    pub fn quantize(&self, price: Price) -> QPrice {
        let scaled = price * Decimal::new(self.scale_factor, 0);
        let quantized = scaled.round().to_i32().unwrap_or(0);
        QPrice(quantized)
    }

    /// Dequantize back to full precision
    pub fn dequantize(&self, qprice: QPrice) -> Price {
        Decimal::new(qprice.0 as i64, 0) / Decimal::new(self.scale_factor, 0)
    }

    /// Fast arithmetic operations on quantized prices
    pub fn add(&self, a: QPrice, b: QPrice) -> QPrice {
        QPrice(a.0 + b.0)
    }

    pub fn subtract(&self, a: QPrice, b: QPrice) -> QPrice {
        QPrice(a.0 - b.0)
    }

    pub fn multiply(&self, a: QPrice, b: QPrice) -> QPrice {
        // Use 64-bit intermediate to avoid overflow
        let result = (a.0 as i64 * b.0 as i64) / self.scale_factor as i64;
        QPrice(result as i32)
    }

    pub fn divide(&self, a: QPrice, b: QPrice) -> QPrice {
        if b.0 == 0 {
            return QPrice(0);
        }
        let result = (a.0 as i64 * self.scale_factor as i64) / b.0 as i64;
        QPrice(result as i32)
    }

    /// Comparison operations
    pub fn greater_than(&self, a: QPrice, b: QPrice) -> bool {
        a.0 > b.0
    }

    pub fn less_than(&self, a: QPrice, b: QPrice) -> bool {
        a.0 < b.0
    }

    pub fn manhattan_distance(&self, a: QPrice, b: QPrice) -> QPrice {
        QPrice((a.0 - b.0).abs())
    }

    /// Percentage change calculation
    pub fn percentage_change(&self, old_price: QPrice, new_price: QPrice) -> QPrice {
        if old_price.0 == 0 {
            return QPrice(0);
        }
        let change = ((new_price.0 as i64 - old_price.0 as i64) * self.scale_factor as i64 * 100)
                    / old_price.0 as i64;
        QPrice(change as i32)
    }
}

/// Signal quantization for confidence values
pub struct SignalQuantizer {
    confidence_scale: f32,
}

impl Default for SignalQuantizer {
    fn default() -> Self {
        Self { confidence_scale: 255.0 } // 8-bit quantization
    }
}

impl SignalQuantizer {
    /// Quantize confidence to 8-bit value
    pub fn quantize_confidence(&self, confidence: f32) -> u8 {
        (confidence.clamp(0.0, 1.0) * self.confidence_scale) as u8
    }

    /// Dequantize confidence back to float
    pub fn dequantize_confidence(&self, qconfidence: u8) -> f32 {
        qconfidence as f32 / self.confidence_scale
    }

    /// Signal strength calculation
    pub fn signal_strength(&self, confidence: u8, direction: i8) -> u8 {
        let abs_direction = direction.abs() as u8;
        (confidence as u16 * abs_direction as u16 / 3) as u8
    }
}

/// Volume quantization utilities
pub struct VolumeQuantizer {
    max_volume: i32,
}

impl Default for VolumeQuantizer {
    fn default() -> Self {
        Self { max_volume: 10_000_000 } // 10M shares max
    }
}

impl VolumeQuantizer {
    /// Quantize volume to 16-bit value
    pub fn quantize(&self, volume: Quantity) -> QQuantity {
        let vol_float = volume.to_f64().unwrap_or(0.0);
        if vol_float <= 0.0 {
            return QQuantity(0);
        }
        if vol_float >= self.max_volume as f64 {
            return QQuantity(i16::MAX);
        }

        // Use logarithmic quantization for large volumes
        if vol_float > 10_000.0 {
            let log_vol = (vol_float.ln() * 1000.0) as i16;
            QQuantity(log_vol.min(i16::MAX as i16))
        } else {
            QQuantity(vol_float as i16)
        }
    }

    /// Dequantize volume back to full precision
    pub fn dequantize(&self, qvolume: QQuantity) -> Quantity {
        if qvolume.0 <= 0 {
            return Decimal::ZERO;
        }
        if qvolume.0 >= i16::MAX {
            return Decimal::from(self.max_volume);
        }

        // Reverse logarithmic quantization
        if qvolume.0 > 10_000 {
            let exp_vol = ((qvolume.0 as f64) / 1000.0).exp();
            Decimal::from_f64(exp_vol).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::from(qvolume.0)
        }
    }

    /// Volume-weighted average price
    pub fn volume_weighted_price(&self, price1: QPrice, vol1: QQuantity,
                               price2: QPrice, vol2: QQuantity) -> QPrice {
        let total_vol = vol1.0 as i32 + vol2.0 as i32;
        if total_vol == 0 {
            return QPrice(0);
        }

        let weighted_sum = price1.0 as i64 * vol1.0 as i64 + price2.0 as i64 * vol2.0 as i64;
        QPrice((weighted_sum / total_vol as i64) as i32)
    }
}

/// Fast approximation functions for expensive operations
pub struct FastApproximations;

impl FastApproximations {
    /// Fast exponential approximation
    pub fn fast_exp(x: f64) -> f64 {
        // Approximation: exp(x) ≈ (1 + x/1024)^1024
        if x.abs() < 1.0 {
            1.0 + x + x * x * 0.5 + x * x * x * 0.16666666666666666
        } else if x > 0.0 {
            2.718281828459045
        } else {
            0.36787944117144233
        }
    }

    /// Fast logarithm approximation
    pub fn fast_log(x: f64) -> f64 {
        if x <= 0.0 {
            f64::NEG_INFINITY
        } else if x < 1.0 {
            // For x < 1, use log(x) = -log(1/x)
            -Self::fast_log(1.0 / x)
        } else {
            // Approximation: log(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3
            let y = x - 1.0;
            y - y * y * 0.5 + y * y * y * 0.3333333333333333
        }
    }

    /// Fast square root using Newton's method approximation
    pub fn fast_sqrt(x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            let mut guess = x * 0.5;
            for _ in 0..3 {  // 3 iterations for good accuracy
                guess = (guess + x / guess) * 0.5;
            }
            guess
        }
    }

    /// Fast normal CDF approximation for risk calculations
    pub fn fast_normal_cdf(x: f64) -> f64 {
        // Abramowitz & Stegun approximation
        let a1 = 0.886226899;
        let a2 = -1.645349621;
        let a3 = 0.914624893;
        let a4 = -0.140543331;

        let y = 1.0 / (1.0 + 0.2316419 * x.abs());
        let z = Self::fast_exp(-x * x * 0.5);

        let cdf = 1.0 - z * (a1 * y + a2 * y * y + a3 * y * y * y + a4 * y * y * y * y);

        if x > 0.0 { cdf } else { 1.0 - cdf }
    }
}

/// Fast variance calculation using Welford's online algorithm
pub struct FastVariance {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Default for FastVariance {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }
}

impl FastVariance {
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    pub fn std_dev(&self) -> f64 {
        Self::fast_sqrt(self.variance())
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Lookup tables for expensive computations
pub struct LookupTables {
    exp_table: Vec<f64>,
    log_table: Vec<f64>,
    sqrt_table: Vec<f64>,
    exp_min: f64,
    exp_max: f64,
    log_min: f64,
    log_max: f64,
    sqrt_max: f64,
    table_size: usize,
}

impl LookupTables {
    pub fn new(table_size: usize) -> Self {
        let mut exp_table = Vec::with_capacity(table_size);
        let mut log_table = Vec::with_capacity(table_size);
        let mut sqrt_table = Vec::with_capacity(table_size);

        let exp_min = -5.0;
        let exp_max = 5.0;
        let log_min = 0.001;
        let log_max = 1000.0;
        let sqrt_max = 1000.0;

        // Initialize lookup tables
        for i in 0..table_size {
            let t = i as f64 / (table_size - 1) as f64;

            // Exponential table
            let x = exp_min + (exp_max - exp_min) * t;
            exp_table.push(FastApproximations::fast_exp(x));

            // Logarithm table
            let x = log_min + (log_max - log_min) * t;
            log_table.push(FastApproximations::fast_log(x));

            // Square root table
            let x = sqrt_max * t;
            sqrt_table.push(FastApproximations::fast_sqrt(x));
        }

        Self {
            exp_table,
            log_table,
            sqrt_table,
            exp_min,
            exp_max,
            log_min,
            log_max,
            sqrt_max,
            table_size,
        }
    }

    pub fn lookup_exp(&self, x: f64) -> f64 {
        if x < self.exp_min {
            FastApproximations::fast_exp(self.exp_min)
        } else if x > self.exp_max {
            FastApproximations::fast_exp(self.exp_max)
        } else {
            let index = ((x - self.exp_min) / (self.exp_max - self.exp_min)
                        * (self.table_size - 1) as f64) as usize;
            let index = index.min(self.table_size - 1);
            self.exp_table[index]
        }
    }

    pub fn lookup_log(&self, x: f64) -> f64 {
        if x < self.log_min {
            FastApproximations::fast_log(self.log_min)
        } else if x > self.log_max {
            FastApproximations::fast_log(self.log_max)
        } else {
            let index = ((x - self.log_min) / (self.log_max - self.log_min)
                        * (self.table_size - 1) as f64) as usize;
            let index = index.min(self.table_size - 1);
            self.log_table[index]
        }
    }

    pub fn lookup_sqrt(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x > self.sqrt_max {
            FastApproximations::fast_sqrt(x)
        } else {
            let index = (x / self.sqrt_max * (self.table_size - 1) as f64) as usize;
            let index = index.min(self.table_size - 1);
            self.sqrt_table[index]
        }
    }
}

impl Default for LookupTables {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Global lookup table instance (lazy static)
use std::sync::OnceLock;
static LOOKUP_TABLES: OnceLock<LookupTables> = OnceLock::new();

pub fn get_lookup_tables() -> &'static LookupTables {
    LOOKUP_TABLES.get_or_init(|| LookupTables::default())
}
