package quantization

import (
	"go-trading-engine/pkg/types"
	"math"
)

// PriceQuantizer handles fast price arithmetic with quantization
type PriceQuantizer struct {
	scaleFactor int64
}

// NewPriceQuantizer creates a new price quantizer
func NewPriceQuantizer(scale int64) *PriceQuantizer {
	return &PriceQuantizer{scaleFactor: scale}
}

// Quantize converts Price to QPrice for fast computation
func (pq *PriceQuantizer) Quantize(price types.Price) types.QPrice {
	scaled := price.Mul(types.DecimalFromFloat(float64(pq.scaleFactor)))
	intVal, _ := scaled.Int64()
	return types.QPrice(int32(intVal))
}

// Dequantize converts QPrice back to Price
func (pq *PriceQuantizer) Dequantize(qprice types.QPrice) types.Price {
	return types.DecimalFromFloat(float64(qprice)).Div(types.DecimalFromFloat(float64(pq.scaleFactor)))
}

// Fast arithmetic operations on quantized prices
func (pq *PriceQuantizer) Add(a, b types.QPrice) types.QPrice {
	return types.QPrice(int32(a) + int32(b))
}

func (pq *PriceQuantizer) Subtract(a, b types.QPrice) types.QPrice {
	return types.QPrice(int32(a) - int32(b))
}

func (pq *PriceQuantizer) Multiply(a, b types.QPrice) types.QPrice {
	result := int64(a) * int64(b) / pq.scaleFactor
	return types.QPrice(int32(result))
}

func (pq *PriceQuantizer) Divide(a, b types.QPrice) types.QPrice {
	if b == 0 {
		return 0
	}
	result := int64(a) * pq.scaleFactor / int64(b)
	return types.QPrice(int32(result))
}

// Comparison operations
func (pq *PriceQuantizer) GreaterThan(a, b types.QPrice) bool {
	return int32(a) > int32(b)
}

func (pq *PriceQuantizer) LessThan(a, b types.QPrice) bool {
	return int32(a) < int32(b)
}

func (pq *PriceQuantizer) ManhattanDistance(a, b types.QPrice) types.QPrice {
	diff := int32(a) - int32(b)
	if diff < 0 {
		return types.QPrice(-diff)
	}
	return types.QPrice(diff)
}

// Percentage change calculation
func (pq *PriceQuantizer) PercentageChange(oldPrice, newPrice types.QPrice) types.QPrice {
	if oldPrice == 0 {
		return 0
	}
	change := int64(newPrice-oldPrice) * pq.scaleFactor * 100 / int64(oldPrice)
	return types.QPrice(int32(change))
}

// Fast approximation functions for expensive operations
type FastApproximations struct{}

// Fast exponential approximation
func (fa FastApproximations) FastExp(x float64) float64 {
	if math.Abs(x) < 1.0 {
		return 1.0 + x + x*x*0.5 + x*x*x*0.16666666666666666
	} else if x > 0.0 {
		return 2.718281828459045
	} else {
		return 0.36787944117144233
	}
}

// Fast logarithm approximation
func (fa FastApproximations) FastLog(x float64) float64 {
	if x <= 0.0 {
		return math.Inf(-1)
	} else if x < 1.0 {
		return -fa.FastLog(1.0 / x)
	} else {
		y := x - 1.0
		return y - y*y*0.5 + y*y*y*0.3333333333333333
	}
}

// Fast square root using Newton's method approximation
func (fa FastApproximations) FastSqrt(x float64) float64 {
	if x <= 0.0 {
		return 0.0
	} else {
		guess := x * 0.5
		for i := 0; i < 3; i++ {
			guess = (guess + x/guess) * 0.5
		}
		return guess
	}
}

// Fast normal CDF approximation for risk calculations
func (fa FastApproximations) FastNormalCDF(x float64) float64 {
	a1 := 0.886226899
	a2 := -1.645349621
	a3 := 0.914624893
	a4 := -1.403402435

	y := 1.0 / (1.0 + 0.2316419*math.Abs(x))
	z := fa.FastExp(-x*x*0.5)

	cdf := 1.0 - z*(a1*y+a2*y*y+a3*y*y*y+a4*y*y*y*y)

	if x > 0.0 {
		return cdf
	} else {
		return 1.0 - cdf
	}
}

// Fast variance calculation using Welford's online algorithm
type FastVariance struct {
	count   uint64
	mean    float64
	m2      float64
}

// Update adds a value to the variance calculation
func (fv *FastVariance) Update(value float64) {
	fv.count++
	delta := value - fv.mean
	fv.mean += delta / float64(fv.count)
	delta2 := value - fv.mean
	fv.m2 += delta * delta2
}

// Variance returns the current variance
func (fv *FastVariance) Variance() float64 {
	if fv.count > 1 {
		return fv.m2 / float64(fv.count-1)
	}
	return 0.0
}

// StdDev returns the standard deviation
func (fv *FastVariance) StdDev() float64 {
	return FastApproximations{}.FastSqrt(fv.Variance())
}

// Reset resets the variance calculation
func (fv *FastVariance) Reset() {
	fv.count = 0
	fv.mean = 0.0
	fv.m2 = 0.0
}

// Lookup tables for expensive computations
type LookupTables struct {
	expTable   []float64
	logTable   []float64
	sqrtTable  []float64
	expMin     float64
	expMax     float64
	logMin     float64
	logMax     float64
	sqrtMax    float64
	tableSize  int
}

var globalLookupTables *LookupTables

// NewLookupTables creates lookup tables for expensive computations
func NewLookupTables(tableSize int) *LookupTables {
	lt := &LookupTables{
		expTable:  make([]float64, tableSize),
		logTable:  make([]float64, tableSize),
		sqrtTable: make([]float64, tableSize),
		expMin:    -5.0,
		expMax:    5.0,
		logMin:    0.001,
		logMax:    1000.0,
		sqrtMax:   1000.0,
		tableSize: tableSize,
	}

	fa := FastApproximations{}

	// Initialize lookup tables
	for i := 0; i < tableSize; i++ {
		t := float64(i) / float64(tableSize-1)

		// Exponential table
		x := lt.expMin + (lt.expMax-lt.expMin)*t
		lt.expTable[i] = fa.FastExp(x)

		// Logarithm table
		x = lt.logMin + (lt.logMax-lt.logMin)*t
		lt.logTable[i] = fa.FastLog(x)

		// Square root table
		x = lt.sqrtMax * t
		lt.sqrtTable[i] = fa.FastSqrt(x)
	}

	return lt
}

// GetGlobalLookupTables returns the global lookup tables instance
func GetGlobalLookupTables() *LookupTables {
	if globalLookupTables == nil {
		globalLookupTables = NewLookupTables(1024)
	}
	return globalLookupTables
}

// LookupExp looks up exponential value from table
func (lt *LookupTables) LookupExp(x float64) float64 {
	if x < lt.expMin {
		return FastApproximations{}.FastExp(lt.expMin)
	} else if x > lt.expMax {
		return FastApproximations{}.FastExp(lt.expMax)
	} else {
		index := int((x-lt.expMin)/(lt.expMax-lt.expMin)*float64(lt.tableSize-1))
		if index >= lt.tableSize {
			index = lt.tableSize - 1
		}
		return lt.expTable[index]
	}
}

// LookupLog looks up logarithm value from table
func (lt *LookupTables) LookupLog(x float64) float64 {
	if x < lt.logMin {
		return FastApproximations{}.FastLog(lt.logMin)
	} else if x > lt.logMax {
		return FastApproximations{}.FastLog(lt.logMax)
	} else {
		index := int((x-lt.logMin)/(lt.logMax-lt.logMin)*float64(lt.tableSize-1))
		if index >= lt.tableSize {
			index = lt.tableSize - 1
		}
		return lt.logTable[index]
	}
}

// LookupSqrt looks up square root value from table
func (lt *LookupTables) LookupSqrt(x float64) float64 {
	if x < 0.0 {
		return 0.0
	} else if x > lt.sqrtMax {
		return FastApproximations{}.FastSqrt(x)
	} else {
		index := int(x/lt.sqrtMax*float64(lt.tableSize-1))
		if index >= lt.tableSize {
			index = lt.tableSize - 1
		}
		return lt.sqrtTable[index]
	}
}
