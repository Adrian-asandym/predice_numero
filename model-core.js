"use strict";

const DigitModel = (() => {
  const EXPECTED_FLOATS = 179926;

  function parseWeights(buffer) {
    const data = new Float32Array(buffer);
    if (data.length !== EXPECTED_FLOATS) {
      throw new Error("Tamaño de pesos inesperado: " + data.length);
    }
    let offset = 0;
    const take = (n) => {
      const out = data.subarray(offset, offset + n);
      offset += n;
      return out;
    };
    return {
      conv1Kernel: take(3 * 3 * 1 * 32),
      conv1Bias: take(32),
      conv2Kernel: take(3 * 3 * 32 * 64),
      conv2Bias: take(64),
      dense1Kernel: take(1600 * 100),
      dense1Bias: take(100),
      dense2Kernel: take(100 * 10),
      dense2Bias: take(10),
    };
  }

  function conv2dValid(input, inH, inW, inC, kernel, bias, outC) {
    const outH = inH - 2;
    const outW = inW - 2;
    const out = new Float32Array(outH * outW * outC);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        for (let oc = 0; oc < outC; oc++) {
          let sum = bias[oc];
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              const inBase = ((y + ky) * inW + (x + kx)) * inC;
              const kBase = (ky * 3 + kx) * inC * outC + oc;
              for (let ic = 0; ic < inC; ic++) {
                sum += input[inBase + ic] * kernel[kBase + ic * outC];
              }
            }
          }
          out[(y * outW + x) * outC + oc] = sum > 0 ? sum : 0;
        }
      }
    }
    return { data: out, h: outH, w: outW, c: outC };
  }

  function maxPool2x2(input, inH, inW, inC) {
    const outH = inH >> 1;
    const outW = inW >> 1;
    const out = new Float32Array(outH * outW * inC);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < outW; x++) {
        for (let c = 0; c < inC; c++) {
          const i0 = ((y * 2) * inW + x * 2) * inC + c;
          const i1 = ((y * 2) * inW + x * 2 + 1) * inC + c;
          const i2 = ((y * 2 + 1) * inW + x * 2) * inC + c;
          const i3 = ((y * 2 + 1) * inW + x * 2 + 1) * inC + c;
          out[(y * outW + x) * inC + c] = Math.max(input[i0], input[i1], input[i2], input[i3]);
        }
      }
    }
    return { data: out, h: outH, w: outW, c: inC };
  }

  function dense(input, kernel, bias, inSize, outSize, relu) {
    const out = new Float32Array(outSize);
    for (let o = 0; o < outSize; o++) {
      let sum = bias[o];
      for (let i = 0; i < inSize; i++) {
        sum += input[i] * kernel[i * outSize + o];
      }
      out[o] = relu ? (sum > 0 ? sum : 0) : sum;
    }
    return out;
  }

  function softmax(logits) {
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    let total = 0;
    const out = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      out[i] = Math.exp(logits[i] - max);
      total += out[i];
    }
    for (let i = 0; i < out.length; i++) out[i] /= total;
    return out;
  }

  function predict(weights, pixels28) {
    const conv1 = conv2dValid(pixels28, 28, 28, 1, weights.conv1Kernel, weights.conv1Bias, 32);
    const pool1 = maxPool2x2(conv1.data, conv1.h, conv1.w, conv1.c);
    const conv2 = conv2dValid(pool1.data, pool1.h, pool1.w, pool1.c, weights.conv2Kernel, weights.conv2Bias, 64);
    const pool2 = maxPool2x2(conv2.data, conv2.h, conv2.w, conv2.c);
    const hidden = dense(pool2.data, weights.dense1Kernel, weights.dense1Bias, 1600, 100, true);
    const logits = dense(hidden, weights.dense2Kernel, weights.dense2Bias, 100, 10, false);
    return softmax(logits);
  }

  function preprocess(sourceData, srcSize, outSize) {
    let minX = srcSize, minY = srcSize, maxX = -1, maxY = -1;
    for (let y = 0; y < srcSize; y++) {
      for (let x = 0; x < srcSize; x++) {
        if (sourceData[(y * srcSize + x) * 4] > 10) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    const scale = 20 / Math.max(boxW, boxH);
    const newW = Math.max(1, Math.round(boxW * scale));
    const newH = Math.max(1, Math.round(boxH * scale));

    const small = new Float32Array(newW * newH);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const srcX = minX + Math.min(boxW - 1, Math.floor(x / scale));
        const srcY = minY + Math.min(boxH - 1, Math.floor(y / scale));
        small[y * newW + x] = sourceData[(srcY * srcSize + srcX) * 4] / 255;
      }
    }

    let mass = 0, comX = 0, comY = 0;
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const v = small[y * newW + x];
        mass += v;
        comX += x * v;
        comY += y * v;
      }
    }
    comX = mass > 0 ? comX / mass : (newW - 1) / 2;
    comY = mass > 0 ? comY / mass : (newH - 1) / 2;

    const out = new Float32Array(outSize * outSize);
    const offX = Math.round(outSize / 2 - 0.5 - comX);
    const offY = Math.round(outSize / 2 - 0.5 - comY);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const dx = x + offX;
        const dy = y + offY;
        if (dx >= 0 && dx < outSize && dy >= 0 && dy < outSize) {
          out[dy * outSize + dx] = small[y * newW + x];
        }
      }
    }
    return out;
  }

  const api = { parseWeights, predict, preprocess, EXPECTED_FLOATS };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  return api;
})();
