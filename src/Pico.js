import * as tf from "@tensorflow/tfjs"

const gelu = (x) => 0.5 * x * (1 + tf.tanh((2/Math.sqrt(Math.PI)) * (x + 0.044715 * Math.pow(x, 3))) )
const softmax = (x) => {
    const n = tf.exp(x - tf.max(x, -1, true))
    const d = tf.sum(n, -1, true)
    return n/d
}

function variance(x) {
    const t = tf.tensor1d(x);
    return t.sub(t.mean()).square().mean();
}

const LayerNorm = (x, g, b, eps=1e-5) => {
    const X = tf.mean(x, -1, true)
    const V = variance(x)
    return g * (x - X) / tf.sqrt(V + eps) + b
}

const Linear = (x, w, b) => {
    return tf.matMul(x, w) + b
}

const FeedForward = (x, l1args, l2args) => {
    return Linear(gelu(Linear(x, ...flayr)), ...slayr)
}
const Attention = (q, k, v, mask) => {
    const sfmx = softmax(tf.matMul(q, tf.transpose(k))/tf.sqrt(q.shape[-1]) + mask)
    return tf.matMul(sfmx, v)
}
const MultiHeadAttention = (x, cAttn, cProj, nHead) => {
    let xA = Linear(x, ...cAttn)
    const qKVHeads = tf.split(xA, 3, -1).map(x => tf.split(x, nHead, -1))
    const outHead = []
    const n = x.shape[0]
    const mask  = (1 - tf.linalg.bandPart(tf.ones([n, n]), -1, 0)) * -1e10
    for (const [q, k, v] of qKVHeads[0].map((_, i) => qKVHeads.map(qkv => qkv[i]))) {
        const at = Attention(q, k, v, mask)
        outHead.push(at)
    }
    xA = Linear(tf.concat(outHead), ...cProj)
}
const TransformerBlock = (x, mlp, attn, ln1, ln2, nHead) => {
    x = x + MultiHeadAttention(LayerNorm(x, ...ln1), ...attn, nHead)
    x = x + FeedForward(LayerNorm(x, ...ln2), ...mlp)
    return x
}
const GPT2 = (inputs, wte, wpe, blocks, ln_f, nHead) => {
    x = wte[inputs] + wpe.slice(0, inputs.length)
    for (const block of blocks) {
        x = TransformerBlock(x, ...block, nHead)
    }
    return tf.matMul(LayerNorm(x, ...ln_f), wte.transpose())
}

const generate = () => {}
