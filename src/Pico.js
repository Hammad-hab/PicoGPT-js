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
    return 
}

const FeedForward = () => {}
const Attention = () => {}
const MultiHeadAttention = () => {}
const TransformerBlock = () => {}
const GPT2 = () => {}

const generate = () => {}
