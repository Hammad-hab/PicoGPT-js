import * as tf from "@tensorflow/tfjs";

const gelu = (x) =>
  tf.mul(
    0.5,
    tf.mul(x, tf.add(1, tf.tanh(tf.mul(Math.sqrt(2 / Math.PI), x))))
  );

const softmax = (x) => tf.softmax(x, -1);

function variance(x) {
  return x.sub(x.mean(-1, true)).square().mean(-1, true);
}

const LayerNorm = ({ x, g, b, eps = 1e-5 }) => {
  const mean = tf.mean(x, -1, true);
  const v = variance(x);
  return tf.add(tf.mul(g, tf.div(tf.sub(x, mean), tf.sqrt(tf.add(v, eps)))), b);
};

const Linear = ({ x, w, b }) => {
    const bT = b instanceof tf.Tensor ? b : tf.tensor(b);
    const outDim = bT.shape[0];
  
    // Reshape flat w → [inDim, outDim]
    const wT = w instanceof tf.Tensor
      ? w
      : tf.tensor(w).reshape([-1, outDim]);  // infers inDim automatically
  
    const xT = x.shape.length === 1 ? x.expandDims(0) : x;
  
    return tf.matMul(xT, wT).add(bT);
  };
// Fix 3: pass {x, ...l2args} as a destructured object
const FeedForward = ({ x, l1args, l2args }) => {
  return Linear({ x: gelu(Linear({ x, ...l1args })), ...l2args });
};

const Attention = ({ q, k, v, mask }) => {
  // Fix 4: use q.shape.length - 1 for last dim
  const depth = q.shape[q.shape.length - 1];
  const scores = tf.add(
    tf.div(tf.matMul(q, tf.transpose(k)), Math.sqrt(depth)),
    mask
  );
  return tf.matMul(softmax(scores), v);
};

const MultiHeadAttention = (x, cAttn, cProj, nHead) => {
    let xA = Linear({ x, w: cAttn.w, b: cAttn.b });
  
    const seqLen = xA.shape[0];
    const totalDim = xA.shape[1];
    const nEmbd = totalDim / 3;
    const headDim = nEmbd / nHead;
  
    const [q, k, v] = tf.split(xA, 3, 1);

    const qHeads = tf.split(q, nHead, 1); // [seq_len, headDim] x nHead
    const kHeads = tf.split(k, nHead, 1);
    const vHeads = tf.split(v, nHead, 1);
  
    const outHeads = [];
    const mask = tf.mul(
      tf.sub(1, tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0)),
      -1e10
    );
  
    for (let i = 0; i < nHead; i++) {
      outHeads.push(Attention({ q: qHeads[i], k: kHeads[i], v: vHeads[i], mask }));
    }
  
    return Linear({ x: tf.concat(outHeads, 1), w: cProj.w, b: cProj.b });
  };

const TransformerBlock = ({ x, mlp, attn, ln1, ln2, nHead }) => {
  
    x = tf.add(
      x,
      MultiHeadAttention(
        LayerNorm({ x, ...ln1 }),
        attn.c_attn,   // { w, b }
        attn.c_proj,   // { w, b }
        nHead
      )
    );
    x = tf.add(
      x,
      FeedForward({
        x: LayerNorm({ x, ...ln2 }),
        l1args: mlp.c_fc,    // { w, b }
        l2args: mlp.c_proj,  // { w, b }
      })
    );
    return x;
  };

const GPT2 = (inputs, wte, wpe, blocks, ln_f, nHead) => {
    const inputTensor = tf.tensor1d(inputs, "int32");
    const positions = tf.range(0, inputs.length, 1, "int32");
  
    // wte is flat Float32Array of length [vocab_size * n_embd]
    // wpe is flat Float32Array of length [n_ctx * n_embd]
    const nEmbd = ln_f.g.length;  // e.g. 1280 for xl — derive from ln_f.g which we know is shaped correctly
    const vocabSize = wte.length / nEmbd;
    const nCtx = wpe.length / nEmbd;
  
    const wteT = tf.tensor(wte).reshape([vocabSize, nEmbd]);
    const wpeT = tf.tensor(wpe).reshape([nCtx, nEmbd]);
  
    let x = tf.add(tf.gather(wteT, inputTensor), tf.gather(wpeT, positions));
  
    for (const block of blocks) {
      x = TransformerBlock({
        x,
        nHead,
        mlp:  block.mlp,
        attn: block.attn,
        ln1:  block.ln_1,
        ln2:  block.ln_2,
      });
    }
  
    // Final layernorm then project back to vocab
    const logits = tf.matMul(LayerNorm({ x, g: ln_f.g, b: ln_f.b }), tf.transpose(wteT));
    return logits;
};

export default GPT2;