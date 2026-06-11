import fs from "node:fs";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";
import { execSync } from "node:child_process";
import path from "node:path";
import { getEncoderFromFiles } from "./Encoder.js";

const MODEL_SIZES = ["124M", "355M", "774M", "1558M"];
const MODEL_FILES = [
  "checkpoint",
  "encoder.json",
  "hparams.json",
  "model.ckpt.data-00000-of-00001",
  "model.ckpt.index",
  "model.ckpt.meta",
  "vocab.bpe",
];
const ENDPOINT = "https://openaipublic.blob.core.windows.net/gpt-2/models";

/**
 * Download all GPT-2 model files into model_dir, waiting for each to finish.
 */
const download_gpt2_files = async (model_size, model_dir) => {
  if (!MODEL_SIZES.includes(model_size)) {
    throw new Error(`Invalid model size "${model_size}". Valid: ${MODEL_SIZES.join(", ")}`);
  }

  for (const file of MODEL_FILES) {
    const filepath = path.join(model_dir, file);
    console.log(`Downloading ${file}...`);
    const res = await fetch(`${ENDPOINT}/${model_size}/${file}`);
    if (!res.ok) throw new Error(`Failed to fetch ${file}: ${res.status} ${res.statusText}`);
    // Await the full stream so the next iteration only starts after this file lands
    await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(filepath));
  }
};

const set_in_nested_dict = (d, keys, val) => {
  if (keys.length === 0) return val;           // was: if (!keys) — always falsy for []
  if (!(keys[0] in d)) d[keys[0]] = {};
  d[keys[0]] = set_in_nested_dict(d[keys[0]], keys.slice(1), val);
  return d;
};


/**
 * Convert a TF checkpoint into a single weights.npz file via Python/TF.
 * @param {string} modelDir  - directory containing the checkpoint (NOT the .ckpt path)
 * @param {string} outputPath - where to write weights.npz
 */
const _tf_conv_weights = (modelDir, outputPath) => {
  console.log("Converting weights, this may take a minute...");
  // Escape any single-quotes in paths to avoid breaking the python string literals
  const safeDir = modelDir.replace(/'/g, "\\'");
  const safeOut = outputPath.replace(/'/g, "\\'");
  execSync(`python3 -c "
import tensorflow as tf
import numpy as np

def save_weights_to_npz(model_dir, output_path):
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    if ckpt_path is None:
        raise ValueError(f'No checkpoint found in: {model_dir}')
    weights = {}
    for name, _ in tf.train.list_variables(ckpt_path):
        array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        weights[name.replace('/', '_')] = array
    np.savez('${safeOut}', **weights)

save_weights_to_npz('${safeDir}', '${safeOut}')
"`, { stdio: "inherit" });
  console.log("Conversion successful");
};

/**
 * List variables in a TF checkpoint, returning [[name, shape], ...].
 * @param {string} ckptPath - full path to the .ckpt prefix
 */
const tf_train_list_variables = (ckptPath) => {
  const safe = ckptPath.replace(/'/g, "\\'");
  const output = execSync(`python3 -c "
import tensorflow as tf, json
vars = [[n, s] for n, s in tf.train.list_variables('${safe}')]
print(json.dumps(vars))
"`).toString();
  return JSON.parse(output);
};

/**
 * Parse a single .npy buffer into a TypedArray.
 * .npy format: magic (6 bytes) + major/minor version (2 bytes) +
 *              header_len (2 or 4 bytes, LE) + header (JSON-like) + data.
 * By Claude
 * @param {Buffer} buf
 * @returns {{ data: Float32Array|Float64Array|Int32Array|..., shape: number[], dtype: string }}
 */
const parse_npy = (buf) => {
  // Magic: \x93NUMPY
  if (buf[0] !== 0x93 || buf.slice(1, 6).toString() !== "NUMPY") {
    throw new Error("Not a valid .npy file");
  }
  const major = buf[6];
  const headerLenBytes = major >= 2 ? 4 : 2;
  const headerLen = major >= 2
    ? buf.readUInt32LE(8)
    : buf.readUInt16LE(8);
  const headerStart = 6 + 2 + headerLenBytes;  // magic+version+headerLen field
  const header = buf.slice(headerStart, headerStart + headerLen).toString("utf8");

  // Parse shape and dtype from the Python-literal header dict
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const dtypeMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
  const orderMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);

  const shape = shapeMatch[1].trim()
    ? shapeMatch[1].split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n))
    : [];
  const dtype = dtypeMatch[1];
  const fortran = orderMatch[1] === "True";
  if (fortran) throw new Error("Fortran-order arrays are not supported");

  const dataOffset = headerStart + headerLen;
  const dataBuffer = buf.slice(dataOffset);

  const TypedArrayMap = {
    "<f4": Float32Array, "|f4": Float32Array,
    "<f8": Float64Array, "|f8": Float64Array,
    "<i4": Int32Array,   "|i4": Int32Array,
    "<i8": BigInt64Array,
    "<u4": Uint32Array,  "|u4": Uint32Array,
    "<u1": Uint8Array,   "|u1": Uint8Array,
    "<b1": Uint8Array,   // bool
  };
  const TypedArray = TypedArrayMap[dtype];
  if (!TypedArray) throw new Error(`Unsupported dtype: ${dtype}`);

  const data = new TypedArray(
    dataBuffer.buffer,
    dataBuffer.byteOffset,
    dataBuffer.byteLength / (TypedArray.BYTES_PER_ELEMENT ?? 1)
  );
  return { data, shape, dtype };
};

/**
 * Load a .npz file (ZIP of .npy entries) into a plain object.
 * Shells out to Python once to re-save as raw .npy files, then parses
 * them natively in Node — no npyjs, no file:// URL issues.
 *
 * @param {string} npzPath
 * @returns {Promise<Record<string, Float32Array>>}
 */
const load_npz = async (npzPath) => {
  const tmpDir  = `${npzPath}_extracted`;
  const safeNpz = npzPath.replace(/'/g, "\\'");
  const safeTmp = tmpDir.replace(/'/g, "\\'");

  fs.mkdirSync(tmpDir, { recursive: true });

  execSync(`python3 -c "
import numpy as np, os
data = np.load('${safeNpz}')
for key in data.files:
    np.save(os.path.join('${safeTmp}', key + '.npy'), data[key])
"`, { stdio: "inherit" });

  const result = {};
  for (const entry of fs.readdirSync(tmpDir)) {
    if (!entry.endsWith(".npy")) continue;
    const key = entry.slice(0, -4);
    const buf = fs.readFileSync(path.join(tmpDir, entry));  // plain fs read, no fetch
    result[key] = parse_npy(buf).data;
  }

  fs.rmSync(tmpDir, { recursive: true, force: true });
  return result;
};

/**
 * Loads (and caches) all weights from the npz alongside the checkpoint.
 * @param {string} ckptPath  - full path to .ckpt prefix, e.g. models/774M/model.ckpt
 * @returns {Promise<Record<string, Float32Array>>}
 */
const load_weight_file = (() => {
  // Per-model-dir cache so loading two models works correctly
  const cache = new Map();
  return async (ckptPath) => {
    const modelDir  = path.dirname(ckptPath);
    const npzPath   = path.join(modelDir, "weights.npz");
    if (cache.has(modelDir)) return cache.get(modelDir);
    if (!fs.existsSync(npzPath)) _tf_conv_weights(modelDir, npzPath);
    const weights = await load_npz(npzPath);
    cache.set(modelDir, weights);
    return weights;
  };
})();

/**
 * Load a single named variable from the checkpoint weight file.
 */
const tf_train_load_variable = async (ckptPath, name) => {
  const weights = await load_weight_file(ckptPath);
  const key = name.replace(/\//g, "_");
  if (!(key in weights)) throw new Error(`Weight not found: "${key}"`);
  return weights[key];
};


const load_gpt2_params_from_tf_ckpt = async (ckptPath, hparams) => {
  const n_layers = hparams["n_layer"];
  const params = { blocks: Array.from({ length: n_layers }, () => ({})) };

  for (const [lbl] of tf_train_list_variables(ckptPath)) {
    const arr  = await tf_train_load_variable(ckptPath, lbl);
    const name = lbl.slice("model/".length);

    if (name.startsWith("h")) {
      const m = name.match(/^h(\d+)\/(.*)/);
      if (!m) continue;
      const n       = Number(m[1]);
      const subName = m[2];
      set_in_nested_dict(params.blocks[n], subName.split("/"), arr);
    } else {
      set_in_nested_dict(params, name.split("/"), arr);
    }
  }

  return params;
};


function latestCheckpoint(modelDir) {
  const checkpointFile = path.join(modelDir, "checkpoint");
  if (!fs.existsSync(checkpointFile)) return null;
  const content = fs.readFileSync(checkpointFile, "utf8");
  const match   = content.match(/model_checkpoint_path:\s*"([^"]+)"/);
  if (!match) return null;
  const ckptPath = match[1];
  return path.isAbsolute(ckptPath) ? ckptPath : path.join(modelDir, ckptPath);
}


const load_encoder_hparams_and_params = async (model_size, models_dir) => {
  if (!MODEL_SIZES.includes(model_size)) {
    throw new Error(`Invalid model size "${model_size}". Valid: ${MODEL_SIZES.join(", ")}`);
  }

  const model_dir = path.join(models_dir, model_size);

  let ckpt_path = latestCheckpoint(model_dir);
  if (!ckpt_path) {
    fs.mkdirSync(model_dir, { recursive: true });
    await download_gpt2_files(model_size, model_dir);
    ckpt_path = latestCheckpoint(model_dir);
    if (!ckpt_path) throw new Error("Download succeeded but no checkpoint found.");
  }

  const encoder = await getEncoderFromFiles(model_size, models_dir);
  const hparams  = JSON.parse(fs.readFileSync(path.join(model_dir, "hparams.json"), "utf-8"));
  const params   = await load_gpt2_params_from_tf_ckpt(ckpt_path, hparams);

  return { encoder, hparams, params };
};


const config = {
  model_size: "774M",
  models_dir: "models",
  n_tokens_to_generate: 40,
};

export { download_gpt2_files, load_encoder_hparams_and_params, config };